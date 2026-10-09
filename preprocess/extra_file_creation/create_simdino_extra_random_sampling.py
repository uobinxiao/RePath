import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import h5py
import numpy as np

ALLOWED_MAGNIFICATIONS = {5, 10, 20, 40}
KNOWN_SLIDE_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

@dataclass
class H5Shard:
    mag: int
    path: Path
    slide_filename: str
    n_tiles: int
    target_magnification: int
    patch_size_level0: int


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TRIDENT coordinate H5 files to SimDINOv2 WSIPatch extra files.",
    )
    parser.add_argument(
        "--h5-dir",
        action="append",
        required=True,
        help="Magnification and H5 directory, e.g. 5:/path/to/5x_h5 or 5=/path/to/5x_h5. Repeat per magnification.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where entries-TRAIN.npy/json/schema.json will be written.",
    )
    parser.add_argument(
        "--sample-per-slide-mag",
        type=int,
        default=5000,
        help="Maximum coordinates sampled from each slide at each magnification.",
    )
    parser.add_argument(
        "--slide-extension",
        default=".svs",
        help="Slide extension to append when the H5 name does not already include one.",
    )
    parser.add_argument(
        "--level0-magnification",
        type=int,
        default=40,
        help="Magnification of level 0, used to infer patch_size_level0 when H5 attrs do not provide it.",
    )
    parser.add_argument(
        "--output-patch-size",
        type=int,
        default=256,
        help="Model input patch size used when inferring patch_size_level0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic coordinate sampling.",
    )
    parser.add_argument(
        "--split",
        default="TRAIN",
        help="Split suffix used in output file names. Defaults to TRAIN.",
    )
    parser.add_argument(
        "--h5-pattern",
        default="*.h5",
        help="Glob pattern used to discover H5 files recursively under each H5 directory.",
    )
    return parser.parse_args(argv)


def parse_h5_dirs(values: Iterable[str]) -> Dict[int, Path]:
    mag_dirs: Dict[int, Path] = {}
    for value in values:
        match = re.match(r"^(\d+)[xX]?[:=](.+)$", value)
        if match is None:
            raise ValueError(f'Expected MAG:DIR or MAG=DIR, got "{value}"')

        mag = int(match.group(1))
        validate_magnification(mag)
        if mag in mag_dirs:
            raise ValueError(f"Duplicate H5 directory for {mag}x")

        path = Path(match.group(2)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Magnification directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Magnification path is not a directory: {path}")
        mag_dirs[mag] = path
    return dict(sorted(mag_dirs.items()))


def validate_magnification(mag: int) -> None:
    if mag not in ALLOWED_MAGNIFICATIONS:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_MAGNIFICATIONS))
        raise ValueError(f"Unsupported magnification {mag}; expected one of: {allowed}")


def clean_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def read_h5_attrs(h5: h5py.File) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    sources = [h5.attrs, h5["coords"].attrs]
    if "features" in h5:
        sources.append(h5["features"].attrs)
    for source in sources:
        for key, value in source.items():
            attrs[str(key)] = clean_attr_value(value)
    return attrs


def as_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def strip_patch_suffix(name: str) -> str:
    for suffix in ("_patches", "_coords"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def slide_filename_from_h5(path: Path, attrs: Dict[str, object], slide_extension: str) -> str:
    raw_name = str(
        attrs.get("slide_filename")
        or attrs.get("file_name")
        or attrs.get("slide_id")
        or attrs.get("slide_submitter_id")
        or attrs.get("name")
        or path.stem
    )
    raw_name = Path(raw_name).name
    raw_name = strip_patch_suffix(raw_name)

    if Path(raw_name).suffix.lower() in KNOWN_SLIDE_SUFFIXES:
        return raw_name

    extension = slide_extension if slide_extension.startswith(".") else f".{slide_extension}"
    return f"{raw_name}{extension}"


def infer_patch_size_level0(
    attrs: Dict[str, object],
    target_magnification: int,
    level0_magnification: int,
    output_patch_size: int,
) -> int:
    patch_size_level0 = as_int(attrs.get("patch_size_level0"), 0)
    if patch_size_level0 > 0:
        return patch_size_level0

    patch_size = as_int(attrs.get("patch_size"), output_patch_size)
    level0_mag = as_int(attrs.get("level0_magnification"), level0_magnification)
    if level0_mag <= 0:
        raise ValueError("level0_magnification must be positive when patch_size_level0 is absent")

    return int(round(patch_size * level0_mag / target_magnification))


def inspect_h5(
    path: Path,
    mag: int,
    slide_extension: str,
    level0_magnification: int,
    output_patch_size: int,
) -> Optional[H5Shard]:
    with h5py.File(path, "r") as h5:
        if "coords" not in h5:
            raise ValueError(f"{path} must contain a 'coords' dataset")

        coords = h5["coords"]
        if coords.ndim != 2 or coords.shape[1] != 2:
            return None
            raise ValueError(f"{path}: coords must have shape (N, 2), got {coords.shape}")
        n_tiles = int(coords.shape[0])
        if n_tiles == 0:
            return None

        attrs = read_h5_attrs(h5)

    target_magnification = as_int(attrs.get("target_magnification"), mag)

    validate_magnification(target_magnification)
    patch_size_level0 = infer_patch_size_level0(
        attrs,
        target_magnification=target_magnification,
        level0_magnification=level0_magnification,
        output_patch_size=output_patch_size,
    )
    slide_filename = slide_filename_from_h5(path, attrs, slide_extension)

    return H5Shard(
        mag=mag,
        path=path,
        slide_filename=slide_filename,
        n_tiles=n_tiles,
        target_magnification=target_magnification,
        patch_size_level0=patch_size_level0,
    )


def discover_shards(
    mag_dirs: Dict[int, Path],
    h5_pattern: str,
    slide_extension: str,
    level0_magnification: int,
    output_patch_size: int,
) -> Dict[int, List[H5Shard]]:
    shards_by_mag: Dict[int, List[H5Shard]] = {}
    for mag, directory in mag_dirs.items():
        paths = sorted(directory.rglob(h5_pattern))
        if not paths:
            raise FileNotFoundError(f"No H5 files matching {h5_pattern} under {directory}")

        shards: List[H5Shard] = []
        for path in paths:
            shard = inspect_h5(
                path,
                mag=mag,
                slide_extension=slide_extension,
                level0_magnification=level0_magnification,
                output_patch_size=output_patch_size,
            )
            if shard is not None:
                shards.append(shard)
        if not shards:
            raise ValueError(f"No non-empty coordinate H5 files under {directory}")
        shards_by_mag[mag] = shards
    return shards_by_mag


def selected_count(shards_by_mag: Dict[int, List[H5Shard]], sample_per_slide_mag: int) -> int:
    return sum(min(sample_per_slide_mag, shard.n_tiles) for shards in shards_by_mag.values() for shard in shards)


def write_slide_mapping(shards_by_mag: Dict[int, List[H5Shard]], output_path: Path) -> Dict[str, int]:
    slide_names = sorted({shard.slide_filename for shards in shards_by_mag.values() for shard in shards})
    slide_to_id = {name: idx for idx, name in enumerate(slide_names)}
    id_to_slide = {str(idx): name for name, idx in slide_to_id.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(id_to_slide, f, indent=2)
    return slide_to_id


def write_schema(output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 2,
                "columns": ["x", "y", "slide_id", "target_magnification", "patch_size_level0"],
                "note": "SimDINO WSIPatch reads level-0 regions of patch_size_level0 and resizes to 256.",
            },
            f,
            indent=2,
        )


def sample_indices(n_tiles: int, sample_per_slide_mag: int, rng: np.random.RandomState) -> np.ndarray:
    if n_tiles <= sample_per_slide_mag:
        return np.arange(n_tiles, dtype=np.int64)
    return np.sort(rng.choice(n_tiles, size=sample_per_slide_mag, replace=False)).astype(np.int64)


def export_entries(
    shards_by_mag: Dict[int, List[H5Shard]],
    output_dir: Path,
    sample_per_slide_mag: int,
    split: str,
    seed: int,
) -> Dict[int, int]:
    if sample_per_slide_mag <= 0:
        raise ValueError("sample_per_slide_mag must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    split = split.upper()
    entries_path = output_dir / f"entries-{split}.npy"
    mapping_path = output_dir / f"entries-{split}.json"
    schema_path = output_dir / f"entries-{split}.schema.json"

    total_selected = selected_count(shards_by_mag, sample_per_slide_mag)
    if total_selected <= 0:
        raise ValueError("No coordinates selected for export")

    entries = np.lib.format.open_memmap(entries_path, mode="w+", dtype="int32", shape=(total_selected, 5))
    slide_to_id = write_slide_mapping(shards_by_mag, mapping_path)

    rng = np.random.RandomState(seed)
    cursor = 0
    counts_by_mag: Dict[int, int] = {}

    for mag, shards in shards_by_mag.items():
        counts_by_mag[mag] = 0
        for shard in shards:
            selected = sample_indices(shard.n_tiles, sample_per_slide_mag, rng)
            with h5py.File(shard.path, "r") as h5:
                coords = h5["coords"][selected]

            count = int(coords.shape[0])
            next_cursor = cursor + count
            entries[cursor:next_cursor, 0:2] = coords.astype(np.int32, copy=False)
            entries[cursor:next_cursor, 2] = slide_to_id[shard.slide_filename]
            entries[cursor:next_cursor, 3] = shard.target_magnification
            entries[cursor:next_cursor, 4] = shard.patch_size_level0

            cursor = next_cursor
            counts_by_mag[mag] += count

    entries.flush()
    write_schema(schema_path)
    return counts_by_mag


def print_summary(shards_by_mag: Dict[int, List[H5Shard]], counts_by_mag: Dict[int, int], output_dir: Path, split: str) -> None:
    n_slides = len({shard.slide_filename for shards in shards_by_mag.values() for shard in shards})
    n_tiles = sum(counts_by_mag.values())
    print(f"Exported SimDINO extra files to: {output_dir}")
    print(f"Slides: {n_slides}")
    print(f"Tiles: {n_tiles}")
    for mag in sorted(counts_by_mag):
        print(f"  {mag}x: {counts_by_mag[mag]}")
    split = split.upper()
    print(f"Entries: {output_dir / f'entries-{split}.npy'}")
    print(f"Slide mapping: {output_dir / f'entries-{split}.json'}")
    print(f"Schema: {output_dir / f'entries-{split}.schema.json'}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    mag_dirs = parse_h5_dirs(args.h5_dir)
    shards_by_mag = discover_shards(
        mag_dirs,
        h5_pattern=args.h5_pattern,
        slide_extension=args.slide_extension,
        level0_magnification=args.level0_magnification,
        output_patch_size=args.output_patch_size,
    )
    counts_by_mag = export_entries(
        shards_by_mag,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        sample_per_slide_mag=args.sample_per_slide_mag,
        split=args.split,
        seed=args.seed,
    )
    print_summary(shards_by_mag, counts_by_mag, Path(args.output_dir).expanduser().resolve(), args.split)


if __name__ == "__main__":
    main()

 #python create_simdino_extra.py --h5-dir 5:__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed/5x_256px_0px_overlap/5x_hsv_filtered_patches/ --h5-dir 10:__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed/10x_256px_0px_overlap/10x_hsv_filtered_patches/ --h5-dir 20:__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed/20x_256px_0px_overlap/20x_hsv_filtered_patches/ --h5-dir 40:__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed/40x_256px_0px_overlap/40x_hsv_filtered_patches/ --output-dir __REPATH_PRIVATE_PROJECT_ROOT_003__/tcga-extra_v3 --sample-per-slide-mag 5000 --level0-magnification 40 --seed 0
