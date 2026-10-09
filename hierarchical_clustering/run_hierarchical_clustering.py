import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
from tqdm import tqdm

import kmeans_gpu as kmg
from offline.common import (
    metadata_by_slide,
    read_wsi_manifest,
    resolve_wsi_path,
)
from tools.generate_hsv_filtered_patches import (
    HSVGenerationTask,
    generate_filter_for_slide,
)


H5_ATTR_KEYS = {
    "name",
    "target_magnification",
    "patch_size",
    "patch_size_level0",
    "level0_magnification",
}

HSV_FILTER_SPEC = {
    "method": "precomputed_coordinate_allowlist",
    "minimum_tissue_coverage": 0.45,
    "hue_range": [90, 180],
    "saturation_range": [8, 255],
    "value_range": [103, 255],
}


def resolved_hsv_filter_spec(
    minimum_tissue_coverage: float,
    *,
    generated: bool,
) -> Dict[str, object]:
    if not generated:
        return dict(HSV_FILTER_SPEC)
    return HSV_FILTER_SPEC | {
        "method": "hsv_tissue_coverage",
        "minimum_tissue_coverage": float(minimum_tissue_coverage),
    }


@dataclass
class H5Shard:
    mag: int
    path: str
    slide_id: str
    n_tiles: int
    feature_dim: int
    source_n_tiles: int = 0
    row_start: int = 0
    target_magnification: int = 0
    patch_size: int = 256
    patch_size_level0: int = 256
    level0_magnification: int = 0
    hsv_filter_path: str | None = None
    selected_row_indices_path: str | None = None
    selected_row_indices_sha256: str | None = None
    _selected_rows: np.ndarray | None = field(default=None, repr=False)

    @property
    def row_end(self) -> int:
        return self.row_start + self.n_tiles

    @property
    def is_upsampled(self) -> bool:
        """Return whether the requested magnification exceeds the WSI's native maximum."""
        return (
            self.level0_magnification > 0
            and self.target_magnification > self.level0_magnification
        )


def parse_mag_dirs(values: Iterable[str]) -> Dict[int, Path]:
    mag_dirs: Dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f'Expected MAG=DIR, got "{value}"')
        mag_text, dir_text = value.split("=", 1)
        mag = int(mag_text.rstrip("xX"))
        path = Path(dir_text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Magnification directory does not exist: {path}")
        mag_dirs[mag] = path
    return dict(sorted(mag_dirs.items()))


def normalized_slide_id(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_patches")] if stem.endswith("_patches") else stem


def discover_filter_h5(directory: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for path in sorted(directory.rglob("*.h5")):
        slide_id = normalized_slide_id(path)
        if slide_id in paths:
            raise ValueError(
                f"Duplicate HSV filter slide ID {slide_id!r}: {paths[slide_id]} and {path}"
            )
        paths[slide_id] = path.resolve()
    if not paths:
        raise FileNotFoundError(f"No HSV filter H5 files under {directory}")
    return paths


def coordinate_rows(path: Path, coords_key: str = "coords") -> tuple[np.ndarray, list[tuple[int, int]]]:
    with h5py.File(path, "r") as handle:
        if coords_key not in handle:
            raise ValueError(f"{path}: missing dataset {coords_key!r}")
        dataset = handle[coords_key]
        coords = np.asarray(dataset[:])
        if coords.size == 0:
            # Some HSV filtering writers serialize an empty coordinate list as
            # shape (0,) instead of the usual (0, 2). Both mean that no patch
            # from this slide passed the tissue filter.
            coords = np.empty((0, 2), dtype=np.int64)
        elif dataset.ndim != 2 or dataset.shape[1] != 2:
            raise ValueError(f"{path}: {coords_key} must have shape (N, 2), got {dataset.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: {coords_key} must use an integer dtype, got {coords.dtype}")
    tuples = [(int(x), int(y)) for x, y in coords]
    duplicate_count = len(tuples) - len(set(tuples))
    if duplicate_count:
        raise ValueError(f"{path}: {coords_key} contains {duplicate_count} duplicate coordinates")
    return coords, tuples


def select_hsv_coordinate_rows(
    mag: int,
    slide_id: str,
    source_path: str,
    filter_path: str,
) -> tuple[int, np.ndarray]:
    """Return source-row indices accepted by one slide's HSV coordinate allowlist."""
    source = Path(source_path)
    filtered = Path(filter_path)
    _, source_coords = coordinate_rows(source)
    _, accepted_coords = coordinate_rows(filtered)
    source_set = set(source_coords)
    missing_coords = sorted(set(accepted_coords) - source_set)
    if missing_coords:
        raise ValueError(
            f"{mag}x {slide_id}: {len(missing_coords)} HSV-filtered coordinates "
            f"are absent from {source}; examples={missing_coords[:5]}"
        )

    accepted_set = set(accepted_coords)
    selected_rows = np.asarray(
        [row for row, coord in enumerate(source_coords) if coord in accepted_set],
        dtype=np.int64,
    )
    return len(source_coords), selected_rows


def metadata_for_shard(
    shard: H5Shard,
    metadata_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    file_id = normalized_slide_id(Path(shard.path))
    candidates = (
        shard.slide_id,
        file_id,
        shard.slide_id.split(".", 1)[0],
        file_id.split(".", 1)[0],
    )
    for candidate in candidates:
        if candidate in metadata_map:
            return metadata_map[candidate]
    raise KeyError(
        f"{shard.mag}x {shard.slide_id}: no metadata found using keys "
        f"{list(dict.fromkeys(candidates))}"
    )


def generated_filter_inventory_key(records: Sequence[Dict[str, object]]) -> str:
    payload = json.dumps(
        list(records),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def generate_hsv_filter_directories(
    shards_by_mag: Dict[int, List[H5Shard]],
    output_dir: Path,
    *,
    metadata_path: Path,
    wsi_root: Path | None,
    wsi_manifest_path: Path | None,
    wsi_path_template: str,
    minimum_tissue_coverage: float,
    workers: int,
    show_progress: bool,
) -> Dict[int, Path]:
    if not 0.0 <= minimum_tissue_coverage <= 1.0:
        raise ValueError("--hsv-min-tissue-coverage must be between 0 and 1")
    if workers <= 0:
        raise ValueError("--hsv-filter-workers must be positive")
    if wsi_root is None and wsi_manifest_path is None:
        raise ValueError(
            "Automatic HSV filtering requires --wsi-root or --wsi-manifest"
        )

    metadata_map = metadata_by_slide(metadata_path)
    manifest = read_wsi_manifest(
        str(wsi_manifest_path.resolve()) if wsi_manifest_path is not None else None
    )
    wsi_config = {
        "wsi": {
            "root": str(wsi_root.resolve()) if wsi_root is not None else None,
            "path_template": wsi_path_template,
        }
    }
    filter_dirs: Dict[int, Path] = {}

    for mag, shards in shards_by_mag.items():
        resolved: list[tuple[H5Shard, Path, Path]] = []
        inventory: list[Dict[str, object]] = []
        for shard in shards:
            metadata = metadata_for_shard(shard, metadata_map)
            wsi_path_value = resolve_wsi_path(metadata, wsi_config, manifest)
            if not wsi_path_value:
                raise ValueError(f"{mag}x {shard.slide_id}: WSI path could not be resolved")
            wsi_path = Path(wsi_path_value).resolve()
            if not wsi_path.is_file():
                raise FileNotFoundError(f"{mag}x {shard.slide_id}: WSI does not exist: {wsi_path}")
            source_path = Path(shard.path).resolve()
            source_stat = source_path.stat()
            wsi_stat = wsi_path.stat()
            inventory.append(
                {
                    "slide_id": shard.slide_id,
                    "source_h5": str(source_path),
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "wsi_path": str(wsi_path),
                    "wsi_size": wsi_stat.st_size,
                    "wsi_mtime_ns": wsi_stat.st_mtime_ns,
                    "patch_size": shard.patch_size,
                    "patch_size_level0": shard.patch_size_level0,
                    "minimum_tissue_coverage": minimum_tissue_coverage,
                }
            )
            resolved.append((shard, source_path, wsi_path))

        inventory_key = generated_filter_inventory_key(inventory)
        mag_dir = (
            output_dir
            / "generated_hsv_filters"
            / f"{mag}x"
            / inventory_key
        )
        mag_dir.mkdir(parents=True, exist_ok=True)
        filter_dirs[mag] = mag_dir
        tasks: list[HSVGenerationTask] = []
        for shard, source_path, wsi_path in resolved:
            source_stat = source_path.stat()
            wsi_stat = wsi_path.stat()
            tasks.append(
                HSVGenerationTask(
                    mag=mag,
                    slide_id=shard.slide_id,
                    source_h5=str(source_path),
                    wsi_path=str(wsi_path),
                    output_h5=str(
                        mag_dir
                        / f"{normalized_slide_id(source_path)}_patches.h5"
                    ),
                    patch_size=shard.patch_size,
                    patch_size_level0=shard.patch_size_level0,
                    minimum_tissue_coverage=minimum_tissue_coverage,
                    source_size=source_stat.st_size,
                    source_mtime_ns=source_stat.st_mtime_ns,
                    wsi_size=wsi_stat.st_size,
                    wsi_mtime_ns=wsi_stat.st_mtime_ns,
                )
            )

        active_workers = min(workers, len(tasks))
        kept_tiles = 0
        empty_slides = 0
        cached_slides = 0
        with tqdm(
            total=len(tasks),
            desc=f"HSV generate {mag}x ({active_workers} proc)",
            unit="slide",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not show_progress,
        ) as progress:
            if active_workers == 1:
                results = (generate_filter_for_slide(task) for task in tasks)
                for result in results:
                    kept_tiles += int(result["accepted_patches"])
                    empty_slides += int(result["accepted_patches"] == 0)
                    cached_slides += int(bool(result["cached"]))
                    progress.set_postfix(
                        kept=f"{kept_tiles:,}",
                        empty=empty_slides,
                        cached=cached_slides,
                        refresh=False,
                    )
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=active_workers) as executor:
                    pending = {
                        executor.submit(generate_filter_for_slide, task): task
                        for task in tasks
                    }
                    for future in as_completed(pending):
                        result = future.result()
                        kept_tiles += int(result["accepted_patches"])
                        empty_slides += int(result["accepted_patches"] == 0)
                        cached_slides += int(bool(result["cached"]))
                        progress.set_postfix(
                            kept=f"{kept_tiles:,}",
                            empty=empty_slides,
                            cached=cached_slides,
                            refresh=False,
                        )
                        progress.update(1)
    return filter_dirs


def apply_hsv_coordinate_filters(
    shards_by_mag: Dict[int, List[H5Shard]],
    excluded_by_mag: Dict[int, List[H5Shard]],
    filter_dirs: Dict[int, Path],
    *,
    show_progress: bool = False,
    workers: int = 1,
) -> None:
    if workers <= 0:
        raise ValueError("--hsv-filter-workers must be positive")
    if set(filter_dirs) != set(shards_by_mag):
        raise ValueError(
            "--hsv-filter-dir magnifications must exactly match --mag-dir; "
            f"feature magnifications={sorted(shards_by_mag)}, "
            f"filter magnifications={sorted(filter_dirs)}"
        )

    for mag, shards in shards_by_mag.items():
        filter_paths = discover_filter_h5(filter_dirs[mag])
        all_shards = [*shards, *excluded_by_mag[mag]]
        shard_by_file_id: Dict[str, H5Shard] = {}
        for shard in all_shards:
            file_id = normalized_slide_id(Path(shard.path))
            if file_id in shard_by_file_id:
                raise ValueError(
                    f"{mag}x duplicate feature filename slide ID {file_id!r}: "
                    f"{shard_by_file_id[file_id].path} and {shard.path}"
                )
            shard_by_file_id[file_id] = shard
        required_ids = {normalized_slide_id(Path(shard.path)) for shard in shards}
        allowed_ids = set(shard_by_file_id)
        missing = sorted(required_ids - set(filter_paths))
        extra = sorted(set(filter_paths) - allowed_ids)
        if missing or extra:
            raise ValueError(
                f"{mag}x HSV filter/feature slide mismatch; "
                f"missing filters={missing[:5]}, extra filters={extra[:5]}"
            )

        tasks = [
            (
                shard,
                filter_paths[normalized_slide_id(Path(shard.path))],
            )
            for shard in shards
        ]
        active_workers = min(workers, len(tasks))
        kept_tiles = 0
        empty_slides = 0

        def apply_result(
            shard: H5Shard,
            filter_path: Path,
            result: tuple[int, np.ndarray],
        ) -> None:
            nonlocal kept_tiles, empty_slides
            source_n_tiles, selected_rows = result
            if source_n_tiles != shard.n_tiles:
                raise ValueError(
                    f"{shard.path}: coordinate count changed during HSV filtering; "
                    f"expected {shard.n_tiles}, got {source_n_tiles}"
                )
            shard.source_n_tiles = source_n_tiles
            shard.n_tiles = int(selected_rows.size)
            shard.hsv_filter_path = str(filter_path)
            shard._selected_rows = selected_rows
            kept_tiles += shard.n_tiles
            empty_slides += int(shard.n_tiles == 0)

        with tqdm(
            total=len(shards),
            desc=f"HSV filter {mag}x ({active_workers} proc)",
            unit="slide",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not show_progress,
        ) as progress:
            if active_workers == 1:
                completed = (
                    (
                        shard,
                        filter_path,
                        select_hsv_coordinate_rows(
                            mag,
                            shard.slide_id,
                            shard.path,
                            str(filter_path),
                        ),
                    )
                    for shard, filter_path in tasks
                )
                for shard, filter_path, result in completed:
                    apply_result(shard, filter_path, result)
                    progress.set_postfix(
                        kept=f"{kept_tiles:,}",
                        empty=empty_slides,
                        refresh=False,
                    )
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=active_workers) as executor:
                    pending = {
                        executor.submit(
                            select_hsv_coordinate_rows,
                            mag,
                            shard.slide_id,
                            shard.path,
                            str(filter_path),
                        ): (shard, filter_path)
                        for shard, filter_path in tasks
                    }
                    for future in as_completed(pending):
                        shard, filter_path = pending[future]
                        apply_result(shard, filter_path, future.result())
                        progress.set_postfix(
                            kept=f"{kept_tiles:,}",
                            empty=empty_slides,
                            refresh=False,
                        )
                        progress.update(1)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def shard_record(shard: H5Shard) -> Dict[str, object]:
    record = asdict(shard)
    record.pop("_selected_rows", None)
    return record


def parse_cluster_counts(value: str) -> List[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise ValueError("--n-clusters must contain at least one integer")
    if any(count <= 0 for count in counts):
        raise ValueError(f"--n-clusters must be positive, got {counts}")
    return counts


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def as_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def scalar_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def read_h5_attrs(h5: h5py.File) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    for source in (h5.attrs, h5["coords"].attrs, h5["features"].attrs):
        for key, value in source.items():
            if key not in H5_ATTR_KEYS:
                continue
            scalar_value = scalar_h5_attr(value)
            if scalar_value is not None:
                attrs[key] = scalar_value
    return attrs


def inspect_h5(path: Path, mag: int) -> H5Shard:
    with h5py.File(path, "r") as h5:
        if "features" not in h5 or "coords" not in h5:
            raise ValueError(f"{path} must contain 'features' and 'coords' datasets")
        features = h5["features"]
        coords = h5["coords"]
        if features.ndim != 2:
            raise ValueError(f"{path}: features must be 2-D, got {features.shape}")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"{path}: coords must have shape (N, 2), got {coords.shape}")
        if features.shape[0] != coords.shape[0]:
            raise ValueError(f"{path}: feature/coord row mismatch")
        attrs = read_h5_attrs(h5)
        n_tiles = int(features.shape[0])
        feature_dim = int(features.shape[1])

    return H5Shard(
        mag=mag,
        path=str(path),
        slide_id=str(attrs.get("name") or path.stem),
        n_tiles=n_tiles,
        feature_dim=feature_dim,
        source_n_tiles=n_tiles,
        target_magnification=as_int(attrs.get("target_magnification"), mag),
        patch_size=as_int(attrs.get("patch_size"), 256),
        patch_size_level0=as_int(attrs.get("patch_size_level0"), 256),
        level0_magnification=as_int(attrs.get("level0_magnification"), 0),
    )


def discover_shards(
    mag_dirs: Dict[int, Path],
    h5_pattern: str,
    exclude_upsampled: bool = True,
) -> tuple[Dict[int, List[H5Shard]], Dict[int, List[H5Shard]]]:
    shards_by_mag: Dict[int, List[H5Shard]] = {}
    excluded_by_mag: Dict[int, List[H5Shard]] = {}
    for mag, directory in mag_dirs.items():
        paths = sorted(directory.rglob(h5_pattern))
        if not paths:
            raise FileNotFoundError(f"No H5 files matching {h5_pattern!r} under {directory}")
        discovered = [inspect_h5(path, mag) for path in paths]
        excluded = [shard for shard in discovered if exclude_upsampled and shard.is_upsampled]
        shards = [shard for shard in discovered if not (exclude_upsampled and shard.is_upsampled)]
        if not shards:
            raise ValueError(
                f"{mag}x: all {len(discovered)} H5 files were excluded because their "
                "target_magnification exceeds level0_magnification"
            )
        dims = {shard.feature_dim for shard in shards}
        if len(dims) != 1:
            raise ValueError(f"{mag}x H5 files have mixed feature dimensions: {sorted(dims)}")
        shards_by_mag[mag] = shards
        excluded_by_mag[mag] = excluded
    return shards_by_mag, excluded_by_mag


def write_upsampled_exclusion_report(
    output_dir: Path,
    mag: int,
    excluded: Sequence[H5Shard],
) -> Path:
    report_path = output_dir / "features" / f"{mag}x" / "excluded_upsampled_h5.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        shard_record(shard)
        | {
            "reason": "target_magnification_gt_level0_magnification",
        }
        for shard in excluded
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return report_path


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms


def validate_finite_features(
    features: np.ndarray,
    h5_path: str,
    row_start: int,
    source_rows: np.ndarray | None = None,
) -> None:
    finite_mask = np.isfinite(features)
    if finite_mask.all():
        return

    np.logical_not(finite_mask, out=finite_mask)
    flat_index = int(finite_mask.argmax())
    local_row, column = np.unravel_index(flat_index, features.shape)
    value = features[local_row, column]
    source_row = (
        int(source_rows[local_row])
        if source_rows is not None
        else row_start + int(local_row)
    )
    raise ValueError(
        f"{h5_path}: non-finite feature value {value!r} "
        f"at row {source_row}, column {int(column)}"
    )


def validate_nonzero_features(
    features: np.ndarray,
    h5_path: str,
    row_start: int,
    source_rows: np.ndarray | None = None,
) -> None:
    norms = np.linalg.norm(features, axis=1)
    invalid_rows = np.flatnonzero(norms <= 1e-12)
    if invalid_rows.size == 0:
        return
    local_row = int(invalid_rows[0])
    source_row = int(source_rows[local_row]) if source_rows is not None else row_start + local_row
    raise ValueError(
        f"{h5_path}: zero or near-zero feature vector at row {source_row} "
        f"(norm={float(norms[local_row]):.8g})"
    )


def pack_features(
    shards: List[H5Shard],
    output_dir: Path,
    feature_chunk_size: int,
    normalize: bool,
    quiet: bool,
) -> Path:
    mag = shards[0].mag
    mag_dir = output_dir / "features" / f"{mag}x"
    mag_dir.mkdir(parents=True, exist_ok=True)
    total_tiles = sum(shard.n_tiles for shard in shards)
    feature_dim = shards[0].feature_dim
    matrix_path = mag_dir / "features.npy"
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float32", shape=(total_tiles, feature_dim))

    selection_dir = mag_dir / "selected_rows"
    if any(shard._selected_rows is not None for shard in shards):
        selection_dir.mkdir(parents=True, exist_ok=True)

    cursor = 0
    for shard_idx, shard in enumerate(shards, start=1):
        shard.row_start = cursor
        if not quiet:
            print(f"{mag}x pack {shard_idx}/{len(shards)} {Path(shard.path).name} rows={shard.n_tiles}")
        with h5py.File(shard.path, "r") as h5:
            features = h5["features"]
            source_rows = shard._selected_rows
            if source_rows is not None:
                selection_path = selection_dir / f"{shard_idx - 1:06d}.npy"
                temporary_selection = selection_path.with_name(f".{selection_path.name}.tmp")
                with open(temporary_selection, "wb") as handle:
                    np.save(handle, source_rows, allow_pickle=False)
                temporary_selection.replace(selection_path)
                shard.selected_row_indices_path = str(selection_path.resolve())
                shard.selected_row_indices_sha256 = file_sha256(selection_path)
            else:
                source_rows = np.arange(shard.source_n_tiles, dtype=np.int64)
            for start in range(0, shard.n_tiles, feature_chunk_size):
                end = min(shard.n_tiles, start + feature_chunk_size)
                chunk_rows = source_rows[start:end]
                chunk = np.asarray(features[chunk_rows], dtype=np.float32)
                validate_finite_features(chunk, shard.path, start, source_rows=chunk_rows)
                validate_nonzero_features(chunk, shard.path, start, source_rows=chunk_rows)
                if normalize:
                    chunk = normalize_rows(chunk).astype(np.float32, copy=False)
                matrix[cursor + start : cursor + end] = chunk
        cursor += shard.n_tiles
    matrix.flush()

    with open(mag_dir / "h5_index.json", "w", encoding="utf-8") as f:
        json.dump([shard_record(shard) for shard in shards], f, indent=2)
    return matrix_path


def sample_fit_indices(total: int, max_fit_samples: int, seed: int) -> np.ndarray:
    if max_fit_samples <= 0:
        raise ValueError("--max-fit-samples must be positive")
    fit_size = min(total, max_fit_samples)
    if fit_size == total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.RandomState(seed)
    selected = set()
    # Floyd's algorithm samples without replacement using memory proportional to fit_size.
    for value in range(total - fit_size, total):
        candidate = int(rng.randint(0, value + 1))
        selected.add(value if candidate in selected else candidate)
    return np.array(sorted(selected), dtype=np.int64)


def distance_chunk_size(n_clusters: int, budget: int, upper_bound: int) -> int:
    if budget <= 0:
        return -1
    return max(1, min(upper_bound, int(budget) // max(1, int(n_clusters))))


def object_array(values) -> np.ndarray:
    out = np.empty((len(values),), dtype=object)
    for idx, value in enumerate(values):
        out[idx] = np.asarray(value, dtype=np.int64)
    return out


def fit_hierarchy(
    matrix_path: Path,
    output_dir: Path,
    mag: int,
    n_clusters: Sequence[int],
    n_iters: int,
    max_fit_samples: int,
    fit_chunk_budget: int,
    seed: int,
    device: torch.device,
    quiet: bool,
) -> List[Dict[str, np.ndarray]]:
    matrix = np.load(matrix_path, mmap_mode="r")
    fit_indices = sample_fit_indices(int(matrix.shape[0]), max_fit_samples, seed + mag)
    fit_data = torch.tensor(np.asarray(matrix[fit_indices], dtype=np.float32), device=device)
    current = fit_data
    rng = np.random.RandomState(seed + mag)

    cluster_dir = output_dir / "clusters" / f"{mag}x"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    np.save(cluster_dir / "fit_indices.npy", fit_indices)

    hierarchy: List[Dict[str, np.ndarray]] = []
    previous_count = int(current.shape[0])
    for level_idx, requested_k in enumerate(n_clusters, start=1):
        k = min(int(requested_k), previous_count)
        if k <= 0:
            raise ValueError(f"{mag}x level{level_idx}: invalid cluster count {k}")
        chunk_size = distance_chunk_size(k, fit_chunk_budget, previous_count)
        if not quiet:
            print(f"{mag}x level{level_idx}: fitting k={k} on {previous_count} rows chunk_size={chunk_size} device={device}")
        centroids, clusters, assignment, pot = kmg.kmeans(
            current,
            n_clusters=k,
            n_iters=n_iters,
            chunk_size=chunk_size,
            num_init=1,
            init_method="kmeans++",
            dist="l2",
            high_precision=torch.float64,
            random_state=rng,
            verbose=not quiet,
        )

        level_dir = cluster_dir / f"level{level_idx}"
        level_dir.mkdir(parents=True, exist_ok=True)
        centroid_np = centroids.detach().cpu().numpy().astype(np.float32)
        assignment_np = np.asarray(assignment, dtype=np.int32)
        np.save(level_dir / "centroids.npy", centroid_np)
        np.save(level_dir / "fit_assignment.npy", assignment_np)
        np.save(level_dir / "clusters.npy", object_array(clusters))
        with open(level_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mag": mag,
                    "level": level_idx,
                    "requested_clusters": int(requested_k),
                    "clusters": int(k),
                    "fit_rows": int(previous_count),
                    "n_iters": int(n_iters),
                    "pot": float(pot),
                },
                f,
                indent=2,
            )

        hierarchy.append({"centroids": centroid_np, "assignment": assignment_np})
        current = centroids.detach()
        previous_count = int(current.shape[0])
    return hierarchy


def assign_level1(
    matrix_path: Path,
    centroids: np.ndarray,
    assignment_path: Path,
    assign_chunk_size: int,
    fit_chunk_budget: int,
    device: torch.device,
    quiet: bool,
) -> Path:
    matrix = np.load(matrix_path, mmap_mode="r")
    out = np.lib.format.open_memmap(assignment_path, mode="w+", dtype="int32", shape=(matrix.shape[0],))
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    inner_chunk_size = distance_chunk_size(centroids.shape[0], fit_chunk_budget, assign_chunk_size)
    for start in range(0, matrix.shape[0], assign_chunk_size):
        end = min(start + assign_chunk_size, matrix.shape[0])
        if not quiet:
            print(f"assign {assignment_path.parent.parent.name} rows {start}:{end}")
        chunk = torch.tensor(np.asarray(matrix[start:end], dtype=np.float32), device=device)
        assigned = kmg.assign_clusters(centroid_tensor, chunk, "l2", chunk_size=inner_chunk_size)
        out[start:end] = assigned.detach().cpu().numpy().astype(np.int32)
    out.flush()
    return assignment_path


def assign_hierarchy(
    matrix_path: Path,
    output_dir: Path,
    mag: int,
    hierarchy: Sequence[Dict[str, np.ndarray]],
    assign_chunk_size: int,
    fit_chunk_budget: int,
    device: torch.device,
    quiet: bool,
) -> List[Path]:
    cluster_dir = output_dir / "clusters" / f"{mag}x"
    assignment_paths: List[Path] = []
    level1_path = cluster_dir / "level1" / "assignments.npy"
    assign_level1(matrix_path, hierarchy[0]["centroids"], level1_path, assign_chunk_size, fit_chunk_budget, device, quiet)
    assignment_paths.append(level1_path)

    previous = np.load(level1_path, mmap_mode="r")
    for level_idx in range(2, len(hierarchy) + 1):
        mapping = np.asarray(hierarchy[level_idx - 1]["assignment"], dtype=np.int32)
        out_path = cluster_dir / f"level{level_idx}" / "assignments.npy"
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype="int32", shape=previous.shape)
        for start in range(0, previous.shape[0], assign_chunk_size):
            end = min(start + assign_chunk_size, previous.shape[0])
            out[start:end] = mapping[np.asarray(previous[start:end], dtype=np.int64)]
        out.flush()
        assignment_paths.append(out_path)
        previous = np.load(out_path, mmap_mode="r")
    return assignment_paths


def write_tile_index(
    shards_by_mag: Dict[int, List[H5Shard]],
    assignment_paths_by_mag: Dict[int, List[Path]],
    output_dir: Path,
) -> None:
    index_dir = output_dir / "tile_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for mag, shards in shards_by_mag.items():
        assignment_arrays = [np.load(path, mmap_mode="r") for path in assignment_paths_by_mag[mag]]
        fields = [
            "slide_id",
            "mag",
            "h5_path",
            "row_idx",
            "x",
            "y",
            "target_magnification",
            "patch_size",
            "patch_size_level0",
            "level0_magnification",
            *[f"level_{idx}" for idx in range(1, len(assignment_arrays) + 1)],
        ]
        with open(index_dir / f"tile_index_{mag}x.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for shard in shards:
                with h5py.File(shard.path, "r") as h5:
                    source_rows = (
                        shard._selected_rows
                        if shard._selected_rows is not None
                        else np.arange(shard.source_n_tiles, dtype=np.int64)
                    )
                    coords = np.asarray(h5["coords"][source_rows], dtype=np.int64)
                for packed_local_idx, (source_row, coord) in enumerate(zip(source_rows, coords)):
                    global_idx = shard.row_start + packed_local_idx
                    row = {
                        "slide_id": shard.slide_id,
                        "mag": mag,
                        "h5_path": shard.path,
                        "row_idx": int(source_row),
                        "x": int(coord[0]),
                        "y": int(coord[1]),
                        "target_magnification": int(shard.target_magnification),
                        "patch_size": int(shard.patch_size),
                        "patch_size_level0": int(shard.patch_size_level0),
                        "level0_magnification": int(shard.level0_magnification),
                    }
                    for level_idx, assignments in enumerate(assignment_arrays, start=1):
                        row[f"level_{level_idx}"] = int(assignments[global_idx])
                    writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mag_dirs = parse_mag_dirs(args.mag_dir)
    hsv_filter_dirs = parse_mag_dirs(args.hsv_filter_dir) if args.hsv_filter_dir else {}
    precomputed_hsv_filters = bool(hsv_filter_dirs)
    n_clusters = parse_cluster_counts(args.n_clusters)
    device = choose_device(args.device)

    print(f"Using device: {device}")
    print(f"Magnifications: {', '.join(str(mag) + 'x' for mag in mag_dirs)}")
    shards_by_mag, excluded_by_mag = discover_shards(
        mag_dirs,
        args.h5_pattern,
        exclude_upsampled=args.exclude_upsampled,
    )
    automatic_hsv_requested = (
        args.metadata_path is not None
        or args.wsi_root is not None
        or args.wsi_manifest is not None
    )
    if not hsv_filter_dirs and automatic_hsv_requested:
        if args.metadata_path is None:
            raise ValueError(
                "Automatic HSV filtering requires --metadata-path"
            )
        if args.wsi_root is None and args.wsi_manifest is None:
            raise ValueError(
                "Automatic HSV filtering requires --wsi-root or --wsi-manifest"
            )
        print(
            "No --hsv-filter-dir supplied; generating HSV coordinate filters "
            f"with minimum tissue coverage {args.hsv_min_tissue_coverage:.2%}"
        )
        hsv_filter_dirs = generate_hsv_filter_directories(
            shards_by_mag,
            output_dir,
            metadata_path=args.metadata_path.expanduser().resolve(),
            wsi_root=(
                args.wsi_root.expanduser().resolve()
                if args.wsi_root is not None
                else None
            ),
            wsi_manifest_path=(
                args.wsi_manifest.expanduser().resolve()
                if args.wsi_manifest is not None
                else None
            ),
            wsi_path_template=args.wsi_path_template,
            minimum_tissue_coverage=args.hsv_min_tissue_coverage,
            workers=args.hsv_filter_workers,
            show_progress=not args.quiet,
        )
    if hsv_filter_dirs:
        apply_hsv_coordinate_filters(
            shards_by_mag,
            excluded_by_mag,
            hsv_filter_dirs,
            show_progress=not args.quiet,
            workers=args.hsv_filter_workers,
        )
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            vars(args)
            | {
                "device_resolved": str(device),
                "hsv_filter_mode": (
                    "precomputed"
                    if precomputed_hsv_filters
                    else "generated"
                    if hsv_filter_dirs
                    else "disabled"
                ),
                "hsv_filter_dirs_resolved": {
                    str(mag): str(path) for mag, path in hsv_filter_dirs.items()
                },
                "hsv_filter_spec": (
                    resolved_hsv_filter_spec(
                        args.hsv_min_tissue_coverage,
                        generated=not precomputed_hsv_filters,
                    )
                    if hsv_filter_dirs
                    else None
                ),
            },
            f,
            indent=2,
            default=str,
        )
    assignment_paths_by_mag: Dict[int, List[Path]] = {}
    summary: Dict[str, object] = {"magnifications": {}}

    for mag, shards in shards_by_mag.items():
        excluded = excluded_by_mag[mag]
        total_tiles = sum(shard.n_tiles for shard in shards)
        source_tiles = sum(shard.source_n_tiles for shard in shards)
        hsv_empty_slides = sum(
            shard.hsv_filter_path is not None and shard.n_tiles == 0 for shard in shards
        )
        if total_tiles <= 0:
            raise ValueError(f"{mag}x: HSV filtering removed every patch")
        excluded_tiles = sum(shard.n_tiles for shard in excluded)
        if excluded:
            print(
                f"{mag}x: excluding {len(excluded)} upsampled H5 files "
                f"({excluded_tiles} tiles)"
            )
        if hsv_filter_dirs:
            print(
                f"{mag}x: HSV tissue filter kept {total_tiles}/{source_tiles} tiles "
                f"({100.0 * total_tiles / max(1, source_tiles):.2f}%)"
            )
            if hsv_empty_slides:
                print(
                    f"{mag}x: skipping {hsv_empty_slides} slides with no "
                    "HSV-accepted patches"
                )
        else:
            print(f"{mag}x: {len(shards)} H5 files, {total_tiles} tiles")
        matrix_path = pack_features(
            shards,
            output_dir,
            feature_chunk_size=args.feature_chunk_size,
            normalize=not args.no_normalize,
            quiet=args.quiet,
        )
        exclusion_report_path = write_upsampled_exclusion_report(output_dir, mag, excluded)
        hierarchy = fit_hierarchy(
            matrix_path=matrix_path,
            output_dir=output_dir,
            mag=mag,
            n_clusters=n_clusters,
            n_iters=args.n_iters,
            max_fit_samples=args.max_fit_samples,
            fit_chunk_budget=args.fit_chunk_budget,
            seed=args.seed,
            device=device,
            quiet=args.quiet,
        )
        assignment_paths_by_mag[mag] = assign_hierarchy(
            matrix_path=matrix_path,
            output_dir=output_dir,
            mag=mag,
            hierarchy=hierarchy,
            assign_chunk_size=args.assign_chunk_size,
            fit_chunk_budget=args.fit_chunk_budget,
            device=device,
            quiet=args.quiet,
        )
        summary["magnifications"][str(mag)] = {
            "discovered_h5_files": len(shards) + len(excluded),
            "h5_files": len(shards),
            "tiles": total_tiles,
            "source_tiles": source_tiles,
            "hsv_filtered_out_tiles": source_tiles - total_tiles,
            "hsv_empty_slides": hsv_empty_slides,
            "hsv_filter_enabled": bool(hsv_filter_dirs),
            "hsv_filter_workers": (
                min(args.hsv_filter_workers, len(shards)) if hsv_filter_dirs else 0
            ),
            "hsv_filter_mode": (
                "precomputed" if precomputed_hsv_filters else "generated"
            )
            if hsv_filter_dirs
            else "disabled",
            "hsv_filter_spec": (
                resolved_hsv_filter_spec(
                    args.hsv_min_tissue_coverage,
                    generated=not precomputed_hsv_filters,
                )
                if hsv_filter_dirs
                else None
            ),
            "excluded_upsampled_h5_files": len(excluded),
            "excluded_upsampled_tiles": excluded_tiles,
            "unknown_level0_magnification_h5_files": sum(
                shard.level0_magnification <= 0 for shard in shards
            ),
            "excluded_upsampled_report": str(exclusion_report_path),
            "feature_dim": shards[0].feature_dim,
            "matrix_path": str(matrix_path),
            "assignment_paths": [str(path) for path in assignment_paths_by_mag[mag]],
        }

    if args.write_tile_index:
        write_tile_index(shards_by_mag, assignment_paths_by_mag, output_dir)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Outputs written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone chunked hierarchical k-means for H5 tile embeddings.")
    parser.add_argument(
        "--mag-dir",
        action="append",
        required=True,
        help="Magnification feature folder in MAG=DIR form, e.g. 5=/path/to/features_conch_v1.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h5-pattern", default="*.h5")
    parser.add_argument(
        "--hsv-filter-dir",
        action="append",
        default=[],
        metavar="MAG=DIR",
        help=(
            "Precomputed HSV-filtered coordinate H5 directory in MAG=DIR form. "
            "When provided, every --mag-dir magnification must have one matching entry. "
            "Only listed patch coordinates enter clustering."
        ),
    )
    parser.add_argument(
        "--hsv-filter-workers",
        type=int,
        default=8,
        help="Number of worker processes used to match HSV coordinates by slide (default: 8).",
    )
    parser.add_argument(
        "--hsv-min-tissue-coverage",
        type=float,
        default=0.45,
        help=(
            "Minimum accepted HSV-tissue pixel fraction when filters are generated "
            "from WSIs (default: 0.45)."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Metadata JSON used to resolve WSI paths for automatic HSV filtering.",
    )
    wsi_group = parser.add_mutually_exclusive_group()
    wsi_group.add_argument("--wsi-root", type=Path)
    wsi_group.add_argument("--wsi-manifest", type=Path)
    parser.add_argument(
        "--wsi-path-template",
        default="{wsi_root}/{file_name}",
    )
    parser.add_argument("--n-clusters", default="4096,512")
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--max-fit-samples", type=int, default=200_000)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--assign-chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--fit-chunk-budget",
        type=int,
        default=100_000_000,
        help="Maximum distance-matrix entries per k-means/assignment chunk. Lower this to reduce GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--exclude-upsampled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude H5 shards where target_magnification exceeds "
            "level0_magnification (default: enabled)."
        ),
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable row normalization before clustering.")
    parser.add_argument("--write-tile-index", action="store_true", help="Write one CSV row per tile with assignments.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_iters < 0:
        raise ValueError("--n-iters must be non-negative")
    if args.feature_chunk_size <= 0 or args.assign_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.hsv_filter_workers <= 0:
        raise ValueError("--hsv-filter-workers must be positive")
    if not 0.0 <= args.hsv_min_tissue_coverage <= 1.0:
        raise ValueError("--hsv-min-tissue-coverage must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
