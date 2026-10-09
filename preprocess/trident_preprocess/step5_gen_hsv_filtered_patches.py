"""Generate HSV-filtered coordinate H5 files for hierarchical clustering.

Run once per magnification, using TRIDENT coordinate or feature H5 files.
The source coordinates are level-0 pixel positions; patch geometry is read
from the H5 attributes rather than inferred from the directory name.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Support direct execution from any working directory without installing RePath.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hierarchical_clustering.tools.generate_hsv_filtered_patches import (
    HSVGenerationTask,
    generate_filter_for_slide,
)


def positive_integer_attr(attrs: dict, key: str, path: Path) -> int:
    value = np.asarray(attrs.get(key))
    try:
        number = float(value.item()) if value.size == 1 else float("nan")
    except (TypeError, ValueError):
        number = float("nan")
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(
            f"{path}: missing or invalid {key!r}; expected a positive integer "
            "in the TRIDENT H5 attributes"
        )
    return int(number)


def read_geometry(path: Path, mag: int) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle or not isinstance(handle["coords"], h5py.Dataset):
            raise ValueError(f"{path}: missing dataset 'coords'")
        coords = handle["coords"]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"{path}: coords must have shape (N, 2), got {coords.shape}")
        if not np.issubdtype(coords.dtype, np.integer):
            raise ValueError(f"{path}: coords must use an integer dtype")
        attrs = dict(handle.attrs)
        attrs.update(coords.attrs)
        if "features" in handle:
            features = handle["features"]
            if (
                not isinstance(features, h5py.Dataset)
                or features.ndim != 2
                or features.shape[0] != coords.shape[0]
            ):
                raise ValueError(f"{path}: features must have one row per coordinate")
            attrs.update(features.attrs)
    if "target_magnification" in attrs:
        recorded_mag = positive_integer_attr(attrs, "target_magnification", path)
        if recorded_mag != mag:
            raise ValueError(f"{path}: target magnification is {recorded_mag}x, not {mag}x")
    return (
        positive_integer_attr(attrs, "patch_size", path),
        positive_integer_attr(attrs, "patch_size_level0", path),
    )


def build_tasks(args: argparse.Namespace) -> list[HSVGenerationTask]:
    if args.mag <= 0 or args.workers <= 0:
        raise ValueError("--mag and --workers must be positive integers")
    if not 0.0 <= args.min_tissue_coverage <= 1.0:
        raise ValueError("--min-tissue-coverage must be between 0 and 1")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    wsi_dir = args.wsi_dir.expanduser().resolve()
    if not wsi_dir.is_dir():
        raise FileNotFoundError(f"WSI directory does not exist: {wsi_dir}")
    if input_path.is_file():
        inputs = [input_path]
        input_dir = input_path.parent
    elif input_path.is_dir():
        inputs = sorted(path.resolve() for path in input_path.rglob(args.pattern) if path.is_file())
        input_dir = input_path
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not inputs:
        raise FileNotFoundError(f"No H5 files matching {args.pattern!r} under {input_path}")
    if output_dir.is_file() or output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("--output must be a separate directory outside the input directory")

    sources: dict[str, Path] = {}
    for path in inputs:
        slide_id = path.stem.removesuffix("_patches")
        if slide_id in sources:
            raise ValueError(f"Duplicate input slide ID {slide_id!r}: {sources[slide_id]} and {path}")
        sources[slide_id] = path

    extensions = {"." + extension.lower().lstrip(".") for extension in args.wsi_ext}
    slides: dict[str, Path] = {}
    for path in sorted(wsi_dir.rglob("*")):
        if path.suffix.lower() not in extensions or path.stem not in sources or not path.is_file():
            continue
        if path.stem in slides:
            raise ValueError(f"Ambiguous WSI for {path.stem!r}: {slides[path.stem]} and {path}")
        slides[path.stem] = path.resolve()

    # Validate the complete input inventory before creating any output files.
    tasks = []
    protected_paths = set(inputs) | set(slides.values())
    for slide_id, source in sources.items():
        if slide_id not in slides:
            raise FileNotFoundError(f"No matching WSI for {source.name} under {wsi_dir}")
        wsi = slides[slide_id]
        output = output_dir / f"{slide_id}_patches.h5"
        if output.resolve() in protected_paths:
            raise ValueError(f"Output would overwrite an input file: {output}")
        patch_size, patch_size_level0 = read_geometry(source, args.mag)
        source_stat = source.stat()
        wsi_stat = wsi.stat()
        tasks.append(
            HSVGenerationTask(
                mag=args.mag,
                slide_id=slide_id,
                source_h5=str(source),
                wsi_path=str(wsi),
                output_h5=str(output),
                patch_size=patch_size,
                patch_size_level0=patch_size_level0,
                minimum_tissue_coverage=args.min_tissue_coverage,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                wsi_size=wsi_stat.st_size,
                wsi_mtime_ns=wsi_stat.st_mtime_ns,
            )
        )
    return tasks


def filter_slides(tasks: list[HSVGenerationTask], workers: int):
    if workers == 1:
        for task in tasks:
            yield generate_filter_for_slide(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(generate_filter_for_slide, task): task for task in tasks}
            for future in as_completed(pending):
                try:
                    yield future.result()
                except Exception as error:
                    task = pending[future]
                    raise RuntimeError(f"HSV filtering failed for {task.slide_id}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TRIDENT H5 file or directory (one magnification).")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory for SLIDE_patches.h5 files.")
    parser.add_argument("--wsi-dir", type=Path, required=True, help="WSI directory, searched recursively by slide filename stem.")
    parser.add_argument("--mag", type=int, required=True, help="Target magnification, e.g. 5, 10, 20, or 40.")
    parser.add_argument("--pattern", default="*.h5", help="Recursive input glob when --input is a directory.")
    parser.add_argument(
        "--wsi-ext", nargs="+", default=["svs", "tif", "tiff", "ndpi", "mrxs", "scn", "vms", "vmu", "bif"],
        help="Allowed WSI extensions (case-insensitive, with or without a leading dot).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel slide workers; use 1 for serial execution (default: 8).")
    parser.add_argument("--min-tissue-coverage", type=float, default=0.45, help="Minimum HSV-tissue pixel fraction (default: 0.45).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and patch geometry without reading slide pixels or writing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = build_tasks(args)
        print(f"{len(tasks)} slide(s) at {args.mag}x; minimum tissue coverage: {args.min_tissue_coverage:g}")
        print(f"Output directory: {args.output.expanduser().resolve()}")
        if args.dry_run:
            print("Dry run: inputs validated; no output files written.")
            return 0

        accepted = source = cached = empty = 0
        results = filter_slides(tasks, min(args.workers, len(tasks)))
        for result in tqdm(results, total=len(tasks), desc="HSV filtering", unit="slide"):
            accepted += result["accepted_patches"]
            source += result["source_patches"]
            cached += int(result["cached"])
            empty += int(result["accepted_patches"] == 0)
        print(f"Kept {accepted:,}/{source:,} patches; {cached} cached slide(s), {empty} empty slide(s).")
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
