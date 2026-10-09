from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class CoordinateData:
    coords: list[Coordinate]
    duplicate_count: int
    feature_rows: int | None


@dataclass(frozen=True)
class Comparison:
    slide_id: str
    hsv_path: Path
    conch_path: Path
    hsv_count: int
    conch_count: int
    feature_rows: int | None
    hsv_duplicate_count: int
    conch_duplicate_count: int
    shared_count: int
    hsv_only: list[Coordinate]
    conch_only: list[Coordinate]
    exact: bool
    subset: bool
    ordered_subset: bool


def normalized_slide_id(path: Path) -> str:
    """Return the slide ID shared by NAME_patches.h5 and NAME.h5."""
    stem = path.stem
    return stem[: -len("_patches")] if stem.endswith("_patches") else stem


def discover_h5(directory: Path) -> dict[str, Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {directory}")

    discovered: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.h5")):
        slide_id = normalized_slide_id(path)
        previous = discovered.get(slide_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate normalized slide ID {slide_id!r}: {previous} and {path}"
            )
        discovered[slide_id] = path.resolve()
    return discovered


def read_coordinates(
    path: Path,
    *,
    coords_key: str,
    read_feature_rows: bool,
) -> CoordinateData:
    with h5py.File(path, "r") as handle:
        if coords_key not in handle:
            raise ValueError(f"{path}: missing dataset {coords_key!r}")
        dataset = handle[coords_key]
        if dataset.ndim != 2 or dataset.shape[1] != 2:
            raise ValueError(
                f"{path}: {coords_key} must have shape (N, 2), got {dataset.shape}"
            )

        array = np.asarray(dataset[:])
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"{path}: {coords_key} must use an integer dtype, got {array.dtype}"
            )
        coords = [(int(x), int(y)) for x, y in array]

        feature_rows: int | None = None
        if read_feature_rows:
            if "features" not in handle:
                raise ValueError(f"{path}: missing dataset 'features'")
            features = handle["features"]
            if features.ndim != 2:
                raise ValueError(f"{path}: features must be 2-D, got {features.shape}")
            feature_rows = int(features.shape[0])
            if feature_rows != len(coords):
                raise ValueError(
                    f"{path}: features has {feature_rows} rows but {coords_key} "
                    f"has {len(coords)} rows"
                )

    return CoordinateData(
        coords=coords,
        duplicate_count=len(coords) - len(set(coords)),
        feature_rows=feature_rows,
    )


def compare_pair(
    slide_id: str,
    hsv_path: Path,
    conch_path: Path,
    *,
    coords_key: str = "coords",
) -> Comparison:
    hsv = read_coordinates(
        hsv_path,
        coords_key=coords_key,
        read_feature_rows=False,
    )
    conch = read_coordinates(
        conch_path,
        coords_key=coords_key,
        read_feature_rows=True,
    )

    hsv_set = set(hsv.coords)
    conch_set = set(conch.coords)
    hsv_only = sorted(hsv_set - conch_set)
    conch_only = sorted(conch_set - hsv_set)
    no_duplicates = hsv.duplicate_count == 0 and conch.duplicate_count == 0
    subset = no_duplicates and not hsv_only
    filtered_conch_order = [coord for coord in conch.coords if coord in hsv_set]

    return Comparison(
        slide_id=slide_id,
        hsv_path=hsv_path,
        conch_path=conch_path,
        hsv_count=len(hsv.coords),
        conch_count=len(conch.coords),
        feature_rows=conch.feature_rows,
        hsv_duplicate_count=hsv.duplicate_count,
        conch_duplicate_count=conch.duplicate_count,
        shared_count=len(hsv_set & conch_set),
        hsv_only=hsv_only,
        conch_only=conch_only,
        exact=no_duplicates and hsv.coords == conch.coords,
        subset=subset,
        ordered_subset=subset and hsv.coords == filtered_conch_order,
    )


def comparison_passes(comparison: Comparison, requirement: str) -> bool:
    if requirement == "exact":
        return comparison.exact
    if requirement == "subset":
        return comparison.subset
    if requirement == "ordered-subset":
        return comparison.ordered_subset
    raise ValueError(f"Unknown requirement: {requirement}")


def format_examples(coords: list[Coordinate], limit: int) -> str:
    shown = coords[:limit]
    suffix = f" ... (+{len(coords) - limit} more)" if len(coords) > limit else ""
    return f"{shown}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match HSV-filtered patch H5 files to CONCH feature H5 files by slide "
            "ID and compare their patch coordinates."
        )
    )
    parser.add_argument("--hsv-dir", default="hsv_filtered_patches", type=Path)
    parser.add_argument("--conch-dir", default="conch_features", type=Path)
    parser.add_argument("--coords-key", default="coords")
    parser.add_argument(
        "--require",
        choices=("ordered-subset", "subset", "exact"),
        default="ordered-subset",
        help=(
            "ordered-subset (default): HSV patches occur in CONCH in the same "
            "relative order; subset: ignore order; exact: require identical rows"
        ),
    )
    parser.add_argument("--max-examples", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")

    try:
        hsv_files = discover_h5(args.hsv_dir)
        conch_files = discover_h5(args.conch_dir)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    hsv_ids = set(hsv_files)
    conch_ids = set(conch_files)
    hsv_only_files = sorted(hsv_ids - conch_ids)
    conch_only_files = sorted(conch_ids - hsv_ids)
    matched_ids = sorted(hsv_ids & conch_ids)
    failures = len(hsv_only_files) + len(conch_only_files)

    for slide_id in hsv_only_files:
        print(f"[FAIL] {slide_id}: HSV file has no matching CONCH file")
    for slide_id in conch_only_files:
        print(f"[FAIL] {slide_id}: CONCH file has no matching HSV file")

    for slide_id in matched_ids:
        try:
            result = compare_pair(
                slide_id,
                hsv_files[slide_id],
                conch_files[slide_id],
                coords_key=args.coords_key,
            )
        except (OSError, ValueError) as error:
            failures += 1
            print(f"[FAIL] {slide_id}: {error}")
            continue

        passed = comparison_passes(result, args.require)
        failures += int(not passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {slide_id}")
        print(
            f"  HSV={result.hsv_count}, CONCH coords={result.conch_count}, "
            f"CONCH features={result.feature_rows}, shared={result.shared_count}"
        )
        print(
            f"  exact={result.exact}, subset={result.subset}, "
            f"ordered_subset={result.ordered_subset}, "
            f"duplicates(HSV/CONCH)="
            f"{result.hsv_duplicate_count}/{result.conch_duplicate_count}"
        )
        if result.hsv_only:
            print(
                "  HSV coordinates missing from CONCH: "
                f"{format_examples(result.hsv_only, args.max_examples)}"
            )
        if result.conch_only:
            print(
                "  Additional CONCH coordinates: "
                f"{format_examples(result.conch_only, args.max_examples)}"
            )

    print(
        f"Summary: matched={len(matched_ids)}, HSV-only files={len(hsv_only_files)}, "
        f"CONCH-only files={len(conch_only_files)}, requirement={args.require}, "
        f"failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
