from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from PIL import Image


HSV_HUE_RANGE = (90.0, 180.0)
HSV_SATURATION_RANGE = (8.0, 255.0)
HSV_VALUE_RANGE = (103.0, 255.0)
HSV_GENERATION_SCHEMA_VERSION = 1

_HSV_SHIFT = 12
_HSV_ROUNDING_DELTA = 1 << (_HSV_SHIFT - 1)
_SATURATION_DIVISION_TABLE = np.zeros(256, dtype=np.int32)
_HUE_DIVISION_TABLE = np.zeros(256, dtype=np.int32)
for _index in range(1, 256):
    _SATURATION_DIVISION_TABLE[_index] = int(
        np.rint((255 << _HSV_SHIFT) / float(_index))
    )
    _HUE_DIVISION_TABLE[_index] = int(
        np.rint((180 << _HSV_SHIFT) / (6.0 * _index))
    )


@dataclass(frozen=True)
class HSVGenerationTask:
    mag: int
    slide_id: str
    source_h5: str
    wsi_path: str
    output_h5: str
    patch_size: int
    patch_size_level0: int
    minimum_tissue_coverage: float
    source_size: int
    source_mtime_ns: int
    wsi_size: int
    wsi_mtime_ns: int

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "schema_version": HSV_GENERATION_SCHEMA_VERSION,
                "task": asdict(self),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def read_coords(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"{path}: missing dataset 'coords'")
        dataset = handle["coords"]
        coords = np.asarray(dataset[:])
        if coords.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if dataset.ndim != 2 or dataset.shape[1] != 2:
            raise ValueError(f"{path}: coords must have shape (N, 2), got {dataset.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords must use an integer dtype, got {coords.dtype}")
    return coords.astype(np.int64, copy=False)


def rgb_to_opencv_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB to OpenCV-scaled HSV (H 0..180, S/V 0..255)."""
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"RGB patch must have shape (H, W, 3), got {values.shape}")
    if values.dtype != np.uint8:
        raise ValueError(f"RGB patch must use uint8 values, got {values.dtype}")
    channels = values.astype(np.int32, copy=False)
    red = channels[..., 0]
    green = channels[..., 1]
    blue = channels[..., 2]
    maximum = channels.max(axis=2)
    minimum = channels.min(axis=2)
    delta = maximum - minimum

    saturation = (
        delta * _SATURATION_DIVISION_TABLE[maximum] + _HSV_ROUNDING_DELTA
    ) >> _HSV_SHIFT
    hue_numerator = np.where(
        maximum == red,
        green - blue,
        np.where(
            maximum == green,
            blue - red + 2 * delta,
            red - green + 4 * delta,
        ),
    )
    hue = (
        hue_numerator * _HUE_DIVISION_TABLE[delta] + _HSV_ROUNDING_DELTA
    ) >> _HSV_SHIFT
    hue[hue < 0] += 180
    return (
        hue.astype(np.uint8),
        saturation.astype(np.uint8),
        maximum.astype(np.uint8),
    )


def tissue_coverage(rgb: np.ndarray) -> float:
    hue, saturation, value = rgb_to_opencv_hsv(rgb)
    accepted = (
        (hue >= HSV_HUE_RANGE[0])
        & (hue <= HSV_HUE_RANGE[1])
        & (saturation >= HSV_SATURATION_RANGE[0])
        & (saturation <= HSV_SATURATION_RANGE[1])
        & (value >= HSV_VALUE_RANGE[0])
        & (value <= HSV_VALUE_RANGE[1])
    )
    return float(accepted.mean()) if accepted.size else 0.0


def extract_patch(slide: Any, x: int, y: int, source_size: int, patch_size: int) -> Image.Image:
    if source_size <= 0 or patch_size <= 0:
        raise ValueError("patch_size and patch_size_level0 must be positive")
    desired_downsample = max(1.0, source_size / float(patch_size))
    level = int(slide.get_best_level_for_downsample(desired_downsample))
    level_downsamples = tuple(float(value) for value in slide.level_downsamples)
    if not level_downsamples:
        raise ValueError("OpenSlide pyramid has no levels")
    level = min(max(level, 0), len(level_downsamples) - 1)
    level_downsample = level_downsamples[level]
    if not math.isfinite(level_downsample) or level_downsample <= 0:
        raise ValueError(f"Invalid OpenSlide level downsample: {level_downsample}")
    level_size = max(1, math.ceil(source_size / level_downsample))
    region = slide.read_region(
        (int(x), int(y)),
        level,
        (level_size, level_size),
    ).convert("RGB")
    if region.size != (patch_size, patch_size):
        region = region.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    return region


def cached_result(task: HSVGenerationTask) -> dict[str, Any] | None:
    output = Path(task.output_h5)
    if not output.is_file():
        return None
    try:
        with h5py.File(output, "r") as handle:
            if handle.attrs.get("generation_fingerprint") != task.fingerprint:
                return None
            if "coords" not in handle or handle["coords"].ndim != 2:
                return None
            if handle["coords"].shape[1] != 2:
                return None
            accepted = int(handle["coords"].shape[0])
            source = int(handle.attrs["source_patch_count"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return {
        "slide_id": task.slide_id,
        "output_h5": str(output),
        "source_patches": source,
        "accepted_patches": accepted,
        "cached": True,
    }


def write_result(
    task: HSVGenerationTask,
    accepted_coords: np.ndarray,
    source_patch_count: int,
) -> None:
    output = Path(task.output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset(
                "coords",
                data=accepted_coords.astype(np.int64, copy=False),
                dtype=np.int64,
            )
            handle.attrs["generation_fingerprint"] = task.fingerprint
            handle.attrs["generation_schema_version"] = HSV_GENERATION_SCHEMA_VERSION
            handle.attrs["method"] = "hsv_tissue_coverage"
            handle.attrs["minimum_tissue_coverage"] = task.minimum_tissue_coverage
            handle.attrs["hue_range"] = np.asarray(HSV_HUE_RANGE, dtype=np.float32)
            handle.attrs["saturation_range"] = np.asarray(
                HSV_SATURATION_RANGE,
                dtype=np.float32,
            )
            handle.attrs["value_range"] = np.asarray(HSV_VALUE_RANGE, dtype=np.float32)
            handle.attrs["source_h5"] = task.source_h5
            handle.attrs["wsi_path"] = task.wsi_path
            handle.attrs["patch_size"] = task.patch_size
            handle.attrs["patch_size_level0"] = task.patch_size_level0
            handle.attrs["source_patch_count"] = source_patch_count
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_filter_for_slide(
    task: HSVGenerationTask,
    open_slide: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    cached = cached_result(task)
    if cached is not None:
        return cached

    coords = read_coords(Path(task.source_h5))
    accepted_rows: list[int] = []
    slide = None
    try:
        if coords.size:
            if open_slide is None:
                try:
                    import openslide
                except (ImportError, OSError) as error:
                    raise ImportError(
                        "openslide-python and the OpenSlide native library are required "
                        "to generate HSV tissue filters"
                    ) from error
                open_slide = openslide.OpenSlide
            slide = open_slide(task.wsi_path)
            for row, (x, y) in enumerate(coords):
                patch = extract_patch(
                    slide,
                    int(x),
                    int(y),
                    task.patch_size_level0,
                    task.patch_size,
                )
                if tissue_coverage(np.asarray(patch, dtype=np.uint8)) >= (
                    task.minimum_tissue_coverage
                ):
                    accepted_rows.append(row)
    finally:
        if slide is not None:
            slide.close()

    accepted_coords = coords[np.asarray(accepted_rows, dtype=np.int64)]
    if not accepted_rows:
        accepted_coords = np.empty((0, 2), dtype=np.int64)
    write_result(task, accepted_coords, len(coords))
    return {
        "slide_id": task.slide_id,
        "output_h5": task.output_h5,
        "source_patches": len(coords),
        "accepted_patches": len(accepted_rows),
        "cached": False,
    }
