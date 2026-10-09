from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from tqdm import tqdm


SCHEMA_VERSION = 1


def discover_h5(inputs: Sequence[str], pattern: str) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        source = Path(value).expanduser().resolve()
        if source.is_file():
            paths.add(source)
        elif source.is_dir():
            paths.update(path.resolve() for path in source.rglob(pattern) if path.is_file())
        else:
            raise FileNotFoundError(f"Input does not exist: {source}")
    if not paths:
        raise FileNotFoundError(f"No H5 files matching {pattern!r} under the requested inputs")
    return sorted(paths)


def json_safe_attr(value: Any) -> Any:
    """Recursively normalize arbitrary H5 attribute values for strict JSON."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return json_safe_attr(value.item() if value.size == 1 else value.tolist())
    if isinstance(value, np.generic):
        return json_safe_attr(value.item())
    if isinstance(value, (list, tuple)):
        return [json_safe_attr(item) for item in value]
    if isinstance(value, dict):
        return {str(json_safe_attr(key)): json_safe_attr(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class NearZeroRuns:
    def __init__(self, max_examples: int) -> None:
        self.max_examples = max_examples
        self.examples: list[dict[str, int]] = []
        self.longest: dict[str, int] | None = None
        self._start: int | None = None
        self._end: int | None = None

    def add_indices(self, indices: np.ndarray) -> None:
        for raw_index in indices:
            index = int(raw_index)
            if self._start is None:
                self._start = self._end = index
            elif index == int(self._end) + 1:
                self._end = index
            else:
                self._close()
                self._start = self._end = index

    def finish(self) -> None:
        self._close()

    def _close(self) -> None:
        if self._start is None or self._end is None:
            return
        run = {
            "start": self._start,
            "end": self._end,
            "length": self._end - self._start + 1,
        }
        if len(self.examples) < self.max_examples:
            self.examples.append(run)
        if self.longest is None or run["length"] > self.longest["length"]:
            self.longest = run
        self._start = None
        self._end = None


def validate_h5(
    path: Path,
    *,
    features_key: str,
    coords_key: str,
    expected_dim: int | None,
    chunk_size: int,
    near_zero_threshold: float,
    max_examples: int,
) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "status": "valid",
        "errors": [],
        "n_rows": None,
        "feature_dim": None,
        "nonfinite_row_count": 0,
        "nonfinite_examples": [],
        "exact_zero_row_count": 0,
        "near_zero_row_count": 0,
        "near_zero_examples": [],
        "near_zero_run_examples": [],
        "longest_near_zero_run": None,
        "valid_feature_norm_min": None,
        "valid_feature_norm_max": None,
    }
    try:
        with h5py.File(path, "r") as handle:
            result["root_attrs"] = {
                key: json_safe_attr(value) for key, value in handle.attrs.items()
            }
            missing = [key for key in (features_key, coords_key) if key not in handle]
            if missing:
                result["errors"].append(f"missing datasets: {missing}")
                result["status"] = "invalid"
                return result

            features = handle[features_key]
            coords = handle[coords_key]
            result["feature_attrs"] = {
                key: json_safe_attr(value) for key, value in features.attrs.items()
            }
            result["coords_attrs"] = {
                key: json_safe_attr(value) for key, value in coords.attrs.items()
            }
            if features.ndim != 2:
                result["errors"].append(f"features must be 2-D, got {features.shape}")
                result["status"] = "invalid"
                return result

            n_rows, feature_dim = (int(features.shape[0]), int(features.shape[1]))
            result["n_rows"] = n_rows
            result["feature_dim"] = feature_dim
            if expected_dim is not None and feature_dim != expected_dim:
                result["errors"].append(
                    f"feature dimension {feature_dim} does not match expected {expected_dim}"
                )
            if coords.shape != (n_rows, 2):
                result["errors"].append(
                    f"coords shape {coords.shape} does not match ({n_rows}, 2)"
                )

            valid_norm_min = float("inf")
            valid_norm_max = 0.0
            runs = NearZeroRuns(max_examples)
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                values = np.asarray(features[start:end], dtype=np.float32)
                finite_elements = np.isfinite(values)
                finite_rows = finite_elements.all(axis=1)
                nonfinite_local_rows = np.flatnonzero(~finite_rows)
                result["nonfinite_row_count"] += int(nonfinite_local_rows.size)
                remaining_examples = max_examples - len(result["nonfinite_examples"])
                if remaining_examples > 0 and nonfinite_local_rows.size:
                    positions = np.argwhere(~finite_elements)
                    for row, column in positions[:remaining_examples]:
                        result["nonfinite_examples"].append(
                            {
                                "row": start + int(row),
                                "column": int(column),
                                "value": repr(float(values[row, column])),
                            }
                        )

                norms = np.full(end - start, np.nan, dtype=np.float64)
                if finite_rows.any():
                    norms[finite_rows] = np.linalg.norm(
                        values[finite_rows].astype(np.float64, copy=False),
                        axis=1,
                    )
                exact_zero = finite_rows & np.all(values == 0.0, axis=1)
                near_zero = finite_rows & (norms <= near_zero_threshold)
                exact_zero_local_rows = np.flatnonzero(exact_zero)
                near_zero_local_rows = np.flatnonzero(near_zero)
                result["exact_zero_row_count"] += int(exact_zero_local_rows.size)
                result["near_zero_row_count"] += int(near_zero_local_rows.size)
                remaining_examples = max_examples - len(result["near_zero_examples"])
                if remaining_examples > 0:
                    result["near_zero_examples"].extend(
                        (start + near_zero_local_rows[:remaining_examples]).astype(int).tolist()
                    )
                runs.add_indices(start + near_zero_local_rows)

                valid_norms = norms[finite_rows & ~near_zero]
                if valid_norms.size:
                    valid_norm_min = min(valid_norm_min, float(valid_norms.min()))
                    valid_norm_max = max(valid_norm_max, float(valid_norms.max()))

            runs.finish()
            result["near_zero_run_examples"] = runs.examples
            result["longest_near_zero_run"] = runs.longest
            if valid_norm_min != float("inf"):
                result["valid_feature_norm_min"] = valid_norm_min
                result["valid_feature_norm_max"] = valid_norm_max
            if result["nonfinite_row_count"] or result["near_zero_row_count"] or result["errors"]:
                result["status"] = "invalid"
    except (OSError, ValueError, TypeError) as error:
        result["status"] = "error"
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream-validate raw CONCH H5 feature shards before clustering or projection."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="H5 file or directory to scan recursively; repeat for multiple inputs.",
    )
    parser.add_argument("--report", required=True, help="Strict JSON report output path.")
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--coords-key", default="coords")
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=512,
        help="Expected raw CONCH feature dimension; use 0 to disable.",
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--near-zero-threshold", type=float, default=1e-12)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.expected_dim < 0:
        raise ValueError("--expected-dim must be non-negative")
    if args.near_zero_threshold < 0:
        raise ValueError("--near-zero-threshold must be non-negative")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")

    paths = discover_h5(args.input, args.pattern)
    records: list[dict[str, Any]] = []
    invalid_count = 0
    total_rows = 0
    total_near_zero = 0
    total_nonfinite = 0
    progress = tqdm(
        paths,
        desc="Validate raw CONCH H5",
        unit="file",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for path in progress:
        record = validate_h5(
            path,
            features_key=args.features_key,
            coords_key=args.coords_key,
            expected_dim=args.expected_dim or None,
            chunk_size=args.chunk_size,
            near_zero_threshold=args.near_zero_threshold,
            max_examples=args.max_examples,
        )
        records.append(record)
        total_rows += int(record["n_rows"] or 0)
        total_near_zero += int(record["near_zero_row_count"])
        total_nonfinite += int(record["nonfinite_row_count"])
        if record["status"] != "valid":
            invalid_count += 1
        progress.set_postfix_str(f"invalid={invalid_count}", refresh=False)
        if args.fail_fast and invalid_count:
            break

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "inputs": [str(Path(value).expanduser().resolve()) for value in args.input],
            "pattern": args.pattern,
            "features_key": args.features_key,
            "coords_key": args.coords_key,
            "expected_dim": args.expected_dim or None,
            "chunk_size": args.chunk_size,
            "near_zero_threshold": args.near_zero_threshold,
            "max_examples": args.max_examples,
            "fail_fast": args.fail_fast,
        },
        "summary": {
            "discovered_files": len(paths),
            "scanned_files": len(records),
            "valid_files": sum(record["status"] == "valid" for record in records),
            "invalid_files": sum(record["status"] == "invalid" for record in records),
            "error_files": sum(record["status"] == "error" for record in records),
            "total_rows": total_rows,
            "near_zero_rows": total_near_zero,
            "nonfinite_rows": total_nonfinite,
        },
        "files": records,
    }
    report = Path(args.report).expanduser().resolve()
    atomic_json(report, payload)
    print(json.dumps(payload["summary"], indent=2, allow_nan=False))
    print(f"Report: {report}")
    if invalid_count:
        print("Invalid raw feature shards:", file=sys.stderr)
        for record in records:
            if record["status"] != "valid":
                print(
                    f"  {record['path']} status={record['status']} "
                    f"near_zero={record['near_zero_row_count']} "
                    f"nonfinite={record['nonfinite_row_count']} "
                    f"errors={record['errors']}",
                    file=sys.stderr,
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
