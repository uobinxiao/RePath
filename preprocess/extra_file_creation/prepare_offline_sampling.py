"""Compile an approved offline semantic index into WSIPatch mmap files.

The source index is produced by ``../hierarchical_clustering``.  This compiler
is deliberately strict: fixture indexes and structurally passing but
unapproved indexes are not accepted for training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
OFFLINE_SCHEMA_VERSION = 1
ALLOWED_MAGNIFICATIONS = {5, 10, 20, 40}
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

REQUIRED_STAGES = (
    "01_build_tile_index",
    "02_build_cluster_centroids",
    "03_score_clusters_with_prompts",
    "04_compute_cluster_statistics",
    "05_build_composite_scores",
    "06_assign_sampling_buckets",
    "07_make_cluster_audit_montages",
    "08_validate_offline_index",
)

BUCKET_BITS = {
    "random_eligible_clusters": 0,
    "bridge_clusters": 1,
    "architecture_bridge_clusters": 2,
    "rare_clean_clusters": 3,
    "tissue_specific_clusters": 4,
    "architecture_specific_clusters": 5,
}

BUCKET_COLUMNS = {
    "random_eligible_clusters": "is_random_eligible",
    "bridge_clusters": "is_bridge",
    "architecture_bridge_clusters": "is_architecture_bridge",
    "architecture_specific_clusters": "is_architecture_specific",
    "rare_clean_clusters": "is_rare_clean",
    "tissue_specific_clusters": "is_tissue_specific",
    "artifact_excluded_clusters": "is_artifact_excluded",
    "support_excluded_clusters": "is_support_excluded",
}

PRIORITY_BUCKETS = (
    "bridge_clusters",
    "architecture_bridge_clusters",
    "architecture_specific_clusters",
    "rare_clean_clusters",
    "tissue_specific_clusters",
)

REQUIRED_BUCKET_COLUMNS = {
    "magnification",
    "fine_cluster_id",
    "coarse_cluster_id",
    "sampling_eligible",
    *BUCKET_COLUMNS.values(),
}

TILE_COLUMNS = (
    "magnification",
    "x",
    "y",
    "patch_size_level0",
    "fine_cluster_id",
    "coarse_cluster_id",
    "wsi_path",
)

ENTRY_COLUMNS = [
    "x",
    "y",
    "slide_id",
    "target_magnification",
    "patch_size_level0",
]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def load_json_object(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    with open(source, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def atomic_save(path: str | Path, value: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _require_schema(name: str, value: Mapping[str, Any]) -> None:
    if int(value.get("schema_version", -1)) != OFFLINE_SCHEMA_VERSION:
        raise ValueError(
            f"{name}: unsupported schema_version={value.get('schema_version')}; "
            f"expected {OFFLINE_SCHEMA_VERSION}"
        )


def _require_true(name: str, value: Any) -> None:
    if value is not True:
        raise ValueError(f"Approved offline index requires {name}=true")


def _require_identity(name: str, value: Any) -> str:
    normalized = str(value)
    if not _valid_sha256(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _validate_audit_render_evidence(root: Path) -> None:
    audit_path = root / "audit_selection.parquet"
    required = (
        "cluster_key",
        "audit_bucket",
        "magnification",
        "fine_cluster_id",
        "render_status",
        "montage_path",
        "montage_sha256",
    )
    schema = pq.read_schema(audit_path)
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValueError(
            f"{audit_path}: missing production audit evidence columns: {', '.join(missing)}"
        )
    table = pq.read_table(audit_path, columns=list(required)).combine_chunks()
    if table.num_rows == 0:
        raise ValueError(f"{audit_path}: approved production audit selection is empty")
    for column in required:
        if table[column].null_count:
            raise ValueError(f"{audit_path}: {column} contains null values")

    values = {column: table[column].to_pylist() for column in required}
    seen = set()
    montage_root = (root / "audit_montages").resolve()
    for row_index in range(table.num_rows):
        cluster_key = str(values["cluster_key"][row_index])
        bucket = str(values["audit_bucket"][row_index])
        mag = int(values["magnification"][row_index])
        fine = int(values["fine_cluster_id"][row_index])
        identity = (cluster_key, bucket)
        if identity in seen:
            raise ValueError(f"{audit_path}: duplicate audit evidence for {identity}")
        seen.add(identity)
        if cluster_key != f"{mag}x:{fine}":
            raise ValueError(f"{audit_path}: row {row_index} cluster identity is inconsistent")
        if str(values["render_status"][row_index]) != "rendered":
            raise ValueError(f"{audit_path}: row {row_index} montage is not rendered")

        expected_relative = (
            Path("audit_montages") / bucket / f"{mag}x" / f"cluster_{fine}.png"
        )
        montage_value = str(values["montage_path"][row_index])
        montage_relative = Path(montage_value)
        if montage_relative.is_absolute() or montage_relative != expected_relative:
            raise ValueError(f"{audit_path}: row {row_index} has an unexpected montage path")
        montage_path = (root / montage_relative).resolve()
        if montage_root != montage_path and montage_root not in montage_path.parents:
            raise ValueError(f"{audit_path}: row {row_index} montage escapes audit_montages")
        if not montage_path.is_file() or montage_path.stat().st_size <= 0:
            raise ValueError(f"{audit_path}: row {row_index} montage is missing or empty")
        expected_digest = str(values["montage_sha256"][row_index]).lower()
        if not _valid_sha256(expected_digest) or sha256_file(montage_path) != expected_digest:
            raise ValueError(f"{audit_path}: row {row_index} montage SHA-256 is invalid")


def _normalize_split(value: str) -> str:
    split = str(value).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", split):
        raise ValueError(f"Invalid split {value!r}; expected letters, digits, or underscore")
    return split


def _read_bucket_records(
    root: Path,
    magnifications: Sequence[int],
    sampling: Mapping[str, Any],
    coarse_index: Mapping[str, Any],
) -> Dict[tuple[int, int], Dict[str, Any]]:
    bucket_path = root / "fine_cluster_buckets.parquet"
    schema = pq.read_schema(bucket_path)
    missing = sorted(REQUIRED_BUCKET_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"{bucket_path}: missing required columns: {', '.join(missing)}")
    for column in ("magnification", "fine_cluster_id", "coarse_cluster_id"):
        if not pa.types.is_integer(schema.field(column).type):
            raise ValueError(f"{bucket_path}: {column} must be an integer column")
    for column in ("sampling_eligible", *BUCKET_COLUMNS.values()):
        if not pa.types.is_boolean(schema.field(column).type):
            raise ValueError(f"{bucket_path}: {column} must be a boolean column")

    ordered_columns = sorted(REQUIRED_BUCKET_COLUMNS)
    table = pq.read_table(bucket_path, columns=ordered_columns).combine_chunks()
    for column in ordered_columns:
        if table[column].null_count:
            raise ValueError(f"{bucket_path}: {column} contains null values")

    columns = {name: table[name].to_pylist() for name in ordered_columns}
    records: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row_index in range(table.num_rows):
        mag = int(columns["magnification"][row_index])
        fine = int(columns["fine_cluster_id"][row_index])
        coarse = int(columns["coarse_cluster_id"][row_index])
        if mag not in magnifications:
            raise ValueError(f"{bucket_path}: unexpected magnification {mag} at row {row_index}")
        if fine < 0 or coarse < 0 or fine > INT32_MAX or coarse > INT32_MAX:
            raise ValueError(f"{bucket_path}: cluster IDs must fit non-negative int32 values")
        key = (mag, fine)
        if key in records:
            raise ValueError(f"{bucket_path}: duplicate cluster identity {mag}x:{fine}")
        memberships = {
            bucket: bool(columns[column][row_index])
            for bucket, column in BUCKET_COLUMNS.items()
        }
        eligible = bool(columns["sampling_eligible"][row_index])
        if memberships["random_eligible_clusters"] != eligible:
            raise ValueError(f"{bucket_path}: {mag}x:{fine} random eligibility is inconsistent")
        excluded = (
            memberships["artifact_excluded_clusters"]
            or memberships["support_excluded_clusters"]
        )
        if eligible != (not excluded):
            raise ValueError(
                f"{bucket_path}: {mag}x:{fine} eligibility must be the complement of exclusions"
            )
        if any(memberships[name] for name in PRIORITY_BUCKETS) and not eligible:
            raise ValueError(f"{bucket_path}: {mag}x:{fine} priority membership is not eligible")
        mask = sum(
            (1 << bit) for bucket, bit in BUCKET_BITS.items() if memberships[bucket]
        )
        records[key] = {
            "magnification": mag,
            "fine_cluster_id": fine,
            "coarse_cluster_id": coarse,
            "memberships": memberships,
            "bucket_mask": mask,
            "eligible": eligible,
        }

    expected_mags = {str(mag) for mag in magnifications}
    sampling_mags = sampling.get("magnifications")
    coarse_mags = coarse_index.get("magnifications")
    if not isinstance(sampling_mags, dict) or set(sampling_mags) != expected_mags:
        raise ValueError("sampling_buckets.json magnifications do not match run_manifest.json")
    if not isinstance(coarse_mags, dict) or set(coarse_mags) != expected_mags:
        raise ValueError("coarse_to_fine_index.json magnifications do not match run_manifest.json")

    for mag in magnifications:
        mag_records = {fine: record for (record_mag, fine), record in records.items() if record_mag == mag}
        if not mag_records:
            raise ValueError(f"fine_cluster_buckets.parquet contains no clusters for {mag}x")
        if not any(record["eligible"] for record in mag_records.values()):
            raise ValueError(f"fine_cluster_buckets.parquet contains no random-eligible clusters for {mag}x")

        sampling_mag = sampling_mags[str(mag)]
        sampling_buckets = sampling_mag.get("buckets") if isinstance(sampling_mag, dict) else None
        if not isinstance(sampling_buckets, dict):
            raise ValueError(f"sampling_buckets.json is missing {mag}x buckets")
        for bucket, column in BUCKET_COLUMNS.items():
            payload = sampling_buckets.get(bucket)
            if not isinstance(payload, dict) or not isinstance(payload.get("fine_cluster_ids"), list):
                raise ValueError(f"sampling_buckets.json is missing {mag}x {bucket}")
            actual = [int(value) for value in payload["fine_cluster_ids"]]
            if len(actual) != len(set(actual)):
                raise ValueError(f"sampling_buckets.json has duplicate IDs in {mag}x {bucket}")
            expected = sorted(
                fine for fine, record in mag_records.items() if record["memberships"][bucket]
            )
            if sorted(actual) != expected:
                raise ValueError(
                    f"sampling_buckets.json {mag}x {bucket} does not match fine_cluster_buckets.parquet"
                )

        expected_by_coarse: Dict[int, list[int]] = defaultdict(list)
        for fine, record in mag_records.items():
            expected_by_coarse[int(record["coarse_cluster_id"])].append(fine)
        sampling_coarse = sampling_mag.get("coarse_clusters")
        if not isinstance(sampling_coarse, dict):
            raise ValueError(f"sampling_buckets.json is missing {mag}x coarse hierarchy")
        coarse_mag = coarse_mags[str(mag)]
        if not isinstance(coarse_mag, dict):
            raise ValueError(f"coarse_to_fine_index.json is missing {mag}x hierarchy")
        expected_coarse_keys = {str(value) for value in expected_by_coarse}
        if set(sampling_coarse) != expected_coarse_keys or set(coarse_mag) != expected_coarse_keys:
            raise ValueError(f"{mag}x coarse hierarchy keys do not match the bucket table")
        for coarse, fine_ids in expected_by_coarse.items():
            expected_fine = sorted(fine_ids)
            sampling_record = sampling_coarse[str(coarse)]
            coarse_record = coarse_mag[str(coarse)]
            if sorted(int(value) for value in sampling_record.get("fine_cluster_ids", [])) != expected_fine:
                raise ValueError(f"sampling_buckets.json has an invalid {mag}x coarse {coarse} hierarchy")
            if sorted(int(value) for value in coarse_record.get("fine_clusters", [])) != expected_fine:
                raise ValueError(f"coarse_to_fine_index.json has an invalid {mag}x coarse {coarse} hierarchy")
            for bucket in BUCKET_COLUMNS:
                expected_bucket = sorted(
                    fine for fine in expected_fine if mag_records[fine]["memberships"][bucket]
                )
                sampling_by_bucket = sampling_record.get("by_bucket", {})
                if sorted(int(value) for value in sampling_by_bucket.get(bucket, [])) != expected_bucket:
                    raise ValueError(
                        f"sampling_buckets.json has an invalid {mag}x coarse {coarse} {bucket} hierarchy"
                    )
                if sorted(int(value) for value in coarse_record.get(bucket, [])) != expected_bucket:
                    raise ValueError(
                        f"coarse_to_fine_index.json has an invalid {mag}x coarse {coarse} {bucket} hierarchy"
                    )
    return records


def _discover_tile_files(root: Path, magnifications: Sequence[int]) -> Dict[int, list[Path]]:
    dataset_root = root / "tile_index.parquet"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing partitioned tile index: {dataset_root}")
    partition_dirs: Dict[int, Path] = {}
    for path in dataset_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"mag_partition=(\d+)x", path.name)
        if match:
            partition_dirs[int(match.group(1))] = path
    if set(partition_dirs) != set(magnifications):
        raise ValueError(
            "tile_index.parquet magnification partitions do not match run_manifest.json: "
            f"found={sorted(partition_dirs)}, expected={list(magnifications)}"
        )
    result: Dict[int, list[Path]] = {}
    for mag in magnifications:
        files = sorted(path for path in partition_dirs[mag].rglob("*.parquet") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No Parquet files in {partition_dirs[mag]}")
        result[mag] = files
    return result


def _validate_approval(root: Path) -> tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    list[int],
]:
    manifest = load_json_object(root / "run_manifest.json")
    report = load_json_object(root / "validation_report.json")
    sampling = load_json_object(root / "sampling_buckets.json")
    coarse_index = load_json_object(root / "coarse_to_fine_index.json")
    summary = load_json_object(root / "tile_index_summary.json")
    for name, value in (
        ("run_manifest.json", manifest),
        ("validation_report.json", report),
        ("sampling_buckets.json", sampling),
        ("coarse_to_fine_index.json", coarse_index),
        ("tile_index_summary.json", summary),
    ):
        _require_schema(name, value)

    if manifest.get("profile") != "production" or report.get("profile") != "production":
        raise ValueError("Only approved production offline indexes may be compiled; sample_smoke is rejected")
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, dict) or resolved.get("profile") != "production":
        raise ValueError("run_manifest.json must contain a resolved production configuration")
    _require_true("run_manifest.approved", manifest.get("approved"))
    _require_true("validation_report.passed", report.get("passed"))
    _require_true("validation_report.approved", report.get("approved"))

    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("run_manifest.json is missing stage records")
    for stage in REQUIRED_STAGES:
        record = stages.get(stage)
        if not isinstance(record, dict) or record.get("status") != "completed":
            raise ValueError(f"run_manifest.json stage {stage} is not completed")
    stage8 = stages["08_validate_offline_index"]
    details = stage8.get("details")
    if not isinstance(details, dict):
        raise ValueError("run_manifest.json stage 08 is missing completion details")
    _require_true("stage 08 details.passed", details.get("passed"))
    _require_true("stage 08 details.approved", details.get("approved"))

    config_hash = _require_identity("run_manifest.config_hash", manifest.get("config_hash"))
    input_fingerprint = _require_identity(
        "run_manifest.input_fingerprint", manifest.get("input_fingerprint")
    )
    if sampling.get("profile") != "production":
        raise ValueError("sampling_buckets.json is not a production artifact")
    if sampling.get("config_hash") != config_hash:
        raise ValueError("sampling_buckets.json config_hash does not match run_manifest.json")
    if sampling.get("input_fingerprint") != input_fingerprint:
        raise ValueError("sampling_buckets.json input_fingerprint does not match run_manifest.json")
    if coarse_index.get("config_hash") != config_hash:
        raise ValueError("coarse_to_fine_index.json config_hash does not match run_manifest.json")

    manifest_review = manifest.get("audit_review")
    report_review = report.get("audit_review")
    if not isinstance(manifest_review, dict) or manifest_review != report_review:
        raise ValueError("run manifest and validation report audit_review records must match exactly")
    binding = manifest_review.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Approved offline index is missing its audit-review binding")
    expected_binding = {
        "run_config_hash": config_hash,
        "run_input_fingerprint": input_fingerprint,
        "audit_selection_sha256": sha256_file(root / "audit_selection.parquet"),
    }
    if binding != expected_binding:
        raise ValueError("Audit-review binding does not match the current approved index")
    _require_identity("audit_review.sha256", manifest_review.get("sha256"))
    _validate_audit_render_evidence(root)

    raw_magnifications = resolved.get("magnifications")
    if not isinstance(raw_magnifications, list) or not raw_magnifications:
        raise ValueError("run_manifest.json resolved_config.magnifications must be a non-empty list")
    magnifications = sorted({int(value) for value in raw_magnifications})
    if len(magnifications) != len(raw_magnifications) or not set(magnifications).issubset(
        ALLOWED_MAGNIFICATIONS
    ):
        raise ValueError("Offline magnifications must be unique values selected from 5, 10, 20, and 40")
    summary_mags = summary.get("magnifications")
    if not isinstance(summary_mags, dict) or set(summary_mags) != {str(mag) for mag in magnifications}:
        raise ValueError("tile_index_summary.json magnifications do not match run_manifest.json")
    return manifest, report, sampling, coarse_index, summary, magnifications


def validate_source(root: str | Path) -> Dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Offline index directory does not exist: {source_root}")
    manifest, report, sampling, coarse_index, summary, magnifications = _validate_approval(
        source_root
    )
    records = _read_bucket_records(source_root, magnifications, sampling, coarse_index)
    tile_files = _discover_tile_files(source_root, magnifications)

    digest_names = (
        "run_manifest.json",
        "validation_report.json",
        "sampling_buckets.json",
        "coarse_to_fine_index.json",
        "fine_cluster_buckets.parquet",
        "tile_index_summary.json",
        "audit_selection.parquet",
    )
    digests = {name: sha256_file(source_root / name) for name in digest_names}
    tile_records = []
    for mag in magnifications:
        for path in tile_files[mag]:
            relative = path.relative_to(source_root).as_posix()
            tile_records.append(
                {
                    "magnification": mag,
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    binding = dict(manifest["audit_review"]["binding"])
    source_provenance = {
        "schema_version": OFFLINE_SCHEMA_VERSION,
        "profile": "production",
        "approved": True,
        "config_hash": str(manifest["config_hash"]),
        "input_fingerprint": str(manifest["input_fingerprint"]),
        "audit_review_binding": binding,
        "run_manifest_sha256": digests["run_manifest.json"],
        "validation_report_sha256": digests["validation_report.json"],
        "sampling_buckets_sha256": digests["sampling_buckets.json"],
        "coarse_to_fine_index_sha256": digests["coarse_to_fine_index.json"],
        "fine_cluster_buckets_sha256": digests["fine_cluster_buckets.parquet"],
        "tile_index_summary_sha256": digests["tile_index_summary.json"],
        "audit_selection_sha256": digests["audit_selection.parquet"],
        "tile_index_files": tile_records,
    }
    source_provenance["source_index_id"] = canonical_hash(source_provenance)
    return {
        "root": source_root,
        "manifest": manifest,
        "report": report,
        "summary": summary,
        "magnifications": magnifications,
        "bucket_records": records,
        "tile_files": tile_files,
        "source": source_provenance,
    }


def _iter_batches(files: Iterable[Path], batch_size: int) -> Iterator[pa.RecordBatch]:
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = sorted(set(TILE_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for column in TILE_COLUMNS[:-1]:
            if not pa.types.is_integer(parquet.schema_arrow.field(column).type):
                raise ValueError(f"{path}: {column} must be an integer column")
        if not (pa.types.is_string(parquet.schema_arrow.field("wsi_path").type) or
                pa.types.is_large_string(parquet.schema_arrow.field("wsi_path").type)):
            raise ValueError(f"{path}: wsi_path must be a string column")
        yield from parquet.iter_batches(batch_size=batch_size, columns=list(TILE_COLUMNS))


def _cluster_lookup(
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    mag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag_records = sorted(
        (fine, record)
        for (record_mag, fine), record in records.items()
        if record_mag == mag
    )
    return (
        np.asarray([fine for fine, _ in mag_records], dtype=np.int64),
        np.asarray(
            [int(record["coarse_cluster_id"]) for _, record in mag_records],
            dtype=np.int64,
        ),
        np.asarray([bool(record["eligible"]) for _, record in mag_records], dtype=bool),
    )


def _validated_batch(
    batch: pa.RecordBatch,
    mag: int,
    lookup: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[pa.Table, np.ndarray]:
    table = pa.Table.from_batches([batch]).combine_chunks()
    for column in TILE_COLUMNS:
        if table[column].null_count:
            raise ValueError(f"{mag}x tile index column {column} contains null values")
    arrays = {
        name: table[name].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        for name in TILE_COLUMNS[:-1]
    }
    if np.any(arrays["magnification"] != mag):
        actual = np.unique(arrays["magnification"]).tolist()
        raise ValueError(f"{mag}x partition contains mismatched magnifications: {actual}")
    for coordinate in ("x", "y"):
        values = arrays[coordinate]
        if values.size and (int(values.min()) < 0 or int(values.max()) > INT32_MAX):
            raise OverflowError(f"{mag}x {coordinate} coordinates must fit non-negative int32 values")
    patch_sizes = arrays["patch_size_level0"]
    if patch_sizes.size and (int(patch_sizes.min()) <= 0 or int(patch_sizes.max()) > INT32_MAX):
        raise OverflowError(f"{mag}x patch_size_level0 must fit positive int32 values")
    fine_ids = arrays["fine_cluster_id"]
    coarse_ids = arrays["coarse_cluster_id"]
    if fine_ids.size and (
        int(fine_ids.min()) < 0
        or int(fine_ids.max()) > INT32_MAX
        or int(coarse_ids.min()) < 0
        or int(coarse_ids.max()) > INT32_MAX
    ):
        raise OverflowError(f"{mag}x cluster IDs must fit non-negative int32 values")

    paths = table["wsi_path"].to_pylist()
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError(f"{mag}x tile index contains an empty wsi_path")

    if fine_ids.size:
        known_fine, expected_coarse, eligible_values = lookup
        positions = np.searchsorted(known_fine, fine_ids)
        in_bounds = positions < len(known_fine)
        safe_positions = np.minimum(positions, len(known_fine) - 1)
        matched = in_bounds & (known_fine[safe_positions] == fine_ids)
        if not np.all(matched):
            missing = np.unique(fine_ids[~matched]).tolist()
            raise ValueError(f"{mag}x tile index references unknown fine clusters: {missing[:20]}")
        if not np.array_equal(expected_coarse[safe_positions], coarse_ids):
            raise ValueError(f"{mag}x tile index coarse/fine hierarchy disagrees with bucket table")
        selected = eligible_values[safe_positions]
    else:
        selected = np.zeros(0, dtype=bool)
    return table, selected


def _first_pass(
    mag: int,
    files: Sequence[Path],
    lookup: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected_rows: int,
    batch_size: int,
) -> tuple[Dict[tuple[int, int, str], int], int]:
    group_counts: Dict[tuple[int, int, str], int] = defaultdict(int)
    total_rows = 0
    for batch in _iter_batches(files, batch_size):
        table, selected = _validated_batch(batch, mag, lookup)
        total_rows += table.num_rows
        if not np.any(selected):
            continue
        selected_table = table.filter(pa.array(selected)).select(
            ["coarse_cluster_id", "fine_cluster_id", "wsi_path", "x"]
        )
        grouped = selected_table.group_by(
            ["coarse_cluster_id", "fine_cluster_id", "wsi_path"]
        ).aggregate([("x", "count")])
        coarse_values = grouped["coarse_cluster_id"].to_pylist()
        fine_values = grouped["fine_cluster_id"].to_pylist()
        path_values = grouped["wsi_path"].to_pylist()
        count_values = grouped["x_count"].to_pylist()
        for coarse, fine, path, count in zip(
            coarse_values, fine_values, path_values, count_values
        ):
            group_counts[(int(coarse), int(fine), str(path))] += int(count)
    if total_rows != expected_rows:
        raise ValueError(
            f"{mag}x tile count differs from tile_index_summary.json: "
            f"actual={total_rows}, expected={expected_rows}"
        )
    return dict(group_counts), total_rows


def _descriptor(path: Path, dtype: str, shape: Sequence[int], **extra: Any) -> Dict[str, Any]:
    return {
        "path": path.name,
        "dtype": dtype,
        "shape": [int(value) for value in shape],
        "sha256": sha256_file(path),
        **extra,
    }


def _compile_magnification(
    context: Mapping[str, Any],
    mag: int,
    partial_root: Path,
    batch_size: int,
) -> Dict[str, Any]:
    destination = partial_root / f"mag-{mag}x"
    temporary = partial_root / f".mag-{mag}x.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    records = context["bucket_records"]
    lookup = _cluster_lookup(records, mag)
    summary_record = context["summary"]["magnifications"][str(mag)]
    expected_rows = int(summary_record["tiles"])
    group_counts, total_rows = _first_pass(
        mag,
        context["tile_files"][mag],
        lookup,
        expected_rows,
        batch_size,
    )

    eligible_records = sorted(
        (
            int(record["coarse_cluster_id"]),
            int(record["fine_cluster_id"]),
            int(record["bucket_mask"]),
        )
        for (record_mag, _), record in records.items()
        if record_mag == mag and record["eligible"]
    )
    groups_by_cluster: Dict[tuple[int, int], list[tuple[int, int, str]]] = defaultdict(list)
    for key in group_counts:
        groups_by_cluster[(key[0], key[1])].append(key)
    for groups in groups_by_cluster.values():
        groups.sort(key=lambda key: key[2])
    missing_clusters = [
        fine
        for coarse, fine, _ in eligible_records
        if not groups_by_cluster.get((coarse, fine))
    ]
    if missing_clusters:
        raise ValueError(f"{mag}x random-eligible clusters contain no tiles: {missing_clusters[:20]}")

    fine_records = np.asarray(
        [[mag, coarse, fine] for coarse, fine, _ in eligible_records], dtype=np.int32
    ).reshape(-1, 3)
    bucket_masks = np.asarray([mask for _, _, mask in eligible_records], dtype=np.uint8)
    fine_slide_offsets = [0]
    slide_entry_offsets = [0]
    ordered_groups: list[tuple[int, int, str]] = []
    for coarse, fine, _ in eligible_records:
        groups = groups_by_cluster[(coarse, fine)]
        ordered_groups.extend(groups)
        fine_slide_offsets.append(len(ordered_groups))
        for key in groups:
            slide_entry_offsets.append(slide_entry_offsets[-1] + group_counts[key])
    selected_rows = int(slide_entry_offsets[-1])
    if selected_rows <= 0:
        raise ValueError(f"{mag}x contains no random-eligible tiles")

    slide_paths = sorted({key[2] for key in ordered_groups})
    slide_to_id = {path: index for index, path in enumerate(slide_paths)}
    group_starts = {
        key: slide_entry_offsets[index] for index, key in enumerate(ordered_groups)
    }
    cursors = dict(group_starts)
    entries_temporary = temporary / ".entries.npy.tmp"
    entries = np.lib.format.open_memmap(
        entries_temporary,
        mode="w+",
        dtype=np.int32,
        shape=(selected_rows, len(ENTRY_COLUMNS)),
    )
    total_rows_second_pass = 0
    try:
        for batch in _iter_batches(context["tile_files"][mag], batch_size):
            table, selected = _validated_batch(batch, mag, lookup)
            total_rows_second_pass += table.num_rows
            if not np.any(selected):
                continue
            selected_table = table.filter(pa.array(selected)).select(
                [
                    "coarse_cluster_id",
                    "fine_cluster_id",
                    "wsi_path",
                    "x",
                    "y",
                    "patch_size_level0",
                ]
            )
            order = pc.sort_indices(
                selected_table,
                sort_keys=[
                    ("coarse_cluster_id", "ascending"),
                    ("fine_cluster_id", "ascending"),
                    ("wsi_path", "ascending"),
                ],
            )
            sorted_table = selected_table.take(order).combine_chunks()
            row_count = sorted_table.num_rows
            if not row_count:
                continue
            coarse_values = sorted_table["coarse_cluster_id"].to_numpy(zero_copy_only=False)
            fine_values = sorted_table["fine_cluster_id"].to_numpy(zero_copy_only=False)
            path_values = sorted_table["wsi_path"]
            changes = np.ones(row_count, dtype=bool)
            if row_count > 1:
                numeric_change = (coarse_values[1:] != coarse_values[:-1]) | (
                    fine_values[1:] != fine_values[:-1]
                )
                path_change = pc.not_equal(path_values.slice(1), path_values.slice(0, row_count - 1))
                changes[1:] = numeric_change | path_change.to_numpy(zero_copy_only=False)
            starts = np.flatnonzero(changes)
            ends = np.r_[starts[1:], row_count]
            x_values = sorted_table["x"].to_numpy(zero_copy_only=False)
            y_values = sorted_table["y"].to_numpy(zero_copy_only=False)
            patch_values = sorted_table["patch_size_level0"].to_numpy(zero_copy_only=False)
            for start, end in zip(starts.tolist(), ends.tolist()):
                path = str(path_values[start].as_py())
                key = (int(coarse_values[start]), int(fine_values[start]), path)
                cursor = cursors[key]
                next_cursor = cursor + (end - start)
                entries[cursor:next_cursor, 0] = x_values[start:end]
                entries[cursor:next_cursor, 1] = y_values[start:end]
                entries[cursor:next_cursor, 2] = slide_to_id[path]
                entries[cursor:next_cursor, 3] = mag
                entries[cursor:next_cursor, 4] = patch_values[start:end]
                cursors[key] = next_cursor
        entries.flush()
    finally:
        del entries
    if total_rows_second_pass != total_rows:
        raise RuntimeError(f"{mag}x source changed between compiler passes")
    incomplete = [
        key
        for key, start in group_starts.items()
        if cursors[key] != start + group_counts[key]
    ]
    if incomplete:
        raise RuntimeError(f"{mag}x compiler did not fill every semantic group")
    os.replace(entries_temporary, temporary / "entries.npy")
    atomic_save(temporary / "fine-records.npy", fine_records)
    atomic_save(
        temporary / "fine-slide-offsets.npy",
        np.asarray(fine_slide_offsets, dtype=np.int64),
    )
    atomic_save(
        temporary / "slide-entry-offsets.npy",
        np.asarray(slide_entry_offsets, dtype=np.int64),
    )
    atomic_save(temporary / "bucket-masks.npy", bucket_masks)
    atomic_json(temporary / "slide-paths.json", slide_paths)

    files = {
        "entries": _descriptor(temporary / "entries.npy", "int32", [selected_rows, 5]),
        "fine_records": _descriptor(
            temporary / "fine-records.npy", "int32", fine_records.shape
        ),
        "fine_slide_offsets": _descriptor(
            temporary / "fine-slide-offsets.npy", "int64", [len(fine_slide_offsets)]
        ),
        "slide_entry_offsets": _descriptor(
            temporary / "slide-entry-offsets.npy", "int64", [len(slide_entry_offsets)]
        ),
        "bucket_masks": _descriptor(
            temporary / "bucket-masks.npy", "uint8", bucket_masks.shape
        ),
        "slide_paths": {
            "path": "slide-paths.json",
            "sha256": sha256_file(temporary / "slide-paths.json"),
        },
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_index_id": context["source"]["source_index_id"],
        "magnification": mag,
        "total_source_rows": total_rows,
        "selected_entries": selected_rows,
        "fine_records": len(fine_records),
        "slide_groups": len(ordered_groups),
        "slides": len(slide_paths),
        "files": files,
    }
    atomic_json(temporary / "meta.json", meta)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return meta


def _verify_intermediate(partial_root: Path, meta: Mapping[str, Any]) -> bool:
    mag = int(meta["magnification"])
    root = partial_root / f"mag-{mag}x"
    if not root.is_dir():
        return False
    try:
        for descriptor in meta["files"].values():
            path = root / descriptor["path"]
            if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
                return False
        persisted = load_json_object(root / "meta.json")
        return persisted == meta
    except (KeyError, OSError, ValueError, TypeError):
        return False


def _load_array(path: Path, dtype: np.dtype, shape: Sequence[int]) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.dtype(dtype) or list(value.shape) != list(shape):
        raise ValueError(
            f"{path}: expected dtype={np.dtype(dtype)} shape={list(shape)}, "
            f"got dtype={value.dtype} shape={list(value.shape)}"
        )
    return value


def _build_publication(
    context: Mapping[str, Any],
    partial_root: Path,
    output_root: Path,
    split: str,
    state: Mapping[str, Any],
) -> Path:
    publication = partial_root / "publication"
    if publication.exists():
        shutil.rmtree(publication)
    semantic_root = publication / f"semantic-index-{split}"
    semantic_root.mkdir(parents=True)
    metas = [state["completed"][str(mag)] for mag in context["magnifications"]]
    total_entries = sum(int(meta["selected_entries"]) for meta in metas)
    total_fine = sum(int(meta["fine_records"]) for meta in metas)
    total_groups = sum(int(meta["slide_groups"]) for meta in metas)
    slide_paths: list[str] = []
    for meta in metas:
        mag_root = partial_root / f"mag-{int(meta['magnification'])}x"
        with open(mag_root / "slide-paths.json", "r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{mag_root / 'slide-paths.json'}: expected a string list")
        slide_paths.extend(values)
    global_paths = sorted(set(slide_paths))
    if len(global_paths) > INT32_MAX:
        raise OverflowError("Number of WSI paths exceeds the int32 slide_id range")
    global_slide_ids = {path: index for index, path in enumerate(global_paths)}

    entries_path = publication / f"entries-{split}.npy"
    entries = np.lib.format.open_memmap(
        entries_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_entries, len(ENTRY_COLUMNS)),
    )
    fine_records_parts = []
    bucket_mask_parts = []
    fine_slide_offsets = [0]
    slide_entry_offsets = [0]
    entry_cursor = 0
    group_cursor = 0
    copy_rows = 1_000_000
    try:
        for meta in metas:
            mag = int(meta["magnification"])
            mag_root = partial_root / f"mag-{mag}x"
            local_entries = _load_array(
                mag_root / "entries.npy",
                np.int32,
                [int(meta["selected_entries"]), 5],
            )
            local_paths = json.loads((mag_root / "slide-paths.json").read_text(encoding="utf-8"))
            remap = np.asarray([global_slide_ids[path] for path in local_paths], dtype=np.int32)
            for start in range(0, len(local_entries), copy_rows):
                end = min(len(local_entries), start + copy_rows)
                destination = entries[entry_cursor + start : entry_cursor + end]
                destination[:] = local_entries[start:end]
                destination[:, 2] = remap[np.asarray(local_entries[start:end, 2], dtype=np.int64)]
            local_fine = _load_array(
                mag_root / "fine-records.npy", np.int32, [int(meta["fine_records"]), 3]
            )
            local_masks = _load_array(
                mag_root / "bucket-masks.npy", np.uint8, [int(meta["fine_records"])]
            )
            local_fine_offsets = _load_array(
                mag_root / "fine-slide-offsets.npy",
                np.int64,
                [int(meta["fine_records"]) + 1],
            )
            local_slide_offsets = _load_array(
                mag_root / "slide-entry-offsets.npy",
                np.int64,
                [int(meta["slide_groups"]) + 1],
            )
            fine_records_parts.append(np.asarray(local_fine))
            bucket_mask_parts.append(np.asarray(local_masks))
            fine_slide_offsets.extend(
                (np.asarray(local_fine_offsets[1:], dtype=np.int64) + group_cursor).tolist()
            )
            slide_entry_offsets.extend(
                (np.asarray(local_slide_offsets[1:], dtype=np.int64) + entry_cursor).tolist()
            )
            entry_cursor += int(meta["selected_entries"])
            group_cursor += int(meta["slide_groups"])
        entries.flush()
    finally:
        del entries
    fine_records = np.concatenate(fine_records_parts, axis=0).astype(np.int32, copy=False)
    bucket_masks = np.concatenate(bucket_mask_parts).astype(np.uint8, copy=False)
    fine_slide_offsets_array = np.asarray(fine_slide_offsets, dtype=np.int64)
    slide_entry_offsets_array = np.asarray(slide_entry_offsets, dtype=np.int64)
    if fine_records.shape != (total_fine, 3):
        raise RuntimeError("Compiled fine-record shape is inconsistent")
    if len(bucket_masks) != total_fine or len(fine_slide_offsets_array) != total_fine + 1:
        raise RuntimeError("Compiled fine-record sidecar lengths are inconsistent")
    if len(slide_entry_offsets_array) != total_groups + 1:
        raise RuntimeError("Compiled slide-group sidecar lengths are inconsistent")
    if int(fine_slide_offsets_array[-1]) != total_groups:
        raise RuntimeError("fine-slide offsets do not terminate at the slide-group count")
    if int(slide_entry_offsets_array[-1]) != total_entries:
        raise RuntimeError("slide-entry offsets do not terminate at the entry count")

    fine_records_path = semantic_root / "fine-records.npy"
    fine_slide_path = semantic_root / "fine-slide-offsets.npy"
    slide_entry_path = semantic_root / "slide-entry-offsets.npy"
    bucket_masks_path = semantic_root / "bucket-masks.npy"
    atomic_save(fine_records_path, fine_records)
    atomic_save(fine_slide_path, fine_slide_offsets_array)
    atomic_save(slide_entry_path, slide_entry_offsets_array)
    atomic_save(bucket_masks_path, bucket_masks)

    mapping_path = publication / f"entries-{split}.json"
    schema_path = publication / f"entries-{split}.schema.json"
    atomic_json(mapping_path, {str(index): path for index, path in enumerate(global_paths)})
    atomic_json(
        schema_path,
        {
            "version": 2,
            "columns": ENTRY_COLUMNS,
            "note": "WSIPatch reads level-0 regions of patch_size_level0 and resizes to 256.",
            "semantic_index": f"semantic-index-{split}/manifest.json",
        },
    )

    entries_descriptor = {
        "path": entries_path.name,
        "dtype": "int32",
        "shape": [total_entries, len(ENTRY_COLUMNS)],
        "columns": ENTRY_COLUMNS,
        "sha256": sha256_file(entries_path),
        "mapping": {"path": mapping_path.name, "sha256": sha256_file(mapping_path)},
        "schema": {"path": schema_path.name, "sha256": sha256_file(schema_path)},
    }
    arrays = {
        "fine_records": _descriptor(
            fine_records_path,
            "int32",
            fine_records.shape,
            columns=["magnification", "coarse_cluster_id", "fine_cluster_id"],
        ),
        "fine_slide_offsets": _descriptor(
            fine_slide_path, "int64", fine_slide_offsets_array.shape
        ),
        "slide_entry_offsets": _descriptor(
            slide_entry_path, "int64", slide_entry_offsets_array.shape
        ),
        "bucket_masks": _descriptor(bucket_masks_path, "uint8", bucket_masks.shape),
    }
    counts_by_mag = {
        str(int(meta["magnification"])): {
            "entries": int(meta["selected_entries"]),
            "fine_records": int(meta["fine_records"]),
            "slide_groups": int(meta["slide_groups"]),
            "slides": int(meta["slides"]),
            "bucket_fine_counts": {
                bucket: int(
                    np.count_nonzero(
                        np.bitwise_and(
                            bucket_masks[
                                fine_records[:, 0] == int(meta["magnification"])
                            ],
                            np.uint8(1 << bit),
                        )
                    )
                )
                for bucket, bit in BUCKET_BITS.items()
            },
        }
        for meta in metas
    }
    manifest_without_id = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "status": "complete",
        "entry_count": total_entries,
        "source": context["source"],
        "entries": entries_descriptor,
        "arrays": arrays,
        "bucket_bits": BUCKET_BITS,
        "sort_order": [
            "magnification",
            "coarse_cluster_id",
            "fine_cluster_id",
            "wsi_path",
        ],
        "counts": {
            "entries": total_entries,
            "fine_records": total_fine,
            "slide_groups": total_groups,
            "slides": len(global_paths),
            "magnifications": counts_by_mag,
        },
    }
    compiled_manifest = {
        **manifest_without_id,
        "compiled_index_id": canonical_hash(manifest_without_id),
    }
    atomic_json(semantic_root / "manifest.json", compiled_manifest)
    return publication


def _generated_paths(output_root: Path, split: str) -> list[Path]:
    return [
        output_root / f"entries-{split}.npy",
        output_root / f"entries-{split}.json",
        output_root / f"entries-{split}.schema.json",
        output_root / f"semantic-index-{split}",
    ]


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _verify_complete_output(
    output_root: Path,
    split: str,
    source_index_id: str,
) -> bool:
    manifest_path = output_root / f"semantic-index-{split}" / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_json_object(manifest_path)
    if manifest.get("split") != split or manifest.get("source", {}).get(
        "source_index_id"
    ) != source_index_id:
        raise FileExistsError(
            f"Existing {split} compiled index was built from a different source; use --overwrite"
        )
    descriptors = [manifest.get("entries", {})]
    descriptors.extend(manifest.get("arrays", {}).values())
    for descriptor in descriptors:
        base = output_root if descriptor is manifest.get("entries") else manifest_path.parent
        path = base / descriptor.get("path", "")
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeError(f"Existing compiled index is corrupt or incomplete: {path}")
    entries = manifest["entries"]
    for key in ("mapping", "schema"):
        descriptor = entries.get(key, {})
        path = output_root / descriptor.get("path", "")
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeError(f"Existing compiled index is corrupt or incomplete: {path}")
    expected_id = canonical_hash(
        {key: value for key, value in manifest.items() if key != "compiled_index_id"}
    )
    if manifest.get("compiled_index_id") != expected_id:
        raise RuntimeError("Existing compiled manifest has an invalid compiled_index_id")
    return True


def compile_index(
    offline_index: str | Path,
    output_dir: str | Path,
    *,
    split: str = "TRAIN",
    overwrite: bool = False,
    batch_size: int = 65_536,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    split = _normalize_split(split)
    context = validate_source(offline_index)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    partial_root = output_root / f".prepare-offline-sampling-{split}.partial"
    generated = _generated_paths(output_root, split)

    if overwrite:
        for path in [*generated, partial_root]:
            _remove_path(path)
    elif _verify_complete_output(
        output_root, split, context["source"]["source_index_id"]
    ):
        if partial_root.exists():
            partial_state_path = partial_root / "state.json"
            partial_state = (
                load_json_object(partial_state_path)
                if partial_state_path.is_file()
                else {}
            )
            if partial_state.get("source_index_id") != context["source"]["source_index_id"]:
                raise RuntimeError(
                    f"Incompatible stale compiler state at {partial_root}; use --overwrite"
                )
            shutil.rmtree(partial_root)
        print(f"Compiled offline sampling index is current: {generated[-1]}")
        return load_json_object(generated[-1] / "manifest.json")

    state_path = partial_root / "state.json"
    if state_path.exists():
        state = load_json_object(state_path)
        if (
            int(state.get("schema_version", -1)) != SCHEMA_VERSION
            or state.get("split") != split
            or state.get("source_index_id") != context["source"]["source_index_id"]
        ):
            raise RuntimeError(
                f"Incompatible partial compiler state at {partial_root}; use --overwrite to discard it"
            )
    else:
        existing = [path for path in generated if path.exists()]
        if existing:
            raise FileExistsError(
                "Generated output paths already exist without matching compiler state: "
                + ", ".join(str(path) for path in existing)
                + "; use --overwrite"
            )
        partial_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "source_index_id": context["source"]["source_index_id"],
            "completed": {},
        }
        atomic_json(state_path, state)

    completed = state.get("completed")
    if not isinstance(completed, dict):
        raise ValueError(f"{state_path}: completed must be an object")
    for mag in context["magnifications"]:
        key = str(mag)
        meta = completed.get(key)
        if isinstance(meta, dict) and _verify_intermediate(partial_root, meta):
            print(f"Compiler resume: {mag}x is already complete")
            continue
        if meta is not None:
            raise RuntimeError(
                f"Partial compiler output for {mag}x is corrupt; use --overwrite to discard it"
            )
        print(f"Compiling approved offline index: {mag}x")
        meta = _compile_magnification(context, mag, partial_root, batch_size)
        completed[key] = meta
        atomic_json(state_path, state)

    current_context = validate_source(context["root"])
    if current_context["source"]["source_index_id"] != context["source"]["source_index_id"]:
        raise RuntimeError(
            "Approved offline index changed while it was being compiled; rerun with --overwrite"
        )
    publication = _build_publication(context, partial_root, output_root, split, state)
    semantic_source = publication / f"semantic-index-{split}"
    for filename in (
        f"entries-{split}.npy",
        f"entries-{split}.json",
        f"entries-{split}.schema.json",
    ):
        os.replace(publication / filename, output_root / filename)
    if generated[-1].exists():
        raise FileExistsError(f"Cannot atomically publish over {generated[-1]}; use --overwrite")
    os.replace(semantic_source, generated[-1])
    manifest = load_json_object(generated[-1] / "manifest.json")
    shutil.rmtree(partial_root)
    print(
        f"Compiled {manifest['counts']['entries']} eligible tiles from approved offline index "
        f"to {output_root}"
    )
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an approved production offline semantic index for SimDINO WSIPatch."
    )
    parser.add_argument("--offline-index", required=True, help="Approved offline index directory")
    parser.add_argument("--output-dir", required=True, help="WSIPatch extra output directory")
    parser.add_argument("--split", default="TRAIN", help="Output split suffix (default: TRAIN)")
    parser.add_argument(
        "--batch-size", type=int, default=65_536, help="Parquet streaming batch size"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard incompatible partial/generated outputs before compiling",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    compile_index(
        args.offline_index,
        args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

# python prepare_offline_sampling.py --offline-index __REPATH_PRIVATE_SCRATCH_ROOT_005__/tcga_conch_v1/offline_index/ --output-dir __REPATH_PRIVATE_PROJECT_ROOT_002__/tcga-extra-clustering_v5
