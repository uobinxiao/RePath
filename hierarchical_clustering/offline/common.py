from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np

from tools.conch_checkpoint import (
    CONCH_MODEL_NAME,
    EMBEDDING_CONTRACT_VERSION,
    EMBEDDING_CONTRACT_KIND_ATTR,
    FIXTURE_ONLY_ATTR,
    SMOKE_FIXTURE_CONTRACT_KIND,
    valid_checkpoint_sha256,
)

from . import SCHEMA_VERSION


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "profile": "production",
    "magnifications": [5, 10, 20, 40],
    "levels": {"fine": 1, "coarse": 2},
    "semantic": {
        "category_top_k": 3,
        "use_artifact_score_in_composite": True,
        "weights": {
            "architecture": 0.30,
            "tissue_compartment": 0.20,
            "pathology_state": 0.20,
            "cellular_nuclear": 0.15,
            "scale_context": 0.15,
        },
    },
    "statistics": {
        "tissue_key": "primary_site",
        "coverage_min_tiles": 20,
        "coverage_min_patients": 3,
    },
    "buckets": {
        "enable_artifact_exclusion": True,
        "top_quantile": 0.80,
        "artifact_exclude_above": 0.80,
        "support_exclude_below_patients": 5,
        "bridge_artifact_below": 0.70,
        "bridge_min_patients": 10,
        "bridge_min_coverage": 3,
        "architecture_min_score": 0.70,
        "architecture_artifact_below": 0.70,
        "architecture_min_patients": 10,
        "architecture_min_coverage": 3,
        "architecture_specific_min_score": 0.70,
        "architecture_specific_artifact_below": 0.70,
        "architecture_specific_min_patients": 10,
        "architecture_specific_min_coverage": 3,
        "architecture_specific_max_entropy": 0.50,
        "rare_artifact_below": 0.60,
        "rare_min_patients": 5,
        "tissue_specific_artifact_below": 0.70,
        "tissue_specific_min_patients": 10,
        "tissue_specific_max_entropy": 0.50,
    },
    "audit": {
        "clusters_per_bucket": 50,
        "tiles_per_cluster": 20,
        "max_tiles_per_wsi": 2,
        "representative_sample_size": 1_000_000,
        "representative_candidate_factor": 10,
        "tile_display_size": 256,
        "max_open_wsi": 16,
        "dominance_batch_size": 262_144,
        "seed": 7,
        "manifest_only": False,
        "strict_wsi": True,
    },
    "wsi": {
        "root": None,
        "manifest": None,
        "path_template": "{wsi_root}/{file_id}/{file_name}",
    },
    "runtime": {
        "feature_chunk_size": 8192,
        "parquet_row_group_size": 65_536,
    },
}


PROMPT_ENSEMBLE_FIELDS = ("polarity", "category", "concept_id")
AUDIT_REVIEW_BINDING_FIELDS = (
    "run_config_hash",
    "run_input_fingerprint",
    "audit_selection_sha256",
)


PATH_FIELDS = {
    "clustering_root",
    "metadata_path",
    "prompt_embeddings_path",
    "output_dir",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand_strings(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_strings(item) for key, item in value.items()}
    return value


def resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    if not isinstance(user_config, dict):
        raise ValueError(f"{config_path}: expected a JSON object")

    config = expand_strings(deep_merge(DEFAULT_CONFIG, user_config))
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema_version={config.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    base_dir = config_path.parent
    for field in PATH_FIELDS:
        config[field] = resolve_path(config.get(field), base_dir)

    projected = config.get("projected_feature_dirs")
    if not isinstance(projected, dict):
        raise ValueError("projected_feature_dirs must be a MAG -> directory mapping")
    config["projected_feature_dirs"] = {
        str(int(mag)): resolve_path(directory, base_dir) for mag, directory in projected.items()
    }
    config["magnifications"] = sorted({int(value) for value in config["magnifications"]})
    for mag in config["magnifications"]:
        if str(mag) not in config["projected_feature_dirs"]:
            raise ValueError(f"projected_feature_dirs is missing {mag}x")

    wsi = config["wsi"]
    wsi["root"] = resolve_path(wsi.get("root"), base_dir)
    wsi["manifest"] = resolve_path(wsi.get("manifest"), base_dir)
    config["_config_path"] = str(config_path)

    required = ["clustering_root", "metadata_path", "prompt_embeddings_path", "output_dir"]
    missing = [field for field in required if not config.get(field)]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    profile = str(config.get("profile", "production"))
    if profile not in {"production", "sample_smoke"}:
        raise ValueError("profile must be either 'production' or 'sample_smoke'")
    fine = int(config["levels"]["fine"])
    coarse = int(config["levels"]["coarse"])
    if fine <= 0 or coarse != fine + 1:
        raise ValueError("levels must be adjacent and satisfy coarse == fine + 1")
    if int(config["runtime"]["feature_chunk_size"]) <= 0:
        raise ValueError("runtime.feature_chunk_size must be positive")
    if int(config["runtime"]["parquet_row_group_size"]) <= 0:
        raise ValueError("runtime.parquet_row_group_size must be positive")
    if int(config["audit"]["max_open_wsi"]) <= 0:
        raise ValueError("audit.max_open_wsi must be positive")
    if int(config["audit"]["dominance_batch_size"]) <= 0:
        raise ValueError("audit.dominance_batch_size must be positive")
    for field in (
        "clusters_per_bucket",
        "tiles_per_cluster",
        "max_tiles_per_wsi",
        "representative_sample_size",
        "representative_candidate_factor",
        "tile_display_size",
    ):
        if int(config["audit"][field]) <= 0:
            raise ValueError(f"audit.{field} must be positive")
    if int(config["semantic"]["category_top_k"]) <= 0:
        raise ValueError("semantic.category_top_k must be positive")
    if not isinstance(config["semantic"]["use_artifact_score_in_composite"], bool):
        raise ValueError("semantic.use_artifact_score_in_composite must be a boolean")
    if not isinstance(config["buckets"]["enable_artifact_exclusion"], bool):
        raise ValueError("buckets.enable_artifact_exclusion must be a boolean")
    if int(config["statistics"]["coverage_min_tiles"]) <= 0:
        raise ValueError("statistics.coverage_min_tiles must be positive")
    if int(config["statistics"]["coverage_min_patients"]) <= 0:
        raise ValueError("statistics.coverage_min_patients must be positive")
    top_quantile = float(config["buckets"]["top_quantile"])
    if not 0.0 <= top_quantile <= 1.0:
        raise ValueError("buckets.top_quantile must be in [0, 1]")
    weights = config["semantic"]["weights"]
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError("semantic.weights must sum to 1")


def public_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def allow_smoke_fixture_contract(config: Mapping[str, Any]) -> bool:
    """Return true only for the explicitly non-production smoke profile."""
    return str(config.get("profile", "production")) == "sample_smoke"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_review_binding(
    manifest: Mapping[str, Any] | None,
    audit_selection_path: str | Path,
) -> Dict[str, str]:
    if manifest is None:
        raise ValueError("Cannot bind an audit review without a current run manifest")
    binding = {
        "run_config_hash": str(manifest.get("config_hash", "")),
        "run_input_fingerprint": str(manifest.get("input_fingerprint", "")),
        "audit_selection_sha256": sha256_file(audit_selection_path),
    }
    missing = [field for field, value in binding.items() if not value]
    if missing:
        raise ValueError(f"Run manifest is missing audit binding field(s): {', '.join(missing)}")
    return binding


def stat_record(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path)
    stat = candidate.stat()
    return {"path": str(candidate.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def clustering_path(config: Mapping[str, Any], mag: int, *parts: str) -> Path:
    return Path(config["clustering_root"]) / "clusters" / f"{mag}x" / Path(*parts)


def feature_index_path(config: Mapping[str, Any], mag: int) -> Path:
    return Path(config["clustering_root"]) / "features" / f"{mag}x" / "h5_index.json"


def upsampled_exclusion_path(config: Mapping[str, Any], mag: int) -> Path:
    return (
        Path(config["clustering_root"])
        / "features"
        / f"{mag}x"
        / "excluded_upsampled_h5.json"
    )


def raw_matrix_path(config: Mapping[str, Any], mag: int) -> Path:
    return Path(config["clustering_root"]) / "features" / f"{mag}x" / "features.npy"


def discover_h5(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*.h5") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No H5 files under {root}")
    return paths


def resolved_wsi_file_records(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Inventory the resolved WSIs used by clustering shards across all scales."""
    metadata_map = metadata_by_slide(config["metadata_path"])
    manifest = read_wsi_manifest(config["wsi"].get("manifest"))
    resolved_paths: set[str] = set()
    for mag in config["magnifications"]:
        for shard in load_h5_index(config, mag):
            submitter_id = slide_submitter_id(shard)
            if submitter_id not in metadata_map:
                raise KeyError(f"{mag}x {shard['slide_id']}: no metadata for {submitter_id}")
            resolved_path = resolve_wsi_path(metadata_map[submitter_id], config, manifest)
            if resolved_path is not None:
                resolved_paths.add(str(Path(resolved_path).expanduser().resolve()))

    records = []
    for path_string in sorted(resolved_paths):
        path = Path(path_string)
        try:
            stat = path.stat()
        except FileNotFoundError:
            records.append(
                {
                    "path": path_string,
                    "exists": False,
                    "size": None,
                    "mtime_ns": None,
                }
            )
        else:
            records.append(
                {
                    "path": path_string,
                    "exists": True,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return records


def input_fingerprint(config: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parent
    pipeline_paths = [
        *sorted(package_dir.glob("*.py")),
        repository_dir / "tools" / "conch_checkpoint.py",
    ]
    records: Dict[str, Any] = {
        "pipeline_code": [
            {
                "path": str(path.relative_to(repository_dir)),
                "sha256": sha256_file(path),
            }
            for path in pipeline_paths
        ],
        "metadata": {
            **stat_record(config["metadata_path"]),
            "sha256": sha256_file(config["metadata_path"]),
        },
        "prompts": {
            **stat_record(config["prompt_embeddings_path"]),
            "sha256": sha256_file(config["prompt_embeddings_path"]),
        },
        "magnifications": {},
    }
    wsi_manifest = config["wsi"].get("manifest")
    records["wsi_manifest"] = (
        {**stat_record(wsi_manifest), "sha256": sha256_file(wsi_manifest)}
        if wsi_manifest
        else None
    )
    records["wsi_files"] = resolved_wsi_file_records(config)
    clustering_controls = []
    for name in ("config.json", "summary.json"):
        path = Path(config["clustering_root"]) / name
        if path.exists():
            clustering_controls.append({**stat_record(path), "sha256": sha256_file(path)})
    records["clustering_controls"] = clustering_controls
    fine_level = int(config["levels"]["fine"])
    coarse_level = int(config["levels"]["coarse"])
    for mag in config["magnifications"]:
        shards = load_h5_index(config, mag)
        paths = [
            feature_index_path(config, mag),
            upsampled_exclusion_path(config, mag),
            raw_matrix_path(config, mag),
            clustering_path(config, mag, "fit_indices.npy"),
            clustering_path(config, mag, f"level{fine_level}", "assignments.npy"),
            clustering_path(config, mag, f"level{fine_level}", "centroids.npy"),
            clustering_path(config, mag, f"level{coarse_level}", "assignments.npy"),
            clustering_path(config, mag, f"level{coarse_level}", "fit_assignment.npy"),
        ]
        projected_paths = discover_h5(config["projected_feature_dirs"][str(mag)])
        clustering_records = []
        for path in paths:
            record = stat_record(path)
            if path.name in {
                "fit_indices.npy",
                "h5_index.json",
                "excluded_upsampled_h5.json",
            }:
                record["sha256"] = sha256_file(path)
            clustering_records.append(record)
        records["magnifications"][str(mag)] = {
            "clustering": clustering_records,
            "raw_shards": [stat_record(shard["path"]) for shard in shards],
            "selected_row_indices": [
                {
                    **stat_record(shard["selected_row_indices_path"]),
                    "sha256": sha256_file(shard["selected_row_indices_path"]),
                }
                for shard in shards
                if shard.get("selected_row_indices_path")
            ],
            "projected": [stat_record(path) for path in projected_paths],
        }
    return canonical_hash(records), records


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)


def output_root(config: Mapping[str, Any]) -> Path:
    return Path(config["output_dir"])


def manifest_path(config: Mapping[str, Any]) -> Path:
    return output_root(config) / "run_manifest.json"


def load_manifest(config: Mapping[str, Any]) -> Dict[str, Any] | None:
    path = manifest_path(config)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_identity(config: Mapping[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    config_hash = canonical_hash(public_config(config))
    fingerprint, inputs = input_fingerprint(config)
    return config_hash, fingerprint, inputs


def initialize_manifest(config: Mapping[str, Any], reset: bool = False) -> Dict[str, Any]:
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    config_hash, fingerprint, inputs = run_identity(config)
    current = load_manifest(config)
    if current is not None and not reset:
        if current.get("config_hash") != config_hash or current.get("input_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing output manifest was created from a different config or input inventory. "
                "Use a new output_dir, or rerun stage 1 with --overwrite."
            )
        return current
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": config.get("profile", "production"),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "git_commit": git_commit(),
        "config_path": config.get("_config_path"),
        "config_hash": config_hash,
        "input_fingerprint": fingerprint,
        "resolved_config": public_config(config),
        "inputs": inputs,
        "stages": {},
        "approved": False,
    }
    atomic_json(manifest_path(config), manifest)
    return manifest


def normalize_output_paths(config: Mapping[str, Any], paths: Sequence[str | Path]) -> list[Path]:
    root = output_root(config).resolve()
    normalized = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Stage output is outside output_dir: {resolved}")
        normalized.append(resolved)
    return normalized


def begin_stage(
    config: Mapping[str, Any],
    stage: str,
    outputs: Sequence[str | Path],
    overwrite: bool = False,
) -> tuple[bool, list[Path]]:
    reset = bool(overwrite and stage.startswith("01"))
    manifest = initialize_manifest(config, reset=reset)
    paths = normalize_output_paths(config, outputs)
    record = manifest.get("stages", {}).get(stage)
    if not overwrite and record and record.get("status") == "completed" and all(path.exists() for path in paths):
        print(f"{stage}: outputs are current; skipping")
        return False, paths
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{stage}: output(s) already exist without a completed matching manifest: "
            + ", ".join(str(path) for path in existing)
        )
    try:
        current_stage_number = int(stage.split("_", 1)[0])
    except ValueError:
        current_stage_number = -1
    if current_stage_number >= 0:
        stale_stages = []
        for stage_name in manifest.get("stages", {}):
            try:
                if int(stage_name.split("_", 1)[0]) >= current_stage_number:
                    stale_stages.append(stage_name)
            except ValueError:
                continue
        for stage_name in stale_stages:
            manifest["stages"].pop(stage_name, None)
        manifest["approved"] = False
        manifest.pop("audit_review", None)
    manifest.setdefault("stages", {})[stage] = {
        "status": "running",
        "started_at": utc_now(),
        "outputs": [str(path.relative_to(output_root(config))) for path in paths],
    }
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path(config), manifest)
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return True, paths


def complete_stage(config: Mapping[str, Any], stage: str, details: Mapping[str, Any] | None = None) -> None:
    manifest = load_manifest(config)
    if manifest is None:
        raise RuntimeError("Run manifest disappeared while completing a stage")
    record = manifest.setdefault("stages", {}).setdefault(stage, {})
    record["status"] = "completed"
    record["completed_at"] = utc_now()
    if details:
        record["details"] = dict(details)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path(config), manifest)


def fail_stage(config: Mapping[str, Any], stage: str, error: BaseException) -> None:
    manifest = load_manifest(config)
    if manifest is None:
        return
    record = manifest.setdefault("stages", {}).setdefault(stage, {})
    record["status"] = "failed"
    record["failed_at"] = utc_now()
    record["error"] = f"{type(error).__name__}: {error}"
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path(config), manifest)


def load_h5_index(config: Mapping[str, Any], mag: int) -> list[Dict[str, Any]]:
    path = feature_index_path(config, mag)
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    expected_start = 0
    seen_slides: set[str] = set()
    for record in records:
        slide_id = str(record.get("slide_id", ""))
        if not slide_id:
            raise ValueError(f"{path}: shard is missing slide_id")
        if slide_id in seen_slides:
            raise ValueError(f"{path}: duplicate slide_id {slide_id}")
        seen_slides.add(slide_id)
        if int(record["row_start"]) != expected_start:
            raise ValueError(f"{path}: non-contiguous row_start at {record.get('path')}")
        selected_path = record.get("selected_row_indices_path")
        if selected_path:
            selected = Path(str(selected_path))
            if not selected.is_file():
                raise ValueError(f"{path}: selected-row index does not exist: {selected}")
            expected_sha256 = str(record.get("selected_row_indices_sha256") or "")
            if not expected_sha256 or sha256_file(selected) != expected_sha256:
                raise ValueError(f"{path}: selected-row index fingerprint mismatch: {selected}")
            rows = np.load(selected, allow_pickle=False)
            if rows.dtype.kind not in "iu" or rows.shape != (int(record["n_tiles"]),):
                raise ValueError(f"{path}: invalid selected-row index shape or dtype: {selected}")
            source_n_tiles = int(record.get("source_n_tiles", record["n_tiles"]))
            if (
                np.any(rows < 0)
                or np.any(rows >= source_n_tiles)
                or np.any(rows[1:] <= rows[:-1])
            ):
                raise ValueError(f"{path}: selected rows must be unique, increasing, and in range")
        expected_start += int(record["n_tiles"])
    return records


def selected_source_rows(shard: Mapping[str, Any]) -> np.ndarray:
    """Map packed shard rows back to their source H5 row indices."""
    selected_path = shard.get("selected_row_indices_path")
    if selected_path:
        return np.asarray(np.load(str(selected_path), allow_pickle=False), dtype=np.int64)
    return np.arange(int(shard["n_tiles"]), dtype=np.int64)


def load_upsampled_exclusions(config: Mapping[str, Any], mag: int) -> list[Dict[str, Any]]:
    path = upsampled_exclusion_path(config, mag)
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list")

    seen_slides: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: exclusion record {index} must be an object")
        slide_id = str(record.get("slide_id", ""))
        if not slide_id:
            raise ValueError(f"{path}: exclusion record {index} is missing slide_id")
        if slide_id in seen_slides:
            raise ValueError(f"{path}: duplicate excluded slide_id {slide_id}")
        seen_slides.add(slide_id)
        try:
            source = float(record["level0_magnification"])
            target = float(record["target_magnification"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: exclusion record {slide_id} has invalid magnification metadata"
            ) from error
        if (
            record.get("reason") != "target_magnification_gt_level0_magnification"
            or not np.isfinite(source)
            or not np.isfinite(target)
            or source <= 0
            or target <= source
        ):
            raise ValueError(f"{path}: invalid upsampled exclusion for {slide_id}")
    return records


def projected_h5_map(config: Mapping[str, Any], mag: int) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in discover_h5(config["projected_feature_dirs"][str(mag)]):
        slide_id = h5_recorded_slide_id(path)
        if slide_id in result:
            raise ValueError(f"Duplicate projected slide_id {slide_id!r} for {mag}x")
        result[slide_id] = path
    return result


def load_metadata(path: str | Path) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list")
    return records


def metadata_file_slide_id(metadata: Mapping[str, Any]) -> str:
    """Return the canonical slide ID encoded in a TCGA WSI filename."""
    file_name = str(metadata.get("file_name", "")).strip()
    if not file_name:
        return ""
    return Path(file_name).name.split(".", 1)[0]


def metadata_by_slide(path: str | Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for record_index, record in enumerate(load_metadata(path)):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: metadata record {record_index} must be an object")
        entity_slide_id = str(record.get("slide_submitter_id", "")).strip()
        if not entity_slide_id:
            raise ValueError(f"{path}: record missing slide_submitter_id")
        file_slide_id = metadata_file_slide_id(record)
        if not file_slide_id:
            raise ValueError(f"{path}: record missing file_name")
        for lookup_key in dict.fromkeys((entity_slide_id, file_slide_id)):
            if lookup_key in result:
                raise ValueError(
                    f"{path}: duplicate or conflicting metadata lookup key {lookup_key!r}"
                )
            result[lookup_key] = record
    return result


def slide_submitter_id(shard: Mapping[str, Any]) -> str:
    return str(shard["slide_id"]).split(".", 1)[0]


def decode_strings(values: Iterable[Any]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def scalar_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def h5_is_upsampled(path: str | Path, target_magnification: int) -> bool:
    """Infer TRIDENT spatial upsampling from recorded source/target magnifications."""
    recorded: Dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        attribute_sources = [handle.attrs]
        for dataset_name in ("coords", "features"):
            if dataset_name in handle:
                attribute_sources.append(handle[dataset_name].attrs)
        for attributes in attribute_sources:
            for key in ("level0_magnification", "target_magnification"):
                if key in attributes:
                    value = scalar_h5_attr(attributes[key])
                    if value is not None:
                        recorded[key] = value

    try:
        source = float(recorded.get("level0_magnification", 0))
        target = float(recorded.get("target_magnification", target_magnification))
    except (TypeError, ValueError):
        return False
    return source > 0 and target > source


def h5_recorded_slide_id(path: str | Path) -> str:
    source_path = Path(path)
    recorded_name = None
    with h5py.File(source_path, "r") as handle:
        attribute_sources = [handle.attrs]
        for dataset_name in ("coords", "features"):
            if dataset_name in handle:
                attribute_sources.append(handle[dataset_name].attrs)
        for attributes in attribute_sources:
            if "name" in attributes:
                candidate = scalar_h5_attr(attributes["name"])
                if candidate not in (None, ""):
                    recorded_name = candidate
    return str(recorded_name or source_path.stem)


def validate_text_aligned_features(
    features: h5py.Dataset,
    path: str | Path,
    expected_rows: int,
    expected_dim: int,
    expected_checkpoint_sha256: str | None = None,
    allow_smoke_fixture: bool = False,
) -> str:
    if features.ndim != 2 or features.shape != (int(expected_rows), int(expected_dim)):
        raise ValueError(
            f"{path}: text-aligned feature shape {features.shape} does not match "
            f"({expected_rows}, {expected_dim})"
        )
    embedding_space = scalar_h5_attr(features.attrs.get("embedding_space"))
    if embedding_space != "conch_v1_text_aligned":
        raise ValueError(
            f"{path}: expected features.attrs['embedding_space']='conch_v1_text_aligned', "
            f"got {embedding_space!r}"
        )
    normalized = scalar_h5_attr(features.attrs.get("normalized"))
    if isinstance(normalized, str):
        normalized = normalized.strip().lower() in {"1", "true", "yes"}
    if normalized is not True:
        raise ValueError(f"{path}: text-aligned features must declare normalized=True")
    contract: Dict[str, Dict[str, Any]] = {}
    for label, attributes in (("root", features.file.attrs), ("features", features.attrs)):
        contract[label] = {
            "embedding_contract_version": scalar_h5_attr(
                attributes.get("embedding_contract_version")
            ),
            "model_name": scalar_h5_attr(attributes.get("model_name")),
            "checkpoint_sha256": scalar_h5_attr(attributes.get("checkpoint_sha256")),
        }
    if contract["root"] != contract["features"]:
        raise ValueError(f"{path}: root/features embedding contract attributes disagree")
    fixture_contract: Dict[str, Dict[str, Any]] = {}
    for label, attributes in (("root", features.file.attrs), ("features", features.attrs)):
        fixture_only = scalar_h5_attr(attributes.get(FIXTURE_ONLY_ATTR, False))
        if isinstance(fixture_only, str):
            fixture_only = fixture_only.strip().lower() in {"1", "true", "yes"}
        fixture_contract[label] = {
            EMBEDDING_CONTRACT_KIND_ATTR: scalar_h5_attr(
                attributes.get(EMBEDDING_CONTRACT_KIND_ATTR)
            ),
            FIXTURE_ONLY_ATTR: fixture_only is True,
        }
    if fixture_contract["root"] != fixture_contract["features"]:
        raise ValueError(f"{path}: root/features fixture contract attributes disagree")
    fixture_values = fixture_contract["features"]
    contract_kind = fixture_values[EMBEDDING_CONTRACT_KIND_ATTR]
    fixture_only = fixture_values[FIXTURE_ONLY_ATTR]
    if fixture_only or contract_kind is not None:
        if not (
            fixture_only
            and contract_kind == SMOKE_FIXTURE_CONTRACT_KIND
        ):
            raise ValueError(f"{path}: invalid fixture-only embedding contract marker")
        if not allow_smoke_fixture:
            raise ValueError(
                f"{path}: fixture-only embeddings are forbidden for production; "
                "they may only be used by the sample_smoke profile"
            )
    values = contract["features"]
    if values["embedding_contract_version"] != EMBEDDING_CONTRACT_VERSION:
        raise ValueError(
            f"{path}: expected embedding_contract_version={EMBEDDING_CONTRACT_VERSION}, "
            f"got {values['embedding_contract_version']!r}"
        )
    if values["model_name"] != CONCH_MODEL_NAME:
        raise ValueError(
            f"{path}: expected model_name={CONCH_MODEL_NAME!r}, got {values['model_name']!r}"
        )
    checkpoint_sha256 = values["checkpoint_sha256"]
    if not valid_checkpoint_sha256(checkpoint_sha256):
        raise ValueError(f"{path}: checkpoint_sha256 must be 64 lowercase hexadecimal characters")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            f"{path}: checkpoint_sha256 {checkpoint_sha256} does not match prompt "
            f"checkpoint {expected_checkpoint_sha256}"
        )
    return checkpoint_sha256


def read_text_aligned_checkpoint_sha256(
    path: str | Path,
    allow_smoke_fixture: bool = False,
) -> str:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise ValueError(f"{path}: expected features dataset")
        features = handle["features"]
        if features.ndim != 2:
            raise ValueError(f"{path}: text-aligned features must be 2-D")
        return validate_text_aligned_features(
            features,
            path,
            int(features.shape[0]),
            int(features.shape[1]),
            allow_smoke_fixture=allow_smoke_fixture,
        )


def load_prompt_h5(
    path: str | Path,
    allow_smoke_fixture: bool = False,
) -> tuple[np.ndarray, list[Dict[str, str]]]:
    with h5py.File(path, "r") as handle:
        if "features" not in handle or "metadata" not in handle:
            raise ValueError(f"{path}: expected features dataset and metadata group")
        feature_dataset = handle["features"]
        if feature_dataset.ndim != 2:
            raise ValueError(f"{path}: prompt features must be 2-D")
        validate_text_aligned_features(
            feature_dataset,
            path,
            int(feature_dataset.shape[0]),
            int(feature_dataset.shape[1]),
            allow_smoke_fixture=allow_smoke_fixture,
        )
        ensemble = scalar_h5_attr(handle.attrs.get("ensemble"))
        if isinstance(ensemble, str):
            ensemble = ensemble.strip().lower() in {"1", "true", "yes"}
        if ensemble is not True:
            raise ValueError(
                f"{path}: prompt features must be concept-level ensembles; "
                "encode with --ensemble-by polarity,category,concept_id"
            )
        ensemble_by = scalar_h5_attr(handle.attrs.get("ensemble_by"))
        ensemble_fields = tuple(
            field.strip() for field in str(ensemble_by or "").split(",") if field.strip()
        )
        if len(ensemble_fields) != len(PROMPT_ENSEMBLE_FIELDS) or set(ensemble_fields) != set(
            PROMPT_ENSEMBLE_FIELDS
        ):
            expected = ",".join(PROMPT_ENSEMBLE_FIELDS)
            raise ValueError(
                f"{path}: prompt ensemble_by must group by {expected}; got {ensemble_by!r}"
            )
        features = np.asarray(feature_dataset[:], dtype=np.float32)
        metadata_group = handle["metadata"]
        required_metadata = set(PROMPT_ENSEMBLE_FIELDS)
        missing_metadata = sorted(required_metadata - set(metadata_group.keys()))
        if missing_metadata:
            raise ValueError(f"{path}: prompt metadata is missing {missing_metadata}")
        fields: Dict[str, list[str]] = {}
        for key in metadata_group.keys():
            dataset = metadata_group[key]
            if dataset.ndim != 1 or len(dataset) != len(features):
                raise ValueError(
                    f"{path}: metadata/{key} must have exactly {len(features)} rows"
                )
            fields[key] = decode_strings(dataset[:])
        for dataset_name in ("prompt_ids", "prompts"):
            if dataset_name in handle:
                dataset = handle[dataset_name]
                if dataset.ndim != 1 or len(dataset) != len(features):
                    raise ValueError(
                        f"{path}: {dataset_name} must have exactly {len(features)} rows"
                    )
        prompt_ids = decode_strings(handle["prompt_ids"][:]) if "prompt_ids" in handle else [str(i) for i in range(len(features))]
        prompts = decode_strings(handle["prompts"][:]) if "prompts" in handle else prompt_ids
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: prompt features contain non-finite values")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"{path}: prompt embeddings must be L2 normalized")
    records = []
    for index in range(features.shape[0]):
        record = {key: values[index] for key, values in fields.items()}
        record["prompt_id"] = prompt_ids[index]
        record["prompt"] = prompts[index]
        records.append(record)
    concept_keys = [
        tuple(record[field].strip() for field in PROMPT_ENSEMBLE_FIELDS)
        for record in records
    ]
    if any(not all(key) for key in concept_keys):
        raise ValueError(f"{path}: prompt ensemble keys must not be empty")
    seen_keys: set[tuple[str, ...]] = set()
    duplicate_keys: set[tuple[str, ...]] = set()
    for key in concept_keys:
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        preview = ["/".join(key) for key in sorted(duplicate_keys)[:5]]
        raise ValueError(f"{path}: duplicate concept-level prompt ensembles: {preview}")
    return features, records


def cluster_key(mag: int, fine_cluster_id: int) -> str:
    return f"{int(mag)}x:{int(fine_cluster_id)}"


def coarse_key(mag: int, coarse_cluster_id: int) -> str:
    return f"{int(mag)}x:{int(coarse_cluster_id)}"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def read_wsi_manifest(path: str | None) -> Dict[str, str]:
    if path is None:
        return {}
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(source)
        rows = frame.to_dict("records")
    else:
        with open(source, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    result: Dict[str, str] = {}
    for row in rows:
        wsi_path = row.get("wsi_path") or row.get("path")
        key = row.get("wsi_id") or row.get("file_name") or row.get("slide_id")
        if not key or not wsi_path:
            raise ValueError(f"{source}: WSI manifest requires a key and wsi_path column")
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError(f"{source}: WSI manifest key must not be empty")
        resolved_path = Path(str(wsi_path)).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = source.parent / resolved_path
        normalized_path = str(resolved_path.resolve())
        if normalized_key in result:
            if result[normalized_key] == normalized_path:
                raise ValueError(f"{source}: duplicate WSI manifest key {normalized_key!r}")
            raise ValueError(
                f"{source}: conflicting WSI manifest key {normalized_key!r}: "
                f"{result[normalized_key]} != {normalized_path}"
            )
        result[normalized_key] = normalized_path
    return result


def resolve_wsi_path(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Mapping[str, str] | None = None,
) -> str | None:
    manifest = manifest or {}
    file_name = str(metadata.get("file_name", ""))
    wsi_id = Path(file_name).stem
    file_slide_id = metadata_file_slide_id(metadata)
    for key in (
        wsi_id,
        file_name,
        file_slide_id,
        str(metadata.get("slide_submitter_id", "")),
    ):
        if key in manifest:
            return str(Path(manifest[key]).expanduser().resolve())
    root = config["wsi"].get("root")
    if not root:
        return None
    fields = dict(metadata)
    fields["wsi_root"] = root
    try:
        return str(Path(config["wsi"]["path_template"].format(**fields)).expanduser().resolve())
    except KeyError as error:
        raise ValueError(f"Unknown field in wsi.path_template: {error.args[0]}") from error


def atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def atomic_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def atomic_parquet(frame, path: str | Path, row_group_size: int = 65_536) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
        row_group_size=row_group_size,
    )
    temporary.replace(output)


def json_distribution(names: Sequence[str], counts: np.ndarray) -> str:
    pairs = [
        {"name": str(name), "count": int(count)}
        for name, count in zip(names, counts)
        if int(count) > 0
    ]
    pairs.sort(key=lambda item: (-item["count"], item["name"]))
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))


def parse_stage_args(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Offline pipeline JSON config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this stage's existing outputs.")
    return parser.parse_args()
