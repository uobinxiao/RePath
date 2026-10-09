#!/usr/bin/env python3
"""Run the production CONCH clustering and offline-index workflow without Slurm.

This is a lightweight orchestrator. Heavy work is delegated to the existing
repository entry points using the Python interpreter that launched this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
MAGNIFICATIONS = (5, 10, 20, 40)
RUNNER_MANIFEST = ".remote_runner_manifest.json"
OWNER_SENTINEL = ".remote_runner_owner.json"
RUNNER_SCHEMA_VERSION = 1
HSV_FILTER_SPEC = {
    "method": "precomputed_coordinate_allowlist",
    "minimum_tissue_coverage": 0.45,
    "hue_range": [90, 180],
    "saturation_range": [8, 255],
    "value_range": [103, 255],
}


class RunnerError(RuntimeError):
    """Raised when an existing or requested production run is unsafe."""


@dataclass(frozen=True)
class Checkpoint:
    reference: str
    resolved_path: str
    sha256: str | None


def canonical_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def parse_magnification_paths(
    values: Sequence[str],
    option_name: str,
    *,
    require_directories: bool,
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise RunnerError(f"{option_name} expects MAG=DIR, got {value!r}")
        mag_text, directory_text = value.split("=", 1)
        try:
            mag = int(mag_text.rstrip("xX"))
        except ValueError as error:
            raise RunnerError(f"{option_name} has invalid magnification {mag_text!r}") from error
        if mag in result:
            raise RunnerError(f"{option_name} contains duplicate {mag}x entries")
        result[mag] = canonical_path(directory_text)
    missing = sorted(set(MAGNIFICATIONS) - set(result))
    extra = sorted(set(result) - set(MAGNIFICATIONS))
    if missing or extra:
        raise RunnerError(
            f"{option_name} must provide exactly 5x, 10x, 20x, and 40x; "
            f"missing={missing}, extra={extra}"
        )
    if require_directories:
        for mag, directory in result.items():
            if not directory.is_dir():
                raise RunnerError(f"{mag}x feature directory does not exist: {directory}")
            if not discover_h5(directory):
                raise RunnerError(f"{mag}x feature directory contains no H5 files: {directory}")
    return result


def discover_h5(directory: Path) -> list[Path]:
    return sorted(path.resolve() for path in directory.rglob("*.h5") if path.is_file())


def file_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted({candidate.resolve() for candidate in paths}):
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return records


def relative_h5_inventory(directory: Path) -> set[str]:
    return {str(path.relative_to(directory.resolve())) for path in discover_h5(directory)}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def run_command(
    label: str,
    command: Sequence[str],
    *,
    dry_run: bool,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print(f"\n=== {label} ===")
    print(command_text(command))
    if dry_run:
        return None
    try:
        return subprocess.run(
            [str(value) for value in command],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as error:
        details = ""
        if capture_output:
            details = (error.stderr or error.stdout or "").strip()
        suffix = f"\n{details}" if details else ""
        raise RunnerError(
            f"{label} failed with exit code {error.returncode}{suffix}"
        ) from error


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def assert_safe_directory_depth(path: Path) -> None:
    resolved = path.resolve()
    if len(resolved.parts) < 4:
        raise RunnerError(
            f"Refusing shallow output directory {resolved}; use a run-specific directory "
            "such as /scratch/USER/cpath_fm/OUTPUT"
        )


def owner_payload(kind: str) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "owner": "run_remote_offline.py",
        "kind": kind,
    }


def mark_owned_directory(path: Path, kind: str, *, dry_run: bool) -> None:
    print(f"Claiming runner-owned {kind} directory: {path}")
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / OWNER_SENTINEL, owner_payload(kind))


def directory_is_owned(path: Path, kind: str) -> bool:
    sentinel = path / OWNER_SENTINEL
    manifest = path / RUNNER_MANIFEST
    for candidate in (sentinel, manifest):
        if not candidate.is_file():
            continue
        try:
            payload = read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if candidate == sentinel:
            if payload == owner_payload(kind):
                return True
        elif payload.get("schema_version") == RUNNER_SCHEMA_VERSION and payload.get("kind") == kind:
            return True
    return False


def owned_file_sidecar(path: Path) -> Path:
    return path.with_name(f".{path.name}.remote_runner_owner.json")


def file_is_owned(path: Path, kind: str) -> bool:
    sidecar = owned_file_sidecar(path)
    try:
        payload = read_json(sidecar)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload == owner_payload(kind)


def mark_owned_file(path: Path, kind: str, *, dry_run: bool) -> None:
    print(f"Claiming runner-owned {kind} file: {path}")
    if not dry_run:
        atomic_json(owned_file_sidecar(path), owner_payload(kind))


def offline_output_is_owned(path: Path) -> bool:
    manifest_path = path / "run_manifest.json"
    try:
        manifest = read_json(manifest_path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    resolved_config = manifest.get("resolved_config")
    if not isinstance(resolved_config, dict):
        return False
    provenance = resolved_config.get("runner_provenance")
    if not isinstance(provenance, dict):
        return False
    configured_output = Path(str(resolved_config.get("output_dir", ""))).expanduser()
    if not configured_output.is_absolute():
        return False
    return bool(
        manifest.get("schema_version") == 1
        and manifest.get("profile") == "production"
        and resolved_config.get("profile") == "production"
        and provenance.get("orchestrator") == "run_remote_offline.py"
        and configured_output.resolve() == path.resolve()
        and valid_sha256(manifest.get("config_hash"))
        and valid_sha256(manifest.get("input_fingerprint"))
        and isinstance(manifest.get("stages"), dict)
    )


def remove_owned_directory(path: Path, kind: str, *, dry_run: bool) -> None:
    assert_safe_directory_depth(path)
    if not directory_is_owned(path, kind):
        raise RunnerError(
            f"Refusing to delete unowned {kind} directory {path}. Adopt and validate it "
            "in a separate run first, remove it manually, or choose a new output path."
        )
    print(f"Removing runner-owned {kind} directory by explicit request: {path}")
    if not dry_run:
        shutil.rmtree(path)


def validate_output_layout(
    raw_directories: Mapping[int, Path],
    projected_directories: Mapping[int, Path],
    clustering_output: Path,
    offline_output: Path,
    prompt_bank_dir: Path,
    prompt_output: Path,
    config_output: Path,
    protected_paths: Sequence[Path],
) -> None:
    destructive_directories = [
        clustering_output,
        offline_output,
        prompt_bank_dir,
        *projected_directories.values(),
    ]
    output_targets = [*destructive_directories, prompt_output, config_output]
    for path in output_targets:
        if path.resolve() in {Path("/").resolve(), REPOSITORY_ROOT.resolve()}:
            raise RunnerError(f"Refusing to use unsafe output path: {path}")
        for raw in raw_directories.values():
            if paths_overlap(path, raw):
                raise RunnerError(f"Output {path} must not overlap raw feature directory {raw}")
        for protected in protected_paths:
            if paths_overlap(path, protected):
                raise RunnerError(f"Output {path} must not overlap protected input {protected}")
    for path in destructive_directories:
        assert_safe_directory_depth(path)
    for index, first in enumerate(output_targets):
        for second in output_targets[index + 1 :]:
            if paths_overlap(first, second):
                raise RunnerError(f"Output paths must not overlap: {first} and {second}")


def validate_checkpoint_reference(reference: str) -> None:
    if reference.startswith("hf_hub:"):
        if "@" not in reference.removeprefix("hf_hub:"):
            raise RunnerError(
                "Production Hugging Face checkpoints must pin a revision: "
                "hf_hub:OWNER/REPO@REVISION"
            )
        return
    path = canonical_path(reference)
    if path.is_dir():
        path = path / "pytorch_model.bin"
    if not path.is_file():
        raise RunnerError(f"CONCH checkpoint does not exist: {path}")


def resolve_checkpoint(
    reference: str,
    *,
    dry_run: bool,
) -> Checkpoint:
    validate_checkpoint_reference(reference)
    if not reference.startswith("hf_hub:"):
        path = canonical_path(reference)
        if path.is_dir():
            path = path / "pytorch_model.bin"
        return Checkpoint(reference, str(path), sha256_file(path))
    print("\n=== resolve pinned CONCH checkpoint ===")
    print(reference)
    if dry_run:
        return Checkpoint(reference, reference, None)
    try:
        from tools.conch_checkpoint import resolve_conch_checkpoint

        resolved = resolve_conch_checkpoint(reference)
    except (ImportError, OSError, ValueError) as error:
        raise RunnerError(f"Failed to resolve CONCH checkpoint {reference!r}: {error}") from error
    return Checkpoint(reference, str(resolved.path), str(resolved.sha256))


def clustering_plan(
    args: argparse.Namespace,
    raw_directories: Mapping[int, Path],
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    hsv_directories = (
        parse_magnification_paths(
            args.hsv_filter_dir,
            "--hsv-filter-dir",
            require_directories=True,
        )
        if args.hsv_filter_dir
        else {}
    )
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "kind": "clustering",
        "attested_checkpoint_sha256": checkpoint_sha256,
        "raw_h5": {
            str(mag): file_inventory(discover_h5(directory))
            for mag, directory in raw_directories.items()
        },
        "hsv_filter_h5": {
            str(mag): file_inventory(discover_h5(directory))
            for mag, directory in hsv_directories.items()
        },
        "generated_hsv_inputs": (
            None
            if hsv_directories
            else {
                "metadata_path": str(canonical_path(args.metadata_path)),
                "metadata_sha256": sha256_file(canonical_path(args.metadata_path)),
                "wsi_root": (
                    str(canonical_path(args.wsi_root)) if args.wsi_root else None
                ),
                "wsi_manifest": (
                    str(canonical_path(args.wsi_manifest))
                    if args.wsi_manifest
                    else None
                ),
                "wsi_manifest_sha256": (
                    sha256_file(canonical_path(args.wsi_manifest))
                    if args.wsi_manifest
                    else None
                ),
                "wsi_path_template": args.wsi_path_template,
            }
        ),
        "parameters": {
            "n_clusters": args.n_clusters,
            "n_iters": args.n_iters,
            "max_fit_samples": args.max_fit_samples,
            "feature_chunk_size": args.feature_chunk_size,
            "hsv_filter_workers": args.hsv_filter_workers,
            "hsv_min_tissue_coverage": args.hsv_min_tissue_coverage,
            "hsv_filter_mode": "precomputed" if hsv_directories else "generated",
            "assign_chunk_size": args.assign_chunk_size,
            "fit_chunk_budget": args.fit_chunk_budget,
            "seed": args.seed,
            "normalize": True,
            "exclude_upsampled": True,
            "hsv_filter_spec": HSV_FILTER_SPEC
            | {
                "method": (
                    "precomputed_coordinate_allowlist"
                    if hsv_directories
                    else "hsv_tissue_coverage"
                ),
                "minimum_tissue_coverage": (
                    HSV_FILTER_SPEC["minimum_tissue_coverage"]
                    if hsv_directories
                    else args.hsv_min_tissue_coverage
                ),
            },
        },
    }


def clustering_artifacts(output: Path) -> list[Path]:
    paths = [output / "config.json", output / "summary.json"]
    for mag in MAGNIFICATIONS:
        paths.extend(
            [
                output / "features" / f"{mag}x" / "features.npy",
                output / "features" / f"{mag}x" / "h5_index.json",
                output / "features" / f"{mag}x" / "excluded_upsampled_h5.json",
                output / "clusters" / f"{mag}x" / "fit_indices.npy",
                output / "clusters" / f"{mag}x" / "level1" / "centroids.npy",
                output / "clusters" / f"{mag}x" / "level1" / "assignments.npy",
                output / "clusters" / f"{mag}x" / "level2" / "centroids.npy",
                output / "clusters" / f"{mag}x" / "level2" / "fit_assignment.npy",
                output / "clusters" / f"{mag}x" / "level2" / "assignments.npy",
            ]
        )
    return paths


def verify_clustering_h5_inventory(
    output: Path,
    raw_directories: Mapping[int, Path],
) -> None:
    for mag in MAGNIFICATIONS:
        records = read_json(output / "features" / f"{mag}x" / "h5_index.json")
        excluded = read_json(
            output / "features" / f"{mag}x" / "excluded_upsampled_h5.json"
        )
        if not isinstance(records, list) or not isinstance(excluded, list):
            raise RunnerError(f"Existing {mag}x clustering H5 inventories are malformed")
        recorded = {str(canonical_path(record["path"])) for record in records}
        excluded_paths = {str(canonical_path(record["path"])) for record in excluded}
        invalid_exclusions = [
            record
            for record in excluded
            if record.get("reason") != "target_magnification_gt_level0_magnification"
            or int(record.get("level0_magnification", 0)) <= 0
            or int(record.get("target_magnification", 0))
            <= int(record.get("level0_magnification", 0))
        ]
        if recorded & excluded_paths or invalid_exclusions:
            raise RunnerError(f"Existing {mag}x upsampled-exclusion report is invalid")
        for record in records:
            selected_path_value = record.get("selected_row_indices_path")
            if not selected_path_value:
                continue
            selected_path = canonical_path(selected_path_value)
            if (
                not selected_path.is_file()
                or sha256_file(selected_path)
                != str(record.get("selected_row_indices_sha256") or "")
            ):
                raise RunnerError(
                    f"Existing {mag}x selected-row index is missing or changed: {selected_path}"
                )
        current = {str(path) for path in discover_h5(raw_directories[mag])}
        if recorded | excluded_paths != current:
            raise RunnerError(
                f"Existing {mag}x clustering references a different raw H5 inventory; "
                "use a new clustering output or --overwrite-clustering"
            )


def verify_clustering_parameters(
    output: Path,
    args: argparse.Namespace,
    raw_directories: Mapping[int, Path],
) -> None:
    stored = read_json(output / "config.json")
    try:
        stored_raw = parse_magnification_paths(
            list(stored["mag_dir"]),
            "stored clustering mag_dir",
            require_directories=False,
        )
        stored_hsv = (
            parse_magnification_paths(
                list(stored.get("hsv_filter_dir") or []),
                "stored clustering hsv_filter_dir",
                require_directories=False,
            )
            if stored.get("hsv_filter_dir")
            else {}
        )
    except (KeyError, TypeError) as error:
        raise RunnerError(f"Existing clustering config is malformed: {output / 'config.json'}") from error
    expected_values = {
        "n_clusters": args.n_clusters,
        "n_iters": args.n_iters,
        "max_fit_samples": args.max_fit_samples,
        "feature_chunk_size": args.feature_chunk_size,
        "hsv_filter_workers": args.hsv_filter_workers,
        "hsv_min_tissue_coverage": args.hsv_min_tissue_coverage,
        "assign_chunk_size": args.assign_chunk_size,
        "fit_chunk_budget": args.fit_chunk_budget,
        "seed": args.seed,
        "no_normalize": False,
        "exclude_upsampled": True,
    }
    mismatches = {
        key: {"expected": value, "actual": stored.get(key)}
        for key, value in expected_values.items()
        if stored.get(key) != value
    }
    if stored_raw != dict(raw_directories):
        mismatches["mag_dir"] = {
            "expected": {str(key): str(value) for key, value in raw_directories.items()},
            "actual": {str(key): str(value) for key, value in stored_raw.items()},
        }
    expected_hsv = (
        parse_magnification_paths(
            args.hsv_filter_dir,
            "--hsv-filter-dir",
            require_directories=True,
        )
        if args.hsv_filter_dir
        else {}
    )
    if stored_hsv != expected_hsv:
        mismatches["hsv_filter_dir"] = {
            "expected": {str(key): str(value) for key, value in expected_hsv.items()},
            "actual": {str(key): str(value) for key, value in stored_hsv.items()},
        }
    if mismatches:
        raise RunnerError(
            "Existing clustering parameters differ from the requested production run: "
            + json.dumps(mismatches, sort_keys=True)
        )


def prepare_clustering(
    args: argparse.Namespace,
    python_bin: str,
    raw_directories: Mapping[int, Path],
    checkpoint: Checkpoint,
) -> None:
    output = canonical_path(args.clustering_output)
    manifest_path = output / RUNNER_MANIFEST
    plan = clustering_plan(args, raw_directories, checkpoint.sha256)
    existing_artifacts = [path for path in clustering_artifacts(output) if path.exists()]
    complete = len(existing_artifacts) == len(clustering_artifacts(output))

    if args.overwrite_clustering and output.exists():
        remove_owned_directory(output, "clustering", dry_run=args.dry_run)
        complete = False
        existing_artifacts = []
    elif complete:
        verify_clustering_parameters(output, args, raw_directories)
        verify_clustering_h5_inventory(output, raw_directories)
        if manifest_path.is_file():
            if read_json(manifest_path) != plan:
                raise RunnerError(
                    "Existing clustering runner manifest does not match current inputs or parameters"
                )
        elif not args.adopt_existing:
            raise RunnerError(
                f"Existing clustering has no {RUNNER_MANIFEST}; rerun with --adopt-existing "
                "after verifying its provenance, or use a new output directory"
            )
        print(f"Reusing complete clustering output: {output}")
        if not manifest_path.exists() and not args.dry_run:
            atomic_json(manifest_path, plan)
        return
    elif existing_artifacts or (output.exists() and any(output.iterdir())):
        raise RunnerError(
            f"Clustering output is incomplete: {output}. Use --overwrite-clustering or a new path."
        )

    command = [
        python_bin,
        str(REPOSITORY_ROOT / "run_hierarchical_clustering.py"),
    ]
    for mag in MAGNIFICATIONS:
        command.extend(["--mag-dir", f"{mag}={raw_directories[mag]}"])
    if args.hsv_filter_dir:
        hsv_directories = parse_magnification_paths(
            args.hsv_filter_dir,
            "--hsv-filter-dir",
            require_directories=True,
        )
        for mag in MAGNIFICATIONS:
            command.extend(["--hsv-filter-dir", f"{mag}={hsv_directories[mag]}"])
    else:
        command.extend(["--metadata-path", str(canonical_path(args.metadata_path))])
        if args.wsi_root:
            command.extend(["--wsi-root", str(canonical_path(args.wsi_root))])
        else:
            command.extend(["--wsi-manifest", str(canonical_path(args.wsi_manifest))])
        command.extend(["--wsi-path-template", args.wsi_path_template])
    command.extend(
        [
            "--output-dir",
            str(output),
            "--n-clusters",
            args.n_clusters,
            "--n-iters",
            str(args.n_iters),
            "--max-fit-samples",
            str(args.max_fit_samples),
            "--feature-chunk-size",
            str(args.feature_chunk_size),
            "--hsv-filter-workers",
            str(args.hsv_filter_workers),
            "--hsv-min-tissue-coverage",
            str(args.hsv_min_tissue_coverage),
            "--assign-chunk-size",
            str(args.assign_chunk_size),
            "--fit-chunk-budget",
            str(args.fit_chunk_budget),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--exclude-upsampled",
        ]
    )
    mark_owned_directory(output, "clustering", dry_run=args.dry_run)
    run_command("hierarchical clustering", command, dry_run=args.dry_run)
    if not args.dry_run:
        missing = [str(path) for path in clustering_artifacts(output) if not path.exists()]
        if missing:
            raise RunnerError(f"Clustering completed without required artifacts: {missing[:5]}")
        atomic_json(manifest_path, plan)


def projection_plan(
    mag: int,
    raw_directory: Path,
    checkpoint: Checkpoint,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "kind": "projection",
        "magnification": mag,
        "raw_h5": file_inventory(discover_h5(raw_directory)),
        "checkpoint_sha256": checkpoint.sha256,
        "batch_size": args.projection_batch_size,
        "compression": args.projection_compression,
    }


def prepare_projections(
    args: argparse.Namespace,
    python_bin: str,
    raw_directories: Mapping[int, Path],
    projected_directories: Mapping[int, Path],
    checkpoint: Checkpoint,
) -> list[tuple[Path, dict[str, Any]]]:
    manifests_to_write: list[tuple[Path, dict[str, Any]]] = []
    for mag in MAGNIFICATIONS:
        raw_directory = raw_directories[mag]
        output = projected_directories[mag]
        manifest_path = output / RUNNER_MANIFEST
        plan = projection_plan(mag, raw_directory, checkpoint, args)
        expected = relative_h5_inventory(raw_directory)
        actual = relative_h5_inventory(output) if output.is_dir() else set()

        if args.overwrite_projections and output.exists():
            remove_owned_directory(output, "projection", dry_run=args.dry_run)
            actual = set()
        elif actual:
            if actual != expected:
                raise RunnerError(
                    f"Existing {mag}x projected H5 inventory is partial or has extras; "
                    "use --overwrite-projections or a new directory"
                )
            if manifest_path.is_file():
                if read_json(manifest_path) != plan:
                    raise RunnerError(
                        f"Existing {mag}x projection manifest does not match current inputs/checkpoint"
                    )
            elif not args.adopt_existing and not directory_is_owned(output, "projection"):
                raise RunnerError(
                    f"Existing {mag}x projections have no {RUNNER_MANIFEST}; use "
                    "--adopt-existing to validate/adopt them, or regenerate into a new directory"
                )
            print(f"Reusing complete {mag}x projected features: {output}")
            if not manifest_path.exists():
                manifests_to_write.append((manifest_path, plan))
            continue
        elif output.exists() and any(output.iterdir()):
            raise RunnerError(f"Projection output contains no usable H5 inventory: {output}")

        command = [
            python_bin,
            str(REPOSITORY_ROOT / "tools" / "convert_conch_trident_features.py"),
            "--input",
            str(raw_directory),
            "--output",
            str(output),
            "--checkpoint-path",
            checkpoint.resolved_path,
            "--batch-size",
            str(args.projection_batch_size),
            "--device",
            args.device,
        ]
        if args.projection_compression:
            command.extend(["--compression", args.projection_compression])
        mark_owned_directory(output, "projection", dry_run=args.dry_run)
        run_command(f"project {mag}x image features", command, dry_run=args.dry_run)
        if not args.dry_run:
            actual = relative_h5_inventory(output)
            if actual != expected:
                raise RunnerError(
                    f"{mag}x conversion completed with a mismatched H5 inventory; "
                    f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
                )
        manifests_to_write.append((manifest_path, plan))
    return manifests_to_write


def prompt_plan(
    prompt_bank_dir: Path,
    checkpoint: Checkpoint,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_paths = [prompt_bank_dir / "prompts.jsonl", prompt_bank_dir / "concept_bank.json"]
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "kind": "prompt_embeddings",
        "sources": file_inventory(source_paths),
        "checkpoint_sha256": checkpoint.sha256,
        "batch_size": args.prompt_batch_size,
        "ensemble_by": "polarity,category,concept_id",
    }


def prepare_prompts(
    args: argparse.Namespace,
    python_bin: str,
    checkpoint: Checkpoint,
) -> list[tuple[Path, dict[str, Any]]]:
    prompt_bank_dir = canonical_path(args.prompt_bank_dir)
    prompt_output = canonical_path(args.prompt_embeddings_output)
    pending_records: list[tuple[Path, dict[str, Any]]] = []
    required_bank = [
        prompt_bank_dir / "prompts.jsonl",
        prompt_bank_dir / "prompts.csv",
        prompt_bank_dir / "prompts.txt",
        prompt_bank_dir / "concept_bank.json",
    ]
    bank_complete = all(path.is_file() for path in required_bank)
    if args.overwrite_prompts or not bank_complete:
        if (
            prompt_bank_dir.exists()
            and any(prompt_bank_dir.iterdir())
            and not directory_is_owned(prompt_bank_dir, "prompt_bank")
        ):
            raise RunnerError(
                f"Refusing to overwrite unowned prompt-bank directory {prompt_bank_dir}; "
                "choose a new directory or adopt it in a non-overwrite run first"
            )
        if prompt_output.exists() and not args.overwrite_prompts and not bank_complete:
            raise RunnerError(
                "Prompt embeddings exist but their prompt bank is incomplete; use "
                "--overwrite-prompts or a new output"
            )
        mark_owned_directory(prompt_bank_dir, "prompt_bank", dry_run=args.dry_run)
        run_command(
            "generate pathology prompt bank",
            [
                python_bin,
                str(REPOSITORY_ROOT / "tools" / "prompt_gen.py"),
                "--out_dir",
                str(prompt_bank_dir),
            ],
            dry_run=args.dry_run,
        )
    else:
        print(f"Reusing prompt bank: {prompt_bank_dir}")
        if args.adopt_existing and not directory_is_owned(prompt_bank_dir, "prompt_bank"):
            pending_records.append(
                (prompt_bank_dir / OWNER_SENTINEL, owner_payload("prompt_bank"))
            )

    if args.dry_run and not bank_complete:
        plan = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "kind": "prompt_embeddings",
            "checkpoint_sha256": checkpoint.sha256,
            "batch_size": args.prompt_batch_size,
            "ensemble_by": "polarity,category,concept_id",
        }
    else:
        plan = prompt_plan(prompt_bank_dir, checkpoint, args)
    manifest_path = prompt_output.with_name(f".{prompt_output.name}.remote_runner_manifest.json")

    if args.overwrite_prompts and prompt_output.exists():
        if not manifest_path.is_file() and not file_is_owned(prompt_output, "prompt_embeddings"):
            raise RunnerError(
                f"Refusing to overwrite unowned prompt embedding file {prompt_output}; "
                "adopt it in a non-overwrite run first or choose a new output"
            )
        existing_manifest = read_json(manifest_path)
        if not isinstance(existing_manifest, dict):
            raise RunnerError(f"Prompt embedding ownership manifest is invalid: {manifest_path}")
        if existing_manifest.get("kind") != "prompt_embeddings":
            raise RunnerError(f"Prompt embedding ownership manifest is invalid: {manifest_path}")

    if prompt_output.is_file() and not args.overwrite_prompts:
        if manifest_path.is_file():
            if read_json(manifest_path) != plan:
                raise RunnerError("Existing prompt embedding manifest does not match current inputs/checkpoint")
        elif not args.adopt_existing and not file_is_owned(prompt_output, "prompt_embeddings"):
            raise RunnerError(
                "Existing prompt embeddings have no runner manifest; use --adopt-existing "
                "to validate/adopt them, or regenerate into a new file"
            )
        print(f"Reusing prompt embeddings: {prompt_output}")
        if not manifest_path.exists():
            pending_records.append((manifest_path, plan))
        return pending_records

    command = [
        python_bin,
        str(REPOSITORY_ROOT / "tools" / "encode_conch_text_prompts.py"),
        "--prompts-file",
        str(prompt_bank_dir / "prompts.jsonl"),
        "--output",
        str(prompt_output),
        "--ensemble-by",
        "polarity,category,concept_id",
        "--checkpoint-path",
        checkpoint.resolved_path,
        "--batch-size",
        str(args.prompt_batch_size),
        "--device",
        args.device,
    ]
    if args.overwrite_prompts and prompt_output.exists():
        command.append("--overwrite")
    mark_owned_file(prompt_output, "prompt_embeddings", dry_run=args.dry_run)
    run_command("encode and ensemble CONCH prompts", command, dry_run=args.dry_run)
    pending_records.append((manifest_path, plan))
    return pending_records


def validate_embedding_contracts(
    prompt_output: Path,
    raw_directories: Mapping[int, Path],
    projected_directories: Mapping[int, Path],
    checkpoint: Checkpoint,
    *,
    dry_run: bool,
) -> None:
    print("\n=== validate production embedding contracts ===")
    if dry_run:
        print("Validate prompt and projected H5 contracts, inventories, slide IDs, and coordinates")
        return

    try:
        import h5py
        import numpy as np
        from tqdm import tqdm

        from offline.common import (
            discover_h5 as discover_contract_h5,
            h5_recorded_slide_id,
            load_prompt_h5,
            read_text_aligned_checkpoint_sha256,
            validate_text_aligned_features,
        )
    except (ImportError, OSError) as error:
        raise RunnerError(
            f"Cannot load embedding-validation dependencies: {error}"
        ) from error

    try:
        prompt_features, records = load_prompt_h5(prompt_output)
        prompt_sha = read_text_aligned_checkpoint_sha256(prompt_output)
        if checkpoint.sha256 and prompt_sha != checkpoint.sha256:
            raise ValueError(
                f"Prompt checkpoint {prompt_sha} != requested checkpoint {checkpoint.sha256}"
            )
        dimension = int(prompt_features.shape[1])
        inventories: dict[int, tuple[Path, dict[Path, Path], dict[Path, Path]]] = {}
        print("Discovering raw/projected H5 inventories...")
        for mag in MAGNIFICATIONS:
            raw_directory = raw_directories[mag].resolve()
            projected_directory = projected_directories[mag].resolve()
            raw_paths = {
                path.relative_to(raw_directory): path
                for path in discover_contract_h5(raw_directory)
            }
            projected_paths = {
                path.relative_to(projected_directory): path
                for path in discover_contract_h5(projected_directory)
            }
            if set(raw_paths) != set(projected_paths):
                raise ValueError(f"Raw/projected H5 inventories differ for {raw_directory}")
            inventories[mag] = (raw_directory, raw_paths, projected_paths)
            print(f"  {mag}x: {len(raw_paths):,} H5 files")

        count = 0
        total_files = sum(len(raw_paths) for _, raw_paths, _ in inventories.values())
        with tqdm(
            total=total_files,
            desc="Embedding contracts",
            unit="file",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress:
            for mag in MAGNIFICATIONS:
                raw_directory, raw_paths, projected_paths = inventories[mag]
                progress.set_postfix_str(f"{mag}x", refresh=False)
                seen_slide_ids: set[str] = set()
                for relative_path, raw_path in raw_paths.items():
                    projected_path = projected_paths[relative_path]
                    slide_id = h5_recorded_slide_id(raw_path)
                    if slide_id in seen_slide_ids:
                        raise ValueError(
                            f"{raw_directory}: duplicate recorded slide ID {slide_id!r}"
                        )
                    seen_slide_ids.add(slide_id)
                    if slide_id != h5_recorded_slide_id(projected_path):
                        raise ValueError(
                            f"{projected_path}: recorded slide ID does not match {raw_path}"
                        )
                    with (
                        h5py.File(raw_path, "r") as raw_handle,
                        h5py.File(projected_path, "r") as projected_handle,
                    ):
                        if "features" not in projected_handle or "coords" not in projected_handle:
                            raise ValueError(
                                f"{projected_path}: projected H5 requires features and coords"
                            )
                        if "features" not in raw_handle or "coords" not in raw_handle:
                            raise ValueError(f"{raw_path}: raw H5 requires features and coords")
                        features = projected_handle["features"]
                        expected_rows = int(raw_handle["features"].shape[0])
                        if features.shape[0] != expected_rows:
                            raise ValueError(
                                f"{projected_path}: projected row count does not match {raw_path}"
                            )
                        if projected_handle["coords"].shape != (expected_rows, 2):
                            raise ValueError(
                                f"{projected_path}: coords do not align with projected features"
                            )
                        if raw_handle["coords"].shape != (expected_rows, 2):
                            raise ValueError(f"{raw_path}: coords do not align with raw features")
                        validate_text_aligned_features(
                            features,
                            projected_path,
                            expected_rows,
                            dimension,
                            expected_checkpoint_sha256=prompt_sha,
                        )
                        for start in range(0, expected_rows, 8192):
                            end = min(start + 8192, expected_rows)
                            if not np.array_equal(
                                raw_handle["coords"][start:end],
                                projected_handle["coords"][start:end],
                            ):
                                raise ValueError(
                                    f"{projected_path}: coords differ from {raw_path} "
                                    f"at rows {start}:{end}"
                                )
                    count += 1
                    progress.update(1)
    except (KeyError, OSError, ValueError) as error:
        raise RunnerError(
            f"Production embedding contract validation failed: {error}"
        ) from error
    print(f"Validated {count} projected H5 files and {len(records)} prompt concepts")


def build_production_config(
    args: argparse.Namespace,
    raw_directories: Mapping[int, Path],
    projected_directories: Mapping[int, Path],
    checkpoint: Checkpoint,
) -> dict[str, Any]:
    base_path = canonical_path(args.base_config)
    config = read_json(base_path)
    if not isinstance(config, dict):
        raise RunnerError(f"Base config must be a JSON object: {base_path}")
    config.update(
        {
            "schema_version": 1,
            "profile": "production",
            "magnifications": list(MAGNIFICATIONS),
            "levels": {"fine": 1, "coarse": 2},
            "clustering_root": str(canonical_path(args.clustering_output)),
            "projected_feature_dirs": {
                str(mag): str(projected_directories[mag]) for mag in MAGNIFICATIONS
            },
            "metadata_path": str(canonical_path(args.metadata_path)),
            "prompt_embeddings_path": str(canonical_path(args.prompt_embeddings_output)),
            "output_dir": str(canonical_path(args.offline_output)),
        }
    )
    config["wsi"] = {
        "root": str(canonical_path(args.wsi_root)) if args.wsi_root else None,
        "manifest": str(canonical_path(args.wsi_manifest)) if args.wsi_manifest else None,
        "path_template": args.wsi_path_template,
    }
    audit = dict(config.get("audit", {}))
    audit["manifest_only"] = False
    audit["strict_wsi"] = True
    config["audit"] = audit
    config["runner_provenance"] = {
        "orchestrator": "run_remote_offline.py",
        "checkpoint_reference": checkpoint.reference,
        "checkpoint_sha256": checkpoint.sha256,
        "raw_features_attested_same_checkpoint": True,
        "raw_feature_dirs": {str(mag): str(raw_directories[mag]) for mag in MAGNIFICATIONS},
        "clustering_parameters": clustering_plan(
            args,
            raw_directories,
            checkpoint.sha256,
        )["parameters"],
    }
    return config


def write_production_config(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    path = canonical_path(args.config_output)
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, dict):
            raise RunnerError(f"Existing production config is not a JSON object: {path}")
        if existing == config:
            print(f"Reusing unchanged production config: {path}")
            return path
        if not args.overwrite_config:
            raise RunnerError(
                f"Production config differs from existing {path}; use --overwrite-config "
                "or select a new config/output path"
            )
        if existing.get("runner_provenance", {}).get("orchestrator") != "run_remote_offline.py":
            raise RunnerError(
                f"Refusing to overwrite config not owned by this runner: {path}. "
                "Choose a new --config-output path."
            )
    print(f"Writing production config: {path}")
    if not args.dry_run:
        atomic_json(path, config)
    return path


def validate_run_arguments(args: argparse.Namespace) -> tuple[dict[int, Path], dict[int, Path]]:
    if args.adopt_existing and any(
        (
            args.overwrite_clustering,
            args.overwrite_projections,
            args.overwrite_prompts,
        )
    ):
        raise RunnerError(
            "--adopt-existing and preparation --overwrite-* flags must be used in separate runs"
        )
    if not args.attest_raw_features_use_checkpoint:
        raise RunnerError(
            "Pass --attest-raw-features-use-checkpoint to confirm that the raw Trident "
            "features were extracted with the same CONCH checkpoint. Legacy raw H5 files "
            "cannot prove this automatically."
        )
    raw = parse_magnification_paths(
        args.raw_feature_dir,
        "--raw-feature-dir",
        require_directories=True,
    )
    projected = parse_magnification_paths(
        args.projected_feature_dir,
        "--projected-feature-dir",
        require_directories=False,
    )
    hsv_directories = (
        parse_magnification_paths(
            args.hsv_filter_dir,
            "--hsv-filter-dir",
            require_directories=True,
        )
        if args.hsv_filter_dir
        else {}
    )
    for value, label, kind in (
        (args.metadata_path, "metadata", "file"),
        (args.base_config, "base config", "file"),
    ):
        path = canonical_path(value)
        if kind == "file" and not path.is_file():
            raise RunnerError(f"{label} does not exist: {path}")
    if args.wsi_root and not canonical_path(args.wsi_root).is_dir():
        raise RunnerError(f"WSI root does not exist: {canonical_path(args.wsi_root)}")
    if args.wsi_manifest and not canonical_path(args.wsi_manifest).is_file():
        raise RunnerError(f"WSI manifest does not exist: {canonical_path(args.wsi_manifest)}")
    for name in (
        "n_iters",
        "max_fit_samples",
        "feature_chunk_size",
        "hsv_filter_workers",
        "assign_chunk_size",
        "fit_chunk_budget",
        "projection_batch_size",
        "prompt_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise RunnerError(f"--{name.replace('_', '-')} must be positive")
    try:
        cluster_counts = [int(value.strip()) for value in args.n_clusters.split(",")]
    except ValueError as error:
        raise RunnerError("--n-clusters must contain two positive integers") from error
    if len(cluster_counts) != 2 or any(value <= 0 for value in cluster_counts):
        raise RunnerError("--n-clusters must contain exactly two positive integers")
    if not 0.0 <= float(args.hsv_min_tissue_coverage) <= 1.0:
        raise RunnerError("--hsv-min-tissue-coverage must be between 0 and 1")
    protected_paths = [
        canonical_path(args.metadata_path),
        canonical_path(args.base_config),
        canonical_path(args.wsi_root or args.wsi_manifest),
    ]
    protected_paths.extend(hsv_directories.values())
    if not args.checkpoint_path.startswith("hf_hub:"):
        protected_paths.append(canonical_path(args.checkpoint_path))
    directory_outputs = [
        canonical_path(args.clustering_output),
        canonical_path(args.offline_output),
        canonical_path(args.prompt_bank_dir),
        *projected.values(),
    ]
    for output in directory_outputs:
        if output.exists() and not output.is_dir():
            raise RunnerError(f"Expected a directory output path, got a file: {output}")
    for value in (args.prompt_embeddings_output, args.config_output):
        output = canonical_path(value)
        if output.exists() and not output.is_file():
            raise RunnerError(f"Expected a file output path, got a directory: {output}")
    validate_output_layout(
        raw,
        projected,
        canonical_path(args.clustering_output),
        canonical_path(args.offline_output),
        canonical_path(args.prompt_bank_dir),
        canonical_path(args.prompt_embeddings_output),
        canonical_path(args.config_output),
        protected_paths,
    )
    clustering_output = canonical_path(args.clustering_output)
    if (
        args.overwrite_clustering
        and clustering_output.exists()
        and not directory_is_owned(clustering_output, "clustering")
    ):
        raise RunnerError(f"Refusing to overwrite unowned clustering directory: {clustering_output}")
    if args.overwrite_projections:
        for output in projected.values():
            if output.exists() and not directory_is_owned(output, "projection"):
                raise RunnerError(f"Refusing to overwrite unowned projection directory: {output}")
    prompt_bank_dir = canonical_path(args.prompt_bank_dir)
    prompt_output = canonical_path(args.prompt_embeddings_output)
    prompt_manifest = prompt_output.with_name(
        f".{prompt_output.name}.remote_runner_manifest.json"
    )
    required_prompt_bank = [
        prompt_bank_dir / "prompts.jsonl",
        prompt_bank_dir / "prompts.csv",
        prompt_bank_dir / "prompts.txt",
        prompt_bank_dir / "concept_bank.json",
    ]
    prompt_bank_complete = all(path.is_file() for path in required_prompt_bank)
    if (
        not prompt_bank_complete
        and prompt_bank_dir.exists()
        and any(prompt_bank_dir.iterdir())
        and not directory_is_owned(prompt_bank_dir, "prompt_bank")
    ):
        raise RunnerError(f"Refusing to generate into incomplete unowned prompt bank: {prompt_bank_dir}")
    if args.overwrite_prompts:
        if (
            prompt_bank_dir.exists()
            and any(prompt_bank_dir.iterdir())
            and not directory_is_owned(prompt_bank_dir, "prompt_bank")
        ):
            raise RunnerError(f"Refusing to overwrite unowned prompt bank: {prompt_bank_dir}")
        if (
            prompt_output.exists()
            and not prompt_manifest.is_file()
            and not file_is_owned(prompt_output, "prompt_embeddings")
        ):
            raise RunnerError(f"Refusing to overwrite unowned prompt embeddings: {prompt_output}")
    config_output = canonical_path(args.config_output)
    if args.overwrite_config and config_output.is_file():
        existing_config = read_json(config_output)
        if not isinstance(existing_config, dict):
            raise RunnerError(f"Refusing to overwrite malformed config: {config_output}")
        if (
            existing_config.get("runner_provenance", {}).get("orchestrator")
            != "run_remote_offline.py"
        ):
            raise RunnerError(f"Refusing to overwrite unowned config: {config_output}")
    offline_output = canonical_path(args.offline_output)
    if (
        args.overwrite_offline
        and offline_output.exists()
        and not offline_output_is_owned(offline_output)
    ):
        raise RunnerError(
            f"Refusing --overwrite-offline for unowned or foreign output: "
            f"{offline_output}"
        )
    return raw, projected


def run_workflow(args: argparse.Namespace) -> None:
    raw_directories, projected_directories = validate_run_arguments(args)
    python_bin = sys.executable
    if args.install_requirements:
        run_command(
            "install repository requirements",
            [python_bin, "-m", "pip", "install", "-r", str(REPOSITORY_ROOT / "requirements.txt")],
            dry_run=args.dry_run,
        )
    checkpoint = resolve_checkpoint(
        args.checkpoint_path,
        dry_run=args.dry_run,
    )
    print(f"Checkpoint SHA-256: {checkpoint.sha256 or '(resolved during real run)'}")
    config = build_production_config(args, raw_directories, projected_directories, checkpoint)
    config_path = write_production_config(args, config)

    prepare_clustering(args, python_bin, raw_directories, checkpoint)
    pending_manifests = prepare_projections(
        args,
        python_bin,
        raw_directories,
        projected_directories,
        checkpoint,
    )
    pending_manifests.extend(prepare_prompts(args, python_bin, checkpoint))
    validate_embedding_contracts(
        canonical_path(args.prompt_embeddings_output),
        raw_directories,
        projected_directories,
        checkpoint,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        for path, manifest in pending_manifests:
            atomic_json(path, manifest)

    output = canonical_path(args.offline_output)
    command = [
        python_bin,
        str(REPOSITORY_ROOT / "run_offline_pipeline.py"),
        "--config",
        str(config_path),
        "--from-stage",
        str(args.from_offline_stage),
        "--to-stage",
        str(args.to_offline_stage),
    ]
    if args.overwrite_offline:
        command.append("--overwrite")
    run_command("offline semantic-index pipeline", command, dry_run=args.dry_run)

    if args.to_offline_stage == 8 and not args.dry_run:
        report = read_json(output / "validation_report.json")
        manifest = read_json(output / "run_manifest.json")
        if report.get("passed") is not True:
            raise RunnerError("Offline validation report did not pass")
        if manifest.get("approved") not in {False, True}:
            raise RunnerError("Offline run manifest has an invalid approval state")
        if report.get("approved") is not manifest.get("approved"):
            raise RunnerError("Validation report and run manifest approval states disagree")
        if manifest.get("approved") is True:
            print("Production index was already approved; no current stages were regenerated.")
    print("\nProduction workflow finished.")
    print(f"Offline index: {output}")
    if args.to_offline_stage >= 7:
        print(f"Audit report: {output / 'audit_report.html'}")
        print(f"Review template: {output / 'audit_review_template.csv'}")
        print("After human review, run this script's finalize subcommand.")


def finalize_workflow(args: argparse.Namespace) -> None:
    python_bin = sys.executable
    config_path = canonical_path(args.config)
    review_path = canonical_path(args.review_csv)
    if not config_path.is_file():
        raise RunnerError(f"Production config does not exist: {config_path}")
    if not review_path.is_file():
        raise RunnerError(f"Completed review CSV does not exist: {review_path}")
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise RunnerError(f"Production config must be a JSON object: {config_path}")
    if config.get("profile") != "production":
        raise RunnerError("Only a production config can be finalized")
    command = [
        python_bin,
        str(REPOSITORY_ROOT / "validate_offline_index.py"),
        "--config",
        str(config_path),
        "--review",
        str(review_path),
        "--overwrite",
    ]
    run_command("finalize reviewed offline index", command, dry_run=args.dry_run)
    if args.dry_run:
        return
    output = configured_output_root(config_path, config)
    manifest = read_json(output / "run_manifest.json")
    if manifest.get("approved") is not True:
        raise RunnerError("Review validation finished but run_manifest.json is not approved")
    print(f"Approved production index: {output}")


def configured_output_root(config_path: Path, config: Mapping[str, Any]) -> Path:
    output_value = Path(str(config["output_dir"])).expanduser()
    return (
        output_value.resolve()
        if output_value.is_absolute()
        else (config_path.parent / output_value).resolve()
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing or running them.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Prepare inputs and build an unapproved production index.")
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--raw-feature-dir",
        action="append",
        required=True,
        metavar="MAG=DIR",
        help="Raw Trident CONCH H5 directory; provide exactly 5, 10, 20, and 40.",
    )
    run_parser.add_argument(
        "--projected-feature-dir",
        action="append",
        required=True,
        metavar="MAG=DIR",
        help="Separate text-aligned output directory; provide all four magnifications.",
    )
    run_parser.add_argument(
        "--hsv-filter-dir",
        action="append",
        default=[],
        metavar="MAG=DIR",
        help=(
            "Precomputed HSV-filtered coordinate H5 directory; when used, provide "
            "exactly 5x, 10x, 20x, and 40x. Only these patch coordinates enter clustering."
        ),
    )
    run_parser.add_argument(
        "--hsv-filter-workers",
        type=int,
        default=8,
        help="Worker processes for HSV generation or coordinate matching (default: 8).",
    )
    run_parser.add_argument(
        "--hsv-min-tissue-coverage",
        type=float,
        default=0.45,
        help=(
            "Minimum HSV-tissue pixel fraction when --hsv-filter-dir is omitted "
            "(default: 0.45)."
        ),
    )
    run_parser.add_argument("--checkpoint-path", required=True)
    run_parser.add_argument("--metadata-path", required=True)
    run_parser.add_argument("--clustering-output", required=True)
    run_parser.add_argument("--prompt-bank-dir", required=True)
    run_parser.add_argument("--prompt-embeddings-output", required=True)
    run_parser.add_argument("--offline-output", required=True)
    run_parser.add_argument("--config-output", required=True)
    run_parser.add_argument(
        "--base-config",
        default=str(REPOSITORY_ROOT / "configs" / "offline_tcga.template.json"),
    )
    wsi_group = run_parser.add_mutually_exclusive_group(required=True)
    wsi_group.add_argument("--wsi-root")
    wsi_group.add_argument("--wsi-manifest")
    run_parser.add_argument(
        "--wsi-path-template",
        default="{wsi_root}/{file_name}",
    )
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--n-clusters", default="4096,512")
    run_parser.add_argument("--n-iters", type=int, default=50)
    run_parser.add_argument("--max-fit-samples", type=int, default=200_000)
    run_parser.add_argument("--feature-chunk-size", type=int, default=8192)
    run_parser.add_argument("--assign-chunk-size", type=int, default=65_536)
    run_parser.add_argument("--fit-chunk-budget", type=int, default=100_000_000)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--projection-batch-size", type=int, default=8192)
    run_parser.add_argument("--prompt-batch-size", type=int, default=256)
    run_parser.add_argument("--projection-compression", choices=("gzip", "lzf"), default=None)
    run_parser.add_argument("--from-offline-stage", type=int, default=1, choices=range(1, 9))
    run_parser.add_argument("--to-offline-stage", type=int, default=8, choices=range(1, 9))
    run_parser.add_argument("--install-requirements", action="store_true")
    run_parser.add_argument("--adopt-existing", action="store_true")
    run_parser.add_argument("--overwrite-clustering", action="store_true")
    run_parser.add_argument("--overwrite-projections", action="store_true")
    run_parser.add_argument("--overwrite-prompts", action="store_true")
    run_parser.add_argument("--overwrite-config", action="store_true")
    run_parser.add_argument("--overwrite-offline", action="store_true")
    run_parser.add_argument(
        "--attest-raw-features-use-checkpoint",
        action="store_true",
        help=(
            "Required attestation that raw Trident features came from the same checkpoint; "
            "legacy raw H5 files cannot prove this cryptographically."
        ),
    )

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Validate a completed human review without regenerating stages 1-7.",
    )
    add_common_arguments(finalize_parser)
    finalize_parser.add_argument("--config", required=True)
    finalize_parser.add_argument("--review-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if args.from_offline_stage > args.to_offline_stage:
            raise RunnerError("--from-offline-stage must not exceed --to-offline-stage")
        run_workflow(args)
    else:
        finalize_workflow(args)


if __name__ == "__main__":
    try:
        main()
    except (RunnerError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
