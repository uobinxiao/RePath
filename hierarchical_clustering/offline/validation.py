from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .buckets import BUCKET_COLUMNS, PRIORITY_COLUMNS
from .common import (
    AUDIT_REVIEW_BINDING_FIELDS,
    allow_smoke_fixture_contract,
    atomic_json,
    audit_review_binding,
    begin_stage,
    fail_stage,
    load_config,
    load_manifest,
    manifest_path,
    output_root,
    read_text_aligned_checkpoint_sha256,
    sha256_file,
    utc_now,
)


STAGE = "08_validate_offline_index"


REQUIRED_UPSTREAM_STAGES = (
    "01_build_tile_index",
    "02_build_cluster_centroids",
    "03_score_clusters_with_prompts",
    "04_compute_cluster_statistics",
    "05_build_composite_scores",
    "06_assign_sampling_buckets",
    "07_make_cluster_audit_montages",
)


def profile_allows_approval(config: Mapping[str, Any]) -> bool:
    """Only production runs can become sampler-approved indexes."""
    return str(config.get("profile", "production")) == "production"


def commit_validation_manifest(
    config: Mapping[str, Any],
    *,
    approved: bool,
    audit_review: Mapping[str, Any] | None,
) -> None:
    """Atomically publish Stage 8 completion and the resulting approval state."""
    manifest = load_manifest(config)
    if manifest is None:
        raise RuntimeError("Run manifest disappeared while completing validation")
    record = manifest.setdefault("stages", {}).setdefault(STAGE, {})
    record["status"] = "completed"
    record["completed_at"] = utc_now()
    record["details"] = {"passed": True, "approved": bool(approved)}
    manifest["approved"] = bool(approved)
    if audit_review is not None:
        manifest["audit_review"] = dict(audit_review)
    else:
        manifest.pop("audit_review", None)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path(config), manifest)


def add_check(checks: list[Dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def missing_current_upstream_stages(manifest: Mapping[str, Any] | None) -> list[str]:
    stages = manifest.get("stages", {}) if manifest else {}
    return [
        stage
        for stage in REQUIRED_UPSTREAM_STAGES
        if stages.get(stage, {}).get("status") != "completed"
    ]


def validate_review(
    review_path: str | Path,
    audit_selection: pd.DataFrame,
    expected_binding: Mapping[str, str],
) -> tuple[bool, Dict[str, Any]]:
    review = pd.read_csv(review_path, keep_default_na=False)
    required = {"cluster_key", "reviewer", "decision", *AUDIT_REVIEW_BINDING_FIELDS}
    missing_columns = sorted(required - set(review.columns))
    if missing_columns:
        return False, {"missing_columns": missing_columns}
    binding_mismatches: Dict[str, Any] = {}
    for field in AUDIT_REVIEW_BINDING_FIELDS:
        expected = str(expected_binding.get(field, ""))
        actual = sorted({str(value).strip() for value in review[field]})
        if not expected or actual != [expected]:
            binding_mismatches[field] = {"expected": expected, "actual": actual}
    review = review.copy()
    review["cluster_key"] = review["cluster_key"].astype(str).str.strip()
    duplicate_clusters = sorted(
        str(value)
        for value in review.loc[
            review["cluster_key"].duplicated(keep=False),
            "cluster_key",
        ].unique()
    )
    required_keys = set(audit_selection["cluster_key"].astype(str))
    review_keys = set(review["cluster_key"])
    decisions = review.drop_duplicates("cluster_key", keep=False).set_index("cluster_key")
    missing_keys = sorted(required_keys - review_keys)
    bad = []
    for cluster_key in sorted(required_keys & set(decisions.index.astype(str))):
        row = decisions.loc[cluster_key]
        if str(row["decision"]).strip().lower() not in {"pass", "approved", "approve"}:
            bad.append({"cluster_key": cluster_key, "reason": "decision_not_approved"})
        elif not str(row["reviewer"]).strip():
            bad.append({"cluster_key": cluster_key, "reason": "missing_reviewer"})
    approved = (
        not missing_keys
        and not bad
        and not binding_mismatches
        and not duplicate_clusters
    )
    return approved, {
        "missing_clusters": missing_keys,
        "unapproved": bad,
        "binding_mismatches": binding_mismatches,
        "duplicate_clusters": duplicate_clusters,
    }


def validate_audit_render_evidence(
    root: str | Path,
    audit_selection: pd.DataFrame,
) -> tuple[bool, Dict[str, Any]]:
    output_root_path = Path(root).resolve()
    montage_root = (output_root_path / "audit_montages").resolve()
    required_columns = {
        "cluster_key",
        "audit_bucket",
        "magnification",
        "fine_cluster_id",
        "render_status",
        "montage_path",
        "montage_sha256",
    }
    missing_columns = sorted(required_columns - set(audit_selection.columns))
    if missing_columns:
        return False, {
            "rows": int(len(audit_selection)),
            "missing_columns": missing_columns,
            "duplicate_rows": [],
            "invalid_rows": [],
        }

    identities = audit_selection[["cluster_key", "audit_bucket"]].astype(str)
    duplicate_mask = identities.duplicated(keep=False)
    duplicate_rows = (
        identities.loc[duplicate_mask]
        .drop_duplicates()
        .sort_values(["cluster_key", "audit_bucket"])
        .to_dict("records")
    )
    invalid_rows: list[Dict[str, Any]] = []
    for row_index, row in audit_selection.iterrows():
        reasons = []
        try:
            mag = int(row["magnification"])
            cluster_id = int(row["fine_cluster_id"])
        except (TypeError, ValueError):
            mag = None
            cluster_id = None
            reasons.append("invalid_cluster_coordinates")
        bucket = str(row["audit_bucket"])
        montage_value = row["montage_path"]
        digest_value = row["montage_sha256"]
        expected_digest = str(digest_value).strip().lower()
        digest_valid = (
            len(expected_digest) == 64
            and all(character in "0123456789abcdef" for character in expected_digest)
        )
        if str(row["render_status"]) != "rendered":
            reasons.append("render_status_not_rendered")
        if not digest_valid:
            reasons.append("invalid_montage_sha256")
        if not isinstance(montage_value, str) or not montage_value.strip():
            reasons.append("missing_montage_path")
        elif mag is not None and cluster_id is not None:
            montage_path = Path(montage_value)
            expected = (
                Path("audit_montages")
                / bucket
                / f"{mag}x"
                / f"cluster_{cluster_id}.png"
            )
            if montage_path.is_absolute() or montage_path != expected:
                reasons.append("unexpected_montage_path")
            else:
                resolved = (output_root_path / montage_path).resolve()
                if montage_root != resolved and montage_root not in resolved.parents:
                    reasons.append("montage_path_outside_output")
                else:
                    try:
                        valid_file = resolved.is_file() and resolved.stat().st_size > 0
                    except OSError:
                        valid_file = False
                    if not valid_file:
                        reasons.append("montage_file_missing_or_empty")
                    elif digest_valid:
                        try:
                            actual_digest = sha256_file(resolved)
                        except OSError:
                            reasons.append("montage_file_unreadable")
                        else:
                            if actual_digest != expected_digest:
                                reasons.append("montage_sha256_mismatch")
        if reasons:
            invalid_rows.append(
                {
                    "row": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
                    "cluster_key": str(row["cluster_key"]),
                    "audit_bucket": bucket,
                    "reasons": reasons,
                }
            )
    passed = not duplicate_rows and not invalid_rows
    return passed, {
        "rows": int(len(audit_selection)),
        "missing_columns": [],
        "duplicate_rows": duplicate_rows,
        "invalid_rows": invalid_rows[:100],
        "invalid_row_count": len(invalid_rows),
    }


def validate(config: Mapping[str, Any], review_path: str | Path | None = None) -> tuple[Dict[str, Any], bool]:
    root = output_root(config)
    checks: list[Dict[str, Any]] = []
    warnings: list[str] = []
    diagnostics: Dict[str, Any] = {"magnifications": {}}
    manifest = load_manifest(config)
    missing_upstream = missing_current_upstream_stages(manifest)
    add_check(
        checks,
        "upstream_stages_current",
        not missing_upstream,
        {"missing_or_incomplete": missing_upstream},
    )
    required_paths = [
        root / "tile_index.parquet",
        root / "tile_index_summary.json",
        root / "fine_cluster_centroids.npz",
        root / "coarse_cluster_centroids.npz",
        root / "representative_tiles.parquet",
        root / "fine_cluster_semantic_scores.parquet",
        root / "fine_cluster_semantic_summary.parquet",
        root / "prompt_concept_metadata.csv",
        root / "fine_cluster_stats.parquet",
        root / "fine_cluster_composite_scores.parquet",
        root / "fine_cluster_buckets.parquet",
        root / "coarse_cluster_stats.parquet",
        root / "sampling_buckets.json",
        root / "coarse_to_fine_index.json",
        root / "audit_selection.parquet",
        root / "bucket_dominance_report.parquet",
        root / "audit_report.html",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    add_check(checks, "required_outputs_exist", not missing, {"missing": missing})
    if missing:
        return {"passed": False, "checks": checks, "warnings": warnings}, False

    summary = load_json(root / "tile_index_summary.json")
    fine_npz = np.load(root / "fine_cluster_centroids.npz")
    coarse_npz = np.load(root / "coarse_cluster_centroids.npz")
    sample_smoke_profile = allow_smoke_fixture_contract(config)
    prompt_checkpoint_sha256 = read_text_aligned_checkpoint_sha256(
        config["prompt_embeddings_path"],
        allow_smoke_fixture=sample_smoke_profile,
    )
    fine_checkpoint_sha256 = (
        str(np.asarray(fine_npz["checkpoint_sha256"]).item())
        if "checkpoint_sha256" in fine_npz
        else None
    )
    coarse_checkpoint_sha256 = (
        str(np.asarray(coarse_npz["checkpoint_sha256"]).item())
        if "checkpoint_sha256" in coarse_npz
        else None
    )
    add_check(
        checks,
        "embedding_checkpoint_alignment",
        fine_checkpoint_sha256 == prompt_checkpoint_sha256
        and coarse_checkpoint_sha256 == prompt_checkpoint_sha256,
        {
            "prompt": prompt_checkpoint_sha256,
            "fine_centroids": fine_checkpoint_sha256,
            "coarse_centroids": coarse_checkpoint_sha256,
        },
    )
    semantic_summary = pd.read_parquet(root / "fine_cluster_semantic_summary.parquet")
    fine_stats = pd.read_parquet(root / "fine_cluster_stats.parquet")
    coarse_stats = pd.read_parquet(root / "coarse_cluster_stats.parquet")
    composite = pd.read_parquet(root / "fine_cluster_composite_scores.parquet")
    buckets = pd.read_parquet(root / "fine_cluster_buckets.parquet")

    expected_fine = 0
    expected_semantic_rows = 0
    prompt_count = len(pd.read_csv(root / "prompt_concept_metadata.csv"))
    for mag in config["magnifications"]:
        mag_summary = summary["magnifications"][str(mag)]
        parquet_path = root / "tile_index.parquet" / f"mag_partition={mag}x" / "tiles.parquet"
        parquet_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        add_check(
            checks,
            f"{mag}x_tile_count",
            parquet_rows == int(mag_summary["tiles"]),
            {"parquet": parquet_rows, "expected": int(mag_summary["tiles"])},
        )
        fine_centroids = np.asarray(fine_npz[f"mag_{mag}"], dtype=np.float32)
        fine_counts = np.asarray(fine_npz[f"mag_{mag}_counts"], dtype=np.int64)
        coarse_centroids = np.asarray(coarse_npz[f"mag_{mag}"], dtype=np.float32)
        coarse_counts = np.asarray(coarse_npz[f"mag_{mag}_counts"], dtype=np.int64)
        add_check(
            checks,
            f"{mag}x_centroid_norms",
            np.allclose(np.linalg.norm(fine_centroids, axis=1), 1.0, atol=1e-4)
            and np.allclose(np.linalg.norm(coarse_centroids, axis=1), 1.0, atol=1e-4),
        )
        add_check(
            checks,
            f"{mag}x_centroid_counts",
            int(fine_counts.sum()) == parquet_rows
            and int(coarse_counts.sum()) == parquet_rows
            and bool((fine_counts > 0).all())
            and bool((coarse_counts > 0).all()),
        )
        mag_stats = fine_stats.loc[fine_stats["magnification"] == mag]
        mag_coarse_stats = coarse_stats.loc[coarse_stats["magnification"] == mag]
        add_check(
            checks,
            f"{mag}x_stats_counts",
            len(mag_stats) == len(fine_centroids)
            and int(mag_stats["n_tiles"].sum()) == parquet_rows
            and len(mag_coarse_stats) == len(coarse_centroids)
            and int(mag_coarse_stats["n_tiles"].sum()) == parquet_rows,
        )
        expected_fine += len(fine_centroids)
        expected_semantic_rows += len(fine_centroids) * prompt_count

    add_check(
        checks,
        "cluster_table_key_alignment",
        len(semantic_summary) == expected_fine
        and len(fine_stats) == expected_fine
        and len(composite) == expected_fine
        and len(buckets) == expected_fine
        and set(semantic_summary["cluster_key"]) == set(fine_stats["cluster_key"]) == set(buckets["cluster_key"]),
    )
    semantic_file = pq.ParquetFile(root / "fine_cluster_semantic_scores.parquet")
    add_check(
        checks,
        "semantic_score_row_count",
        semantic_file.metadata.num_rows == expected_semantic_rows,
        {"actual": semantic_file.metadata.num_rows, "expected": expected_semantic_rows},
    )
    semantic_ranges = pd.read_parquet(
        root / "fine_cluster_semantic_scores.parquet",
        columns=["raw_cosine", "z_score", "percentile_score"],
    )
    semantic_values = semantic_ranges.to_numpy(dtype=np.float64)
    add_check(
        checks,
        "semantic_scores_finite_and_bounded",
        np.isfinite(semantic_values).all()
        and semantic_ranges["percentile_score"].between(0.0, 1.0).all(),
    )
    category_columns = [
        "architecture_score",
        "compartment_score",
        "pathology_score",
        "cellular_score",
        "scale_context_score",
        "artifact_score",
    ]
    add_check(
        checks,
        "category_scores_bounded",
        semantic_summary[category_columns].apply(lambda column: column.between(0.0, 1.0).all()).all(),
    )

    if bool(config["semantic"]["use_artifact_score_in_composite"]):
        cleanliness = 1.0 - composite["artifact_score"]
    else:
        cleanliness = pd.Series(1.0, index=composite.index)
    expected_clean = composite["semantic_usefulness"] * cleanliness
    expected_bridge = (
        expected_clean
        * composite["tissue_entropy"]
        * composite["tissue_coverage_norm"]
        * composite["patient_support"]
    )
    expected_architecture_specific = (
        composite["architecture_score"]
        * (1.0 - composite["tissue_entropy"])
        * composite["patient_support"]
        * cleanliness
    )
    expected_rare = composite["rarity_score"] * expected_clean * composite["patient_support"] * cleanliness
    add_check(
        checks,
        "composite_formula_consistency",
        np.allclose(composite["clean_semantic_score"], expected_clean, atol=1e-6)
        and np.allclose(composite["bridge_score"], expected_bridge, atol=1e-6)
        and np.allclose(
            composite["architecture_specific_score"],
            expected_architecture_specific,
            atol=1e-6,
        )
        and np.allclose(composite["rare_clean_score"], expected_rare, atol=1e-6),
    )
    excluded = buckets["is_artifact_excluded"] | buckets["is_support_excluded"]
    add_check(
        checks,
        "excluded_clusters_not_prioritized",
        not buckets.loc[excluded, PRIORITY_COLUMNS].to_numpy(dtype=bool).any(),
    )
    add_check(
        checks,
        "artifact_exclusion_switch_respected",
        bool(config["buckets"]["enable_artifact_exclusion"])
        or not buckets["is_artifact_excluded"].any(),
    )
    add_check(
        checks,
        "random_eligible_matches_eligibility",
        bool((buckets["is_random_eligible"] == buckets["sampling_eligible"]).all()),
    )
    for mag, mag_frame in buckets.groupby("magnification", sort=True):
        artifact_fraction = float(mag_frame["is_artifact_excluded"].mean())
        bucket_counts = {
            bucket_name: int(mag_frame[column].sum())
            for bucket_name, column in BUCKET_COLUMNS.items()
        }
        diagnostics["magnifications"][str(int(mag))] = {
            "artifact_excluded_fraction": artifact_fraction,
            "bucket_counts": bucket_counts,
        }
        if artifact_fraction > 0.5:
            warnings.append(
                f"{int(mag)}x artifact_excluded fraction is {artifact_fraction:.1%}; "
                "inspect artifact max-percentile calibration before production use."
            )
        for bucket_name in (
            "bridge_clusters",
            "architecture_bridge_clusters",
            "architecture_specific_clusters",
            "rare_clean_clusters",
            "tissue_specific_clusters",
        ):
            if bucket_counts[bucket_name] == 0:
                warnings.append(f"{int(mag)}x {bucket_name} is empty.")

    coarse_index = load_json(root / "coarse_to_fine_index.json")
    mapping_ok = True
    for mag in config["magnifications"]:
        expected_ids = set(
            buckets.loc[buckets["magnification"] == mag, "fine_cluster_id"].astype(int).tolist()
        )
        mapped_ids = []
        for record in coarse_index["magnifications"][str(mag)].values():
            mapped_ids.extend(int(value) for value in record["fine_clusters"])
        if set(mapped_ids) != expected_ids or len(mapped_ids) != len(set(mapped_ids)):
            mapping_ok = False
    add_check(checks, "coarse_to_fine_is_exhaustive_and_unique", mapping_ok)

    sampling = load_json(root / "sampling_buckets.json")
    sampling_ok = True
    for mag in config["magnifications"]:
        mag_frame = buckets.loc[buckets["magnification"] == mag]
        for bucket_name, column in BUCKET_COLUMNS.items():
            expected_ids = sorted(mag_frame.loc[mag_frame[column], "fine_cluster_id"].astype(int).tolist())
            actual_ids = sampling["magnifications"][str(mag)]["buckets"][bucket_name]["fine_cluster_ids"]
            if expected_ids != actual_ids:
                sampling_ok = False
    add_check(checks, "sampling_json_matches_bucket_table", sampling_ok)

    audit_selection = pd.read_parquet(root / "audit_selection.parquet")
    add_check(checks, "audit_selection_nonempty", len(audit_selection) > 0)
    render_evidence_ok, render_evidence_details = validate_audit_render_evidence(
        root,
        audit_selection,
    )
    production_audit_required = profile_allows_approval(config)
    render_evidence_details["required_for_profile"] = production_audit_required
    diagnostics["audit_render_evidence"] = render_evidence_details
    if production_audit_required:
        add_check(
            checks,
            "production_audit_montages_rendered",
            render_evidence_ok,
            render_evidence_details,
        )
    dominance = pd.read_parquet(root / "bucket_dominance_report.parquet")
    dominance_fraction_columns = [column for column in dominance if column.endswith("_fraction")]
    add_check(
        checks,
        "bucket_dominance_fractions_bounded",
        dominance[dominance_fraction_columns].apply(lambda column: column.between(0.0, 1.0).all()).all(),
    )
    review_approved = False
    review_binding = None
    if review_path is None:
        warnings.append("Audit review has not been supplied; structural validation can pass but the index is not approved.")
    else:
        review_binding = audit_review_binding(manifest, root / "audit_selection.parquet")
        review_approved, review_details = validate_review(
            review_path,
            audit_selection,
            review_binding,
        )
        add_check(checks, "audit_review_approved", review_approved, review_details)

    passed = all(check["passed"] for check in checks)
    if sample_smoke_profile:
        warnings.append(
            "sample_smoke indexes are fixture-only validation artifacts and can never be approved."
        )
    approved = bool(
        passed
        and review_approved
        and render_evidence_ok
        and profile_allows_approval(config)
    )
    report = {
        "schema_version": 1,
        "validated_at": utc_now(),
        "passed": passed,
        "approved": approved,
        "profile": config.get("profile", "production"),
        "checks": checks,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
    if review_path is not None:
        report["audit_review"] = {
            "path": str(Path(review_path).expanduser().resolve()),
            "sha256": sha256_file(review_path),
            "binding": review_binding,
        }
    return report, approved


def run(
    config: Mapping[str, Any],
    overwrite: bool = False,
    review_path: str | Path | None = None,
) -> None:
    root = output_root(config)
    report_path = root / "validation_report.json"
    should_run, _ = begin_stage(config, STAGE, [report_path], overwrite=overwrite)
    if not should_run:
        return
    try:
        report, approved = validate(config, review_path=review_path)
        atomic_json(report_path, report)
        if not report["passed"]:
            failed = [check["name"] for check in report["checks"] if not check["passed"]]
            raise RuntimeError(f"Offline index validation failed: {', '.join(failed)}")
        commit_validation_manifest(
            config,
            approved=approved,
            audit_review=report.get("audit_review") if review_path is not None else None,
        )
    except BaseException as error:
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an offline semantic sampling index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--review", default=None, help="Completed audit review CSV; all selected clusters must pass.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), overwrite=args.overwrite, review_path=args.review)


if __name__ == "__main__":
    main()
