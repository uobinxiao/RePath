from __future__ import annotations

import json
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from . import SCHEMA_VERSION
from .common import (
    atomic_json,
    atomic_parquet,
    begin_stage,
    complete_stage,
    fail_stage,
    load_config,
    load_manifest,
    output_root,
)


STAGE = "06_assign_sampling_buckets"


BUCKET_COLUMNS = {
    "bridge_clusters": "is_bridge",
    "architecture_bridge_clusters": "is_architecture_bridge",
    "architecture_specific_clusters": "is_architecture_specific",
    "rare_clean_clusters": "is_rare_clean",
    "tissue_specific_clusters": "is_tissue_specific",
    "random_eligible_clusters": "is_random_eligible",
    "artifact_excluded_clusters": "is_artifact_excluded",
    "support_excluded_clusters": "is_support_excluded",
}


PRIORITY_COLUMNS = [
    "is_bridge",
    "is_architecture_bridge",
    "is_architecture_specific",
    "is_rare_clean",
    "is_tissue_specific",
]


def score_threshold(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
    quantile: float,
) -> float | None:
    values = frame.loc[eligible, column].dropna()
    if values.empty:
        return None
    return float(values.quantile(quantile))


def meets_threshold(frame: pd.DataFrame, column: str, threshold: float | None) -> pd.Series:
    if threshold is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column] >= threshold


def membership_json(row: pd.Series) -> str:
    memberships = [name for name, column in BUCKET_COLUMNS.items() if bool(row[column])]
    return json.dumps(memberships, separators=(",", ":"))


def exclusion_json(row: pd.Series) -> str:
    reasons = []
    if bool(row["is_artifact_excluded"]):
        reasons.append("artifact_score_above_threshold")
    if bool(row["is_support_excluded"]):
        reasons.append("insufficient_patient_support")
    return json.dumps(reasons, separators=(",", ":"))


def assign_buckets(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    bucket = config["buckets"]
    output_frames = []
    thresholds: Dict[str, Any] = {}
    for mag, group in frame.groupby("magnification", sort=True):
        group = group.copy()
        artifact_exclusion_enabled = bool(bucket["enable_artifact_exclusion"])
        if artifact_exclusion_enabled:
            group["is_artifact_excluded"] = (
                group["artifact_score"] > float(bucket["artifact_exclude_above"])
            )
            bridge_artifact_allowed = (
                group["artifact_score"] < float(bucket["bridge_artifact_below"])
            )
            architecture_artifact_allowed = (
                group["artifact_score"] < float(bucket["architecture_artifact_below"])
            )
            architecture_specific_artifact_allowed = (
                group["artifact_score"]
                < float(bucket["architecture_specific_artifact_below"])
            )
            rare_artifact_allowed = (
                group["artifact_score"] < float(bucket["rare_artifact_below"])
            )
            tissue_specific_artifact_allowed = (
                group["artifact_score"] < float(bucket["tissue_specific_artifact_below"])
            )
        else:
            group["is_artifact_excluded"] = False
            artifact_allowed = pd.Series(True, index=group.index, dtype=bool)
            bridge_artifact_allowed = artifact_allowed
            architecture_artifact_allowed = artifact_allowed
            architecture_specific_artifact_allowed = artifact_allowed
            rare_artifact_allowed = artifact_allowed
            tissue_specific_artifact_allowed = artifact_allowed
        group["is_support_excluded"] = group["n_patients"] < int(bucket["support_exclude_below_patients"])
        group["sampling_eligible"] = ~(
            group["is_artifact_excluded"] | group["is_support_excluded"]
        )
        quantile = float(bucket["top_quantile"])
        threshold_values = {
            "bridge_score": score_threshold(group, "bridge_score", group["sampling_eligible"], quantile),
            "architecture_bridge_score": score_threshold(
                group, "architecture_bridge_score", group["sampling_eligible"], quantile
            ),
            "architecture_specific_score": score_threshold(
                group,
                "architecture_specific_score",
                group["sampling_eligible"],
                quantile,
            ),
            "rare_clean_score": score_threshold(
                group, "rare_clean_score", group["sampling_eligible"], quantile
            ),
            "tissue_specific_score": score_threshold(
                group, "tissue_specific_score", group["sampling_eligible"], quantile
            ),
        }
        thresholds[str(int(mag))] = threshold_values
        group["is_bridge"] = (
            group["sampling_eligible"]
            & meets_threshold(group, "bridge_score", threshold_values["bridge_score"])
            & bridge_artifact_allowed
            & (group["n_patients"] >= int(bucket["bridge_min_patients"]))
            & (group["tissue_coverage"] >= int(bucket["bridge_min_coverage"]))
        )
        group["is_architecture_bridge"] = (
            group["sampling_eligible"]
            & meets_threshold(
                group,
                "architecture_bridge_score",
                threshold_values["architecture_bridge_score"],
            )
            & (group["architecture_score"] >= float(bucket["architecture_min_score"]))
            & architecture_artifact_allowed
            & (group["n_patients"] >= int(bucket["architecture_min_patients"]))
            & (group["tissue_coverage"] >= int(bucket["architecture_min_coverage"]))
        )
        group["is_architecture_specific"] = (
            group["sampling_eligible"]
            & meets_threshold(
                group,
                "architecture_specific_score",
                threshold_values["architecture_specific_score"],
            )
            & (
                group["architecture_score"]
                >= float(bucket["architecture_specific_min_score"])
            )
            & architecture_specific_artifact_allowed
            & (
                group["n_patients"]
                >= int(bucket["architecture_specific_min_patients"])
            )
            & (
                group["tissue_coverage"]
                >= int(bucket["architecture_specific_min_coverage"])
            )
            & (
                group["tissue_entropy"]
                <= float(bucket["architecture_specific_max_entropy"])
            )
        )
        group["is_rare_clean"] = (
            group["sampling_eligible"]
            & meets_threshold(group, "rare_clean_score", threshold_values["rare_clean_score"])
            & rare_artifact_allowed
            & (group["n_patients"] >= int(bucket["rare_min_patients"]))
        )
        group["is_tissue_specific"] = (
            group["sampling_eligible"]
            & meets_threshold(
                group,
                "tissue_specific_score",
                threshold_values["tissue_specific_score"],
            )
            & tissue_specific_artifact_allowed
            & (group["n_patients"] >= int(bucket["tissue_specific_min_patients"]))
            & (group["tissue_entropy"] <= float(bucket["tissue_specific_max_entropy"]))
        )
        group["is_random_eligible"] = group["sampling_eligible"]
        group["bucket_memberships"] = group.apply(membership_json, axis=1)
        group["exclusion_reasons"] = group.apply(exclusion_json, axis=1)
        group["bucket_quantile"] = quantile
        for score_name, value in threshold_values.items():
            group[f"{score_name}_threshold"] = value
        excluded = group["is_artifact_excluded"] | group["is_support_excluded"]
        if group.loc[excluded, PRIORITY_COLUMNS].to_numpy(dtype=bool).any():
            raise RuntimeError(f"{mag}x excluded clusters entered a priority bucket")
        output_frames.append(group)
    output = pd.concat(output_frames, ignore_index=True)
    output = output.sort_values(["magnification", "fine_cluster_id"]).reset_index(drop=True)
    return output, thresholds


def bucket_payload(frame: pd.DataFrame, thresholds: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = load_manifest(config) or {}
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": manifest.get("config_hash"),
        "input_fingerprint": manifest.get("input_fingerprint"),
        "profile": config.get("profile", "production"),
        "thresholds": dict(thresholds),
        "magnifications": {},
    }
    for mag, group in frame.groupby("magnification", sort=True):
        mag_payload: Dict[str, Any] = {"buckets": {}, "coarse_clusters": {}}
        for bucket_name, column in BUCKET_COLUMNS.items():
            selected = group.loc[group[column]].sort_values("fine_cluster_id")
            mag_payload["buckets"][bucket_name] = {
                "fine_cluster_ids": selected["fine_cluster_id"].astype(int).tolist(),
                "cluster_keys": selected["cluster_key"].tolist(),
            }
        for coarse_id, coarse_group in group.groupby("coarse_cluster_id", sort=True):
            record: Dict[str, Any] = {
                "fine_cluster_ids": coarse_group["fine_cluster_id"].astype(int).sort_values().tolist(),
                "by_bucket": {},
            }
            for bucket_name, column in BUCKET_COLUMNS.items():
                record["by_bucket"][bucket_name] = (
                    coarse_group.loc[coarse_group[column], "fine_cluster_id"].astype(int).sort_values().tolist()
                )
            mag_payload["coarse_clusters"][str(int(coarse_id))] = record
        payload["magnifications"][str(int(mag))] = mag_payload
    return payload


def coarse_to_fine_payload(frame: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = load_manifest(config) or {}
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": manifest.get("config_hash"),
        "magnifications": {},
    }
    for mag, group in frame.groupby("magnification", sort=True):
        coarse_records: Dict[str, Any] = {}
        for coarse_id, coarse_group in group.groupby("coarse_cluster_id", sort=True):
            record: Dict[str, Any] = {
                "fine_clusters": coarse_group["fine_cluster_id"].astype(int).sort_values().tolist()
            }
            for bucket_name, column in BUCKET_COLUMNS.items():
                record[bucket_name] = (
                    coarse_group.loc[coarse_group[column], "fine_cluster_id"].astype(int).sort_values().tolist()
                )
            coarse_records[str(int(coarse_id))] = record
        result["magnifications"][str(int(mag))] = coarse_records
    return result


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    buckets_path = root / "fine_cluster_buckets.parquet"
    sampling_path = root / "sampling_buckets.json"
    coarse_path = root / "coarse_to_fine_index.json"
    should_run, _ = begin_stage(
        config,
        STAGE,
        [buckets_path, sampling_path, coarse_path],
        overwrite=overwrite,
    )
    if not should_run:
        return
    try:
        frame = pd.read_parquet(root / "fine_cluster_composite_scores.parquet")
        result, thresholds = assign_buckets(frame, config)
        atomic_parquet(result, buckets_path, int(config["runtime"]["parquet_row_group_size"]))
        atomic_json(sampling_path, bucket_payload(result, thresholds, config))
        atomic_json(coarse_path, coarse_to_fine_payload(result, config))
        counts = {column: int(result[column].sum()) for column in BUCKET_COLUMNS.values()}
        complete_stage(config, STAGE, counts)
    except BaseException as error:
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Assign multi-label semantic sampling buckets to fine clusters.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
