from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import (
    atomic_parquet,
    begin_stage,
    complete_stage,
    fail_stage,
    load_config,
    output_root,
)


STAGE = "05_build_composite_scores"


def build_scores(
    semantic: pd.DataFrame,
    statistics: pd.DataFrame,
    weights: Mapping[str, float],
    use_artifact_score: bool = True,
) -> pd.DataFrame:
    merged = statistics.merge(
        semantic,
        on=["cluster_key", "magnification", "fine_cluster_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_semantic"),
    )
    if len(merged) != len(statistics) or len(merged) != len(semantic):
        raise ValueError("Semantic summary and cluster statistics do not have identical cluster keys")
    score_columns = {
        "architecture": "architecture_score",
        "tissue_compartment": "compartment_score",
        "pathology_state": "pathology_score",
        "cellular_nuclear": "cellular_score",
        "scale_context": "scale_context_score",
    }
    usefulness = np.zeros(len(merged), dtype=np.float64)
    for category, weight in weights.items():
        if category not in score_columns:
            raise ValueError(f"Unknown semantic weight category {category!r}")
        usefulness += float(weight) * merged[score_columns[category]].to_numpy(dtype=np.float64)
    if use_artifact_score:
        artifact_cleanliness = 1.0 - merged["artifact_score"].to_numpy(dtype=np.float64)
    else:
        artifact_cleanliness = np.ones(len(merged), dtype=np.float64)
    merged["semantic_usefulness"] = usefulness.astype(np.float32)
    merged["clean_semantic_score"] = (usefulness * artifact_cleanliness).astype(np.float32)
    merged["bridge_score"] = (
        merged["clean_semantic_score"]
        * merged["tissue_entropy"]
        * merged["tissue_coverage_norm"]
        * merged["patient_support"]
    ).astype(np.float32)
    merged["architecture_bridge_score"] = (
        merged["architecture_score"]
        * merged["tissue_entropy"]
        * merged["tissue_coverage_norm"]
        * merged["patient_support"]
        * artifact_cleanliness
    ).astype(np.float32)
    merged["architecture_specific_score"] = (
        merged["architecture_score"]
        * (1.0 - merged["tissue_entropy"])
        * merged["patient_support"]
        * artifact_cleanliness
    ).astype(np.float32)
    merged["rare_clean_score"] = (
        merged["rarity_score"]
        * merged["clean_semantic_score"]
        * merged["patient_support"]
        * artifact_cleanliness
    ).astype(np.float32)
    merged["tissue_specific_score"] = (
        merged["clean_semantic_score"]
        * (1.0 - merged["tissue_entropy"])
        * merged["patient_support"]
    ).astype(np.float32)
    merged["uses_independent_quality_score"] = False
    merged["uses_artifact_score_in_composite"] = bool(use_artifact_score)
    merged["score_formula_version"] = (
        "v2_architecture_specific_with_artifact"
        if use_artifact_score
        else "v2_architecture_specific_without_artifact"
    )
    score_names = [
        "semantic_usefulness",
        "clean_semantic_score",
        "bridge_score",
        "architecture_bridge_score",
        "architecture_specific_score",
        "rare_clean_score",
        "tissue_specific_score",
    ]
    if not np.isfinite(merged[score_names].to_numpy(dtype=np.float64)).all():
        raise ValueError("Composite scores contain non-finite values")
    return merged.sort_values(["magnification", "fine_cluster_id"]).reset_index(drop=True)


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    output_path = root / "fine_cluster_composite_scores.parquet"
    should_run, _ = begin_stage(config, STAGE, [output_path], overwrite=overwrite)
    if not should_run:
        return
    try:
        semantic = pd.read_parquet(root / "fine_cluster_semantic_summary.parquet")
        statistics = pd.read_parquet(root / "fine_cluster_stats.parquet")
        result = build_scores(
            semantic,
            statistics,
            config["semantic"]["weights"],
            use_artifact_score=bool(config["semantic"]["use_artifact_score_in_composite"]),
        )
        atomic_parquet(result, output_path, int(config["runtime"]["parquet_row_group_size"]))
        complete_stage(config, STAGE, {"fine_clusters": len(result)})
    except BaseException as error:
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args(
        "Build clean, bridge, architecture-specific, rare, and tissue-specific cluster scores."
    )
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
