from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from .common import (
    allow_smoke_fixture_contract,
    atomic_csv,
    atomic_parquet,
    begin_stage,
    cluster_key,
    complete_stage,
    fail_stage,
    load_config,
    load_prompt_h5,
    output_root,
    read_text_aligned_checkpoint_sha256,
)


STAGE = "03_score_clusters_with_prompts"


SCORE_SCHEMA = pa.schema(
    [
        ("cluster_key", pa.string()),
        ("magnification", pa.int16()),
        ("fine_cluster_id", pa.int32()),
        ("prompt_index", pa.int32()),
        ("prompt_id", pa.string()),
        ("concept_id", pa.string()),
        ("concept", pa.string()),
        ("polarity", pa.string()),
        ("category", pa.string()),
        ("raw_cosine", pa.float32()),
        ("z_score", pa.float32()),
        ("percentile_score", pa.float32()),
    ]
)


CATEGORY_OUTPUTS = {
    "architecture": "architecture_score",
    "tissue_compartment": "compartment_score",
    "pathology_state": "pathology_score",
    "cellular_nuclear": "cellular_score",
    "scale_context": "scale_context_score",
    "anatomical_structures": "anatomical_structures_score",
    "tissue_identity": "tissue_identity_score",
}


def concept_percentiles(raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = raw_scores.mean(axis=0, keepdims=True)
    stds = raw_scores.std(axis=0, keepdims=True)
    z_scores = np.divide(
        raw_scores - means,
        stds,
        out=np.zeros_like(raw_scores, dtype=np.float32),
        where=stds > 0,
    )
    if raw_scores.shape[0] == 1:
        percentiles = np.full(raw_scores.shape, 0.5, dtype=np.float32)
    else:
        ranks = rankdata(raw_scores, method="average", axis=0)
        percentiles = ((ranks - 1.0) / float(raw_scores.shape[0] - 1)).astype(np.float32)
    return z_scores.astype(np.float32, copy=False), percentiles


def top_k_mean(values: np.ndarray, indices: Sequence[int], top_k: int) -> np.ndarray:
    if not indices:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    selected = values[:, np.asarray(indices, dtype=np.int64)]
    k = min(max(int(top_k), 1), selected.shape[1])
    if k == selected.shape[1]:
        return selected.mean(axis=1, dtype=np.float32)
    partitioned = np.partition(selected, selected.shape[1] - k, axis=1)
    return partitioned[:, -k:].mean(axis=1, dtype=np.float32)


def top_concepts_json(
    raw_scores: np.ndarray,
    percentile_scores: np.ndarray,
    prompt_records: Sequence[Mapping[str, str]],
    indices: Sequence[int],
    cluster_id: int,
    limit: int = 5,
) -> str:
    ordered = sorted(
        indices,
        key=lambda idx: (
            -float(percentile_scores[cluster_id, idx]),
            -float(raw_scores[cluster_id, idx]),
            prompt_records[idx].get("concept_id", ""),
        ),
    )[:limit]
    payload = [
        {
            "concept_id": prompt_records[idx].get("concept_id", ""),
            "concept": prompt_records[idx].get("concept", prompt_records[idx].get("prompt", "")),
            "category": prompt_records[idx].get("category", ""),
            "percentile": round(float(percentile_scores[cluster_id, idx]), 6),
            "raw_cosine": round(float(raw_scores[cluster_id, idx]), 6),
        }
        for idx in ordered
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def score_table(
    mag: int,
    raw_scores: np.ndarray,
    z_scores: np.ndarray,
    percentiles: np.ndarray,
    prompt_records: Sequence[Mapping[str, str]],
) -> pa.Table:
    num_clusters, num_prompts = raw_scores.shape
    cluster_ids = np.repeat(np.arange(num_clusters, dtype=np.int32), num_prompts)
    prompt_indices = np.tile(np.arange(num_prompts, dtype=np.int32), num_clusters)
    cluster_keys = [cluster_key(mag, cluster_id) for cluster_id in range(num_clusters) for _ in range(num_prompts)]

    def repeated_prompt_field(field: str) -> list[str]:
        base = [record.get(field, "") for record in prompt_records]
        return base * num_clusters

    return pa.Table.from_arrays(
        [
            pa.array(cluster_keys, type=pa.string()),
            pa.array(np.full(cluster_ids.shape[0], mag, dtype=np.int16), type=pa.int16()),
            pa.array(cluster_ids, type=pa.int32()),
            pa.array(prompt_indices, type=pa.int32()),
            pa.array(repeated_prompt_field("prompt_id"), type=pa.string()),
            pa.array(repeated_prompt_field("concept_id"), type=pa.string()),
            pa.array(repeated_prompt_field("concept"), type=pa.string()),
            pa.array(repeated_prompt_field("polarity"), type=pa.string()),
            pa.array(repeated_prompt_field("category"), type=pa.string()),
            pa.array(raw_scores.reshape(-1).astype(np.float32, copy=False), type=pa.float32()),
            pa.array(z_scores.reshape(-1).astype(np.float32, copy=False), type=pa.float32()),
            pa.array(percentiles.reshape(-1).astype(np.float32, copy=False), type=pa.float32()),
        ],
        schema=SCORE_SCHEMA,
    )


def semantic_summary(
    mag: int,
    raw_scores: np.ndarray,
    percentiles: np.ndarray,
    prompt_records: Sequence[Mapping[str, str]],
    top_k: int,
) -> list[Dict[str, Any]]:
    indices_by_category: Dict[str, list[int]] = {}
    for category in CATEGORY_OUTPUTS:
        indices_by_category[category] = [
            index
            for index, record in enumerate(prompt_records)
            if record.get("category") == category
            and (record.get("polarity") == "positive" or category == "tissue_identity")
        ]
    required_categories = {
        "architecture",
        "tissue_compartment",
        "pathology_state",
        "cellular_nuclear",
        "scale_context",
    }
    missing_categories = sorted(
        category for category in required_categories if not indices_by_category[category]
    )
    if missing_categories:
        raise ValueError(f"Prompt bank is missing required positive categories: {missing_categories}")
    artifact_indices = [
        index
        for index, record in enumerate(prompt_records)
        if record.get("category") == "artifact_quality" and record.get("polarity") == "negative"
    ]
    if not artifact_indices:
        raise ValueError("Prompt bank has no negative artifact_quality concepts")
    positive_indices = [
        index for index, record in enumerate(prompt_records) if record.get("polarity") == "positive"
    ]
    category_scores = {
        output: top_k_mean(percentiles, indices_by_category[category], top_k)
        for category, output in CATEGORY_OUTPUTS.items()
    }
    artifact_score = percentiles[:, np.asarray(artifact_indices, dtype=np.int64)].max(axis=1)

    records: list[Dict[str, Any]] = []
    for cluster_id in range(raw_scores.shape[0]):
        record: Dict[str, Any] = {
            "cluster_key": cluster_key(mag, cluster_id),
            "magnification": mag,
            "fine_cluster_id": cluster_id,
            "artifact_score": float(artifact_score[cluster_id]),
            "top_positive_concepts": top_concepts_json(
                raw_scores, percentiles, prompt_records, positive_indices, cluster_id
            ),
            "top_artifact_concepts": top_concepts_json(
                raw_scores, percentiles, prompt_records, artifact_indices, cluster_id
            ),
        }
        for output, values in category_scores.items():
            record[output] = float(values[cluster_id])
        records.append(record)
    return records


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    scores_path = root / "fine_cluster_semantic_scores.parquet"
    summary_path = root / "fine_cluster_semantic_summary.parquet"
    prompt_metadata_path = root / "prompt_concept_metadata.csv"
    should_run, _ = begin_stage(
        config,
        STAGE,
        [scores_path, summary_path, prompt_metadata_path],
        overwrite=overwrite,
    )
    if not should_run:
        return

    temporary_scores = scores_path.with_name(f".{scores_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        allow_smoke_fixture = allow_smoke_fixture_contract(config)
        prompt_features, prompt_records = load_prompt_h5(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        prompt_checkpoint_sha256 = read_text_aligned_checkpoint_sha256(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        centroids = np.load(root / "fine_cluster_centroids.npz")
        if "checkpoint_sha256" not in centroids:
            raise ValueError("fine_cluster_centroids.npz is missing checkpoint provenance")
        centroid_checkpoint_sha256 = str(np.asarray(centroids["checkpoint_sha256"]).item())
        if centroid_checkpoint_sha256 != prompt_checkpoint_sha256:
            raise ValueError(
                "Fine centroid checkpoint does not match the prompt embedding checkpoint: "
                f"{centroid_checkpoint_sha256} != {prompt_checkpoint_sha256}"
            )
        writer = pq.ParquetWriter(
            temporary_scores,
            SCORE_SCHEMA,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        summaries: list[Dict[str, Any]] = []
        total_rows = 0
        for mag in config["magnifications"]:
            key = f"mag_{mag}"
            if key not in centroids:
                raise KeyError(f"fine_cluster_centroids.npz is missing {key}")
            fine_centroids = np.asarray(centroids[key], dtype=np.float32)
            if fine_centroids.shape[1] != prompt_features.shape[1]:
                raise ValueError(f"{mag}x centroid/prompt embedding dimensions differ")
            raw_scores = (fine_centroids @ prompt_features.T).astype(np.float32, copy=False)
            if not np.isfinite(raw_scores).all():
                raise ValueError(f"{mag}x semantic scores contain non-finite values")
            z_scores, percentiles = concept_percentiles(raw_scores)
            table = score_table(mag, raw_scores, z_scores, percentiles, prompt_records)
            writer.write_table(table, row_group_size=int(config["runtime"]["parquet_row_group_size"]))
            total_rows += len(table)
            summaries.extend(
                semantic_summary(
                    mag,
                    raw_scores,
                    percentiles,
                    prompt_records,
                    int(config["semantic"]["category_top_k"]),
                )
            )
            print(f"{STAGE} {mag}x: {fine_centroids.shape[0]} clusters x {len(prompt_records)} concepts")
        writer.close()
        writer = None
        temporary_scores.replace(scores_path)
        atomic_parquet(
            pd.DataFrame.from_records(summaries),
            summary_path,
            int(config["runtime"]["parquet_row_group_size"]),
        )
        metadata_fields = sorted({key for record in prompt_records for key in record})
        preferred = ["prompt_id", "concept_id", "concept", "polarity", "category", "prompt"]
        fieldnames = preferred + [field for field in metadata_fields if field not in preferred]
        atomic_csv(prompt_metadata_path, fieldnames, prompt_records)
        complete_stage(
            config,
            STAGE,
            {"semantic_score_rows": total_rows, "cluster_rows": len(summaries), "concepts": len(prompt_records)},
        )
    except BaseException as error:
        if writer is not None:
            writer.close()
        if temporary_scores.exists():
            temporary_scores.unlink()
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Score fine cluster centroids with CONCH concept embeddings.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
