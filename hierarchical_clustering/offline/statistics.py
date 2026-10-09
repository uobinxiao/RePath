from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    atomic_parquet,
    begin_stage,
    cluster_key,
    clustering_path,
    coarse_key,
    complete_stage,
    fail_stage,
    json_distribution,
    load_config,
    load_h5_index,
    metadata_by_slide,
    output_root,
    slide_submitter_id,
)


STAGE = "04_compute_cluster_statistics"


def normalized_entropy(counts: np.ndarray) -> np.ndarray:
    if counts.shape[1] <= 1:
        return np.zeros(counts.shape[0], dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -(probabilities * log_probabilities).sum(axis=1) / np.log(float(counts.shape[1]))
    return entropy.astype(np.float32)


def log_support(counts: np.ndarray) -> np.ndarray:
    maximum = int(counts.max(initial=0))
    if maximum <= 0:
        return np.zeros(counts.shape, dtype=np.float32)
    return (np.log1p(counts) / np.log1p(maximum)).astype(np.float32)


def presence_for_shards(
    assignments: np.ndarray,
    shards: Sequence[Mapping[str, Any]],
    num_clusters: int,
) -> np.ndarray:
    present = np.zeros(num_clusters, dtype=bool)
    for shard in shards:
        start = int(shard["row_start"])
        end = start + int(shard["n_tiles"])
        present[np.unique(np.asarray(assignments[start:end], dtype=np.int32))] = True
    return present


def compute_level_statistics(
    *,
    config: Mapping[str, Any],
    mag: int,
    level_name: str,
    shards: list[Dict[str, Any]],
    metadata_map: Mapping[str, Mapping[str, Any]],
    assignments: np.ndarray,
    num_clusters: int,
) -> tuple[list[Dict[str, Any]], Dict[str, np.ndarray]]:
    contexts = [(shard, metadata_map[slide_submitter_id(shard)]) for shard in shards]
    tissue_key = str(config["statistics"]["tissue_key"])
    tissue_names = sorted({str(metadata[tissue_key]) for _, metadata in contexts})
    project_names = sorted({str(metadata["project_id"]) for _, metadata in contexts})
    tissue_index = {name: index for index, name in enumerate(tissue_names)}
    project_index = {name: index for index, name in enumerate(project_names)}

    n_tiles = np.zeros(num_clusters, dtype=np.int64)
    n_slides = np.zeros(num_clusters, dtype=np.int64)
    n_patients = np.zeros(num_clusters, dtype=np.int64)
    tissue_tiles = np.zeros((num_clusters, len(tissue_names)), dtype=np.int64)
    tissue_patients = np.zeros((num_clusters, len(tissue_names)), dtype=np.int64)
    project_tiles = np.zeros((num_clusters, len(project_names)), dtype=np.int64)

    patients: Dict[str, list[tuple[Dict[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for shard, metadata in contexts:
        start = int(shard["row_start"])
        end = start + int(shard["n_tiles"])
        ids = np.asarray(assignments[start:end], dtype=np.int32)
        if ids.size != int(shard["n_tiles"]):
            raise ValueError(f"{mag}x {level_name}: assignment slice length mismatch for {shard['slide_id']}")
        if ids.min(initial=0) < 0 or ids.max(initial=-1) >= num_clusters:
            raise ValueError(f"{mag}x {level_name}: assignment id outside [0, {num_clusters})")
        counts = np.bincount(ids, minlength=num_clusters)
        n_tiles += counts
        n_slides[np.flatnonzero(counts)] += 1
        tissue_tiles[:, tissue_index[str(metadata[tissue_key])]] += counts
        project_tiles[:, project_index[str(metadata["project_id"])]] += counts
        patients[str(metadata["patient_id"])].append((shard, metadata))

    for patient_shards in patients.values():
        overall_presence = presence_for_shards(
            assignments,
            [shard for shard, _ in patient_shards],
            num_clusters,
        )
        n_patients[overall_presence] += 1
        by_tissue: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for shard, metadata in patient_shards:
            by_tissue[str(metadata[tissue_key])].append(shard)
        for tissue, patient_tissue_shards in by_tissue.items():
            presence = presence_for_shards(assignments, patient_tissue_shards, num_clusters)
            tissue_patients[presence, tissue_index[tissue]] += 1

    if np.any(n_tiles == 0):
        empty = np.flatnonzero(n_tiles == 0).tolist()
        raise ValueError(f"{mag}x {level_name}: empty clusters {empty[:20]}")

    tissue_entropy = normalized_entropy(tissue_tiles)
    project_entropy = normalized_entropy(project_tiles)
    coverage_mask = (
        (tissue_tiles >= int(config["statistics"]["coverage_min_tiles"]))
        & (tissue_patients >= int(config["statistics"]["coverage_min_patients"]))
    )
    tissue_coverage = coverage_mask.sum(axis=1).astype(np.int32)
    max_coverage = int(tissue_coverage.max(initial=0))
    if max_coverage > 0:
        coverage_norm = (tissue_coverage / float(max_coverage)).astype(np.float32)
    else:
        coverage_norm = np.zeros(num_clusters, dtype=np.float32)
    patient_support = log_support(n_patients)
    slide_support = log_support(n_slides)
    median_size = float(np.median(n_tiles))
    rarity_score = np.clip(np.sqrt(median_size / n_tiles), 0.5, 3.0).astype(np.float32)

    dominant_tissue_indices = tissue_tiles.argmax(axis=1)
    dominant_project_indices = project_tiles.argmax(axis=1)
    dominant_tissue_fraction = tissue_tiles.max(axis=1) / n_tiles
    dominant_project_fraction = project_tiles.max(axis=1) / n_tiles
    records: list[Dict[str, Any]] = []
    for cluster_id in range(num_clusters):
        if level_name == "fine":
            key = cluster_key(mag, cluster_id)
            id_field = "fine_cluster_id"
        else:
            key = coarse_key(mag, cluster_id)
            id_field = "coarse_cluster_id"
        record: Dict[str, Any] = {
            "cluster_key": key,
            "magnification": mag,
            id_field: cluster_id,
            "n_tiles": int(n_tiles[cluster_id]),
            "n_patients": int(n_patients[cluster_id]),
            "n_slides": int(n_slides[cluster_id]),
            "n_tissues": int(np.count_nonzero(tissue_tiles[cluster_id])),
            "n_projects": int(np.count_nonzero(project_tiles[cluster_id])),
            "n_scale_bins": 1,
            "scale_coverage": 1.0,
            "tissue_entropy": float(tissue_entropy[cluster_id]),
            "project_entropy": float(project_entropy[cluster_id]),
            "tissue_coverage": int(tissue_coverage[cluster_id]),
            "tissue_coverage_norm": float(coverage_norm[cluster_id]),
            "tissue_coverage_fraction": float(tissue_coverage[cluster_id] / max(1, len(tissue_names))),
            "patient_support": float(patient_support[cluster_id]),
            "slide_support": float(slide_support[cluster_id]),
            "rarity_score": float(rarity_score[cluster_id]),
            "dominant_tissue": tissue_names[int(dominant_tissue_indices[cluster_id])],
            "dominant_tissue_fraction": float(dominant_tissue_fraction[cluster_id]),
            "dominant_project": project_names[int(dominant_project_indices[cluster_id])],
            "dominant_project_fraction": float(dominant_project_fraction[cluster_id]),
            "tissue_distribution": json_distribution(tissue_names, tissue_tiles[cluster_id]),
            "project_distribution": json_distribution(project_names, project_tiles[cluster_id]),
            "quality_score": np.nan,
            "mean_quality": np.nan,
            "median_quality": np.nan,
            "tissue_ratio": np.nan,
            "blur_score": np.nan,
            "quality_source": "none_semantic_artifact_only",
        }
        records.append(record)
    arrays = {
        "n_tiles": n_tiles,
        "n_patients": n_patients,
        "n_slides": n_slides,
        "tissue_tiles": tissue_tiles,
        "tissue_patients": tissue_patients,
        "project_tiles": project_tiles,
    }
    return records, arrays


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    fine_path = root / "fine_cluster_stats.parquet"
    coarse_path = root / "coarse_cluster_stats.parquet"
    should_run, _ = begin_stage(config, STAGE, [fine_path, coarse_path], overwrite=overwrite)
    if not should_run:
        return
    try:
        metadata_map = metadata_by_slide(config["metadata_path"])
        centroids = np.load(root / "fine_cluster_centroids.npz")
        coarse_centroids = np.load(root / "coarse_cluster_centroids.npz")
        fine_records: list[Dict[str, Any]] = []
        coarse_records: list[Dict[str, Any]] = []
        fine_level = int(config["levels"]["fine"])
        coarse_level = int(config["levels"]["coarse"])

        for mag in config["magnifications"]:
            shards = load_h5_index(config, mag)
            for shard in shards:
                submitter_id = slide_submitter_id(shard)
                if submitter_id not in metadata_map:
                    raise KeyError(f"{mag}x {shard['slide_id']}: missing metadata")
            fine_assignments = np.load(
                clustering_path(config, mag, f"level{fine_level}", "assignments.npy"), mmap_mode="r"
            )
            coarse_assignments = np.load(
                clustering_path(config, mag, f"level{coarse_level}", "assignments.npy"), mmap_mode="r"
            )
            fine_to_coarse = np.load(
                clustering_path(config, mag, f"level{coarse_level}", "fit_assignment.npy")
            ).astype(np.int32, copy=False)
            num_fine = int(centroids[f"mag_{mag}"].shape[0])
            num_coarse = int(coarse_centroids[f"mag_{mag}"].shape[0])
            mag_fine_records, _ = compute_level_statistics(
                config=config,
                mag=mag,
                level_name="fine",
                shards=shards,
                metadata_map=metadata_map,
                assignments=fine_assignments,
                num_clusters=num_fine,
            )
            for record in mag_fine_records:
                record["coarse_cluster_id"] = int(fine_to_coarse[int(record["fine_cluster_id"])])
            mag_coarse_records, _ = compute_level_statistics(
                config=config,
                mag=mag,
                level_name="coarse",
                shards=shards,
                metadata_map=metadata_map,
                assignments=coarse_assignments,
                num_clusters=num_coarse,
            )
            fine_counts_per_coarse = np.bincount(fine_to_coarse, minlength=num_coarse)
            for record in mag_coarse_records:
                record["n_fine_clusters"] = int(fine_counts_per_coarse[int(record["coarse_cluster_id"])])
            fine_records.extend(mag_fine_records)
            coarse_records.extend(mag_coarse_records)
            print(f"{STAGE} {mag}x: {num_fine} fine, {num_coarse} coarse clusters")

        row_group_size = int(config["runtime"]["parquet_row_group_size"])
        atomic_parquet(pd.DataFrame.from_records(fine_records), fine_path, row_group_size)
        atomic_parquet(pd.DataFrame.from_records(coarse_records), coarse_path, row_group_size)
        complete_stage(
            config,
            STAGE,
            {"fine_clusters": len(fine_records), "coarse_clusters": len(coarse_records)},
        )
    except BaseException as error:
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Compute exact TCGA cluster distribution statistics.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()

