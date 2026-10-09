from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np
import pandas as pd

from .common import (
    allow_smoke_fixture_contract,
    atomic_npz,
    atomic_parquet,
    begin_stage,
    cluster_key,
    clustering_path,
    complete_stage,
    fail_stage,
    load_config,
    load_h5_index,
    load_prompt_h5,
    metadata_by_slide,
    normalize_rows,
    output_root,
    projected_h5_map,
    raw_matrix_path,
    read_text_aligned_checkpoint_sha256,
    read_wsi_manifest,
    resolve_wsi_path,
    selected_source_rows,
    slide_submitter_id,
    validate_text_aligned_features,
)


STAGE = "02_build_cluster_centroids"


def negative_squared_l2_scores(
    features: np.ndarray,
    centroids: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    differences = features - centroids[cluster_ids]
    return -np.einsum("ij,ij->i", differences, differences).astype(np.float32, copy=False)


def bounded_sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    sample_size = min(max(int(sample_size), 0), int(total))
    if sample_size == total:
        return np.arange(total, dtype=np.int64)
    if sample_size == 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    selected: set[int] = set()
    for value in range(total - sample_size, total):
        candidate = int(rng.randint(0, value + 1))
        selected.add(value if candidate in selected else candidate)
    return np.asarray(sorted(selected), dtype=np.int64)


def select_diverse_candidate_rows(
    sample_indices: np.ndarray,
    sampled_ids: np.ndarray,
    scores: np.ndarray,
    row_ends: np.ndarray,
    num_clusters: int,
    tiles_per_cluster: int,
    max_tiles_per_wsi: int,
    candidate_limit: int,
) -> list[tuple[int, int, float, int]]:
    if not (sample_indices.shape == sampled_ids.shape == scores.shape):
        raise ValueError("Representative sample indices, cluster IDs, and scores must align")
    order = np.lexsort((sample_indices, -scores, sampled_ids))
    candidates: Dict[int, list[tuple[int, float]]] = defaultdict(list)
    per_wsi: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for position in order:
        cluster_id = int(sampled_ids[position])
        if cluster_id < 0 or cluster_id >= num_clusters:
            raise ValueError(f"Representative cluster ID is out of range: {cluster_id}")
        if len(candidates[cluster_id]) >= candidate_limit:
            continue
        global_row = int(sample_indices[position])
        shard_idx = int(np.searchsorted(row_ends, global_row, side="right"))
        if shard_idx >= len(row_ends):
            raise ValueError(f"Representative row is outside the shard inventory: {global_row}")
        if per_wsi[cluster_id][shard_idx] >= max_tiles_per_wsi:
            continue
        per_wsi[cluster_id][shard_idx] += 1
        candidates[cluster_id].append((global_row, float(scores[position])))

    chosen: list[tuple[int, int, float, int]] = []
    for cluster_id in range(num_clusters):
        for rank, (global_row, score) in enumerate(
            candidates.get(cluster_id, [])[:tiles_per_cluster],
            start=1,
        ):
            chosen.append((cluster_id, global_row, score, rank))
    return chosen


def validate_fine_to_coarse(
    fine_assignments: np.ndarray,
    coarse_assignments: np.ndarray,
    mapping: np.ndarray,
    chunk_size: int,
    mag: int,
) -> None:
    if fine_assignments.shape != coarse_assignments.shape:
        raise ValueError(f"{mag}x fine/coarse assignment shapes differ")
    for start in range(0, fine_assignments.shape[0], chunk_size):
        end = min(fine_assignments.shape[0], start + chunk_size)
        expected = mapping[np.asarray(fine_assignments[start:end], dtype=np.int64)]
        actual = np.asarray(coarse_assignments[start:end], dtype=np.int32)
        if not np.array_equal(expected, actual):
            raise ValueError(f"{mag}x fine->coarse mapping disagrees with tile assignments at {start}:{end}")


def accumulate_projected_centroids(
    config: Mapping[str, Any],
    mag: int,
    shards: list[Dict[str, Any]],
    projected_map: Mapping[str, Path],
    fine_assignments: np.ndarray,
    num_fine: int,
    embedding_dim: int,
    prompt_checkpoint_sha256: str,
    allow_smoke_fixture: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros((num_fine, embedding_dim), dtype=np.float64)
    counts = np.zeros(num_fine, dtype=np.int64)
    chunk_size = int(config["runtime"]["feature_chunk_size"])
    for shard_index, shard in enumerate(shards, start=1):
        path = projected_map[str(shard["slide_id"])]
        raw_path = Path(shard["path"])
        with h5py.File(raw_path, "r") as raw_handle, h5py.File(path, "r") as handle:
            source_n_tiles = int(shard.get("source_n_tiles", shard["n_tiles"]))
            expected_coord_shape = (source_n_tiles, 2)
            for source_handle, label in ((raw_handle, "raw"), (handle, "projected")):
                if "coords" not in source_handle:
                    raise ValueError(f"{source_handle.filename}: {label} H5 is missing coords")
                if source_handle["coords"].shape != expected_coord_shape:
                    raise ValueError(
                        f"{source_handle.filename}: {label} coords shape "
                        f"{source_handle['coords'].shape} does not match {expected_coord_shape}"
                    )
            features = handle["features"]
            validate_text_aligned_features(
                features,
                path,
                source_n_tiles,
                embedding_dim,
                expected_checkpoint_sha256=prompt_checkpoint_sha256,
                allow_smoke_fixture=allow_smoke_fixture,
            )
            global_start = int(shard["row_start"])
            source_rows = selected_source_rows(shard)
            for local_start in range(0, int(shard["n_tiles"]), chunk_size):
                local_end = min(int(shard["n_tiles"]), local_start + chunk_size)
                embedding_rows = source_rows[local_start:local_end]
                raw_coords = np.asarray(
                    raw_handle["coords"][embedding_rows],
                    dtype=np.int64,
                )
                projected_coords = np.asarray(
                    handle["coords"][embedding_rows],
                    dtype=np.int64,
                )
                if not np.array_equal(raw_coords, projected_coords):
                    row_description = (
                        f"rows {local_start}:{local_end}"
                        if np.array_equal(
                            embedding_rows,
                            np.arange(local_start, local_end, dtype=np.int64),
                        )
                        else f"selected source rows {embedding_rows[:5].tolist()}"
                    )
                    raise ValueError(
                        f"{mag}x {shard['slide_id']}: raw/projected coords differ "
                        f"at {row_description}"
                    )
                chunk = np.asarray(features[embedding_rows], dtype=np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError(f"{path}: projected features contain non-finite values")
                norms = np.linalg.norm(chunk, axis=1)
                normalized = np.isclose(norms, 1.0, atol=1e-3)
                if not normalized.all():
                    bad_local_row = int(np.flatnonzero(~normalized)[0])
                    raise ValueError(
                        f"{path}: projected feature at row {int(embedding_rows[bad_local_row])} "
                        f"is not L2 normalized (norm={float(norms[bad_local_row]):.8g}, "
                        f"chunk_min={float(norms.min()):.8g}, "
                        f"chunk_max={float(norms.max()):.8g})"
                    )
                ids = np.asarray(
                    fine_assignments[global_start + local_start : global_start + local_end],
                    dtype=np.int64,
                )
                np.add.at(sums, ids, chunk.astype(np.float64, copy=False))
                counts += np.bincount(ids, minlength=num_fine)
        print(f"{STAGE} {mag}x: accumulated {shard_index}/{len(shards)} {shard['slide_id']}")
    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"{mag}x has empty fine clusters: {empty[:20]}")
    centroids = normalize_rows(sums.astype(np.float32)).astype(np.float32, copy=False)
    return centroids, counts, sums


def sampled_representatives(
    config: Mapping[str, Any],
    mag: int,
    shards: list[Dict[str, Any]],
    projected_map: Mapping[str, Path],
    fine_assignments: np.ndarray,
    fine_to_coarse: np.ndarray,
    raw_centroids: np.ndarray,
) -> list[Dict[str, Any]]:
    total = int(fine_assignments.shape[0])
    audit = config["audit"]
    tiles_per_cluster = int(audit["tiles_per_cluster"])
    candidate_limit = max(
        tiles_per_cluster,
        tiles_per_cluster * int(audit["representative_candidate_factor"]),
    )
    fit_indices = np.load(clustering_path(config, mag, "fit_indices.npy"))
    extra = bounded_sample_indices(total, int(audit["representative_sample_size"]), int(audit["seed"]) + mag)
    sample_indices = np.unique(np.concatenate([fit_indices.astype(np.int64, copy=False), extra]))

    raw_matrix = np.load(raw_matrix_path(config, mag), mmap_mode="r")
    clustering_centroids = np.asarray(raw_centroids, dtype=np.float32)
    sampled_ids = np.asarray(fine_assignments[sample_indices], dtype=np.int32)
    scores = np.empty(sample_indices.shape[0], dtype=np.float32)
    chunk_size = int(config["runtime"]["feature_chunk_size"])
    for start in range(0, sample_indices.shape[0], chunk_size):
        end = min(sample_indices.shape[0], start + chunk_size)
        features = np.asarray(raw_matrix[sample_indices[start:end]], dtype=np.float32)
        ids = sampled_ids[start:end].astype(np.int64, copy=False)
        scores[start:end] = negative_squared_l2_scores(features, clustering_centroids, ids)

    row_ends = np.asarray([int(shard["row_start"]) + int(shard["n_tiles"]) for shard in shards], dtype=np.int64)
    chosen = select_diverse_candidate_rows(
        sample_indices=sample_indices,
        sampled_ids=sampled_ids,
        scores=scores,
        row_ends=row_ends,
        num_clusters=raw_centroids.shape[0],
        tiles_per_cluster=tiles_per_cluster,
        max_tiles_per_wsi=int(audit["max_tiles_per_wsi"]),
        candidate_limit=candidate_limit,
    )

    metadata_map = metadata_by_slide(config["metadata_path"])
    wsi_manifest = read_wsi_manifest(config["wsi"].get("manifest"))
    chosen_by_shard: Dict[int, list[tuple[int, int, float, int]]] = defaultdict(list)
    for cluster_id, global_row, score, rank in chosen:
        shard_idx = int(np.searchsorted(row_ends, global_row, side="right"))
        chosen_by_shard[shard_idx].append((cluster_id, global_row, score, rank))

    records: list[Dict[str, Any]] = []
    for shard_idx, selections in sorted(chosen_by_shard.items()):
        shard = shards[shard_idx]
        metadata = metadata_map[slide_submitter_id(shard)]
        raw_path = Path(shard["path"])
        projected_path = projected_map[str(shard["slide_id"])]
        wsi_path = resolve_wsi_path(metadata, config, wsi_manifest)
        source_rows = selected_source_rows(shard)
        with h5py.File(raw_path, "r") as handle:
            for cluster_id, global_row, score, rank in selections:
                local_row = global_row - int(shard["row_start"])
                embedding_row = int(source_rows[local_row])
                coord = np.asarray(handle["coords"][embedding_row], dtype=np.int64)
                records.append(
                    {
                        "cluster_key": cluster_key(mag, cluster_id),
                        "magnification": mag,
                        "fine_cluster_id": cluster_id,
                        "coarse_cluster_id": int(fine_to_coarse[cluster_id]),
                        "representative_rank": rank,
                        "representative_similarity": score,
                        "representative_score": score,
                        "representative_distance": -score,
                        "representative_metric": "squared_l2",
                        "packed_row_index": global_row,
                        "embedding_row_index": embedding_row,
                        "wsi_id": Path(str(metadata["file_name"])).stem,
                        "slide_id": slide_submitter_id(shard),
                        "patient_id": str(metadata["patient_id"]),
                        "tcga_project": str(metadata["project_id"]),
                        "primary_site": str(metadata["primary_site"]),
                        "x": int(coord[0]),
                        "y": int(coord[1]),
                        "patch_size_level0": int(shard["patch_size_level0"]),
                        "raw_embedding_h5_path": str(raw_path.resolve()),
                        "projected_embedding_h5_path": str(projected_path.resolve()),
                        "wsi_path": wsi_path,
                    }
                )
    records.sort(key=lambda item: (item["magnification"], item["fine_cluster_id"], item["representative_rank"]))
    return records


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    fine_path = root / "fine_cluster_centroids.npz"
    coarse_path = root / "coarse_cluster_centroids.npz"
    representatives_path = root / "representative_tiles.parquet"
    should_run, _ = begin_stage(
        config,
        STAGE,
        [fine_path, coarse_path, representatives_path],
        overwrite=overwrite,
    )
    if not should_run:
        return
    try:
        allow_smoke_fixture = allow_smoke_fixture_contract(config)
        prompt_features, _ = load_prompt_h5(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        prompt_checkpoint_sha256 = read_text_aligned_checkpoint_sha256(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        embedding_dim = int(prompt_features.shape[1])
        fine_arrays: Dict[str, np.ndarray] = {
            "checkpoint_sha256": np.asarray(prompt_checkpoint_sha256)
        }
        coarse_arrays: Dict[str, np.ndarray] = {
            "checkpoint_sha256": np.asarray(prompt_checkpoint_sha256)
        }
        representative_records: list[Dict[str, Any]] = []
        fine_level = int(config["levels"]["fine"])
        coarse_level = int(config["levels"]["coarse"])
        chunk_size = int(config["runtime"]["feature_chunk_size"])

        for mag in config["magnifications"]:
            shards = load_h5_index(config, mag)
            projected_map = projected_h5_map(config, mag)
            fine_assignments = np.load(
                clustering_path(config, mag, f"level{fine_level}", "assignments.npy"), mmap_mode="r"
            )
            coarse_assignments = np.load(
                clustering_path(config, mag, f"level{coarse_level}", "assignments.npy"), mmap_mode="r"
            )
            raw_centroids = np.load(
                clustering_path(config, mag, f"level{fine_level}", "centroids.npy")
            ).astype(np.float32, copy=False)
            fine_to_coarse = np.load(
                clustering_path(config, mag, f"level{coarse_level}", "fit_assignment.npy")
            ).astype(np.int32, copy=False)
            if fine_to_coarse.shape != (raw_centroids.shape[0],):
                raise ValueError(f"{mag}x coarse fit_assignment must map every fine centroid")
            validate_fine_to_coarse(
                fine_assignments,
                coarse_assignments,
                fine_to_coarse,
                chunk_size,
                mag,
            )
            fine_centroids, fine_counts, fine_sums = accumulate_projected_centroids(
                config,
                mag,
                shards,
                projected_map,
                fine_assignments,
                raw_centroids.shape[0],
                embedding_dim,
                prompt_checkpoint_sha256,
                allow_smoke_fixture=allow_smoke_fixture,
            )
            num_coarse = int(fine_to_coarse.max()) + 1
            coarse_sums = np.zeros((num_coarse, embedding_dim), dtype=np.float64)
            coarse_counts = np.zeros(num_coarse, dtype=np.int64)
            for fine_id in range(fine_centroids.shape[0]):
                coarse_id = int(fine_to_coarse[fine_id])
                coarse_sums[coarse_id] += fine_sums[fine_id]
                coarse_counts[coarse_id] += fine_counts[fine_id]
            if np.any(coarse_counts == 0):
                raise ValueError(f"{mag}x has empty coarse clusters")
            coarse_centroids = normalize_rows(coarse_sums.astype(np.float32)).astype(np.float32, copy=False)

            key = f"mag_{mag}"
            fine_arrays[key] = fine_centroids
            fine_arrays[f"{key}_counts"] = fine_counts
            coarse_arrays[key] = coarse_centroids
            coarse_arrays[f"{key}_counts"] = coarse_counts
            representative_records.extend(
                sampled_representatives(
                    config,
                    mag,
                    shards,
                    projected_map,
                    fine_assignments,
                    fine_to_coarse,
                    raw_centroids,
                )
            )

        atomic_npz(fine_path, fine_arrays)
        atomic_npz(coarse_path, coarse_arrays)
        frame = pd.DataFrame.from_records(representative_records)
        atomic_parquet(frame, representatives_path, int(config["runtime"]["parquet_row_group_size"]))
        complete_stage(
            config,
            STAGE,
            {
                "representative_tiles": len(representative_records),
                "embedding_dim": embedding_dim,
                "checkpoint_sha256": prompt_checkpoint_sha256,
            },
        )
    except BaseException as error:
        fail_stage(config, STAGE, error)
        raise


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Build projected fine/coarse centroids and representative tile manifests.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
