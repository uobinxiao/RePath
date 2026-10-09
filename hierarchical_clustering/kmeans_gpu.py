# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted as a standalone local dependency for this clustering runner.

import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState instance")


def create_clusters_from_cluster_assignment(cluster_assignment: np.ndarray, num_clusters: int):
    order = np.argsort(cluster_assignment)
    sorted_assignment = cluster_assignment[order]
    split_points = np.searchsorted(sorted_assignment, list(range(num_clusters)))
    return np.array(np.split(order, split_points[1:]), dtype=object)


def matmul_transpose(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y.T)


def compute_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    y_squared_norms: torch.Tensor,
    dist: str = "l2",
    x_squared_norm: Optional[torch.Tensor] = None,
):
    if dist == "cos":
        return 2 - 2 * matmul_transpose(F.normalize(x, dim=1), F.normalize(y, dim=1))
    if dist == "l2":
        if x_squared_norm is None:
            x_squared_norm = torch.linalg.vector_norm(x, dim=1) ** 2
        distances = x_squared_norm[:, None] - 2 * matmul_transpose(x, y) + y_squared_norms[None, :]
        return torch.clamp_min(distances, 0.0)
    raise ValueError(f'dist = "{dist}" not supported')


def compute_kmeans_potential(x: torch.Tensor, centroids: torch.Tensor, clusters, dist: str) -> float:
    pot = 0.0
    for cluster_idx, point_indices in enumerate(clusters):
        if len(point_indices) == 0:
            continue
        point_feats = x[point_indices.astype(int)]
        x_squared_norms = torch.linalg.vector_norm(point_feats, dim=1) ** 2
        distances = compute_distance(
            centroids[cluster_idx, None],
            point_feats,
            x_squared_norms,
            dist,
        )
        pot += torch.sum(distances).item()
    return pot


def kmeans_plusplus(
    x: torch.Tensor,
    n_clusters: int,
    x_squared_norms: torch.Tensor,
    dist: str,
    random_state=None,
    n_local_trials=None,
    high_precision=torch.float64,
    verbose: bool = False,
):
    random_state = check_random_state(random_state)
    n_samples, n_features = x.shape
    centers = torch.empty((n_clusters, n_features), dtype=x.dtype, device=x.device)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.randint(n_samples)
    centers[0] = x[center_id]
    closest_dist_sq = compute_distance(x[center_id, None], x, x_squared_norms, dist)[0].type(high_precision)
    current_pot = closest_dist_sq.sum()

    iterates = range(1, n_clusters)
    if verbose:
        iterates = tqdm(
            iterates,
            desc="Kmeans++ initialization",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )

    for center_idx in iterates:
        rand_vals = torch.tensor(
            random_state.uniform(size=n_local_trials),
            device=current_pot.device,
            dtype=current_pot.dtype,
        ) * current_pot
        candidate_ids = torch.searchsorted(torch.cumsum(closest_dist_sq, dim=0), rand_vals)
        candidate_ids.clamp_(max=closest_dist_sq.shape[0] - 1)

        distance_to_candidates = compute_distance(x[candidate_ids], x, x_squared_norms, dist).type(high_precision)
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        centers[center_idx] = x[candidate_ids[best_candidate]]

    return centers


def assign_clusters(
    centroids: torch.Tensor,
    x: torch.Tensor,
    dist: str,
    chunk_size: int = -1,
    verbose: bool = False,
) -> torch.Tensor:
    n_samples, _ = x.shape
    x_squared_norms = torch.linalg.vector_norm(x, dim=1) ** 2
    centroid_squared_norms = torch.linalg.vector_norm(centroids, dim=1) ** 2

    if chunk_size < 0:
        distances = compute_distance(centroids, x, x_squared_norms, dist, centroid_squared_norms)
        return torch.argmin(distances, dim=0)

    cluster_ids = []
    n_iters = (n_samples + chunk_size - 1) // chunk_size
    iterates = range(n_iters)
    if verbose:
        iterates = tqdm(
            iterates,
            desc="Assigning data points to centroids",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    for chunk_idx in iterates:
        begin = chunk_idx * chunk_size
        end = min(n_samples, (chunk_idx + 1) * chunk_size)
        distances = compute_distance(
            centroids,
            x[begin:end],
            x_squared_norms[begin:end],
            dist,
            centroid_squared_norms,
        )
        cluster_ids.append(torch.argmin(distances, dim=0))
        del distances
    return torch.cat(cluster_ids)


def compute_centroids(
    centroids: torch.Tensor,
    cluster_assignment: np.ndarray,
    n_clusters: int,
    x: torch.Tensor,
    high_precision=torch.float32,
) -> torch.Tensor:
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    new_centroids = torch.zeros_like(centroids)
    for cluster_idx in range(n_clusters):
        if len(clusters[cluster_idx]) > 0:
            new_centroids[cluster_idx] = torch.mean(
                x[clusters[cluster_idx].astype(int)].type(high_precision),
                dim=0,
            )
        else:
            new_centroids[cluster_idx] = centroids[cluster_idx]
    return new_centroids


def _kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_iters: int,
    chunk_size: int = -1,
    init_method: str = "kmeans++",
    dist: str = "l2",
    high_precision=torch.float32,
    random_state=None,
    verbose: bool = False,
):
    random_state = check_random_state(random_state)
    x_squared_norms = torch.linalg.vector_norm(x, dim=1) ** 2

    if init_method == "kmeans++":
        centroids = kmeans_plusplus(
            x,
            n_clusters,
            x_squared_norms,
            dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        indices = np.sort(random_state.choice(range(len(x)), n_clusters, replace=False))
        centroids = torch.tensor(x[indices], device=x.device, dtype=x.dtype)

    cluster_assignment = assign_clusters(centroids, x, dist, chunk_size).cpu().numpy()
    for _ in range(n_iters):
        centroids = compute_centroids(centroids, cluster_assignment, n_clusters, x, high_precision)
        cluster_assignment = assign_clusters(centroids, x, dist, chunk_size).cpu().numpy()

    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    pot = compute_kmeans_potential(x, centroids, clusters, dist)
    return centroids, clusters, cluster_assignment, pot


def kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_iters: int,
    chunk_size: int = -1,
    num_init: int = 1,
    init_method: str = "kmeans++",
    dist: str = "l2",
    high_precision=torch.float32,
    random_state=None,
    verbose: bool = False,
):
    random_state = check_random_state(random_state)
    n_clusters = min(int(n_clusters), int(x.shape[0]))
    best_centroids, best_clusters, best_assignment, best_pot = None, None, None, np.inf

    for _ in range(num_init):
        centroids, clusters, assignment, pot = _kmeans(
            x,
            n_clusters,
            n_iters,
            chunk_size=chunk_size,
            init_method=init_method,
            dist=dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
        if pot < best_pot:
            best_centroids, best_clusters, best_assignment, best_pot = centroids, clusters, assignment, pot
    return best_centroids, best_clusters, best_assignment, best_pot
