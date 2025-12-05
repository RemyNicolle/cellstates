"""
Prototype multi-sweep greedy loop using JAX sweep primitive.
"""

from __future__ import annotations

import numpy as np

from .jax_mcmc import greedy_partition_sweep_jax


def run_greedy_partition_jax(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray,
    sweeps: int = 3,
    device: str | None = None,
    enable_x64: bool = False,
    dtype=None,
    cluster_chunk: int = 128,
    candidate_topk: int | None = None,
    seed: int = 0,
):
    """
    Run multiple greedy sweeps with shuffling between sweeps.

    Parameters are forwarded to greedy_partition_sweep_jax. Shuffles cell
    order between sweeps using a NumPy RNG on host (cheap) for now.
    """
    rng = np.random.default_rng(seed)
    clusters_curr = clusters.copy()
    total_moves = 0
    total_delta = 0.0
    for s in range(sweeps):
        order = rng.permutation(data.shape[1])
        data_perm = data[:, order]
        clusters_perm = clusters_curr[order]
        new_clusters_perm, moves, delta = greedy_partition_sweep_jax(
            data_perm,
            clusters_perm,
            lam,
            device=device,
            enable_x64=enable_x64,
            dtype=dtype,
            cluster_chunk=cluster_chunk,
            candidate_topk=candidate_topk,
        )
        # unpermute back
        inv = np.argsort(order)
        clusters_curr = new_clusters_perm[inv]
        total_moves += moves
        total_delta += delta
        if moves == 0:
            break
    return clusters_curr, total_moves, total_delta
