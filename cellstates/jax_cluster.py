"""
Experimental JAX implementations for cluster hierarchy construction.

This mirrors the greedy merge logic of Cluster.get_cluster_hierarchy but runs
the merge-distance calculations in JAX. It is not yet optimized for very large
numbers of clusters (the pairwise search is O(K^2)), so use it on already
aggregated cluster counts with a modest K.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    HAS_JAX = True
except Exception:  # pragma: no cover - import guard
    jax = None
    jnp = None
    gammaln = None
    HAS_JAX = False


def _select_device(device: str | None):
    if device is None:
        return None
    devices = [d for d in jax.devices() if d.platform == ("gpu" if device == "mps" else device)]
    if devices:
        return devices[0]
    all_devices = jax.devices()
    if all_devices:
        return all_devices[0]
    raise ValueError(f"No JAX devices found; requested platform '{device}'.")


def _cluster_ll(counts, lam, B, lam_sum):
    n_sum = jnp.sum(counts, axis=0)
    return B - gammaln(n_sum + lam_sum) + jnp.sum(gammaln(counts + lam[:, None]), axis=0)


def get_cluster_hierarchy_jax_from_counts(
    counts: np.ndarray,
    lam: np.ndarray,
    LL_threshold: float = 0.0,
    device: str | None = None,
    enable_x64: bool = False,
    dtype=None,
    pair_chunk_size: int | None = None,
):
    """
    Greedy cluster hierarchy using JAX for merge distances.

    Parameters
    ----------
    counts : array (G, K)
        Aggregated counts per gene (rows) and cluster (columns).
    lam : array (G,)
        Dirichlet pseudocounts.
    LL_threshold : float, default 0.0
        Stop merging when best delta_LL <= threshold.
    device : {'cpu','gpu','tpu','mps'}, optional
        Device preference for JAX arrays.
    enable_x64 : bool, default False
        If True, enable float64; otherwise float32.
    dtype : jnp.dtype, optional
        Override dtype; defaults to float64 if enable_x64 else float32.
    pair_chunk_size : int, optional
        If set, compute pairwise deltas in host-side chunks of this many pairs
        to reduce peak memory versus forming the full matrix. Smaller chunks
        reduce memory but increase host/device transfers.

    Returns
    -------
    merge_hierarchy : list[tuple[int, int]]
        Sequence of (cluster_new, cluster_old) merges.
    delta_LL_history : list[float]
        Corresponding delta log-likelihoods.
    """
    if not HAS_JAX:
        raise ImportError("JAX is not available; install jax to use this function.")

    if enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    if dtype is None:
        dtype = jnp.float64 if enable_x64 else jnp.float32

    counts_j = jnp.asarray(counts, dtype=dtype)
    lam_j = jnp.asarray(lam, dtype=dtype)

    target_device = _select_device(device)
    if target_device:
        counts_j = jax.device_put(counts_j, target_device)
        lam_j = jax.device_put(lam_j, target_device)

    lam_sum = jnp.sum(lam_j)
    B = gammaln(lam_sum) - jnp.sum(gammaln(lam_j))

    # active mask for clusters with nonzero size
    cluster_sizes = jnp.sum(counts_j, axis=0)
    active = np.array(np.array(cluster_sizes > 0, dtype=bool))
    merge_hierarchy = []
    delta_LL_history = []

    # jit a delta calculator for batches of pairs
    def _delta_for_pairs(counts_arr, pair_idx):
        c1 = counts_arr[:, pair_idx[:, 0]]
        c2 = counts_arr[:, pair_idx[:, 1]]
        merged = c1 + c2
        ll1 = _cluster_ll(c1, lam_j, B, lam_sum)
        ll2 = _cluster_ll(c2, lam_j, B, lam_sum)
        ll_merge = _cluster_ll(merged, lam_j, B, lam_sum)
        return ll_merge - (ll1 + ll2)

    delta_fn = jax.jit(_delta_for_pairs, donate_argnums=(0,))

    while active.sum() > 1:
        # materialize list of active indices on host
        idx = np.nonzero(active)[0]
        K = len(idx)
        if K < 2:
            break

        # build all pairs (upper triangle)
        pairs = np.asarray(np.triu_indices(K, k=1)).T
        if pairs.size == 0:
            break
        # map back to original cluster indices
        full_pair_idx = np.stack([idx[pairs[:, 0]], idx[pairs[:, 1]]], axis=1).astype(np.int32)

        # optional chunking to avoid large allocations
        best_delta = -np.inf
        best_pair = None
        if pair_chunk_size is None:
            pair_idx_dev = jnp.asarray(full_pair_idx, dtype=jnp.int32)
            deltas = delta_fn(counts_j, pair_idx_dev)
            best_pos = int(jnp.argmax(deltas))
            best_delta = float(deltas[best_pos])
            best_pair = full_pair_idx[best_pos]
        else:
            for start in range(0, full_pair_idx.shape[0], pair_chunk_size):
                chunk = full_pair_idx[start : start + pair_chunk_size]
                pair_idx_dev = jnp.asarray(chunk, dtype=jnp.int32)
                deltas = delta_fn(counts_j, pair_idx_dev)
                local_best = int(jnp.argmax(deltas))
                local_delta = float(deltas[local_best])
                if local_delta > best_delta:
                    best_delta = local_delta
                    best_pair = chunk[local_best]

        i_new, i_old = map(int, best_pair)

        if best_delta <= LL_threshold:
            break

        merge_hierarchy.append((i_new, i_old))
        delta_LL_history.append(best_delta)

        # update counts for merged cluster and deactivate old cluster
        counts_j = counts_j.at[:, i_new].set(counts_j[:, i_new] + counts_j[:, i_old])
        counts_j = counts_j.at[:, i_old].set(0)
        active[i_old] = False

    return merge_hierarchy, delta_LL_history
