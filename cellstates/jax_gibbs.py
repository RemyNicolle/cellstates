"""
Minimal JAX-backed Gibbs sampler prototype.

Notes:
- Operates on counts shaped (G, N) with integer clusters.
- Uses float32 by default.
- Intended for small/medium synthetic benchmarks; not a replacement for the
  Cython MCMC.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    HAS_JAX = True
except Exception:  # pragma: no cover
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
    return None


def _ll_cluster(counts, lam, B, lam_sum):
    lam_col = lam[:, None]
    n_sum = jnp.sum(counts, axis=0)
    return B - gammaln(n_sum + lam_sum) + jnp.sum(gammaln(counts + lam_col), axis=0)


def run_gibbs_partition_jax(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray,
    sweeps: int = 3,
    device: str | None = None,
    enable_x64: bool = False,
    dtype=None,
    seed: int = 0,
):
    """
    Simple Gibbs sampler over cluster labels.

    Parameters
    ----------
    data : (G, N) array
        Gene counts per cell.
    clusters : (N,) array
        Initial cluster labels.
    lam : (G,) array
        Dirichlet pseudocounts.
    sweeps : int
        Number of full passes over cells.
    device : {'cpu','gpu','tpu','mps'}, optional
    enable_x64 : bool
    dtype : jnp.dtype, optional
    seed : int

    Returns
    -------
    labels : np.ndarray
    moves : int
    total_delta : float
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

    G, N = data.shape
    clusters_np = np.asarray(clusters, dtype=np.int32)
    K = int(clusters_np.max()) + 1

    data_j = jnp.asarray(data, dtype=jnp.int32)
    lam_vec = jnp.asarray(lam, dtype=dtype).reshape(-1)
    lam_sum = jnp.sum(lam_vec)
    B = gammaln(lam_sum) - jnp.sum(gammaln(lam_vec))

    target = _select_device(device)
    if target:
        data_j = jax.device_put(data_j, target)
        lam_vec = jax.device_put(lam_vec, target)

    # cluster counts on host for simplicity
    counts_np = np.zeros((G, K), dtype=np.int64)
    for k in range(K):
        mask = clusters_np == k
        if mask.any():
            counts_np[:, k] = np.sum(data[:, mask], axis=1)

    moves = 0
    total_delta = 0.0
    rng = np.random.default_rng(seed)

    def ll_for_cluster(counts_col):
        c_j = jnp.asarray(counts_col, dtype=dtype)
        return float(np.asarray(_ll_cluster(c_j[:, None], lam_vec, B, lam_sum))[0])

    for _ in range(sweeps):
        for i in range(N):
            c_old = clusters_np[i]
            cell_vec = data[:, i]
            if counts_np[:, c_old].sum() > 0:
                counts_np[:, c_old] -= cell_vec

            ll_current = ll_for_cluster(counts_np[:, c_old]) if counts_np[:, c_old].sum() > 0 else 0.0
            ll_base = [ll_for_cluster(counts_np[:, k]) for k in range(K)]
            ll_new = []
            for k in range(K):
                ll_new.append(ll_for_cluster(counts_np[:, k] + cell_vec))
            ll_new = np.array(ll_new)
            ll_base = np.array(ll_base)

            logits = ll_new - ll_base
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            c_new = rng.choice(K, p=probs)

            counts_np[:, c_new] += cell_vec
            clusters_np[i] = c_new

            if c_new != c_old:
                moves += 1
                total_delta += float(ll_new[c_new] - ll_current)

    return clusters_np, moves, total_delta
