"""
Prototype JAX-backed greedy sweep for cluster reassignment.

This is not a full MCMC replica; it performs one sweep over cells, proposing
moves to any cluster and accepting only moves that improve total likelihood.
Chunking over clusters keeps memory bounded on TPU/GPU at the cost of more
host/device roundtrips.

Assumptions:
- Operates on an existing partition; no prior optimization.
- Uses float32 by default for speed/memory; optional float64.
- Suitable for modest numbers of clusters; for very large K, increase
  `cluster_chunk` to keep memory down.
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

try:  # pragma: no cover - import guard
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - import guard
    torch = None
    HAS_TORCH = False


def _select_device(device: str | None):
    if device is None:
        return None
    devices = [d for d in jax.devices() if d.platform == ("gpu" if device == "mps" else device)]
    if devices:
        return devices[0]
    # Fallback: let JAX pick the default device if requested platform is absent.
    return None


def _ll_cluster(counts, lam, B, lam_sum):
    lam_col = lam[:, None]
    n_sum = jnp.sum(counts, axis=0)
    return B - gammaln(n_sum + lam_sum) + jnp.sum(gammaln(counts + lam_col), axis=0)


def _ll_remove(counts_old, cell_counts, lam, B, lam_sum, size_old):
    """LL of old cluster after removing the cell."""
    lam_col = lam[:, None]
    n_sum = jnp.sum(counts_old, axis=0) - jnp.sum(cell_counts)
    ll = B - gammaln(n_sum + lam_sum) + jnp.sum(gammaln(counts_old - cell_counts + lam_col), axis=0)
    return jnp.where(size_old > 1, ll, 0.0)


def _ll_add(counts_new, cell_counts, lam, B, lam_sum):
    lam_col = lam[:, None]
    n_sum = jnp.sum(counts_new, axis=0) + jnp.sum(cell_counts)
    return B - gammaln(n_sum + lam_sum) + jnp.sum(gammaln(counts_new + cell_counts + lam_col), axis=0)


def greedy_partition_sweep_jax(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray,
    device: str | None = None,
    enable_x64: bool = False,
    dtype=None,
    cluster_chunk: int = 128,
    candidate_topk: int | None = None,
):
    """
    One greedy sweep of cell reassignments in JAX.

    Parameters
    ----------
    data : array (G, N)
        Gene counts per cell.
    clusters : array (N,)
        Initial cluster labels (int).
    lam : array (G,)
        Dirichlet pseudocounts.
    device : {'cpu','gpu','tpu','mps'}, optional
        Device preference.
    enable_x64 : bool, default False
        Enable float64 precision (default float32).
    dtype : jnp.dtype, optional
        Override dtype; defaults to float64 if enable_x64 else float32.
    cluster_chunk : int, default 128
        Number of candidate clusters evaluated per chunk (trade memory vs speed).
    candidate_topk : int, optional
        If set, restrict proposals to this many best clusters per cell based on
        a cached delta estimate. Reduces O(N*K) work at the cost of an
        approximate search (still accepts only improving moves).

    Returns
    -------
    new_clusters : np.ndarray
        Updated cluster assignments.
    moves_made : int
        Number of cells moved.
    total_delta : float
        Sum of likelihood deltas over accepted moves.
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

    data_j = jnp.asarray(data, dtype=jnp.int32)
    lam_vec = jnp.asarray(lam, dtype=dtype).reshape(-1)
    lam_sum = jnp.sum(lam_vec)
    B = gammaln(lam_sum) - jnp.sum(gammaln(lam_vec))

    clusters_np = np.array(clusters, dtype=np.int32, copy=True)
    K = int(clusters_np.max()) + 1

    target_device = _select_device(device)
    if target_device:
        data_j = jax.device_put(data_j, target_device)
        lam_vec = jax.device_put(lam_vec, target_device)

    # Build cluster aggregates on device
    def _cluster_counts_for_gene(gene_row):
        return jnp.bincount(clusters_np, weights=gene_row, length=K)

    counts_j = jax.vmap(_cluster_counts_for_gene)(data_j)  # (G, K)
    sizes_np = np.bincount(clusters_np, minlength=K).astype(np.int32)
    sizes_j = jax.device_put(sizes_np, target_device) if target_device else jnp.asarray(sizes_np)

    ll_j = _ll_cluster(counts_j, lam_vec, B, lam_sum)  # (K,)

    G = counts_j.shape[0]

    @jax.jit
    def _best_delta_for_chunk(counts_arr, ll_arr, sizes_arr, cell_vec, c_old, chunk_idx):
        start = chunk_idx * cluster_chunk
        end = jnp.minimum(start + cluster_chunk, counts_arr.shape[1])
        idx = jnp.arange(start, end, dtype=jnp.int32)
        counts_chunk = jnp.take(counts_arr, idx, axis=1)
        ll_chunk = jnp.take(ll_arr, idx, axis=0)
        size_chunk = jnp.take(sizes_arr, idx, axis=0)

        old_slice = jax.lax.dynamic_slice(counts_arr, (0, c_old), (G, 1))
        ll_old_after = _ll_remove(old_slice, cell_vec[:, None], lam_vec, B, lam_sum, sizes_arr[c_old])
        ll_new = _ll_add(counts_chunk, cell_vec[:, None], lam_vec, B, lam_sum)

        # delta = new + old_after - old_before - cand_before
        delta = ll_new + ll_old_after - ll_arr[c_old] - ll_chunk
        delta = jnp.where(idx == c_old, -jnp.inf, delta)

        best_pos = jnp.argmax(delta)
        return delta[best_pos], idx[best_pos]

    moves = 0
    total_delta = 0.0

    for m in range(data_j.shape[1]):
        cell_vec = data_j[:, m]
        c_old = int(clusters_np[m])
        best_delta = -np.inf
        best_cluster = c_old

        # choose candidate clusters
        if candidate_topk is not None and candidate_topk > 0:
            candidates = np.arange(min(candidate_topk, K), dtype=np.int32)
            if c_old not in candidates:
                candidates = np.unique(np.concatenate([[c_old], candidates]))
        else:
            candidates = np.arange(K, dtype=np.int32)

        cand_chunks = [candidates[i : i + cluster_chunk] for i in range(0, len(candidates), cluster_chunk)]

        for chunk in cand_chunks:
            chunk_pad = jnp.asarray(chunk, dtype=jnp.int32)
            # reuse best-delta logic on this chunk only
            def _delta_for_indices(counts_arr, ll_arr, sizes_arr, cell_vec, idx):
                counts_chunk = jnp.take(counts_arr, idx, axis=1)
                ll_chunk = jnp.take(ll_arr, idx, axis=0)
                size_chunk = jnp.take(sizes_arr, idx, axis=0)
                old_slice = jax.lax.dynamic_slice(counts_arr, (0, c_old), (G, 1))
                ll_old_after = _ll_remove(old_slice, cell_vec[:, None], lam_vec, B, lam_sum, sizes_arr[c_old])
                ll_new = _ll_add(counts_chunk, cell_vec[:, None], lam_vec, B, lam_sum)
                delta = ll_new + ll_old_after - ll_arr[c_old] - ll_chunk
                delta = jnp.where(idx == c_old, -jnp.inf, delta)
                pos = jnp.argmax(delta)
                return delta[pos], idx[pos]

            delta_val, cand = jax.jit(_delta_for_indices, donate_argnums=(0,))(counts_j, ll_j, sizes_j, cell_vec, chunk_pad)
            delta_host = float(delta_val)
            cand_host = int(cand)
            if delta_host > best_delta:
                best_delta = delta_host
                best_cluster = cand_host

        if best_cluster != c_old and best_delta > 0:
            moves += 1
            total_delta += best_delta

            # update aggregates on host via device arrays
            counts_j = counts_j.at[:, c_old].add(-cell_vec)
            counts_j = counts_j.at[:, best_cluster].add(cell_vec)
            sizes_np[c_old] -= 1
            sizes_np[best_cluster] += 1
            sizes_j = jax.device_put(sizes_np, target_device) if target_device else jnp.asarray(sizes_np)

            # recompute LL for affected clusters
            slice_old = jax.lax.dynamic_slice(counts_j, (0, c_old), (G, 1))
            slice_new = jax.lax.dynamic_slice(counts_j, (0, best_cluster), (G, 1))
            ll_j = ll_j.at[c_old].set(_ll_cluster(slice_old, lam_vec, B, lam_sum)[0] if sizes_np[c_old] > 0 else 0.0)
            ll_j = ll_j.at[best_cluster].set(_ll_cluster(slice_new, lam_vec, B, lam_sum)[0])

            clusters_np[m] = best_cluster

    return clusters_np, moves, total_delta


def stochastic_partition_jax(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray | float | None = None,
    sweeps: int = 3,
    proposals_per_cell: int = 16,
    device: str | None = None,
    enable_x64: bool = False,
    dtype=None,
    seed: int = 0,
    lam_alpha: float = 0.001,
):
    """
    Alternative stochastic JAX partitioner.

    Randomly samples candidate clusters per cell each sweep (always includes
    the current cluster) and accepts only improving moves. Keeps working state
    on device; suitable for TPU with small proposal sets. Not a full MCMC
    replica, but a lower-memory alternative to the greedy sweep.
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

    data_j = jnp.asarray(data, dtype=jnp.int32)
    if lam is None:
        lam = float(lam_alpha)
    if np.isscalar(lam):
        lam_vec = jnp.full(data.shape[0], float(lam), dtype=dtype)
    else:
        lam_vec = jnp.asarray(lam, dtype=dtype).reshape(-1)
    lam_sum = jnp.sum(lam_vec)
    B = gammaln(lam_sum) - jnp.sum(gammaln(lam_vec))

    clusters_np = np.array(clusters, dtype=np.int32, copy=True)
    K = int(clusters_np.max()) + 1

    target_device = _select_device(device)
    if target_device:
        data_j = jax.device_put(data_j, target_device)
        lam_vec = jax.device_put(lam_vec, target_device)

    def _cluster_counts_for_gene(gene_row):
        return jnp.bincount(clusters_np, weights=gene_row, length=K)

    counts_j = jax.vmap(_cluster_counts_for_gene)(data_j)  # (G, K)
    sizes_np = np.bincount(clusters_np, minlength=K).astype(np.int32)
    sizes_j = jax.device_put(sizes_np, target_device) if target_device else jnp.asarray(sizes_np)
    ll_j = _ll_cluster(counts_j, lam_vec, B, lam_sum)  # (K,)

    G = counts_j.shape[0]

    def _delta_for_indices(counts_arr, ll_arr, sizes_arr, cell_vec, idx, c_old):
        counts_chunk = jnp.take(counts_arr, idx, axis=1)
        ll_chunk = jnp.take(ll_arr, idx, axis=0)
        old_slice = jax.lax.dynamic_slice(counts_arr, (0, c_old), (G, 1))
        ll_old_after = _ll_remove(old_slice, cell_vec[:, None], lam_vec, B, lam_sum, sizes_arr[c_old])
        ll_new = _ll_add(counts_chunk, cell_vec[:, None], lam_vec, B, lam_sum)
        delta = ll_new + ll_old_after - ll_arr[c_old] - ll_chunk
        delta = jnp.where(idx == c_old, -jnp.inf, delta)
        pos = jnp.argmax(delta)
        return delta[pos], idx[pos]

    rng = np.random.default_rng(seed)
    total_moves = 0
    total_delta = 0.0

    for _ in range(sweeps):
        order = rng.permutation(data_j.shape[1])
        for m in order:
            cell_vec = data_j[:, m]
            c_old = int(clusters_np[m])
            # sample a fixed-size proposal set to keep shapes static
            size = proposals_per_cell if proposals_per_cell > 0 else 1
            cand = rng.choice(K, size=size, replace=True).astype(np.int32)
            cand[0] = c_old  # ensure current cluster is included
            cand_dev = jnp.asarray(cand, dtype=jnp.int32)
            delta_val, cand_best = _delta_for_indices(counts_j, ll_j, sizes_j, cell_vec, cand_dev, c_old)
            delta_host = float(delta_val)
            cand_host = int(cand_best)
            if cand_host != c_old and delta_host > 0:
                total_moves += 1
                total_delta += delta_host
                counts_j = counts_j.at[:, c_old].add(-cell_vec)
                counts_j = counts_j.at[:, cand_host].add(cell_vec)
                sizes_np[c_old] -= 1
                sizes_np[cand_host] += 1
                sizes_j = jax.device_put(sizes_np, target_device) if target_device else jnp.asarray(sizes_np)
                slice_old = jax.lax.dynamic_slice(counts_j, (0, c_old), (G, 1))
                slice_new = jax.lax.dynamic_slice(counts_j, (0, cand_host), (G, 1))
                ll_j = ll_j.at[c_old].set(_ll_cluster(slice_old, lam_vec, B, lam_sum)[0] if sizes_np[c_old] > 0 else 0.0)
                ll_j = ll_j.at[cand_host].set(_ll_cluster(slice_new, lam_vec, B, lam_sum)[0])
                clusters_np[m] = cand_host
        if total_moves == 0:
            break

    return clusters_np, total_moves, total_delta


# --------- PyTorch alternatives (GPU/CPU/MPS) ---------

def _select_torch_device(device: str | None):
    if not HAS_TORCH:
        return None
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = device.lower()
    if dev in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(device)


def _torch_ll_cluster(counts, lam_vec, B, lam_sum):
    n_sum = counts.sum()
    return B - torch.lgamma(n_sum + lam_sum) + torch.lgamma(counts + lam_vec).sum()


def stochastic_partition_torch(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray | float | None = None,
    sweeps: int = 3,
    proposals_per_cell: int = 16,
    device: str | None = None,
    dtype=None,
    seed: int = 0,
    lam_alpha: float = 0.001,
):
    """
    PyTorch alternative to stochastic_partition_jax. Uses simple Python loops
    and torch tensors on the selected device (CPU/GPU/MPS).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is not available; install torch to use this function.")

    torch_device = _select_torch_device(device)
    if dtype is None:
        dtype = torch.float32

    data_t = torch.as_tensor(data, dtype=torch.int64, device=torch_device)
    G, N = data_t.shape
    clusters_np = np.asarray(clusters, dtype=np.int32)
    K = int(clusters_np.max()) + 1

    if lam is None:
        lam = float(lam_alpha)
    if np.isscalar(lam):
        lam_vec = torch.full((G,), float(lam), dtype=dtype, device=torch_device)
    else:
        lam_vec = torch.as_tensor(lam, dtype=dtype, device=torch_device).view(-1)
    lam_sum = lam_vec.sum()
    B = torch.lgamma(lam_sum) - torch.lgamma(lam_vec).sum()

    counts = torch.zeros((G, K), dtype=torch.int64, device=torch_device)
    for k in range(K):
        mask = clusters_np == k
        if mask.any():
            counts[:, k] = data_t[:, mask].sum(dim=1)
    sizes = torch.as_tensor(np.bincount(clusters_np, minlength=K), dtype=torch.int64, device=torch_device)

    ll_clusters = torch.zeros(K, dtype=dtype, device=torch_device)
    for k in range(K):
        if sizes[k] > 0:
            ll_clusters[k] = _torch_ll_cluster(counts[:, k].to(dtype), lam_vec, B, lam_sum)

    rng = np.random.default_rng(seed)
    total_moves = 0
    total_delta = 0.0

    for _ in range(sweeps):
        order = rng.permutation(N)
        for m in order:
            c_old = int(clusters_np[m])
            cell_vec = data_t[:, m].to(dtype)

            size = proposals_per_cell if proposals_per_cell > 0 else 1
            cand = rng.choice(K, size=size, replace=True).astype(np.int64)
            cand[0] = c_old

            best_delta = float("-inf")
            best_cand = c_old

            counts_old = counts[:, c_old].to(dtype)
            ll_old_before = ll_clusters[c_old] if sizes[c_old] > 0 else torch.tensor(0.0, device=torch_device, dtype=dtype)
            ll_old_after = _torch_ll_cluster(counts_old - cell_vec, lam_vec, B, lam_sum) if sizes[c_old] > 1 else torch.tensor(0.0, device=torch_device, dtype=dtype)

            for c_new in cand:
                if c_new == c_old:
                    continue
                c_new_int = int(c_new)
                ll_new_before = ll_clusters[c_new_int] if sizes[c_new_int] > 0 else torch.tensor(0.0, device=torch_device, dtype=dtype)
                ll_new_after = _torch_ll_cluster(counts[:, c_new_int].to(dtype) + cell_vec, lam_vec, B, lam_sum)
                delta = float((ll_new_after + ll_old_after - ll_old_before - ll_new_before).cpu().item())
                if delta > best_delta and delta > 0:
                    best_delta = delta
                    best_cand = c_new_int

            if best_cand != c_old:
                total_moves += 1
                total_delta += best_delta
                counts[:, c_old] -= cell_vec.to(torch.int64)
                counts[:, best_cand] += cell_vec.to(torch.int64)
                sizes[c_old] -= 1
                sizes[best_cand] += 1
                ll_clusters[c_old] = _torch_ll_cluster(counts[:, c_old].to(dtype), lam_vec, B, lam_sum) if sizes[c_old] > 0 else torch.tensor(0.0, device=torch_device, dtype=dtype)
                ll_clusters[best_cand] = _torch_ll_cluster(counts[:, best_cand].to(dtype), lam_vec, B, lam_sum)
                clusters_np[m] = best_cand

    return clusters_np, total_moves, total_delta


def run_gibbs_partition_torch(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray | float | None = None,
    sweeps: int = 3,
    device: str | None = None,
    dtype=None,
    seed: int = 0,
    lam_alpha: float = 0.001,
):
    """
    PyTorch alternative to run_gibbs_partition_jax. Performs Gibbs sampling on
    CPU/GPU using torch.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is not available; install torch to use this function.")

    torch_device = _select_torch_device(device)
    if dtype is None:
        dtype = torch.float32

    data_t = torch.as_tensor(data, dtype=torch.int64, device=torch_device)
    G, N = data_t.shape
    clusters_np = np.asarray(clusters, dtype=np.int32)
    K = int(clusters_np.max()) + 1

    if lam is None:
        lam = float(lam_alpha)
    if np.isscalar(lam):
        lam_vec = torch.full((G,), float(lam), dtype=dtype, device=torch_device)
    else:
        lam_vec = torch.as_tensor(lam, dtype=dtype, device=torch_device).view(-1)
    lam_sum = lam_vec.sum()
    B = torch.lgamma(lam_sum) - torch.lgamma(lam_vec).sum()

    counts = torch.zeros((G, K), dtype=torch.int64, device=torch_device)
    for k in range(K):
        mask = clusters_np == k
        if mask.any():
            mask_t = torch.from_numpy(mask).to(torch_device)
            counts[:, k] = data_t[:, mask_t].sum(dim=1)
    sizes = torch.as_tensor(np.bincount(clusters_np, minlength=K), dtype=torch.int64, device=torch_device)
    clusters_t = torch.as_tensor(clusters_np, dtype=torch.int64, device=torch_device)

    def ll_col(idx):
        if sizes[idx] == 0:
            return torch.tensor(0.0, device=torch_device, dtype=dtype)
        return _torch_ll_cluster(counts[:, idx].to(dtype), lam_vec, B, lam_sum)

    gen = torch.Generator(device=torch_device)
    gen.manual_seed(int(seed))
    moves = 0
    total_delta = 0.0

    for _ in range(sweeps):
        order = torch.randperm(N, generator=gen, device=torch_device)
        for m_t in order:
            m = int(m_t.item())
            c_old = int(clusters_t[m].item())
            cell_vec = data_t[:, m].to(dtype)
            counts[:, c_old] -= cell_vec.to(torch.int64)
            sizes[c_old] -= 1

            ll_base = torch.stack([ll_col(k) for k in range(K)])
            ll_new = torch.stack(
                [_torch_ll_cluster(counts[:, k].to(dtype) + cell_vec, lam_vec, B, lam_sum) for k in range(K)]
            )

            probs = torch.softmax(ll_new - ll_base, dim=0)
            c_new = int(torch.multinomial(probs, 1, generator=gen).item())

            counts[:, c_new] += cell_vec.to(torch.int64)
            sizes[c_new] += 1
            clusters_t[m] = c_new

            if c_new != c_old:
                moves += 1
                total_delta += float((ll_new[c_new] - ll_base[c_new]).item())

    clusters_np = clusters_t.cpu().numpy()
    return clusters_np, moves, total_delta
