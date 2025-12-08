"""
Optional JAX-backed helpers to speed up pure-Python routines.
The functions here mirror logic in helpers.py but run on JAX devices.
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


def jax_available() -> bool:
    """Return True if JAX is importable."""
    return HAS_JAX


def available_jax_devices(platform: str | None = None):
    """
    List JAX devices, optionally filtered by platform (cpu/gpu/tpu/mps).

    Parameters
    ----------
    platform : str, optional
        Filter devices by platform. Accepts 'cpu', 'gpu', 'tpu', or 'mps'
        (an alias for gpu on Apple silicon).
    """
    if not HAS_JAX:
        return []

    if platform == "mps":
        platform = "gpu"
    if platform:
        return [d for d in jax.devices() if d.platform == platform]
    return list(jax.devices())


def _select_device(device: str | None):
    if device is None:
        return None
    devices = available_jax_devices(device)
    if devices:
        return devices[0]
    all_devices = available_jax_devices(None)
    if all_devices:
        return all_devices[0]
    raise ValueError(f"No JAX devices found; requested platform '{device}'.")


def _binomial_p_jax(n, lam):
    lam_sum = jnp.sum(lam)
    n_sum = jnp.sum(n)
    return (
        gammaln(lam_sum)
        - gammaln(lam)
        - gammaln(lam_sum - lam)
        + gammaln(n + lam)
        + gammaln(n_sum + lam_sum - n - lam)
        - gammaln(n_sum + lam_sum)
    )


def _gene_contribution_jax(n1, n2, lam):
    return _binomial_p_jax(n1 + n2, lam) - _binomial_p_jax(n1, lam) - _binomial_p_jax(n2, lam)


def gene_contribution_table_jax(
    clst,
    hierarchy_df,
    device: str | None = None,
    enable_x64: bool = True,
    dtype=None,
):
    """
    JAX-accelerated equivalent of helpers.gene_contribution_table.

    Parameters
    ----------
    clst : cellstates.Cluster
        Cluster instance containing counts.
    hierarchy_df : pandas.DataFrame
        Output of helpers.get_hierarchy_df.
    device : str, optional
        Desired device platform ('cpu', 'gpu', 'tpu', 'mps'). Defaults to
        JAX's default device selection.
    enable_x64 : bool, default=True
        Enable float64 precision for numerical parity with numpy/scipy.
    dtype : jnp.dtype, optional
        Override dtype for the computation. If None, use float64 when
        enable_x64=True else float32. Passing bfloat16/float32 can
        significantly lower TPU/GPU memory usage (at the cost of precision).
    """
    if not HAS_JAX:
        raise ImportError("JAX is not available; install jax to use the accelerated path.")

    if enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    if dtype is None:
        dtype = jnp.float64 if enable_x64 else jnp.float32
    lam = jnp.asarray(clst.dirichlet_pseudocounts, dtype=dtype)
    counts = jnp.asarray(clst.cluster_umi_counts, dtype=dtype)
    merges = jnp.asarray(hierarchy_df.loc[:, ["cluster_old", "cluster_new"]].to_numpy(np.int32))

    target_device = _select_device(device)
    if target_device:
        lam = jax.device_put(lam, target_device)
        counts = jax.device_put(counts, target_device)
        merges = jax.device_put(merges, target_device)

    def _scan(counts_arr, merges_arr, lam_arr):
        def merge_step(current_counts, pair):
            c_old, c_new = pair
            n1 = current_counts[:, c_old]
            n2 = current_counts[:, c_new]
            score = _gene_contribution_jax(n1, n2, lam_arr)
            updated = current_counts.at[:, c_new].set(n1 + n2)
            updated = updated.at[:, c_old].set(0)
            return updated, score

        _, scores = jax.lax.scan(merge_step, counts_arr, merges_arr)
        return scores

    # donate buffers so XLA can reuse memory and reduce peak footprint
    scan_fn = jax.jit(_scan, donate_argnums=(0, 1))
    scores = scan_fn(counts, merges, lam)
    return np.asarray(scores)
