"""
Backend-agnostic acceleration helpers for stochastic and Gibbs partitioning.
Dispatches to JAX or PyTorch implementations and handles device selection.
"""

from __future__ import annotations

import numpy as np


def detect_device(backend: str) -> str:
    """Return preferred device string for the given backend ('jax' or 'torch')."""
    backend = backend.lower()
    if backend == "jax":
        try:
            import jax

            devs = jax.devices()
            if devs:
                return devs[0].platform
        except Exception:
            pass
        return "cpu"
    if backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    raise ValueError("backend must be 'jax' or 'torch'")


def verify_acceleration(backend: str):
    """
    Check whether acceleration is available for the backend.
    Returns a dict with keys: backend, ok (bool), devices (list), message (str).
    """
    backend = backend.lower()
    info = {"backend": backend, "ok": False, "devices": [], "message": ""}
    if backend == "jax":
        try:
            import jax

            devs = jax.devices()
            info["devices"] = devs
            info["ok"] = len(devs) > 0
            info["message"] = f"jax {jax.__version__}, jaxlib {jax.lib.version}"
        except Exception as err:  # pragma: no cover - runtime env check
            info["message"] = f"JAX not available: {err}"
        return info
    if backend == "torch":
        try:
            import torch

            devices = []
            if torch.cuda.is_available():
                devices.append(f"cuda:{torch.cuda.current_device()}:{torch.cuda.get_device_name(torch.cuda.current_device())}")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devices.append("mps")
            if not devices:
                devices.append("cpu")
            info["devices"] = devices
            info["ok"] = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            info["message"] = f"torch {torch.__version__}"
        except Exception as err:  # pragma: no cover - runtime env check
            info["message"] = f"Torch not available: {err}"
        return info
    raise ValueError("backend must be 'jax' or 'torch'")


def run_stochastic_accel(
    data: np.ndarray,
    clusters: np.ndarray,
    lam: np.ndarray | float | None = None,
    backend: str = "jax",
    device: str | None = None,
    sweeps: int = 3,
    proposals_per_cell: int = 8,
    seed: int = 0,
    lam_alpha: float = 0.001,
):
    """
    Run stochastic partition using the selected backend (JAX or torch).
    Returns (labels, moves, delta), backend_label.
    """
    backend = backend.lower()
    if device is None:
        device = detect_device(backend)

    lam_local = lam
    if lam_local is None:
        lam_local = np.full(data.shape[0], float(lam_alpha), dtype=np.float32)

    if backend == "jax":
        import jax
        from .jax_mcmc import stochastic_partition_jax

        result = stochastic_partition_jax(
            data,
            clusters,
            lam=lam_local,
            sweeps=sweeps,
            proposals_per_cell=proposals_per_cell,
            device=device,
            enable_x64=False,
            dtype=jax.numpy.float32,
            seed=seed,
        )
        return result, f"jax:{device}"

    if backend == "torch":
        import torch
        from .jax_mcmc import stochastic_partition_torch

        result = stochastic_partition_torch(
            data,
            clusters,
            lam=lam_local,
            sweeps=sweeps,
            proposals_per_cell=proposals_per_cell,
            device=device,
            dtype=torch.float32,
            seed=seed,
        )
        return result, f"torch:{device}"

    raise ValueError("backend must be 'jax' or 'torch'")


def run_gibbs_accel(
    data: np.ndarray,
    clusters: np.ndarray,
    backend: str = "jax",
    device: str | None = None,
    sweeps: int = 3,
    seed: int = 0,
    lam_alpha: float = 0.001,
    max_sweeps: int | None = None,
    stop_tol: float | None = None,
    use_mem_heuristic: bool = True,
    return_sweeps: bool = False,
):
    """
    Run Gibbs partition using the selected backend (JAX or torch).
    If stop_tol or max_sweeps are provided, runs adaptively until moves per cell
    fall below stop_tol or max_sweeps is reached. When return_sweeps=True,
    returns (labels, moves, delta, sweeps_done).
    """
    backend = backend.lower()
    if device is None:
        device = detect_device(backend)

    sweeps_per_call = max(1, int(sweeps))
    max_sweeps = max_sweeps if max_sweeps is not None else sweeps_per_call

    def _heuristic_cap(cap: int) -> int:
        if not use_mem_heuristic:
            return cap
        if backend == "torch":
            try:
                import torch

                if torch.cuda.is_available():
                    _, total = torch.cuda.mem_get_info()
                    total_gb = total / 1e9
                    if total_gb < 8:
                        return min(cap, 20)
                    if total_gb < 16:
                        return min(cap, 40)
                    if total_gb < 32:
                        return min(cap, 70)
            except Exception:
                pass
        return cap

    max_sweeps = _heuristic_cap(int(max_sweeps))
    labels_local = np.asarray(clusters, dtype=np.int32)
    total_moves = 0
    total_delta = 0.0
    sweeps_done = 0
    backend_label = f"{backend}:{device}"

    if backend == "jax":
        import jax
        from .jax_gibbs import run_gibbs_partition_jax

        while sweeps_done < max_sweeps:
            result = run_gibbs_partition_jax(
                data,
                labels_local,
                lam=None,
                sweeps=min(sweeps_per_call, max_sweeps - sweeps_done),
                device=device,
                enable_x64=False,
                dtype=jax.numpy.float32,
                seed=seed + sweeps_done,
                lam_alpha=lam_alpha,
            )
            labels_local, moves, delta = result
            total_moves += moves
            total_delta += delta
            sweeps_done += min(sweeps_per_call, max_sweeps - sweeps_done)
            if stop_tol is not None and moves / labels_local.shape[0] < stop_tol:
                break
        backend_label = f"jax:{device}"

    elif backend == "torch":
        import torch
        from .jax_mcmc import run_gibbs_partition_torch

        while sweeps_done < max_sweeps:
            result = run_gibbs_partition_torch(
                data,
                labels_local,
                lam=None,
                sweeps=min(sweeps_per_call, max_sweeps - sweeps_done),
                device=device,
                dtype=torch.float32,
                seed=seed + sweeps_done,
                lam_alpha=lam_alpha,
            )
            labels_local, moves, delta = result
            total_moves += moves
            total_delta += delta
            sweeps_done += min(sweeps_per_call, max_sweeps - sweeps_done)
            if stop_tol is not None and moves / labels_local.shape[0] < stop_tol:
                break
        backend_label = f"torch:{device}"

    else:
        raise ValueError("backend must be 'jax' or 'torch'")

    result = (labels_local, total_moves, total_delta, sweeps_done) if return_sweeps else (labels_local, total_moves, total_delta)
    return result, backend_label


def run_gibbs_adaptive(
    data: np.ndarray,
    clusters: np.ndarray,
    backend: str = "jax",
    device: str | None = None,
    lam_alpha: float = 0.001,
    seed: int = 0,
    max_sweeps: int = 100,
    stop_tol: float = 0.001,
    sweeps_per_call: int = 1,
    use_mem_heuristic: bool = True,
):
    """
    Convenience wrapper that runs Gibbs with adaptive stopping.
    Returns (labels, moves, delta, sweeps_done), backend_label.
    """
    result, backend_label = run_gibbs_accel(
        data,
        clusters,
        backend=backend,
        device=device,
        sweeps=sweeps_per_call,
        seed=seed,
        lam_alpha=lam_alpha,
        max_sweeps=max_sweeps,
        stop_tol=stop_tol,
        use_mem_heuristic=use_mem_heuristic,
        return_sweeps=True,
    )
    return result, backend_label
