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
):
    """
    Run Gibbs partition using the selected backend (JAX or torch).
    Returns (labels, moves, delta), backend_label.
    """
    backend = backend.lower()
    if device is None:
        device = detect_device(backend)

    if backend == "jax":
        import jax
        from .jax_gibbs import run_gibbs_partition_jax

        result = run_gibbs_partition_jax(
            data,
            clusters,
            lam=None,
            sweeps=sweeps,
            device=device,
            enable_x64=False,
            dtype=jax.numpy.float32,
            seed=seed,
            lam_alpha=lam_alpha,
        )
        return result, f"jax:{device}"

    if backend == "torch":
        import torch
        from .jax_mcmc import run_gibbs_partition_torch

        result = run_gibbs_partition_torch(
            data,
            clusters,
            lam=None,
            sweeps=sweeps,
            device=device,
            dtype=torch.float32,
            seed=seed,
            lam_alpha=lam_alpha,
        )
        return result, f"torch:{device}"

    raise ValueError("backend must be 'jax' or 'torch'")

