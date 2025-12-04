#!/usr/bin/env python
"""
Quick benchmark comparing the original numpy/scipy gene_contribution_table
to the JAX-accelerated implementation on a small synthetic dataset.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import pandas as pd

from cellstates import (
    Cluster,
    gene_contribution_table,
    gene_contribution_table_jax,
    get_hierarchy_df,
    jax_available,
)


def _time_call(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _load_counts(path: str, max_genes: int, max_cells: int) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[0] and not np.issubdtype(df.dtypes[0], np.number):
        df = df.iloc[1:, 1:]
    counts = df.to_numpy(dtype=np.int64)
    if max_genes > 0:
        counts = counts[:max_genes, :]
    if max_cells > 0:
        counts = counts[:, :max_cells]
    return counts


def _synthetic_counts(genes: int, cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.5, 5.0, size=(genes, 1))
    return rng.poisson(rates, size=(genes, cells)).astype(np.int64)


def _build_cluster(counts: np.ndarray, threads: int, seed: int) -> Tuple[Cluster, np.ndarray]:
    clst = Cluster(counts, num_threads=threads, seed=seed)
    hierarchy, delta_ll = clst.get_cluster_hierarchy()
    hierarchy_df = get_hierarchy_df(hierarchy, delta_ll)
    return clst, hierarchy_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=int, default=256, help="Number of genes for synthetic data.")
    parser.add_argument("--cells", type=int, default=256, help="Number of cells for synthetic data.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--threads", type=int, default=1, help="Threads used by cython routines.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional path to counts table; defaults to synthetic data if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="JAX device preference: auto, cpu, gpu, mps, or tpu.",
    )
    args = parser.parse_args()

    if args.data:
        counts = _load_counts(args.data, args.genes, args.cells)
        print(f"Loaded counts from {args.data} with shape {counts.shape}")
    else:
        counts = _synthetic_counts(args.genes, args.cells, args.seed)
        print(f"Synthetic counts shape: {counts.shape}")

    clst, hierarchy_df = _build_cluster(counts, args.threads, args.seed)
    print(f"Hierarchy merges: {hierarchy_df.shape[0]}, genes: {clst.G}")

    baseline_scores, baseline_time = _time_call(lambda: gene_contribution_table(clst, hierarchy_df))
    print(f"Baseline numpy/scipy runtime: {baseline_time:.4f} s")

    if not jax_available():
        print("JAX is not installed; skipping JAX benchmark.")
        return

    device_choice = None if args.device == "auto" else args.device
    warm_scores, jax_first = _time_call(
        lambda: gene_contribution_table_jax(clst, hierarchy_df, device=device_choice)
    )
    repeat_scores, jax_second = _time_call(
        lambda: gene_contribution_table_jax(clst, hierarchy_df, device=device_choice)
    )

    max_diff = np.max(np.abs(baseline_scores - repeat_scores))

    print(f"JAX runtime (compile + execute): {jax_first:.4f} s")
    print(f"JAX runtime (cached): {jax_second:.4f} s")
    print(f"Max |baseline - jax|: {max_diff:.3e}")


if __name__ == "__main__":
    main()
