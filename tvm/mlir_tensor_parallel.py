#!/usr/bin/env python3
"""
Formal tensor-parallel (TP) walkthrough matching mlir_tensor_parallel.mlir.

What this file does
-------------------
1. Print the sharding + all-reduce placement story (column-parallel matmul
   followed by row-parallel matmul).
2. Run a NumPy SPMD-style simulation for a two-layer matmul block:

     H = X @ W1        (W1 column-sharded; X replicated)
     Y = H @ W2       (W2 row-sharded; each rank computes a partial sum)
     Y = all_reduce_sum(y_partial) across ranks

Run::

    cd /home/xuzhiyuan/ai_compiler_path/tvm
    source ./env.sh
    python mlir_tensor_parallel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_REPLICAS = 4

# Shapes (must be divisible by NUM_REPLICAS along the sharded dimensions)
BATCH = 16
K = 8
M = 16  # hidden dim, column/row-sharded => M % NUM_REPLICAS == 0
N = 6

MLIR_PATH = Path(__file__).with_name("mlir_tensor_parallel.mlir")


def explain_ir() -> None:
    print("=" * 72)
    print("MLIR / StableHLO tensor parallel (formal sketch)")
    print("=" * 72)
    print(
        f"""
Goal: implement the common Megatron TP pattern

  X replicated
  W1 column-parallel  -> H is sharded on dim=1 (second dim)
  W2 row-parallel     -> each rank computes y_partial
  AllReduce(SUM)(y_partial) gives the full Y
"""
    )

    if MLIR_PATH.is_file():
        print(f"Companion IR: {MLIR_PATH}")
        print("-" * 72)
        text = MLIR_PATH.read_text(encoding="utf-8")
        # Print only the key lines so the output stays short.
        for line in text.splitlines():
            if "forward_tp_two_layer" in line:
                print(line)
            if "stablehlo.dot" in line or "stablehlo.all_reduce" in line:
                print(line.strip())


def tp_two_layer_column_row_allreduce(
    x: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    """
    NumPy SPMD simulation for:
      H = X @ W1       with W1 column-sharded
      Y_partial = H @ W2 with W2 row-sharded
      Y = AllReduce(SUM)(Y_partial)
    """

    assert x.shape == (BATCH, K)
    assert w1.shape == (K, M)
    assert w2.shape == (M, N)
    assert M % NUM_REPLICAS == 0

    m_per_rank = M // NUM_REPLICAS

    y_partials: list[np.ndarray] = []
    for rank in range(NUM_REPLICAS):
        # Column-parallel for W1: shard columns (dim=1).
        w1_r = w1[:, rank * m_per_rank : (rank + 1) * m_per_rank]  # (K, M/np)
        h_r = x @ w1_r  # (B, M/np)

        # Row-parallel for W2: shard rows (dim=0).
        w2_r = w2[rank * m_per_rank : (rank + 1) * m_per_rank, :]  # (M/np, N)
        y_partial_r = h_r @ w2_r  # (B, N)
        y_partials.append(y_partial_r)

    # AllReduce(SUM): each rank would end with the same final Y.
    return sum(y_partials)


def simulate(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(BATCH, K)).astype(np.float64)
    w1 = rng.normal(size=(K, M)).astype(np.float64)
    w2 = rng.normal(size=(M, N)).astype(np.float64)

    y_tp = tp_two_layer_column_row_allreduce(x, w1, w2)
    y_ref = (x @ w1) @ w2

    diff = np.max(np.abs(y_tp - y_ref))
    print()
    print(f"NUM_REPLICAS = {NUM_REPLICAS}")
    print(f"max |Y_tp - Y_ref| = {diff:.3e}")
    print("equivalence:", "OK" if diff < 1e-10 else "MISMATCH")


def main() -> None:
    explain_ir()
    simulate()


if __name__ == "__main__":
    main()

