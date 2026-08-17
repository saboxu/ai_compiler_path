#!/usr/bin/env python3
"""Formal data-parallel (DP) walkthrough matching mlir_data_parallel.mlir.

What this file does
-------------------
1. Print the StableHLO / GSPMD annotation story (shard / replicate / all-reduce).
2. Run a NumPy SPMD simulation of 4-way DP on a tiny linear model so the
   IR steps are numerically checkable without XLA.

Run::

    cd /home/xuzhiyuan/ai_compiler_path/tvm
    source ./env.sh
    python mlir_data_parallel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_REPLICAS = 4
GLOBAL_BATCH = 1024
LOCAL_BATCH = GLOBAL_BATCH // NUM_REPLICAS  # 256
IN_DIM = 784
OUT_DIM = 10
LR = 0.1
MLIR_PATH = Path(__file__).with_name("mlir_data_parallel.mlir")


def explain_ir() -> None:
    print("=" * 72)
    print("MLIR / StableHLO data parallel (formal sketch)")
    print("=" * 72)
    print(
        """
Pseudocode ops                 Formal StableHLO / GSPMD form
-----------------------------  -------------------------------------------------
mhlo.shard(data, dim=0)        tensor arg + {mhlo.sharding = "{devices=[4,1]<=[4]}"}
mhlo.replicate(forward)        weights {mhlo.sharding = "{replicated}"} + same body
mhlo.all_reduce(grad, add)     stablehlo.all_reduce + replica_groups=[[0,1,2,3]]
mhlo.apply_gradient            local W := W - lr * (sum_grad / N)

SPMD idea:
  - Programmer writes *one* program with global shapes + sharding attrs.
  - Compiler rewrites it into a *per-device* program (local batch = 256).
  - Collectives (all_reduce) keep replicated state (weights) consistent.
"""
    )
    if MLIR_PATH.is_file():
        print(f"Companion IR: {MLIR_PATH}")
        print("-" * 72)
        # Show the train_step signature region briefly.
        text = MLIR_PATH.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "func.func @train_step_dp" in line or "stablehlo.all_reduce" in line:
                print(line)
            if 'mhlo.sharding = "{devices=[4,1]<=[4]}"' in line:
                print(line.strip())
            if 'mhlo.sharding = "{replicated}"' in line and "%w:" in line:
                print(line.strip())


def softmax_cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """dL/dlogits for mean softmax-CE (per local batch)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    grad = probs
    grad[np.arange(labels.shape[0]), labels] -= 1.0
    return grad / labels.shape[0]


def forward(data: np.ndarray, w: np.ndarray) -> np.ndarray:
    return data @ w


def weight_grad(data: np.ndarray, dlogits: np.ndarray) -> np.ndarray:
    # dW = X^T @ dlogits
    return data.T @ dlogits


def all_reduce_sum(local_grads: list[np.ndarray]) -> np.ndarray:
    """Simulate stablehlo.all_reduce(add) over replica_groups=[[0,1,2,3]]."""
    return sum(local_grads)


def simulate_dp(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # Global batch, then shard along dim 0 (like devices=[4,1]<=[4]).
    data = rng.normal(size=(GLOBAL_BATCH, IN_DIM)).astype(np.float64)
    labels = rng.integers(0, OUT_DIM, size=(GLOBAL_BATCH,), dtype=np.int64)
    w = rng.normal(scale=0.02, size=(IN_DIM, OUT_DIM)).astype(np.float64)

    shards_data = np.split(data, NUM_REPLICAS, axis=0)
    shards_label = np.split(labels, NUM_REPLICAS, axis=0)

    print()
    print("=" * 72)
    print("NumPy SPMD simulation (4 replicas)")
    print("=" * 72)
    print(f"global data {data.shape} -> local {[s.shape for s in shards_data]}")
    print(f"weights replicated: {w.shape}")

    local_grads: list[np.ndarray] = []
    for rank, (xd, yd) in enumerate(zip(shards_data, shards_label)):
        logits = forward(xd, w)  # replicated W
        dlogits = softmax_cross_entropy_grad(logits, yd)
        g = weight_grad(xd, dlogits)
        local_grads.append(g)
        print(f"  rank{rank}: local_grad frobenius={np.linalg.norm(g):.6f}")

    # Cross-device sync (sum), then mean — matches the .mlir comments.
    global_sum = all_reduce_sum(local_grads)
    mean_grad = global_sum / NUM_REPLICAS
    w_new = w - LR * mean_grad

    # Reference: single-device training on the full batch should match DP mean.
    logits_ref = forward(data, w)
    dlogits_ref = softmax_cross_entropy_grad(logits_ref, labels)
    grad_ref = weight_grad(data, dlogits_ref)
    # Careful: local CE used mean over local batch; global mean over global batch
    # with equal shard sizes is equivalent to averaging local mean-grads.
    w_ref = w - LR * grad_ref

    diff = np.max(np.abs(w_new - w_ref))
    print()
    print(f"max |W_dp - W_single| = {diff:.3e}")
    print("DP equivalence:", "OK" if diff < 1e-10 else "MISMATCH")
    print()
    print("Takeaway: shard batch → local forward/grad → all_reduce → same W update.")


def main() -> None:
    explain_ir()
    simulate_dp()


if __name__ == "__main__":
    main()
