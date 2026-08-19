#!/usr/bin/env python3
"""
Minimal StableHLO collective insertion tool (teaching-grade).

Given a StableHLO module that matches the specific TP sketch pattern:
  - there is a %y_partial produced by a stablehlo.dot
  - the function returns %y_partial directly

This tool inserts:
  %y = stablehlo.all_reduce(%y_partial, add) across mhlo.num_replicas ranks
and rewrites the return to return %y.

This is not a full compiler pass (no graph/symbolic sharding propagation).
It is a focused "landing code" demo for column→row→AllReduce(SUM).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


NUM_REPLICAS_RE = re.compile(
    r"mhlo\.num_replicas\s*=\s*(\d+)\s*:\s*i32", re.MULTILINE
)
RETURN_Y_PARTIAL_RE = re.compile(
    r"return\s+%y_partial\s*:\s*(tensor<[^>]+>)\s*",
    re.MULTILINE,
)
TENSOR_ELEM_RE = re.compile(r"tensor<.*x([a-zA-Z0-9]+)>\s*$")


def _extract_num_replicas(text: str) -> int:
    m = NUM_REPLICAS_RE.search(text)
    if not m:
        raise ValueError("Cannot find mhlo.num_replicas = N : i32 in module.")
    return int(m.group(1))


def _extract_elem_type(tensor_type: str) -> str:
    # Examples:
    #   tensor<16x6xf32> -> f32
    #   tensor<4x4xbf16> -> bf16
    m = TENSOR_ELEM_RE.search(tensor_type)
    if not m:
        raise ValueError(f"Cannot parse element type from: {tensor_type}")
    return m.group(1)


def _replica_groups_dense(n: int) -> str:
    # stablehlo example style:
    #   dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    ranks = ", ".join(str(i) for i in range(n))
    return f"dense<[[{ranks}]]> : tensor<1x{n}xi64>"


def insert_allreduce_if_missing(input_mlir: str) -> str:
    # Only treat it as "already inserted" when we see the actual StableHLO
    # op form: `"stablehlo.all_reduce"(%ssa)`, not just a mention in comments.
    if '"stablehlo.all_reduce"' in input_mlir:
        return input_mlir

    n = _extract_num_replicas(input_mlir)

    m = RETURN_Y_PARTIAL_RE.search(input_mlir)
    if not m:
        raise ValueError(
            "Pattern not found: expected `return %y_partial : tensor<...>`."
        )
    y_tensor_type = m.group(1)
    elem_type = _extract_elem_type(y_tensor_type)

    # Note: StableHLO uses `{}` braces for regions/attributes, so we must
    # escape them in Python string formatting.
    allreduce_op = """
    %y = "stablehlo.all_reduce"(%y_partial) ({{
      ^bb0(%lhs: tensor<{elem_type}>, %rhs: tensor<{elem_type}>):
        %sum = stablehlo.add %lhs, %rhs : tensor<{elem_type}>
        stablehlo.return %sum : tensor<{elem_type}>
    }}) {{
      replica_groups = {replica_groups},
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
    }} : ({y_tensor_type}) -> ({y_tensor_type})
""".format(
        elem_type=elem_type,
        replica_groups=_replica_groups_dense(n),
        y_tensor_type=y_tensor_type,
    )

    # Insert right before the return %y_partial line.
    split_at = m.start()
    before = input_mlir[:split_at]
    after = input_mlir[split_at:]

    # Rewrite the return line.
    after = after.replace(
        f"return %y_partial : {y_tensor_type}",
        f"return %y : {y_tensor_type}",
        1,
    )
    return before + allreduce_op + after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_mlir", type=str)
    ap.add_argument(
        "-o", "--output", type=str, default="", help="Write output to a file."
    )
    args = ap.parse_args()

    input_path = Path(args.input_mlir)
    text = input_path.read_text(encoding="utf-8")
    out = insert_allreduce_if_missing(text)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()

