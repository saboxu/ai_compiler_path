#!/usr/bin/env python3
"""Symbolic shape *inference* — derive output dims from inputs, do not hardcode them.

The common textbook sketch writes::

    output_shape = (b, s, 4*d)  # 「推理结果」写死在代码里

That is only the *answer*. Real inference is a function of the op + operand shapes::

    infer_matmul((b, s, d), (d, 4*d))  -->  (b, s, 4*d)
                   ^ known               ^ derived from weight's N dim

Run::

    cd /home/xuzhiyuan/ai_compiler_path/tvm
    source ./env.sh
    python symbolic_shape_inference.py
"""

from __future__ import annotations

from typing import Callable

import sympy as sp

Shape = tuple[sp.Expr, ...]


# ---------------------------------------------------------------------------
# Inference rules (this is the actual "reasoning")
# ---------------------------------------------------------------------------


def infer_matmul(a: Shape, b: Shape) -> Shape:
    """[..., K] @ [K, N] -> [..., N].  N is *taken from* b, not written by hand."""
    if len(b) != 2:
        raise ValueError(f"rhs must be rank-2, got {b}")
    if len(a) < 1:
        raise ValueError(f"lhs must be at least rank-1, got {a}")
    *batch, k_left = a
    k_right, n_out = b
    # Emit the contract-dim proof obligation (symbolic equality).
    if sp.simplify(sp.Eq(k_left, k_right)) is sp.false:
        # Still allow symbols that are algebraically equal, e.g. d vs d.
        if sp.simplify(k_left - k_right) != 0:
            raise ValueError(f"matmul contract mismatch: {k_left} vs {k_right}")
    # >>> Inference step: keep batch dims from A, take output width from B[1]
    return (*batch, n_out)


def infer_softmax(x: Shape, axis: int = -1) -> Shape:
    """Elementwise-ish on shape: softmax preserves shape."""
    _ = axis
    return x


def infer_add(a: Shape, b: Shape) -> Shape:
    """Broadcasting add under symbolic dims (same rank, dims equal or 1)."""
    if len(a) != len(b):
        raise ValueError(f"add rank mismatch: {a} vs {b}")
    out: list[sp.Expr] = []
    for da, db in zip(a, b):
        if sp.simplify(da - db) == 0:
            out.append(da)
        elif sp.simplify(da - 1) == 0:
            out.append(db)
        elif sp.simplify(db - 1) == 0:
            out.append(da)
        else:
            raise ValueError(f"cannot broadcast {da} with {db}")
    return tuple(out)


RULES: dict[str, Callable[..., Shape]] = {
    "matmul": infer_matmul,
    "softmax": infer_softmax,
    "add": infer_add,
}


def infer(op: str, *args: Shape, **kwargs) -> Shape:
    if op not in RULES:
        raise KeyError(op)
    return RULES[op](*args, **kwargs)


# ---------------------------------------------------------------------------
# Demo: print each derivation step so "inference" is visible
# ---------------------------------------------------------------------------


def demo_sympy() -> None:
    print("=" * 64)
    print("Symbolic shape inference (derived, not hardcoded)")
    print("=" * 64)

    b, s, d = sp.symbols("b s d", integer=True, positive=True)

    x = (b, s, d)
    w1 = (d, 4 * d)  # only place 4*d appears as a *known weight attribute*
    w2 = (4 * d, d)

    print("\n[known operands]")
    print(f"  x  : {x}")
    print(f"  w1 : {w1}   # weight.shape, given")
    print(f"  w2 : {w2}")

    print("\n[infer] h1 = matmul(x, w1)")
    print(f"  rule: out = (*x[:-1], w1[1])  and require x[-1] == w1[0]")
    print(f"  check contract: x[-1]={x[-1]}, w1[0]={w1[0]}, equal? "
          f"{sp.simplify(x[-1] - w1[0]) == 0}")
    h1 = infer("matmul", x, w1)
    print(f"  => h1 = {h1}")
    # Prove we did not hardcode 4*d: it is identically w1[1]
    assert sp.simplify(h1[-1] - w1[1]) == 0
    print(f"  proof: h1[-1] is w1[1]  ({h1[-1]} == {w1[1]})")

    print("\n[infer] h2 = softmax(h1)")
    h2 = infer("softmax", h1)
    print(f"  => h2 = {h2}   # same as h1")

    print("\n[infer] y = matmul(h2, w2)")
    y = infer("matmul", h2, w2)
    print(f"  => y  = {y}")
    assert sp.simplify(y[0] - b) == 0
    assert sp.simplify(y[1] - s) == 0
    assert sp.simplify(y[2] - d) == 0
    print("  proof: y == (b, s, d)  (FFN round-trip closes)")

    print("\n[infer] z = add(x, y)   # residual")
    z = infer("add", x, y)
    print(f"  => z  = {z}")

    constraints = [sp.Le(s, 512), sp.Eq(d, 768)]
    print(f"\n[constraints] {constraints}")
    subst = {b: 2, s: 128, d: 768}
    print(f"[specialize] {subst}")
    for name, shape in ("x", x), ("h1", h1), ("y", y), ("z", z):
        concrete = tuple(int(dim.subs(subst)) for dim in shape)
        print(f"  {name}: {shape}  ->  {concrete}")
    for c in constraints:
        print(f"  check {c}: {bool(c.subs(subst))}")


def demo_contrast_hardcoded() -> None:
    print()
    print("=" * 64)
    print("Contrast: hardcoded answer vs inferred answer")
    print("=" * 64)
    b, s, d = sp.symbols("b s d", integer=True, positive=True)
    hardcoded = (b, s, 4 * d)  # textbook sketch — just writes the result
    inferred = infer("matmul", (b, s, d), (d, 4 * d))
    print(f"  hardcoded : {hardcoded}")
    print(f"  inferred  : {inferred}")
    print(f"  equal?    : {sp.simplify(sp.Matrix(hardcoded) - sp.Matrix(inferred)) == sp.Matrix.zeros(3, 1)}")
    print("  difference: inferred *computes* last dim as weight[1]; hardcoded asserts it.")


def demo_tvm_relax() -> None:
    """Relax side: annotations name dims; matmul still has to line them up."""
    print()
    print("=" * 64)
    print("TVM Relax: dynamic dims (annotation + matmul)")
    print("=" * 64)
    print(
        """
  R.Tensor(("b","s","d")) is like sympy symbols — a *declaration*.
  Shape inference inside Relax still runs when you build/normalize the
  function: matmul's output last-dim becomes w's N (here named "d4").

  Important: writing
      -> R.Tensor(("b","s","d4"))
  on the return type is an annotation / check, analogous to asserting
  the inferred shape. The rule itself is still matmul's infer_struct_info.
"""
    )
    from tvm.script import ir as I
    from tvm.script import relax as R

    @I.ir_module
    class LinearFFN:
        @R.function
        def main(
            x: R.Tensor(("b", "s", "d"), "float32"),
            w1: R.Tensor(("d", "d4"), "float32"),
        ) -> R.Tensor(("b", "s", "d4"), "float32"):
            out = R.matmul(x, w1)
            return out

    print(LinearFFN)
    # Show the body binding's inferred TensorStructInfo (post-parse).
    fn = LinearFFN["main"]
    # Walk to the matmul binding if present.
    try:
        blocks = fn.body.blocks
        bind = blocks[0].bindings[0]
        print("matmul binding struct_info:", bind.var.struct_info)
    except Exception as exc:
        print(f"(could not print binding struct_info: {exc})")


def main() -> None:
    demo_sympy()
    demo_contrast_hardcoded()
    try:
        demo_tvm_relax()
    except Exception as exc:  # pragma: no cover
        print(f"\n[TVM demo skipped] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
