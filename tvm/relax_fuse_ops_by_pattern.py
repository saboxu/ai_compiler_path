"""Relax operator fusion via Dataflow Pattern Language (DPL).

Relay counterpart roughly maps as:
  relay.dataflow_pattern + MergeComposite(pattern, convert)
    →  relax.dpl + FuseOpsByPattern(patterns)

Unlike Relay's MergeComposite convert callback, FuseOpsByPattern extracts the
matched subgraph into a composite Relax function (optionally annotated for BYOC).
Custom rewrite (e.g. replace with a packed kernel) uses rewrite_call / @R.rewriter.
"""

from __future__ import annotations

import tvm
from tvm import relax
from tvm.relax.dpl import (
    is_op,
    is_tuple_get_item,
    make_fused_bias_activation_pattern,
    rewrite_call,
    wildcard,
)
from tvm.script import ir as I
from tvm.script import relax as R


# ---------------------------------------------------------------------------
# 1) Sample graphs
# ---------------------------------------------------------------------------


@I.ir_module
class ConvBnRelu:
    """Inference-style conv2d → batch_norm → relu (use BN output tuple item 0)."""

    @R.function
    def main(
        data: R.Tensor((1, 3, 16, 16), "float32"),
        weight: R.Tensor((8, 3, 3, 3), "float32"),
        gamma: R.Tensor((8,), "float32"),
        beta: R.Tensor((8,), "float32"),
        moving_mean: R.Tensor((8,), "float32"),
        moving_var: R.Tensor((8,), "float32"),
    ) -> R.Tensor((1, 8, 14, 14), "float32"):
        with R.dataflow():
            conv = R.nn.conv2d(data, weight, padding=(0, 0))
            bn = R.nn.batch_norm(
                conv,
                gamma,
                beta,
                moving_mean,
                moving_var,
                axis=1,
                epsilon=1e-5,
                center=True,
                scale=True,
                training=False,
            )
            # batch_norm returns (output, new_moving_mean, new_moving_var)
            bn_out = bn[0]
            out = R.nn.relu(bn_out)
            R.output(out)
        return out


@I.ir_module
class ConvRelu:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight: R.Tensor((64, 64, 3, 3), "float32"),
    ) -> R.Tensor((1, 64, 56, 56), "float32"):
        with R.dataflow():
            conv = R.nn.conv2d(data, weight, padding=(1, 1))
            out = R.nn.relu(conv)
            R.output(out)
        return out


# ---------------------------------------------------------------------------
# 2) Define fuseable patterns (DPL)
# ---------------------------------------------------------------------------


def pattern_conv_bn_relu():
    """Match relax.nn.conv2d → batch_norm → TupleGetItem(0) → relu.

    Mirrors the Relay DFPattern:
      conv = is_op('nn.conv2d')(...)
      bn   = is_op('nn.batch_norm')(conv, ...)
      relu = is_op('nn.relu')(bn.astuple()[0])
    """
    data = wildcard()
    weight = wildcard()
    gamma = wildcard()
    beta = wildcard()
    moving_mean = wildcard()
    moving_var = wildcard()

    conv = is_op("relax.nn.conv2d")(data, weight)
    bn = is_op("relax.nn.batch_norm")(conv, gamma, beta, moving_mean, moving_var)
    bn_out = is_tuple_get_item(bn, 0)
    relu = is_op("relax.nn.relu")(bn_out)

    annotations = {
        "data": data,
        "weight": weight,
        "conv": conv,
        "bn": bn,
        "bn_out": bn_out,
        "relu": relu,
    }
    return relu, annotations


def pattern_conv_relu():
    """Convenience helper: conv2d (+ optional bias) + relu."""
    return make_fused_bias_activation_pattern(
        "relax.nn.conv2d",
        with_bias=False,
        activation="relax.nn.relu",
    )


# ---------------------------------------------------------------------------
# 3) Fuse via FuseOpsByPattern  (≈ Relay MergeComposite)
# ---------------------------------------------------------------------------


def fuse_by_pattern(mod: tvm.IRModule, patterns, *, annotate_codegen: bool = False):
    """Group each match into a composite Relax function.

    patterns entries are either:
      (name, pattern)
      (name, pattern, annotation_dict)
      (name, pattern, annotation_dict, check_fn)

    With annotate_codegen=True, outer functions get Codegen=<prefix> for BYOC
    (prefix taken from pattern name before the first '.').
    """
    return relax.transform.FuseOpsByPattern(
        patterns,
        bind_constants=True,
        annotate_codegen=annotate_codegen,
    )(mod)


# ---------------------------------------------------------------------------
# 4) Optional: custom rewrite  (≈ Relay convert callback)
# ---------------------------------------------------------------------------


def rewrite_conv_relu_to_packed(func: relax.Function) -> relax.Function:
    """Replace matched conv+relu with a single call_dps_packed placeholder.

    This is the Relax analogue of writing a MergeComposite `convert` that
    returns `relay.Call(fused_op, ...)`. Here we leave a packed call stub so
    you can plug in a custom kernel / BYOC codegen later.
    """
    data = wildcard()
    weight = wildcard()
    conv = is_op("relax.nn.conv2d")(data, weight)
    relu = is_op("relax.nn.relu")(conv)

    def rewriter(expr, matchings):
        x = matchings[data]
        w = matchings[weight]
        # Keep original output type from the matched relu expr.
        return relax.op.call_dps_packed(
            "my_backend.fused_conv2d_relu",
            [x, w],
            out_ty=expr.ty,
        )

    return rewrite_call(relu, rewriter, func)


def legalize_and_fuse_tir(mod: tvm.IRModule) -> tvm.IRModule:
    """Lower high-level ops to TIR, then fuse PrimFuncs.

    FuseOps / FuseTIR operate on call_tir graphs (after LegalizeOps +
    AnnotateTIROpPattern). High-level nn.* matching for BYOC stays in
    FuseOpsByPattern above.
    """
    return tvm.ir.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )(mod)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _print_section(title: str, mod) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(mod)


def main() -> None:
    # --- A. Pattern-based fusion: conv + bn + relu ---
    root, ann = pattern_conv_bn_relu()
    patterns_bn = [("demo.conv_bn_relu", root, ann)]

    _print_section("Before FuseOpsByPattern (conv + bn + relu)", ConvBnRelu)
    fused_bn = fuse_by_pattern(ConvBnRelu, patterns_bn, annotate_codegen=True)
    _print_section("After FuseOpsByPattern (Composite + Codegen)", fused_bn)

    # --- B. Pattern-based fusion: conv + relu (helper) ---
    patterns_relu = [("demo.conv_relu", pattern_conv_relu())]
    _print_section("Before FuseOpsByPattern (conv + relu)", ConvRelu)
    fused_relu = fuse_by_pattern(ConvRelu, patterns_relu, annotate_codegen=False)
    _print_section("After FuseOpsByPattern (Composite only)", fused_relu)

    # --- C. Automatic TIR fusion pipeline ---
    auto = legalize_and_fuse_tir(ConvRelu)
    _print_section(
        "After LegalizeOps → AnnotateTIROpPattern → FuseOps → FuseTIR",
        auto,
    )

    # --- D. Custom rewrite to a packed kernel stub ---
    rewritten = ConvRelu.clone()
    rewritten["main"] = rewrite_conv_relu_to_packed(rewritten["main"])
    _print_section("After rewrite_call → call_dps_packed stub", rewritten)


if __name__ == "__main__":
    main()
