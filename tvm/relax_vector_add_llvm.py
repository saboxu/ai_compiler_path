"""Relax + TIRx vector_add targeting LLVM (CPU).

Modern replacement for the old TE ``create_schedule`` + ``tvm.build`` example.
"""

import numpy as np

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

N = 1024


@T.prim_func(s_tir=True)
def vector_add_prim(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    C: T.Buffer((N,), "float32"),
):
    T.func_attr({"tirx.noalias": True, "global_symbol": "vector_add"})
    for i in range(N):
        with T.sblock("C"):
            vi = T.axis.spatial(N, i)
            C[vi] = A[vi] + B[vi]


@I.ir_module(s_tir=True)
class VectorAdd:
    @T.prim_func(private=True, s_tir=True)
    def vector_add(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i in range(N):
            with T.sblock("C"):
                vi = T.axis.spatial(N, i)
                C[vi] = A[vi] + B[vi]

    @R.function
    def main(
        a: R.Tensor((N,), "float32"),
        b: R.Tensor((N,), "float32"),
    ) -> R.Tensor((N,), "float32"):
        cls = VectorAdd
        return R.call_tir(cls.vector_add, (a, b), R.Tensor((N,), dtype="float32"))


if __name__ == "__main__":
    print("=== TIRx / Relax 模块 ===")
    print(VectorAdd)

    # 1) 对公开 PrimFunc 做 LLVM codegen，并打印 IR 片段
    #    （对应旧代码里的 tvm.build(...); func.get_source("llvm")）
    target = tvm.target.Target("llvm")
    llvm_mod = tvm.build(vector_add_prim, target=target)
    llvm_ir = llvm_mod.inspect_source()
    print("\n--- 生成的 LLVM IR 片段 ---")
    print(llvm_ir[:500])

    # 2) 走 Relax VM 构建并跑通数值
    ex = relax.build(VectorAdd, target=target)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)

    a_np = np.random.uniform(size=(N,)).astype(np.float32)
    b_np = np.random.uniform(size=(N,)).astype(np.float32)
    c = vm["main"](tvm.runtime.tensor(a_np, dev), tvm.runtime.tensor(b_np, dev))
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5, atol=1e-5)
    print("\nvector_add 计算完成！")
