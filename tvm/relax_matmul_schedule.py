"""Relax + TIRx matmul with tiling / GPU thread binding.

Modern replacement for the old TE ``create_schedule`` + ``tile`` + ``bind`` example.
Compute lives in a TIRx PrimFunc; Relax calls it via ``R.call_tir``. Scheduling uses
``tvm.s_tir.Schedule``.
"""

import numpy as np

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

N = 1024
TILE = 32


@I.ir_module(s_tir=True)
class MatmulModule:
    @T.prim_func(private=True, s_tir=True)
    def matmul(
        A: T.Buffer((N, N), "float32"),
        B: T.Buffer((N, N), "float32"),
        C: T.Buffer((N, N), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i, j, k in T.grid(N, N, N):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        a: R.Tensor((N, N), "float32"),
        b: R.Tensor((N, N), "float32"),
    ) -> R.Tensor((N, N), "float32"):
        cls = MatmulModule
        return R.call_tir(cls.matmul, (a, b), R.Tensor((N, N), dtype="float32"))


def apply_gpu_schedule(mod: tvm.IRModule) -> tvm.IRModule:
    """Tile 32x32 and bind outer/inner loops to CUDA block/thread axes."""
    sch = tvm.s_tir.Schedule(mod)
    sch.work_on("matmul")
    block_c = sch.get_sblock("C")
    i, j, k = sch.get_loops(block_c)
    bx, tx = sch.split(i, factors=[N // TILE, TILE])
    by, ty = sch.split(j, factors=[N // TILE, TILE])
    sch.reorder(bx, by, tx, ty, k)
    sch.bind(bx, "blockIdx.x")
    sch.bind(by, "blockIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.bind(ty, "threadIdx.y")
    return sch.mod


def apply_cpu_tile_schedule(mod: tvm.IRModule) -> tvm.IRModule:
    """Same tiling without GPU thread binding (for llvm)."""
    sch = tvm.s_tir.Schedule(mod)
    sch.work_on("matmul")
    block_c = sch.get_sblock("C")
    i, j, k = sch.get_loops(block_c)
    io, ii = sch.split(i, factors=[N // TILE, TILE])
    jo, ji = sch.split(j, factors=[N // TILE, TILE])
    sch.reorder(io, jo, ii, ji, k)
    return sch.mod


def run(mod: tvm.IRModule, target: str, device: tvm.runtime.Device) -> None:
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)
    a_np = np.random.uniform(size=(N, N)).astype(np.float32)
    b_np = np.random.uniform(size=(N, N)).astype(np.float32)
    a = tvm.runtime.tensor(a_np, device)
    b = tvm.runtime.tensor(b_np, device)
    c = vm["main"](a, b)
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-3, atol=1e-3)
    print("矩阵乘法计算完成！")


if __name__ == "__main__":
    print("默认调度（未优化的 TIRx）：")
    print(MatmulModule["matmul"])

    gpu_mod = apply_gpu_schedule(MatmulModule)
    print("\n优化后调度（tile + GPU thread bind）：")
    print(gpu_mod["matmul"])

    if tvm.cuda().exist:
        print("\n在 CUDA 上构建并运行…")
        run(gpu_mod, target="cuda", device=tvm.cuda(0))
    else:
        print("\n未检测到 CUDA，改用 llvm + CPU tiling 运行…")
        cpu_mod = apply_cpu_tile_schedule(MatmulModule)
        run(cpu_mod, target="llvm", device=tvm.cpu())
