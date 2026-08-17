#!/usr/bin/env python3
"""Compare small GEMM implementations side-by-side.

Implementations
---------------
1. C++ naive triple-loop          (gemm_naive)
2. C++ blocked + AVX2/FMA 4x8     (gemm_avx2_blocked)  ← your micro-kernel sketch
3. NumPy / OpenBLAS ``A @ B``
4. TVM Relax + TIRx tiled schedule (llvm)

Run::

    cd /home/xuzhiyuan/ai_compiler_path/tvm
    source ./env.sh
    python compare_small_gemm.py
    python compare_small_gemm.py --sizes 128,256,512 --repeat 20
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CPP = ROOT / "gemm_avx2_blocked.cpp"
SO = ROOT / "libgemm_avx2.so"


def build_shared_lib() -> ctypes.CDLL:
    cmd = [
        "g++",
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
        "-shared",
        "-fPIC",
        "-o",
        str(SO),
        str(CPP),
    ]
    print("build:", " ".join(cmd))
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(str(SO))
    for name in ("gemm_avx2_blocked", "gemm_naive"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = None
    return lib


def as_ptr(x: np.ndarray):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def bench(fn, repeat: int) -> float:
    # warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def gflops(M: int, N: int, K: int, seconds: float) -> float:
    return (2.0 * M * N * K) / seconds / 1e9


def build_tvm_matmul(n: int):
    import tvm
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    from tvm.script import tirx as T

    tile = 32 if n >= 32 else max(1, n // 4)

    @I.ir_module(s_tir=True)
    class MatmulModule:
        @T.prim_func(private=True, s_tir=True)
        def matmul(
            A: T.Buffer((n, n), "float32"),
            B: T.Buffer((n, n), "float32"),
            C: T.Buffer((n, n), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(n, n, n):
                with T.sblock("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            a: R.Tensor((n, n), "float32"),
            b: R.Tensor((n, n), "float32"),
        ) -> R.Tensor((n, n), "float32"):
            cls = MatmulModule
            return R.call_tir(cls.matmul, (a, b), R.Tensor((n, n), dtype="float32"))

    sch = tvm.s_tir.Schedule(MatmulModule)
    sch.work_on("matmul")
    block_c = sch.get_sblock("C")
    i, j, k = sch.get_loops(block_c)
    if n % tile == 0 and tile > 1:
        io, ii = sch.split(i, factors=[n // tile, tile])
        jo, ji = sch.split(j, factors=[n // tile, tile])
        sch.reorder(io, jo, ii, ji, k)
    ex = relax.build(sch.mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return tvm, vm


def run_size(lib: ctypes.CDLL, n: int, repeat: int) -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n), dtype=np.float32)
    B = rng.standard_normal((n, n), dtype=np.float32)
    C_ref = A @ B

    C_naive = np.empty_like(C_ref)
    C_avx = np.empty_like(C_ref)

    def call_naive():
        lib.gemm_naive(as_ptr(A), as_ptr(B), as_ptr(C_naive), n, n, n)

    def call_avx():
        lib.gemm_avx2_blocked(as_ptr(A), as_ptr(B), as_ptr(C_avx), n, n, n)

    def call_numpy():
        return A @ B

    tvm, vm = build_tvm_matmul(n)
    a_t = tvm.runtime.tensor(A)
    b_t = tvm.runtime.tensor(B)

    def call_tvm():
        return vm["main"](a_t, b_t).numpy()

    # correctness
    call_naive()
    call_avx()
    C_tvm = call_tvm()
    err_naive = float(np.max(np.abs(C_naive - C_ref)))
    err_avx = float(np.max(np.abs(C_avx - C_ref)))
    err_tvm = float(np.max(np.abs(C_tvm - C_ref)))

    t_naive = bench(call_naive, max(1, repeat // 4 if n >= 512 else repeat))
    t_avx = bench(call_avx, repeat)
    t_np = bench(call_numpy, repeat)
    t_tvm = bench(call_tvm, repeat)

    print(f"\n=== GEMM {n}x{n}x{n}  (repeat={repeat}) ===")
    print(f"{'impl':<22} {'ms':>10} {'GFLOP/s':>10} {'max|err|':>12}")
    rows = [
        ("C++ naive", t_naive, err_naive),
        ("C++ AVX2 blocked 4x8", t_avx, err_avx),
        ("NumPy (BLAS)", t_np, 0.0),
        ("TVM tiled llvm", t_tvm, err_tvm),
    ]
    for name, sec, err in rows:
        print(
            f"{name:<22} {sec*1e3:10.3f} {gflops(n,n,n,sec):10.2f} {err:12.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="64,128,256",
        help="comma-separated square sizes (default: 64,128,256)",
    )
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    print(
        """
Small GEMM comparison
---------------------
Your sketch:
  - outer OpenMP blocked loops (cache)
  - inner AVX2/FMA micro-kernel C[4x8] += a[4] * b[8] over k

This formalizes that kernel (with packing + edge handling) and contrasts it
against naive C++, NumPy/BLAS, and a TVM Relax tiled llvm schedule.
"""
    )
    lib = build_shared_lib()
    for n in sizes:
        run_size(lib, n, args.repeat)


if __name__ == "__main__":
    main()
