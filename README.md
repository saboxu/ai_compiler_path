# AI Compiler Path

AI 编译器入门练习：从 **TVM Relax / TIRx** 写算子与调度，到用 **MLIR** 定义 Toy 方言并实现常量折叠 Pass。

## Clone

```bash
git clone --recursive git@github.com:saboxu/ai_compiler_path.git
# 已 clone 过：
git submodule update --init --recursive
```

`tiny-accel-mlir` 为独立子模块：[saboxu/tiny-accel-mlir](https://github.com/saboxu/tiny-accel-mlir)

## 目录结构

```
ai_compiler_path/
├── tiny-accel-mlir/                   # 迷你加速器 MLIR 后端（tinyaccel 方言）
│   ├── build_and_run.sh
│   ├── compile_native.sh              # 降到 LLVM IR → 本机 x86-64
│   ├── examples/
│   └── llvm_target_skeleton/          # 自研 ISA 的 LLVM Target 说明（不链接）
├── stablehlo_tp_allreduce_pass/      # StableHLO TP：C++ Pass 插入 stablehlo.all_reduce
│   ├── src/                          # out-of-tree pass 代码
│   ├── build.sh                      # 用系统 MLIR/LLVM 22 构建 pass so
│   └── README.md
└── third_party/
    └── stablehlo/                    # openxla/stablehlo 子模块（用于 dialect/ops & 验证）
└── tvm/
    ├── relax_basic.py                 # Relax IR 入门：(x+y)*(x-y)
    ├── relax_vector_add_llvm.py       # TIRx vector_add → LLVM / Relax VM
    ├── relax_matmul_schedule.py       # matmul tiling + GPU/CPU schedule
    ├── relax_fuse_ops_by_pattern.py   # DPL 算子融合（FuseOpsByPattern）
    ├── mlir_data_parallel.mlir        # StableHLO/GSPMD 数据并行注解（正式示意）
    ├── mlir_data_parallel.py          # 对照讲解 + NumPy 4 卡 DP 仿真
    ├── mlir_tensor_parallel.mlir     # StableHLO/GSPMD 张量并行注解（column→row→AllReduce）
    ├── mlir_tensor_parallel.py       # 对照讲解 + NumPy TP 仿真（2-layer matmul）
    ├── gemm_avx2_blocked.cpp          # 手写分块 GEMM + AVX2/FMA 4×8 micro-kernel
    ├── compare_small_gemm.py          # 小型 GEMM：AVX2 / NumPy / TVM 对比
    ├── symbolic_shape_inference.py    # 符号形状推理：sympy + Relax 动态维
    ├── generated/                     # mlir-tblgen 生成的参考产物
    └── toy_mlir/                      # 可编译运行的 out-of-tree Toy 方言
        ├── include/Toy/
        ├── lib/
        ├── tools/toy-opt.cpp
        ├── examples/constant_fold.mlir
        └── build_and_run.sh
```

## 环境依赖

| 部分 | 依赖 |
|------|------|
| Relax / TIRx / 对比脚本 | `tvm/` 下 **uv** 虚拟环境（Python 3.12，`apache-tvm`、`numpy`、`sympy` 等）；可选 CUDA |
| 手写 AVX2 GEMM | `g++`（`-mavx2 -mfma -fopenmp`），由 `compare_small_gemm.py` 自动编译 |
| `tiny-accel-mlir` | **MLIR / LLVM 22**（`mlir-tblgen`、`libMLIR`、`clang++`；可用 apt 安装） |
| `toy_mlir` | 同上，或 conda 安装的 MLIR / LLVM 22 |

### TVM Python 环境

推荐用目录内 uv 环境（已写入 `tvm/pyproject.toml`）：

```bash
cd tvm
# 若尚未创建：
#   uv venv --python 3.12 .venv
#   uv pip install --python .venv/bin/python -e .
source ./env.sh    # 激活 .venv，并可选挂上本地 TVM_HOME
```

若要用自己编译的 TVM 源码而非 PyPI wheel：

```bash
export TVM_HOME=/path/to/your/tvm
source ./env.sh
```

`build_and_run.sh` 从 `CONDA_PREFIX`（未设置时为 `/Users/saboxu/miniconda3`）查找 MLIR。

## 快速开始

### 1. TVM Relax

```bash
cd tvm
source ./env.sh                       # 配置本地 TVM（否则 import tvm 会失败）

python relax_basic.py                 # 打印 Relax IR
python relax_vector_add_llvm.py       # LLVM IR 片段 + Relax VM 数值校验
python relax_matmul_schedule.py       # schedule；有 CUDA 则上 GPU，否则 llvm
python relax_fuse_ops_by_pattern.py   # conv+bn+relu / conv+relu 模式融合
python mlir_data_parallel.py          # StableHLO 数据并行注解讲解 + NumPy 仿真
python mlir_tensor_parallel.py       # StableHLO 张量并行注解（column→row→AllReduce）+ NumPy 仿真
python compare_small_gemm.py          # 小型 GEMM：手写 AVX2 分块 vs NumPy vs TVM
python symbolic_shape_inference.py    # 符号形状推理：(b,s,d)@ (d,4d)->(b,s,4d)
```

配套 IR 见 `mlir_data_parallel.mlir`（GSPMD sharding + `stablehlo.all_reduce`）以及
`mlir_tensor_parallel.mlir`（column→row→AllReduce(SUM)）。
手写内核见 `gemm_avx2_blocked.cpp`（OpenMP 分块 + AVX2/FMA 4×8 micro-kernel）。

### 2. tiny-accel-mlir（MLIR 加速器后端）

```bash
cd tiny-accel-mlir
./build_and_run.sh
# arith → tinyaccel → fuse mac → LLVM x86 机器码（dot_like = 10）
./compile_native.sh 2 3 4
```

### 3. StableHLO TP AllReduce Pass（C++ / local driver）

```bash
cd stablehlo_tp_allreduce_pass
./build.sh
./test/run_regression.sh
```

推荐用本地 driver（不依赖 plugin 加载）：

```bash
./build/stablehlo-tp-opt \
  --stablehlo-tp-allreduce \
  test/tp_two_layer.mlir -o output.mlir
```

Pass 通过 `dot_dimension_numbers` 的 contracting dims + RHS global/local shape
对比识别 row-parallel matmul，并在其后插入 `stablehlo.all_reduce(SUM)`。

### 4. MLIR Toy 方言

```bash
cd tvm/toy_mlir
./build_and_run.sh
```

或构建后手动跑：

```bash
./build/toy-opt examples/constant_fold.mlir
./build/toy-opt examples/constant_fold.mlir --toy-constant-fold
```

`--toy-constant-fold` 会把 `dense<[1,2]> + dense<[3,4]>` 折叠为 `dense<[4,6]>`。

## 学习路径建议

1. **`relax_basic.py`** — `@I.ir_module` / `@R.function` 与基本算子
2. **`relax_vector_add_llvm.py`** — TIRx PrimFunc、`R.call_tir`、LLVM codegen、Relax VM
3. **`relax_matmul_schedule.py`** — `tvm.s_tir.Schedule`：tile、reorder、bind
4. **`relax_fuse_ops_by_pattern.py`** — DPL 模式匹配 + `FuseOpsByPattern` / `FuseTIR`
5. **`mlir_data_parallel.mlir` / `.py`** — 数据并行：sharding 注解、All-Reduce、与单卡等价性
6. **`mlir_tensor_parallel.mlir` / `.py`** — 张量并行：column-parallel / row-parallel / AllReduce(SUM)
7. **`compare_small_gemm.py`** — 手写 AVX2 分块 GEMM vs NumPy/BLAS vs TVM tile
8. **`symbolic_shape_inference.py`** — 符号维推理规则（matmul/softmax/add），非仅硬编码结果
9. **`tiny-accel-mlir/`** — 迷你加速器后端：lowering / fuse / 本机 x86 codegen（+ LLVM Target 骨架说明）
10. **`toy_mlir/`** — TableGen 定义方言，再写 Pass 做常量折叠

## 参考

- [Apache TVM](https://tvm.apache.org/)
- [Relax Dataflow Pattern Language](https://tvm.apache.org/docs/deep_dive/relax/dpl.html)
- [MLIR](https://mlir.llvm.org/)
- `tiny-accel-mlir/README.md` — 迷你加速器后端（lowering / fuse / 本机 x86 codegen）
- `tvm/toy_mlir/README.md` — Toy 方言构建说明

## License

[MIT](https://opensource.org/licenses/MIT)
