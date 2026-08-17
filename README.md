# AI Compiler Path

AI 编译器入门练习：从 **TVM Relax / TIRx** 写算子与调度，到用 **MLIR** 定义 Toy 方言并实现常量折叠 Pass。

## 目录结构

```
ai_compiler_path/
└── tvm/
    ├── relax_basic.py                 # Relax IR 入门：(x+y)*(x-y)
    ├── relax_vector_add_llvm.py       # TIRx vector_add → LLVM / Relax VM
    ├── relax_matmul_schedule.py       # matmul tiling + GPU/CPU schedule
    ├── relax_fuse_ops_by_pattern.py   # DPL 算子融合（FuseOpsByPattern）
    ├── mlir_data_parallel.mlir        # StableHLO/GSPMD 数据并行注解（正式示意）
    ├── mlir_data_parallel.py          # 对照讲解 + NumPy 4 卡 DP 仿真
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
| `toy_mlir` | conda 安装的 **MLIR / LLVM 22**（`mlir-tblgen`、`libMLIR`、`clang++`） |

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
python compare_small_gemm.py          # 小型 GEMM：手写 AVX2 分块 vs NumPy vs TVM
python symbolic_shape_inference.py    # 符号形状推理：(b,s,d)@ (d,4d)->(b,s,4d)
```

配套 IR 见 `mlir_data_parallel.mlir`（GSPMD sharding + `stablehlo.all_reduce`）。
手写内核见 `gemm_avx2_blocked.cpp`（OpenMP 分块 + AVX2/FMA 4×8 micro-kernel）。

### 2. MLIR Toy 方言

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
6. **`compare_small_gemm.py`** — 手写 AVX2 分块 GEMM vs NumPy/BLAS vs TVM tile
7. **`symbolic_shape_inference.py`** — 符号维推理规则（matmul/softmax/add），非仅硬编码结果
8. **`toy_mlir/`** — TableGen 定义方言，再写 Pass 做常量折叠

## 参考

- [Apache TVM](https://tvm.apache.org/)
- [Relax Dataflow Pattern Language](https://tvm.apache.org/docs/deep_dive/relax/dpl.html)
- [MLIR](https://mlir.llvm.org/)
- `tvm/toy_mlir/README.md` — Toy 方言构建说明

## License

[MIT](https://opensource.org/licenses/MIT)
