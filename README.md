# AI Compiler Path

AI 编译器入门练习：从 **TVM Relax / TIRx** 写算子与调度，到用 **MLIR** 定义 Toy 方言并实现常量折叠 Pass。

## 目录结构

```
ai_compiler_path/
└── tvm/
    ├── relax_basic.py              # Relax IR 入门：(x+y)*(x-y)
    ├── relax_vector_add_llvm.py    # TIRx vector_add → LLVM / Relax VM
    ├── relax_matmul_schedule.py    # matmul tiling + GPU/CPU schedule
    ├── generated/                  # mlir-tblgen 生成的参考产物
    └── toy_mlir/                   # 可编译运行的 out-of-tree Toy 方言
        ├── include/Toy/
        ├── lib/
        ├── tools/toy-opt.cpp
        ├── examples/constant_fold.mlir
        └── build_and_run.sh
```

## 环境依赖

| 部分 | 依赖 |
|------|------|
| Relax / TIRx 脚本 | Python 3、`tvm`（含 Relax）、`numpy`；可选 CUDA |
| `toy_mlir` | conda 安装的 **MLIR / LLVM 22**（`mlir-tblgen`、`libMLIR`、`clang++`） |

`build_and_run.sh` 从 `CONDA_PREFIX`（未设置时为 `/Users/saboxu/miniconda3`）查找 MLIR。

## 快速开始

### 1. TVM Relax

```bash
cd tvm

python relax_basic.py              # 打印 Relax IR
python relax_vector_add_llvm.py    # LLVM IR 片段 + Relax VM 数值校验
python relax_matmul_schedule.py    # schedule；有 CUDA 则上 GPU，否则 llvm
```

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
4. **`toy_mlir/`** — TableGen 定义方言，再写 Pass 做常量折叠

## 参考

- [Apache TVM](https://tvm.apache.org/)
- [MLIR](https://mlir.llvm.org/)
- `tvm/toy_mlir/README.md` — Toy 方言构建说明

## License

[MIT](https://opensource.org/licenses/MIT)
