# tiny-accel-mlir

最小可运行的 MLIR **加速器后端骨架**（方言 `tinyaccel`）：lower arith → fuse MAC → emit teaching ISA。

对应课堂里那三段拼盘，这里拆成**能编过、能跑通**的完整小项目：

| 教学片段 | 本仓库落点 | 是否可执行 |
|----------|------------|------------|
| MLIR lowering：`arith.addf` → `tinyaccel.add` | `--convert-arith-to-tinyaccel` | ✅ |
| MLIR 优化：`mul+add` → `mac` | `--tinyaccel-fuse-mul-add` | ✅ |
| 「Codegen」：落到简单指令 | `--tinyaccel-emit-isa` + `python_sim/` | ✅ |
| LLVM `Target` 注册 / ISel | `llvm_target_skeleton/` | 📖 说明骨架（需进 llvm-project） |

> **Lowering** = 把高层 IR 改写成更靠近硬件的 IR。  
> 本项目里：`arith.*`（标准方言）→ `tinyaccel.*`（加速器方言）就是一次 lowering。

## 依赖（本机已可用 apt）

```bash
sudo apt-get install -y mlir-22-tools libmlir-22-dev llvm-22-dev clang-22
```

默认 `LLVM_ROOT=/usr/lib/llvm-22`。

## 一键构建 + 演示

```bash
cd /home/xuzhiyuan/ai_compiler_path/tiny-accel-mlir
chmod +x build_and_run.sh
./build_and_run.sh
```

你会依次看到：

1. 原始 `arith.mulf` / `arith.addf`
2. lowering 后的 `tinyaccel.mul` / `tinyaccel.add`
3. fuse 后的 `tinyaccel.mac`
4. stderr 上的 textual ISA（`MAC arg0, arg1, arg2`）

## 手动跑

```bash
OPT=./build/tinyaccel-opt
$OPT examples/mul_add.mlir --convert-arith-to-tinyaccel
$OPT examples/mul_add.mlir --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add
$OPT examples/mul_add.mlir \
  --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add --tinyaccel-emit-isa
```

ISA 解释器（不依赖 MLIR）：

```bash
python3 python_sim/run_isa_sim.py --a 2 --b 3 --c 4   # => 10.0
```

## 目录

```
tiny-accel-mlir/
├── include/TinyAccel/     # TableGen + C++ 头
├── lib/                  # Dialect / Lowering / Fuse / EmitISA
├── tools/tinyaccel-opt.cpp
├── examples/             # 输入 .mlir
├── llvm_target_skeleton/ # LLVM Target 教学骨架（不参与链接）
├── python_sim/           # ISA 解释器
└── build_and_run.sh
```

## 实践建议（和笔记对齐）

1. **先只支持少量指令**：本仓库 ISA 只有 `CONST/ADD/MUL/MAC/LOAD_ARG`。
2. **先打通 lowering + fuse + emit**，再考虑进 llvm-project 做真 ISel。
3. 真 LLVM 后端就绪后，用 `llc -debug-only=isel` 观察指令选择（见
   `llvm_target_skeleton/README.md`）。

## 和下一步（TVM BYOC / Target）的关系

- 这里练的是 **MLIR 方言路线**（不绑 TVM）。
- 若接到 TVM：可以把 `tinyaccel-opt` 管道当成 BYOC 外包编译器；或把 emit 的
  ISA/二进制挂进自研 runtime。
