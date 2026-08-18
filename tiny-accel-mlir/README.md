# tiny-accel-mlir

最小可运行的 MLIR **加速器后端骨架**（方言 `tinyaccel`）：

`arith` → `tinyaccel` → fuse MAC → LLVM IR → **本机 x86-64 机器码**。

| 步骤 | 落点 |
|------|------|
| Lowering：`arith.addf` → `tinyaccel.add` | `--convert-arith-to-tinyaccel` |
| 优化：`mul+add` → `mac` | `--tinyaccel-fuse-mul-add` |
| 合法化回 CPU | `--convert-tinyaccel-to-arith` |
| 真 ISel / 可执行文件 | `./compile_native.sh`（LLVM x86 Target + clang） |
| 自研芯片 Target | `llvm_target_skeleton/`（阅读用；本机不需要） |

> **Lowering** = 把高层 IR 改写成更靠近硬件的 IR：`arith.*` → `tinyaccel.*`。  
> 本机没有自定义 NPU，codegen 走现成 **x86 LLVM backend**。

## 依赖

```bash
sudo apt-get install -y mlir-22-tools libmlir-22-dev llvm-22-dev clang-22
```

默认 `LLVM_ROOT=/usr/lib/llvm-22`。

## 一键构建 + 跑通机器码

```bash
cd /home/xuzhiyuan/ai_compiler_path/tiny-accel-mlir
./build_and_run.sh
```

会看到 IR 变换，最后编译并执行：

```text
dot_like(2, 3, 4) = 10
```

产物在 `build/native/`：`dot_like.ll`、`dot_like.s`、`run_dot_like`。

单独再跑：

```bash
./compile_native.sh 2 3 4
```

## 手动看 IR

```bash
OPT=./build/tinyaccel-opt
$OPT examples/mul_add.mlir --convert-arith-to-tinyaccel
$OPT examples/mul_add.mlir --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add
$OPT examples/mul_add.mlir \
  --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add --convert-tinyaccel-to-arith
```

## 目录

```
tiny-accel-mlir/
├── include/TinyAccel/
├── lib/                    # Dialect / Lowering / Fuse / 合法化到 arith
├── tools/tinyaccel-opt.cpp
├── examples/
├── compile_native.sh       # LLVM IR → x86-64 ELF
├── llvm_target_skeleton/   # 自研 ISA 的 LLVM Target 说明（不链接）
└── build_and_run.sh
```

## 和下一步

接到 TVM 时，可以把这条管线当成 BYOC 外包编译器；本机执行则继续走 LLVM x86。
