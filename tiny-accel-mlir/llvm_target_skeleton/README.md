# LLVM Target skeleton (educational — NOT used for native run)

本仓库在这台 x86 服务器上的真代码生成走 **LLVM 自带的 x86 Target**
（见 `../compile_native.sh`）。这个目录只说明：如果要为**自研 ISA**
做 ISel，需要把 Target 接到 `llvm-project` 里。

## Why it is separate

A real LLVM CodeGen backend lives **inside `llvm-project`**. It needs
instruction TableGen, ISel, asm printer, MC layer — not a tiny
standalone binary like `tinyaccel-opt`.

## What you would do in llvm-project

1. Add `llvm/lib/Target/TinyAccel/` with `TinyAccel.td`, `TinyAccelISelLowering`, …
2. Register the target in `LLVMInitializeTinyAccelTarget()` (see
   `TinyAccelTarget.h` stub).
3. Build LLVM with your target enabled.
4. Compile with `llc -mtriple=tinyaccel-unknown-elf -debug-only=isel`

## Relation to the runnable pipeline (this server)

```
arith
    --convert-arith-to-tinyaccel-->   tinyaccel.*
    --tinyaccel-fuse-mul-add------>   tinyaccel.mac
    --convert-tinyaccel-to-arith-->   arith.*
    mlir-opt / llc (x86 Target) -->   native ELF     (compile_native.sh)

For a custom chip, replace the last two steps with this skeleton's ISel.
```
