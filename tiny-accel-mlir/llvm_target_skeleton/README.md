# LLVM Target skeleton (educational — NOT built by build_and_run.sh)

This folder mirrors the first teaching snippet (`TinyAccelTarget : llvm::Target`).

## Why it is separate

A real LLVM CodeGen backend lives **inside `llvm-project`** (or an
out-of-tree target hooked into LLVM's CMake). It needs TableGen for
instructions (`.td` for ISD / InstInfo), instruction selection (`ISel`),
asm printer, MC layer, etc. That is a multi-thousand-line undertaking and
cannot be linked as a tiny standalone binary the way our MLIR dialect can.

So this tree is a **reading / stub reference**, while the runnable pipeline
in `../` is the MLIR dialect + lowering + fuse + ISA dump.

## What you would do in llvm-project

1. Add `llvm/lib/Target/TinyAccel/` with `TinyAccel.td`, `TinyAccelISelLowering`, …
2. Register the target in `LLVMInitializeTinyAccelTarget()` (see
   `TinyAccelTarget.h` stub).
3. Build LLVM with your target enabled.
4. Compile with `llc -mtriple=tinyaccel-unknown-elf -debug-only=isel`
   to watch instruction selection.

## Relation to the MLIR pipeline

```
arith / high-level IR
    --convert-arith-to-tinyaccel-->   tinyaccel.*     (this repo, runnable)
    --tinyaccel-fuse-mul-add------>   tinyaccel.mac
    --tinyaccel-emit-isa---------->   textual ISA    (toy codegen)

Later / separately:
    tinyaccel or LLVM IR
    --llc ISel------------------->   TinyAccel machine instrs  (this skeleton)
```

Practice tip from the notes: start with ADD/MUL/LOAD only, then grow the
ISA; use `-debug-only=isel` once the LLVM target actually builds.
