#!/usr/bin/env bash
# Compile examples/mul_add.mlir through tinyaccel, then LLVM, to a native binary.
#
#   arith → tinyaccel → fuse mac → arith → llvm ir → x86-64 ELF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LLVM_ROOT="${LLVM_ROOT:-/usr/lib/llvm-22}"
BUILD="$ROOT/build"
OPT="$BUILD/tinyaccel-opt"
A="${1:-2}"
B="${2:-3}"
C="${3:-4}"

if [[ ! -x "$OPT" ]]; then
  echo "error: $OPT missing; run ./build_and_run.sh first" >&2
  exit 1
fi

mkdir -p "$BUILD/native"
IR_MLIR="$BUILD/native/dot_like.tinyaccel.mlir"
ARITH_MLIR="$BUILD/native/dot_like.arith.mlir"
LLVM_MLIR="$BUILD/native/dot_like.llvm.mlir"
LL="$BUILD/native/dot_like.ll"
S="$BUILD/native/dot_like.s"
BIN="$BUILD/native/run_dot_like"

echo "[1/5] tinyaccel lowering + fuse"
"$OPT" "$ROOT/examples/mul_add.mlir" \
  --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add \
  -o "$IR_MLIR"
cat "$IR_MLIR"

echo
echo "[2/5] tinyaccel → arith (CPU legalization; mac expands to mul+add)"
"$OPT" "$IR_MLIR" --convert-tinyaccel-to-arith -o "$ARITH_MLIR"
cat "$ARITH_MLIR"

echo
echo "[3/5] arith/func → LLVM dialect → LLVM IR"
"$LLVM_ROOT/bin/mlir-opt" "$ARITH_MLIR" \
  --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
  -o "$LLVM_MLIR"
"$LLVM_ROOT/bin/mlir-translate" --mlir-to-llvmir "$LLVM_MLIR" -o "$LL"
echo "----- LLVM IR -----"
cat "$LL"

echo
echo "[4/5] llc → x86-64 asm + clang link"
"$LLVM_ROOT/bin/llc" -O2 -march=x86-64 -mattr=+fma "$LL" -o "$S"
echo "----- asm (look for vfmadd / mulss+addss) -----"
cat "$S"
"$LLVM_ROOT/bin/clang" -O2 -o "$BIN" "$LL" "$ROOT/examples/run_dot_like.c"

echo
echo "[5/5] run native binary"
echo "file: $BIN"
file "$BIN"
"$BIN" "$A" "$B" "$C"
echo "expected: $(python3 -c "print(float('$A')*float('$B')+float('$C'))")"
