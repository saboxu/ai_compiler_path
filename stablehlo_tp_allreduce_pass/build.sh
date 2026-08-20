#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
LLVM_DIR="${LLVM_DIR:-$ROOT/../third_party/stablehlo/llvm-build/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-$ROOT/../third_party/stablehlo/llvm-build/lib/cmake/mlir}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"

mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR"
cmake --build "$BUILD_DIR" -j"${BUILD_JOBS:-4}"

echo "Built: $BUILD_DIR/stablehlo_tp_allreduce_pass.so"
echo "Built: $BUILD_DIR/stablehlo-tp-opt"

if [[ "${RUN_REGRESSION:-0}" == "1" ]]; then
  RUN_REGRESSION=1 "$ROOT/test/run_regression.sh"
fi

