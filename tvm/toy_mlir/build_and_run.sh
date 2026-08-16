#!/usr/bin/env bash
# Build and run the Toy constant-fold demo against conda MLIR 22.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_PREFIX="${CONDA_PREFIX:-/Users/saboxu/miniconda3}"
BUILD="$ROOT/build"
INC_GEN="$BUILD/include/Toy"
mkdir -p "$INC_GEN" "$BUILD/obj"

TBLGEN="$CONDA_PREFIX/bin/mlir-tblgen"
TD="$ROOT/include/Toy/ToyOps.td"
MLIR_INC="$CONDA_PREFIX/include"
CXX="${CXX:-clang++}"
LLVM_CONFIG="$CONDA_PREFIX/bin/llvm-config"

echo "[1/3] TableGen..."
"$TBLGEN" --gen-dialect-decls "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -dialect=toy -o "$INC_GEN/ToyOpsDialect.h.inc"
"$TBLGEN" --gen-dialect-defs "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -dialect=toy -o "$INC_GEN/ToyOpsDialect.cpp.inc"
"$TBLGEN" --gen-op-decls "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -o "$INC_GEN/ToyOps.h.inc"
"$TBLGEN" --gen-op-defs "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -o "$INC_GEN/ToyOps.cpp.inc"

COMMON_FLAGS=(
  -std=c++17
  -fno-exceptions
  -fno-rtti
  -fPIC
  -I"$ROOT/include"
  -I"$BUILD/include"
  -I"$MLIR_INC"
)

echo "[2/3] Compile..."
OBJS=()
for src in \
  "$ROOT/lib/ToyDialect.cpp" \
  "$ROOT/lib/ToyOps.cpp" \
  "$ROOT/lib/ToyConstantFold.cpp" \
  "$ROOT/tools/toy-opt.cpp"
do
  obj="$BUILD/obj/$(basename "${src%.cpp}").o"
  echo "  CC $(basename "$src")"
  "$CXX" "${COMMON_FLAGS[@]}" -c "$src" -o "$obj"
  OBJS+=("$obj")
done

echo "[3/3] Link toy-opt..."
"$CXX" "${OBJS[@]}" -o "$BUILD/toy-opt" \
  -L"$CONDA_PREFIX/lib" \
  -Wl,-rpath,"$CONDA_PREFIX/lib" \
  "$CONDA_PREFIX/lib/libMLIR.22.1.dylib" \
  "$CONDA_PREFIX/lib/libLLVM.dylib" \
  -lc++ -lm

echo "Built: $BUILD/toy-opt"
echo
echo "===== Before pass ====="
"$BUILD/toy-opt" "$ROOT/examples/constant_fold.mlir"
echo
echo "===== After --toy-constant-fold ====="
"$BUILD/toy-opt" "$ROOT/examples/constant_fold.mlir" --toy-constant-fold
