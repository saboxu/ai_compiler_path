#!/usr/bin/env bash
# Build & demo the TinyAccel accelerator teaching backend (MLIR 22).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LLVM_ROOT="${LLVM_ROOT:-/usr/lib/llvm-22}"
BUILD="$ROOT/build"
INC_GEN="$BUILD/include/TinyAccel"
mkdir -p "$INC_GEN" "$BUILD/obj"

TBLGEN="$LLVM_ROOT/bin/mlir-tblgen"
CXX="${CXX:-$LLVM_ROOT/bin/clang++}"
TD="$ROOT/include/TinyAccel/TinyAccelOps.td"
MLIR_INC="$LLVM_ROOT/include"

if [[ ! -x "$TBLGEN" ]]; then
  echo "error: mlir-tblgen not found at $TBLGEN" >&2
  echo "Install: sudo apt-get install -y mlir-22-tools libmlir-22-dev llvm-22-dev clang-22" >&2
  exit 1
fi

echo "[1/4] TableGen..."
"$TBLGEN" --gen-dialect-decls "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -dialect=tinyaccel -o "$INC_GEN/TinyAccelOpsDialect.h.inc"
"$TBLGEN" --gen-dialect-defs "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -dialect=tinyaccel -o "$INC_GEN/TinyAccelOpsDialect.cpp.inc"
"$TBLGEN" --gen-op-decls "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -o "$INC_GEN/TinyAccelOps.h.inc"
"$TBLGEN" --gen-op-defs "$TD" -I"$MLIR_INC" -I"$ROOT/include" \
  -o "$INC_GEN/TinyAccelOps.cpp.inc"

COMMON_FLAGS=(
  -std=c++17
  -fno-exceptions
  -fno-rtti
  -fPIC
  -I"$ROOT/include"
  -I"$BUILD/include"
  -I"$MLIR_INC"
)

echo "[2/4] Compile..."
SRCS=(
  "$ROOT/lib/TinyAccelDialect.cpp"
  "$ROOT/lib/TinyAccelOps.cpp"
  "$ROOT/lib/LowerArithToTinyAccel.cpp"
  "$ROOT/lib/LowerTinyAccelToArith.cpp"
  "$ROOT/lib/FuseMulAddPass.cpp"
  "$ROOT/tools/tinyaccel-opt.cpp"
)
OBJS=()
for src in "${SRCS[@]}"; do
  obj="$BUILD/obj/$(basename "${src%.cpp}").o"
  echo "  CC $(basename "$src")"
  "$CXX" "${COMMON_FLAGS[@]}" -c "$src" -o "$obj"
  OBJS+=("$obj")
done

echo "[3/4] Link tinyaccel-opt..."
# Prefer the umbrella shared lib when present.
MLIR_LIB="$LLVM_ROOT/lib/libMLIR.so"
LLVM_LIB="$LLVM_ROOT/lib/libLLVM.so"
"$CXX" "${OBJS[@]}" -o "$BUILD/tinyaccel-opt" \
  -L"$LLVM_ROOT/lib" \
  -Wl,-rpath,"$LLVM_ROOT/lib" \
  "$MLIR_LIB" "$LLVM_LIB" \
  -lpthread -lm -ldl

echo "Built: $BUILD/tinyaccel-opt"
echo
OPT="$BUILD/tinyaccel-opt"
EX="$ROOT/examples/mul_add.mlir"

echo "[4/4] Demo pipeline on examples/mul_add.mlir"
echo
echo "===== (0) original arith ====="
"$OPT" "$EX"
echo
echo "===== (1) after --convert-arith-to-tinyaccel  (lowering) ====="
"$OPT" "$EX" --convert-arith-to-tinyaccel
echo
echo "===== (2) after lowering + --tinyaccel-fuse-mul-add ====="
"$OPT" "$EX" --convert-arith-to-tinyaccel --tinyaccel-fuse-mul-add
echo
echo "[4/4] Native x86-64 codegen (LLVM ISel → ELF)"
"$ROOT/compile_native.sh" 2 3 4
