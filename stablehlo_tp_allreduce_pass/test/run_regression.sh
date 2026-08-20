#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPT="${OPT:-$ROOT/build/stablehlo-tp-opt}"
TEST_DIR="$ROOT/test"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if [[ ! -x "$OPT" ]]; then
  echo "error: stablehlo-tp-opt not found at $OPT (run ./build.sh first)" >&2
  exit 1
fi

run_diff_case() {
  local name=$1
  local input=$2
  local expected=$3
  echo "[regression] $name"
  "$OPT" --stablehlo-tp-allreduce "$input" -o "$TMP"
  diff -u "$expected" "$TMP"
}

run_diff_case "tp_two_layer" \
  "$TEST_DIR/tp_two_layer.mlir" \
  "$TEST_DIR/tp_two_layer.expected.mlir"

run_diff_case "tp_dot_general_row" \
  "$TEST_DIR/tp_dot_general_row.mlir" \
  "$TEST_DIR/tp_dot_general_row.expected.mlir"

run_diff_case "tp_column_only" \
  "$TEST_DIR/tp_column_only.mlir" \
  "$TEST_DIR/tp_column_only.expected.mlir"

echo "All regression tests passed."
