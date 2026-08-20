# StableHLO TP AllReduce Pass (C++ / MLIR)

This folder contains an out-of-tree C++ MLIR pass that inserts
`stablehlo.all_reduce` after row-parallel matmul ops in a Megatron-style
tensor-parallel block.

The recommended entrypoint is a local driver executable:
- `stablehlo-tp-opt`

The old plugin `.so` is still built, but the driver is the reliable path when
the host `stablehlo-opt` and plugin do not share the same pass registry / ABI.

## Detection logic

The pass identifies row-parallel partial sums using `dot_dimension_numbers`:

- For `stablehlo.dot`, default contracting dims are used
  (`lhs: [rank-1]`, `rhs: [0]`).
- For `stablehlo.dot_general`, `dot_dimension_numbers` is parsed from the op.

It then compares the RHS **global** shape (from the function argument or from
the operand of `SPMD_shard_to_full_shape_inverse`) against the local RHS shape.
If a contracting dimension is smaller locally than globally, the matmul is treated
as row-parallel and `stablehlo.all_reduce(SUM)` is inserted.

Column-parallel matmuls shard the RHS on a non-contracting dimension, so they
are skipped.

Replica count is inferred from the global/local shard ratio when possible, with
`mhlo.sharding`'s `devices=[...]` as a fallback.

## Build

Assuming the local StableHLO submodule and its pinned LLVM/MLIR have already
been built:

```bash
cd /home/xuzhiyuan/ai_compiler_path/stablehlo_tp_allreduce_pass
./build.sh
```

Outputs:
- `build/stablehlo-tp-opt`
- `stablehlo_tp_allreduce_pass.so`

## Run

Preferred: run the local driver directly.

```bash
./build/stablehlo-tp-opt \
  --stablehlo-tp-allreduce \
  input.mlir -o output.mlir
```

Example:

```bash
./build/stablehlo-tp-opt \
  --stablehlo-tp-allreduce \
  test/tp_two_layer.mlir
```

## Regression tests

```bash
chmod +x test/run_regression.sh
./test/run_regression.sh
```

Or build and test together:

```bash
RUN_REGRESSION=1 ./build.sh
```

Test layout:
- `test/tp_two_layer.mlir` — column + row matmul; expect one all_reduce
- `test/tp_column_only.mlir` — column matmul only; expect none
- `test/tp_dot_general_row.mlir` — row matmul via `dot_general`
- `test/*.expected.mlir` — golden outputs from `stablehlo-tp-opt`

Optional fallback: load the plugin into another driver if you really need
dynamic loading, but this is more sensitive to ABI / registry mismatches.
