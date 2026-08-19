# StableHLO TP AllReduce Pass (C++ / MLIR)

This folder contains an out-of-tree C++ MLIR pass that inserts
`stablehlo.all_reduce` after certain `stablehlo.dot_general`/`stablehlo.dot`
matmul ops based on `mhlo.sharding` annotations.

The recommended entrypoint is now a local driver executable:
- `stablehlo-tp-opt`

The old plugin `.so` is still built, but the driver is the reliable path when
the host `stablehlo-opt` and plugin do not share the same pass registry / ABI.

## Important limitations (by design for now)

- Full, general sharding propagation for arbitrary sharding specs is complex
  and depends on StableHLO/MHLO sharding semantics.
- This pass currently implements a conservative heuristic:
  - If both operands of a matmul are annotated as non-replicated
    (`mhlo.sharding` != `{replicated}`), then the matmul output is treated as a
    "partial sum" candidate and the pass inserts `stablehlo.all_reduce`.
  - If at least one operand is replicated (or missing sharding info), the pass
    skips insertion (to avoid the typical column-parallel case).

The code is intentionally written so you can extend:
- sharding parsing (devices/T(...) parsing),
- "partial sum" detection rules using `dot_dimension_numbers`, and
- replica_groups construction.

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
  -stablehlo-tp-allreduce \
  input.mlir -o output.mlir
```

Example:

```bash
./build/stablehlo-tp-opt \
  -stablehlo-tp-allreduce \
  ../tvm/mlir_tensor_parallel_no_allreduce_with_intermediate_sharding.mlir
```

Optional fallback: load the plugin into another driver if you really need
dynamic loading, but this is more sensitive to ABI / registry mismatches.

