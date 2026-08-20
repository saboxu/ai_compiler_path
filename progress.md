# Progress

## 2026-08-20

- Improved TP AllReduce pass detection to use `dot_dimension_numbers` and
  global/local RHS shape comparison along contracting dims, instead of the
  previous "both operands non-replicated" sharding heuristic.
- Motivation: column-parallel matmuls should never get an all_reduce, even
  when intermediate activations carry sharding annotations; row-parallel matmuls
  should be detected from Megatron-style weight sharding on the contracting dim.
- Solution:
  - parse contracting dims for `stablehlo.dot` / `stablehlo.dot_general`,
  - trace global RHS shapes through `SPMD_shard_to_full_shape_inverse`, and
  - add `test/` regression inputs, expected outputs, and `run_regression.sh`.
- Status: pass + regression tests verified locally via `stablehlo-tp-opt`.

## 2026-08-19

- Added a local StableHLO driver path in `stablehlo_tp_allreduce_pass` so the
  TP AllReduce pass can be validated without relying on plugin loading into a
  separately built `stablehlo-opt`.
- Motivation: the plugin path was loading the shared object but still failed to
  expose the pass reliably because host/plugin registry and ABI expectations
  were drifting across builds.
- Solution:
  - exported `registerStablehloTPAllReducePass()` through a small header,
  - added `src/StablehloTPOptMain.cpp` to register core MLIR + StableHLO
    dialects and the TP pass, and
  - updated `CMakeLists.txt`, `build.sh`, and `README.md` to build and run
    `build/bin/stablehlo-tp-opt`.
- Status: implementation wired up; next step is to finish linking/build
  validation and run the driver on the sample StableHLO input.

