# Progress

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

