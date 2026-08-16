# Toy MLIR dialect demo

Minimal out-of-tree MLIR dialect (`toy.constant` / `toy.add`) plus a
`--toy-constant-fold` pass. Built against conda MLIR 22.

## Build & run

```bash
./build_and_run.sh
```

Or after a successful build:

```bash
./build/toy-opt examples/constant_fold.mlir
./build/toy-opt examples/constant_fold.mlir --toy-constant-fold
```

Expected after folding: `dense<[1,2]> + dense<[3,4]>` → `dense<[4,6]>`.
