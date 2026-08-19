#include "StablehloTPAllReducePass.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerStablehloTPAllReducePass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "StableHLO TP optimizer driver\n", registry));
}

