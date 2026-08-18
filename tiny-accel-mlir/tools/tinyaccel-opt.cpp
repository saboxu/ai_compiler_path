//===----------------------------------------------------------------------===//
// tinyaccel-opt.cpp - driver for the TinyAccel accelerator teaching backend
//===----------------------------------------------------------------------===//

#include "TinyAccel/TinyAccelDialect.h"
#include "TinyAccel/TinyAccelOps.h"
#include "TinyAccel/TinyAccelPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  tinyaccel::TinyAccelDialect>();
  tinyaccel::registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TinyAccel accelerator dialect optimizer\n", registry));
}
