//===----------------------------------------------------------------------===//
// toy-opt.cpp - tiny driver for the Toy dialect demo
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/ToyPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, toy::ToyDialect>();
  toy::registerToyConstantFoldPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Toy dialect optimizer driver\n", registry));
}
