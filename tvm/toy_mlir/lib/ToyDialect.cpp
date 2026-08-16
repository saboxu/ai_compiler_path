//===----------------------------------------------------------------------===//
// ToyDialect.cpp
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

using namespace mlir;
using namespace toy;

#include "Toy/ToyOpsDialect.cpp.inc"

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
      >();
}
