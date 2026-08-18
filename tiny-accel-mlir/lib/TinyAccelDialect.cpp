//===----------------------------------------------------------------------===//
// TinyAccelDialect.cpp
//===----------------------------------------------------------------------===//

#include "TinyAccel/TinyAccelDialect.h"
#include "TinyAccel/TinyAccelOps.h"

using namespace mlir;
using namespace tinyaccel;

#include "TinyAccel/TinyAccelOpsDialect.cpp.inc"

void TinyAccelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TinyAccel/TinyAccelOps.cpp.inc"
      >();
}
