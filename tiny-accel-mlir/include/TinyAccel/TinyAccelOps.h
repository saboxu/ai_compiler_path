//===----------------------------------------------------------------------===//
// TinyAccelOps.h
//===----------------------------------------------------------------------===//

#ifndef TINYACCEL_TINYACCELOPS_H
#define TINYACCEL_TINYACCELOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TinyAccel/TinyAccelDialect.h"

#define GET_OP_CLASSES
#include "TinyAccel/TinyAccelOps.h.inc"

#endif // TINYACCEL_TINYACCELOPS_H
