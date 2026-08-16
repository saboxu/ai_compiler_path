//===----------------------------------------------------------------------===//
// ToyOps.h
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYOPS_H
#define TOY_TOYOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Toy/ToyDialect.h"

#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif // TOY_TOYOPS_H
