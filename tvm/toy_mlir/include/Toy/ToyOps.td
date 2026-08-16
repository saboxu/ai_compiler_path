//===----------------------------------------------------------------------===//
// ToyOps.td - Minimal Toy dialect for constant-fold demo
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A toy dialect for MLIR learning";
  let description = [{
    Minimal tensor dialect used to demonstrate TableGen + a constant-fold pass.
  }];
  let cppNamespace = "::toy";
  // Only builtin types/attrs are used; disable custom dialect printers.
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant", [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "constant operation";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$result);
  // ElementsAttr prints as `dense<...> : tensor<...>`.
  let assemblyFormat = "$value attr-dict";
}

def AddOp : Toy_Op<"add", [Pure]> {
  let summary = "element-wise addition";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

#endif // TOY_OPS_TD
