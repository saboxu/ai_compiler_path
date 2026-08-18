//===----------------------------------------------------------------------===//
// TinyAccelOps.td - Minimal AI-accelerator dialect (add / mul / mac / load)
//===----------------------------------------------------------------------===//

#ifndef TINYACCEL_OPS_TD
#define TINYACCEL_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TinyAccel_Dialect : Dialect {
  let name = "tinyaccel";
  let summary = "Tiny custom accelerator dialect for learning backends";
  let description = [{
    A teaching dialect with a handful of ops (add, mul, mac, load, store, constant).
    Used to demonstrate: dialect + lowering + fuse pass + simple ISA emission.
  }];
  let cppNamespace = "::tinyaccel";
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class TinyAccel_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyAccel_Dialect, mnemonic, traits>;

def TinyAccel_ConstantOp
    : TinyAccel_Op<"constant", [Pure]> {
  let summary = "scalar f32 constant";
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$result);
  let assemblyFormat = "$value attr-dict";
}

def TinyAccel_AddOp
    : TinyAccel_Op<"add", [Pure, AllTypesMatch<["lhs","rhs","result"]>]> {
  let summary = "scalar add (maps to accelerator ADD)";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def TinyAccel_MulOp
    : TinyAccel_Op<"mul", [Pure, AllTypesMatch<["lhs","rhs","result"]>]> {
  let summary = "scalar mul (maps to accelerator MUL)";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

// Fused multiply-add: result = a * b + c  (hardware MAC unit)
def TinyAccel_MacOp
    : TinyAccel_Op<"mac", [Pure, AllTypesMatch<["a", "b", "c", "result"]>]> {
  let summary = "fused multiply-add (maps to accelerator MAC)";
  let arguments = (ins F32:$a, F32:$b, F32:$c);
  let results = (outs F32:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` type($result)
  }];
}

// Educational "load" from a function argument slot id (not a full memref model).
def TinyAccel_LoadArgOp
    : TinyAccel_Op<"load_arg", [Pure]> {
  let summary = "load function argument by index into a register-like SSA value";
  let arguments = (ins I32Attr:$index);
  let results = (outs F32:$result);
  let assemblyFormat = "$index attr-dict `:` type($result)";
}

#endif // TINYACCEL_OPS_TD
