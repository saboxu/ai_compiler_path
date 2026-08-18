//===----------------------------------------------------------------------===//
// EmitSimpleISA.cpp
//
// "Codegen" for the teaching accelerator: print a tiny textual ISA.
// Real backends would emit binary / LLVM MIR / vendor objects instead.
//===----------------------------------------------------------------------===//

#include "TinyAccel/TinyAccelOps.h"
#include "TinyAccel/TinyAccelPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static std::string valName(Value v, DenseMap<Value, unsigned> &ids) {
  auto it = ids.find(v);
  if (it != ids.end())
    return "r" + std::to_string(it->second);
  if (auto barg = dyn_cast<BlockArgument>(v))
    return "arg" + std::to_string(barg.getArgNumber());
  unsigned id = ids.size();
  ids[v] = id;
  return "r" + std::to_string(id);
}

struct EmitSimpleISAPass
    : public PassWrapper<EmitSimpleISAPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitSimpleISAPass)

  StringRef getArgument() const final { return "tinyaccel-emit-isa"; }
  StringRef getDescription() const final {
    return "Emit a textual ISA listing for tinyaccel ops (teaching codegen)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    raw_ostream &os = llvm::errs();
    os << "; === TinyAccel simple ISA dump ===\n";

    for (auto func : module.getOps<func::FuncOp>()) {
      os << "FUNC " << func.getSymName() << "\n";
      DenseMap<Value, unsigned> ids;
      // Pre-assign names for results in order.
      func.walk([&](Operation *op) {
        for (Value r : op->getResults())
          (void)valName(r, ids);
      });

      func.walk([&](Operation *op) {
        if (auto c = dyn_cast<tinyaccel::ConstantOp>(op)) {
          os << "  " << valName(c.getResult(), ids) << " = CONST "
             << c.getValue().convertToFloat() << "\n";
        } else if (auto a = dyn_cast<tinyaccel::AddOp>(op)) {
          os << "  " << valName(a.getResult(), ids) << " = ADD "
             << valName(a.getLhs(), ids) << ", " << valName(a.getRhs(), ids)
             << "\n";
        } else if (auto m = dyn_cast<tinyaccel::MulOp>(op)) {
          os << "  " << valName(m.getResult(), ids) << " = MUL "
             << valName(m.getLhs(), ids) << ", " << valName(m.getRhs(), ids)
             << "\n";
        } else if (auto mac = dyn_cast<tinyaccel::MacOp>(op)) {
          os << "  " << valName(mac.getResult(), ids) << " = MAC "
             << valName(mac.getA(), ids) << ", " << valName(mac.getB(), ids)
             << ", " << valName(mac.getC(), ids) << "\n";
        } else if (auto ld = dyn_cast<tinyaccel::LoadArgOp>(op)) {
          os << "  " << valName(ld.getResult(), ids) << " = LOAD_ARG "
             << ld.getIndex() << "\n";
        }
      });
      os << "ENDFUNC\n";
    }
    os << "; === end ISA dump ===\n";
  }
};

} // namespace

void tinyaccel::registerEmitSimpleISAPass() {
  PassRegistration<EmitSimpleISAPass>();
}

void tinyaccel::registerAllPasses() {
  registerConvertArithToTinyAccelPass();
  registerFuseMulAddPass();
  registerEmitSimpleISAPass();
}
