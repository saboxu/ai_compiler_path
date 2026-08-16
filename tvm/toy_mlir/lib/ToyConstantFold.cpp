//===----------------------------------------------------------------------===//
// ToyConstantFold.cpp
//===----------------------------------------------------------------------===//

#include "Toy/ToyOps.h"
#include "Toy/ToyPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

struct ToyConstantFoldPass
    : public PassWrapper<ToyConstantFoldPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyConstantFoldPass)

  StringRef getArgument() const final { return "toy-constant-fold"; }
  StringRef getDescription() const final {
    return "Fold toy.add of two toy.constant operands into one constant";
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();

    SmallVector<toy::AddOp, 8> addOps;
    function.walk([&](toy::AddOp op) { addOps.push_back(op); });

    for (toy::AddOp op : addOps) {
      auto lhsConst =
          dyn_cast_or_null<toy::ConstantOp>(op.getLhs().getDefiningOp());
      auto rhsConst =
          dyn_cast_or_null<toy::ConstantOp>(op.getRhs().getDefiningOp());
      if (!lhsConst || !rhsConst)
        continue;

      auto lhsVal = dyn_cast<DenseElementsAttr>(lhsConst.getValue());
      auto rhsVal = dyn_cast<DenseElementsAttr>(rhsConst.getValue());
      if (!lhsVal || !rhsVal)
        continue;
      if (lhsVal.getType() != rhsVal.getType())
        continue;

      SmallVector<APFloat, 4> newValues;
      auto lhsIt = lhsVal.value_begin<APFloat>();
      auto rhsIt = rhsVal.value_begin<APFloat>();
      auto lhsEnd = lhsVal.value_end<APFloat>();
      for (; lhsIt != lhsEnd; ++lhsIt, ++rhsIt) {
        APFloat v = *lhsIt;
        v.add(*rhsIt, APFloat::rmNearestTiesToEven);
        newValues.push_back(v);
      }

      OpBuilder builder(op);
      auto newAttr = DenseElementsAttr::get(
          cast<ShapedType>(lhsVal.getType()), ArrayRef<APFloat>(newValues));
      auto newConst = toy::ConstantOp::create(builder, op.getLoc(),
                                              op.getType(), newAttr);

      op.replaceAllUsesWith(newConst.getResult());
      op.erase();
      if (lhsConst->use_empty())
        lhsConst.erase();
      if (rhsConst->use_empty())
        rhsConst.erase();
    }
  }
};

} // namespace

void toy::registerToyConstantFoldPass() {
  PassRegistration<ToyConstantFoldPass>();
}
