//===----------------------------------------------------------------------===//
// FuseMulAddPass.cpp
//
// Hardware-oriented optimization: mul + add --> mac
//   %t = tinyaccel.mul %a, %b
//   %r = tinyaccel.add %t, %c
// becomes
//   %r = tinyaccel.mac %a, %b, %c
//===----------------------------------------------------------------------===//

#include "TinyAccel/TinyAccelOps.h"
#include "TinyAccel/TinyAccelPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct FuseMulAddPattern : public OpRewritePattern<tinyaccel::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tinyaccel::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    auto mulOp = addOp.getLhs().getDefiningOp<tinyaccel::MulOp>();
    Value c = addOp.getRhs();
    if (!mulOp) {
      mulOp = addOp.getRhs().getDefiningOp<tinyaccel::MulOp>();
      c = addOp.getLhs();
    }
    if (!mulOp)
      return failure();
    // Only fuse if the mul has a single use (the add).
    if (!mulOp->hasOneUse())
      return failure();

    rewriter.replaceOpWithNewOp<tinyaccel::MacOp>(
        addOp, addOp.getType(), mulOp.getLhs(), mulOp.getRhs(), c);
    rewriter.eraseOp(mulOp);
    return success();
  }
};

struct FuseMulAddPass
    : public PassWrapper<FuseMulAddPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMulAddPass)

  StringRef getArgument() const final { return "tinyaccel-fuse-mul-add"; }
  StringRef getDescription() const final {
    return "Fuse tinyaccel.mul + tinyaccel.add into tinyaccel.mac";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMulAddPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void tinyaccel::registerFuseMulAddPass() {
  PassRegistration<FuseMulAddPass>();
}
