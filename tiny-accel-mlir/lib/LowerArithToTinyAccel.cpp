//===----------------------------------------------------------------------===//
// LowerArithToTinyAccel.cpp
//
// Lowering: arith.addf / arith.mulf  -->  tinyaccel.add / tinyaccel.mul
// This is the MLIR "ConversionPattern" story from the teaching snippets.
//===----------------------------------------------------------------------===//

#include "TinyAccel/TinyAccelOps.h"
#include "TinyAccel/TinyAccelPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct AddfToTinyAccel : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isF32())
      return failure();
    rewriter.replaceOpWithNewOp<tinyaccel::AddOp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
    return success();
  }
};

struct MulfToTinyAccel : public OpRewritePattern<arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isF32())
      return failure();
    rewriter.replaceOpWithNewOp<tinyaccel::MulOp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
    return success();
  }
};

struct ConstantToTinyAccel : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto attr = dyn_cast<FloatAttr>(op.getValue());
    if (!attr || !op.getType().isF32())
      return failure();
    rewriter.replaceOpWithNewOp<tinyaccel::ConstantOp>(op, op.getType(), attr);
    return success();
  }
};

struct ConvertArithToTinyAccelPass
    : public PassWrapper<ConvertArithToTinyAccelPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArithToTinyAccelPass)

  StringRef getArgument() const final { return "convert-arith-to-tinyaccel"; }
  StringRef getDescription() const final {
    return "Lower arith.{constant,addf,mulf} (f32) into tinyaccel dialect ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tinyaccel::TinyAccelDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantToTinyAccel, AddfToTinyAccel, MulfToTinyAccel>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void tinyaccel::registerConvertArithToTinyAccelPass() {
  PassRegistration<ConvertArithToTinyAccelPass>();
}
