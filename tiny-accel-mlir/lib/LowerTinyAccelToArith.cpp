//===----------------------------------------------------------------------===//
// LowerTinyAccelToArith.cpp
//
// CPU codegen path: tinyaccel.*  -->  arith.*  (then upstream LLVM lowering).
// tinyaccel.mac expands to mul + add; x86 ISel may re-fuse it into FMA.
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

struct TinyAccelAddToArith : public OpRewritePattern<tinyaccel::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tinyaccel::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

struct TinyAccelMulToArith : public OpRewritePattern<tinyaccel::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tinyaccel::MulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

struct TinyAccelConstToArith : public OpRewritePattern<tinyaccel::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tinyaccel::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto cst = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getValueAttr());
    rewriter.replaceOp(op, cst.getResult());
    return success();
  }
};

struct TinyAccelMacToArith : public OpRewritePattern<tinyaccel::MacOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tinyaccel::MacOp op,
                                PatternRewriter &rewriter) const override {
    // a * b + c  — real x86 codegen; -mfma may select vfmaddss.
    auto mul = rewriter.create<arith::MulFOp>(op.getLoc(), op.getA(), op.getB());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, mul, op.getC());
    return success();
  }
};

struct ConvertTinyAccelToArithPass
    : public PassWrapper<ConvertTinyAccelToArithPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTinyAccelToArithPass)

  StringRef getArgument() const final { return "convert-tinyaccel-to-arith"; }
  StringRef getDescription() const final {
    return "Lower tinyaccel ops back to arith so LLVM/clang can emit native code";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TinyAccelAddToArith, TinyAccelMulToArith, TinyAccelConstToArith,
                 TinyAccelMacToArith>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void tinyaccel::registerConvertTinyAccelToArithPass() {
  PassRegistration<ConvertTinyAccelToArithPass>();
}

void tinyaccel::registerAllPasses() {
  registerConvertArithToTinyAccelPass();
  registerConvertTinyAccelToArithPass();
  registerFuseMulAddPass();
}
