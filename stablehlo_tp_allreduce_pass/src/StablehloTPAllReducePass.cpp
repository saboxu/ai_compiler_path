#include "StablehloTPAllReducePass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static std::optional<std::string> getMhloShardingString(Value v) {
  // We treat "sharding on a value" as "sharding on its defining op" for
  // OpResults. BlockArgument sharding is ignored for now.
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto attr = defOp->getAttr("mhlo.sharding")) {
    // `mhlo.sharding` is not guaranteed to be a plain StringAttr across
    // StableHLO/MHLO versions; print the attribute to recover the sharding
    // spec string used by the heuristic parsing below.
    if (auto strAttr = llvm::dyn_cast<StringAttr>(attr))
      return strAttr.getValue().str();

    std::string printed;
    llvm::raw_string_ostream os(printed);
    attr.print(os);
    os.flush();
    return printed;
  }
  return std::nullopt;
}

static bool isReplicatedSharding(const std::optional<std::string> &sharding) {
  if (!sharding.has_value())
    return true; // Missing sharding info => assume replicated.
  return sharding->find("replicated") != std::string::npos;
}

static std::optional<int64_t> parseNumDevicesFromSharding(
    const std::optional<std::string> &sharding) {
  if (!sharding.has_value())
    return std::nullopt;

  // Examples observed in StableHLO sharding strings:
  //   "{devices=[4]<=[4]}"
  //   "{devices=[4,1]<=[4]}"
  //
  // Heuristic: extract all ints inside `devices=[...]` and multiply them.
  llvm::StringRef s = sharding->c_str();
  llvm::StringRef key = "devices=[";
  size_t pos = s.find(key);
  if (pos == llvm::StringRef::npos) {
    return std::nullopt;
  }
  size_t start = pos + key.size();
  size_t end = s.find(']', start);
  if (end == llvm::StringRef::npos || end <= start) {
    return std::nullopt;
  }

  llvm::StringRef inside = s.slice(start, end);

  int64_t prod = 1;
  bool any = false;
  while (!inside.empty()) {
    llvm::StringRef token = inside.take_until([](char c) { return c == ','; });
    token = token.trim();
    if (!token.empty()) {
      int64_t v = 0;
      if (!token.getAsInteger(10, v)) {
        prod *= v;
        any = true;
      }
    }
    if (auto commaPos = inside.find(',');
        commaPos != llvm::StringRef::npos) {
      inside = inside.drop_front(commaPos + 1);
    } else {
      inside = {};
    }
  }

  if (!any) {
    return std::nullopt;
  }

  return std::optional<int64_t>(prod);
}

static bool alreadyHasAllReduceOn(Value v) {
  for (auto &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user)
      continue;
    if (user->getName().getStringRef() == "stablehlo.all_reduce")
      return true;
  }
  return false;
}

/// Create a stablehlo.all_reduce that uses stablehlo.add as computation.
///
/// This builds unknown ops by name; correctness relies on the driver being
/// StableHLO/MHLO aware so the ops are registered/verified.
static Operation *createStablehloAllReduce(
    OpBuilder &builder, Location loc, Value operand, RankedTensorType resultTy,
    int64_t numReplicas) {
  auto elemType = resultTy.getElementType();
  auto scalarTensorTy = RankedTensorType::get({}, elemType);

  // replica_groups: dense<[[0,1,2,...]]> : tensor<1xNxi64>
  SmallVector<int64_t> groups;
  groups.reserve(numReplicas);
  for (int64_t i = 0; i < numReplicas; ++i)
    groups.push_back(i);

  auto repGroupsTy = RankedTensorType::get({1, numReplicas},
                                           builder.getI64Type());
  auto replicaGroupsAttr = DenseIntElementsAttr::get(
      repGroupsTy, groups);

  // Build the all_reduce operation with a computation region:
  //   ^bb0(%lhs: tensor<elem>, %rhs: tensor<elem>):
  //     %sum = stablehlo.add %lhs, %rhs : tensor<elem>
  //     stablehlo.return %sum : tensor<elem> -> ()
  //
  // stablehlo.all_reduce is defined with a single computation region.
  OperationState st(loc, "stablehlo.all_reduce");
  st.addOperands(operand);
  st.addTypes(resultTy);
  st.addAttribute("replica_groups", replicaGroupsAttr);
  st.addRegion();

  Operation *allReduceOp = builder.create(st);

  Region &computation = allReduceOp->getRegion(0);
  computation.push_back(new Block());
  Block *block = &computation.front();
  block->addArgument(scalarTensorTy, loc); // lhs
  block->addArgument(scalarTensorTy, loc); // rhs

  OpBuilder bodyBuilder(builder);
  bodyBuilder.setInsertionPointToStart(block);
  Value lhs = block->getArgument(0);
  Value rhs = block->getArgument(1);

  // %sum = stablehlo.add %lhs, %rhs : tensor<elem>
  OperationState addSt(loc, "stablehlo.add");
  addSt.addOperands({lhs, rhs});
  addSt.addTypes(scalarTensorTy);
  Operation *addOp = bodyBuilder.create(addSt);
  Value sum = addOp->getResult(0);

  // stablehlo.return %sum : tensor<elem>
  OperationState retSt(loc, "stablehlo.return");
  retSt.addOperands(sum);
  bodyBuilder.create(retSt);

  return allReduceOp;
}

struct InsertStablehloAllReduceAfterMatmul
    : public PassWrapper<InsertStablehloAllReduceAfterMatmul,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      InsertStablehloAllReduceAfterMatmul)

  StringRef getArgument() const final { return "stablehlo-tp-allreduce"; }
  StringRef getDescription() const final {
    return "Insert stablehlo.all_reduce after sharded matmuls (heuristic TP)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    SmallVector<Operation *> dotOpsToProcess;
    func.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "stablehlo.dot_general" || name == "stablehlo.dot")
        dotOpsToProcess.push_back(op);
    });

    OpBuilder builder(ctx);
    for (Operation *dotOp : dotOpsToProcess) {
      if (!dotOp->getNumResults())
        continue;
      Value dotRes = dotOp->getResult(0);

      // Skip if already all-reduced.
      if (alreadyHasAllReduceOn(dotRes))
        continue;

      // Must be tensor result.
      auto resultTy =
          llvm::dyn_cast<RankedTensorType>(dotRes.getType());
      if (!resultTy)
        continue;

      // Heuristic partial-sum candidate:
      // - dot_general operands are expected to be (lhs, rhs)
      // - require both operands to be annotated as non-replicated
      //   (missing sharding => treated as replicated).
      if (dotOp->getNumOperands() < 2)
        continue;
      Value lhs = dotOp->getOperand(0);
      Value rhs = dotOp->getOperand(1);

      auto lhsSharding = getMhloShardingString(lhs);
      auto rhsSharding = getMhloShardingString(rhs);
      bool replicatedLhs = isReplicatedSharding(lhsSharding);
      bool replicatedRhs = isReplicatedSharding(rhsSharding);

      // Determine num replicas from sharding on either operand.
      auto nFromLhs = parseNumDevicesFromSharding(lhsSharding);
      auto nFromRhs = parseNumDevicesFromSharding(rhsSharding);
      int64_t numReplicas = 0;
      if (nFromLhs.has_value())
        numReplicas = *nFromLhs;
      else if (nFromRhs.has_value())
        numReplicas = *nFromRhs;

      if (replicatedLhs || replicatedRhs) {
        // Likely column-parallel case: one side replicated.
        continue;
      }

      if (numReplicas <= 1)
        continue;

      // Insert after dotOp within the same block.
      builder.setInsertionPointAfter(dotOp);
      Operation *allReduceOp = createStablehloAllReduce(
          builder, dotOp->getLoc(), dotRes, resultTy, numReplicas);
      Value allReduceRes = allReduceOp->getResult(0);

      // Replace all uses of dot result with all-reduced result, except the
      // new all_reduce op itself (which should keep the original operand).
      dotRes.replaceAllUsesExcept(allReduceRes, allReduceOp);
    }
  }
};

} // namespace

void mlir::registerStablehloTPAllReducePass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<InsertStablehloAllReduceAfterMatmul>();
  });
}

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION, "stablehlo_tp_allreduce_pass", "0.1",
      []() {
        mlir::registerStablehloTPAllReducePass();
      }};
}

