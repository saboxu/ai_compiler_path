/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ToyDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace toy {

/// A toy dialect for MLIR learning
/// The Toy dialect models a simple tensor-based language for
///     educational purposes in MLIR compiler construction.
class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};

} // namespace toy
MLIR_DECLARE_EXPLICIT_TYPE_ID(::toy::ToyDialect)
