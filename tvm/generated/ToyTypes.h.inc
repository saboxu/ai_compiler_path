/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace toy {

/// 64-bit floating point tensor type
class F64TensorType;
namespace detail {

struct F64TensorTypeStorage;

} // namespace detail
class F64TensorType : public ::mlir::Type::TypeBase<F64TensorType, ::mlir::Type, detail::F64TensorTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "toy.tensor";
  static constexpr ::llvm::StringLiteral dialectName = "toy";
  static F64TensorType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
};

} // namespace toy
MLIR_DECLARE_EXPLICIT_TYPE_ID(::toy::F64TensorType)

#endif // GET_TYPEDEF_CLASSES

