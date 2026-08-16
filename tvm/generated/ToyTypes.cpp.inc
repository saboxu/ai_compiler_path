/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::toy::F64TensorType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::toy::F64TensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::toy::F64TensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::toy::F64TensorType>([&](auto t) {
      printer << ::toy::F64TensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace toy {

namespace detail {

struct F64TensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>>;
  F64TensorTypeStorage(::llvm::ArrayRef<int64_t> shape) : shape(std::move(shape)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static F64TensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<F64TensorTypeStorage>()) F64TensorTypeStorage(std::move(shape));
  }

  ::llvm::ArrayRef<int64_t> shape;
};

} // namespace detail
F64TensorType F64TensorType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape) {
  return Base::get(context, std::move(shape));
}

::mlir::Type F64TensorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_shape;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'shape'
  _result_shape = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
  if (::mlir::failed(_result_shape)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Toy_F64Tensor parameter 'shape' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_shape));
  return F64TensorType::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_shape)));
}

void F64TensorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getShape());
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> F64TensorType::getShape() const {
  return getImpl()->shape;
}


} // namespace toy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::toy::F64TensorType)

#endif // GET_TYPEDEF_CLASSES

