// Input in *standard* arith dialect (before lowering).
// Pipeline:
//   --convert-arith-to-tinyaccel
//   --tinyaccel-fuse-mul-add
//   --convert-tinyaccel-to-arith   (then mlir-opt/llc → native x86)

module {
  func.func @dot_like(%a: f32, %b: f32, %c: f32) -> f32 {
    %t = arith.mulf %a, %b : f32
    %r = arith.addf %t, %c : f32
    return %r : f32
  }
}
