// Input in *standard* arith dialect (before lowering).
// Pipeline:
//   --convert-arith-to-tinyaccel
//   --tinyaccel-fuse-mul-add
//   --tinyaccel-emit-isa

module {
  func.func @dot_like(%a: f32, %b: f32, %c: f32) -> f32 {
    %t = arith.mulf %a, %b : f32
    %r = arith.addf %t, %c : f32
    return %r : f32
  }
}
