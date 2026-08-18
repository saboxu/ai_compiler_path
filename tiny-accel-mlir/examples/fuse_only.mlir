// Already in tinyaccel dialect — only need fuse + emit.
module {
  func.func @already_lowered(%a: f32, %b: f32, %c: f32) -> f32 {
    %t = tinyaccel.mul %a, %b : f32
    %r = tinyaccel.add %t, %c : f32
    return %r : f32
  }
}
