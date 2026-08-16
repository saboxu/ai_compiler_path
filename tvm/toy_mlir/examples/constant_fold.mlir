module {
  func.func @add_constants() -> tensor<2xf64> {
    %0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
    %1 = toy.constant dense<[3.0, 4.0]> : tensor<2xf64>
    %2 = toy.add %0, %1 : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %2 : tensor<2xf64>
  }
}
