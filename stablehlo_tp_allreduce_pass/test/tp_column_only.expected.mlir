module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_column_only(%arg0: tensor<16x8xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<8x16xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> tensor<16x4xf32> {
    %0 = stablehlo.custom_call @SPMD_shard_to_full_shape_inverse(%arg1) {mhlo.sharding = "{devices=[4]<=[4]}"} : (tensor<8x16xf32>) -> tensor<8x4xf32>
    %1 = stablehlo.dot %arg0, %0 : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>
    return %1 : tensor<16x4xf32>
  }
}

