module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_tp_two_layer(%arg0: tensor<16x8xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<8x16xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}, %arg2: tensor<16x6xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> tensor<16x6xf32> {
    %0 = stablehlo.custom_call @SPMD_shard_to_full_shape_inverse(%arg1) {mhlo.sharding = "{devices=[4]<=[4]}"} : (tensor<8x16xf32>) -> tensor<8x4xf32>
    %1 = stablehlo.custom_call @SPMD_shard_to_full_shape_inverse(%arg2) {mhlo.sharding = "{devices=[4]<=[4]}"} : (tensor<16x6xf32>) -> tensor<4x6xf32>
    %2 = stablehlo.dot %arg0, %0 {mhlo.sharding = "{devices=[4]<=[4]}"} : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>
    %3 = stablehlo.dot %2, %1 : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>
    %4 = "stablehlo.all_reduce"(%3) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<16x6xf32>) -> tensor<16x6xf32>
    return %4 : tensor<16x6xf32>
  }
}

