module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_dot_general_row(%arg0: tensor<16x4xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}, %arg1: tensor<16x6xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> tensor<16x6xf32> {
    %0 = stablehlo.custom_call @SPMD_shard_to_full_shape_inverse(%arg1) {mhlo.sharding = "{devices=[4]<=[4]}"} : (tensor<16x6xf32>) -> tensor<4x6xf32>
    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>
    %2 = "stablehlo.all_reduce"(%1) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<16x6xf32>) -> tensor<16x6xf32>
    return %2 : tensor<16x6xf32>
  }
}

