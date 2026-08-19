// Same as mlir_tensor_parallel_no_allreduce.mlir but with mhlo.sharding
// attached to the intermediate activation so the C++ heuristic pass can
// detect a row-parallel / partial-sum candidate.

module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_tp_two_layer_no_allreduce_with_intermediate_sharding(
      %x: tensor<16x8xf32> {mhlo.sharding = "{replicated}"},
      %w1: tensor<8x16xf32> {mhlo.sharding = "{devices=[4]<=[4]}"},
      %w2: tensor<16x6xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}
    ) -> tensor<16x6xf32> {

    %w1_local = "stablehlo.custom_call"(%w1) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<8x16xf32>) -> tensor<8x4xf32>

    %w2_local = "stablehlo.custom_call"(%w2) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<16x6xf32>) -> tensor<4x6xf32>

    // Column-parallel matmul: result is sharded.
    %h_local = "stablehlo.dot"(%x, %w1_local) {
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>

    // Row-parallel matmul: produces a partial sum that must be reduced.
    %y_partial = "stablehlo.dot"(%h_local, %w2_local)
      : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>

    return %y_partial : tensor<16x6xf32>
  }
}

