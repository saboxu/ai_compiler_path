// Regression input: two-layer Megatron TP without the final all_reduce.
// Column-parallel dot should stay unchanged; row-parallel dot gets all_reduce.

module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_tp_two_layer(
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

    %h_local = "stablehlo.dot"(%x, %w1_local) {
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>

    %y_partial = "stablehlo.dot"(%h_local, %w2_local)
      : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>

    return %y_partial : tensor<16x6xf32>
  }
}
