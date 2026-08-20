// Regression input: column-parallel matmul only. Pass must not insert all_reduce.

module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_column_only(
      %x: tensor<16x8xf32> {mhlo.sharding = "{replicated}"},
      %w1: tensor<8x16xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}
    ) -> tensor<16x4xf32> {

    %w1_local = "stablehlo.custom_call"(%w1) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<8x16xf32>) -> tensor<8x4xf32>

    %h_local = "stablehlo.dot"(%x, %w1_local)
      : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>

    return %h_local : tensor<16x4xf32>
  }
}
