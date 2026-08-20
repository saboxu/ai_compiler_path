// Regression input: row-parallel matmul expressed as stablehlo.dot_general.

module attributes {mhlo.num_replicas = 4 : i32} {
  func.func @forward_dot_general_row(
      %h: tensor<16x4xf32> {mhlo.sharding = "{devices=[4]<=[4]}"},
      %w2: tensor<16x6xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}
    ) -> tensor<16x6xf32> {

    %w2_local = "stablehlo.custom_call"(%w2) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<16x6xf32>) -> tensor<4x6xf32>

    %y_partial = "stablehlo.dot_general"(%h, %w2_local) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >
    } : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>

    return %y_partial : tensor<16x6xf32>
  }
}
