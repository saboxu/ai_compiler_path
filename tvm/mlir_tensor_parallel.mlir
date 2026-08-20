// Formal tensor-parallel (TP) sketch in StableHLO / MHLO sharding style.
//
// Teaching IR for GSPMD-style annotation — not meant to be fed directly into
// toy-opt. Real stacks (JAX/XLA, Shardy) lower the sharding attributes into
// per-device programs and insert collectives.
//
// Scenario
// --------
// Two-layer MLP matmul block:
//
//   H = X @ W1        (W1 column-parallel)
//   Y = H @ W2        (W2 row-parallel, then reduction)
//
// With the common Megatron TP pattern:
//  - X is replicated (input not sharded)
//  - W1 is sharded on its output/column dimension => H becomes sharded
//  - W2 is sharded on its input/row dimension => each rank produces a partial sum
//  - AllReduce(SUM) is needed after the row-parallel matmul
//
// Device mesh:
//   num_replicas = 4
//
// This file focuses on showing where the all_reduce goes, and what the
// tensor layouts conceptually look like.

module attributes {mhlo.num_replicas = 4 : i32} {
  // Note: tensor shapes below are examples (global shapes).
  // After SPMD partitioning, each replica sees smaller local shapes for the
  // sharded weights, and sharded intermediate activations.

  func.func @forward_tp_two_layer(
      %x: tensor<16x8xf32> {mhlo.sharding = "{replicated}"},
      %w1: tensor<8x16xf32> {mhlo.sharding = "{devices=[4]<=[4]}"}, // column shard on dim=1
      %w2: tensor<16x6xf32> {mhlo.sharding = "{devices=[4]<=[4]}"} // row shard on dim=0
    ) -> tensor<16x6xf32> {
    // Per-replica local view after partitioning:
    //   x_local    : tensor<16x8xf32>       (replicated)
    //   w1_local   : tensor<8x4xf32>        (column shard => M/num_replicas)
    //   h_local    : tensor<16x4xf32>       (sharded on its second dim)
    //   w2_local   : tensor<4x6xf32>        (row shard => M/num_replicas)
    //   y_partial  : tensor<16x6xf32>       (partial sum over the split M)
    //   y_local    : AllReduce(SUM)(y_partial)

    %w1_local = "stablehlo.custom_call"(%w1) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<8x16xf32>) -> tensor<8x4xf32>

    %w2_local = "stablehlo.custom_call"(%w2) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<16x6xf32>) -> tensor<4x6xf32>

    // 1) Column-parallel matmul: no cross-replica reduction yet.
    %h_local = "stablehlo.dot"(%x, %w1_local)
      : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>

    // 2) Row-parallel matmul: each replica produces a partial sum.
    %y_partial = "stablehlo.dot"(%h_local, %w2_local)
      : (tensor<16x4xf32>, tensor<4x6xf32>) -> tensor<16x6xf32>

    // 3) Reduction to get the full output.
    %y = "stablehlo.all_reduce"(%y_partial) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }) {
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      // channel_handle/type are stablehlo-specific; keep it as a placeholder
      // matching the teaching style of mlir_data_parallel.mlir.
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
    } : (tensor<16x6xf32>) -> tensor<16x6xf32>

    return %y : tensor<16x6xf32>
  }
}

