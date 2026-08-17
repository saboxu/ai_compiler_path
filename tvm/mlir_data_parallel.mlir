// Formal data-parallel (DP) sketch in StableHLO / MHLO sharding style.
//
// This is teaching IR for GSPMD-style annotation — not meant to be fed
// directly into toy-opt. Real stacks (JAX/XLA, Shardy) lower the sharding
// attributes into per-device programs and insert collectives.
//
// Mapping from the earlier pedagogical pseudocode:
//   mhlo.shard(data, split_dim=0)  ->  mhlo.sharding on the tensor argument
//   mhlo.replicate(forward)        ->  weights {replicated}; same compute on each rank
//   mhlo.all_reduce(grad)          ->  stablehlo.all_reduce over replica_groups
//   mhlo.apply_gradient            ->  local weight update with synced grads
//
// Device mesh: 4 replicas along the batch axis.
// Global batch 1024  ->  local batch 256 per device after SPMD partition.

module attributes {mhlo.num_replicas = 4 : i32} {

  // Tiny forward: logits = data @ W  (no bias, for clarity).
  func.func private @forward_pass(
      %data: tensor<256x784xf32>,
      %w: tensor<784x10xf32>
  ) -> tensor<256x10xf32> {
    %logits = "stablehlo.dot"(%data, %w)
        : (tensor<256x784xf32>, tensor<784x10xf32>) -> tensor<256x10xf32>
    return %logits : tensor<256x10xf32>
  }

  // Local gradient w.r.t. W (illustration: outer product of data^T and dlogits).
  // In a real graph this comes from autodiff; here it is an opaque call.
  func.func private @compute_weight_grad(
      %data: tensor<256x784xf32>,
      %dlogits: tensor<256x10xf32>
  ) -> tensor<784x10xf32>

  // One training step under data parallelism.
  //
  // Logical (unpartitioned) types stay global; sharding attrs tell the
  // compiler how to slice them across the mesh.
  func.func @train_step_dp(
      %data: tensor<1024x784xf32>
          {mhlo.sharding = "{devices=[4,1]<=[4]}"},
      %label: tensor<1024xi32>
          {mhlo.sharding = "{devices=[4]<=[4]}"},
      %w: tensor<784x10xf32>
          {mhlo.sharding = "{replicated}"}
  ) -> (tensor<784x10xf32> {mhlo.sharding = "{replicated}"}) {

    // After SPMD partitioning, each replica's *local* view is:
    //   data_local  : tensor<256x784xf32>   // shard of dim 0
    //   label_local : tensor<256xi32>
    //   w_local     : tensor<784x10xf32>    // full replica of weights
    //
    // Compilers rewrite the body to use those local shapes. Below we write
    // the post-partition (per-replica) program explicitly for readability.

    %data_local = "stablehlo.custom_call"(%data) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4,1]<=[4]}"
    } : (tensor<1024x784xf32>) -> tensor<256x784xf32>

    %label_local = "stablehlo.custom_call"(%label) {
      call_target_name = "SPMD_shard_to_full_shape_inverse",
      mhlo.sharding = "{devices=[4]<=[4]}"
    } : (tensor<1024xi32>) -> tensor<256xi32>

    // 1) Forward on every replica (weights replicated).
    %logits = func.call @forward_pass(%data_local, %w)
        : (tensor<256x784xf32>, tensor<784x10xf32>) -> tensor<256x10xf32>

    // 2) Local loss gradient w.r.t. logits (placeholder).
    %dlogits = "stablehlo.custom_call"(%logits, %label_local) {
      call_target_name = "local_softmax_cross_entropy_grad"
    } : (tensor<256x10xf32>, tensor<256xi32>) -> tensor<256x10xf32>

    // 3) Local weight gradient (NOT yet synchronized).
    %local_grad = func.call @compute_weight_grad(%data_local, %dlogits)
        : (tensor<256x784xf32>, tensor<256x10xf32>) -> tensor<784x10xf32>

    // 4) Cross-replica All-Reduce (sum). Each device ends with the same grad.
    //    Optimizers usually scale by 1/num_replicas afterwards (mean).
    %global_grad = "stablehlo.all_reduce"(%local_grad) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }) {
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
    } : (tensor<784x10xf32>) -> tensor<784x10xf32>

    // Optional: convert sum -> mean for DP SGD.
    %c4 = stablehlo.constant dense<4.0> : tensor<f32>
    %c4b = stablehlo.broadcast_in_dim %c4, dims = []
        : (tensor<f32>) -> tensor<784x10xf32>
    %mean_grad = stablehlo.divide %global_grad, %c4b
        : tensor<784x10xf32>

    // 5) Apply gradient (SGD): W := W - lr * mean_grad
    %lr = stablehlo.constant dense<0.1> : tensor<f32>
    %lrb = stablehlo.broadcast_in_dim %lr, dims = []
        : (tensor<f32>) -> tensor<784x10xf32>
    %delta = stablehlo.multiply %mean_grad, %lrb : tensor<784x10xf32>
    %w_new = stablehlo.subtract %w, %delta : tensor<784x10xf32>

    return %w_new : tensor<784x10xf32>
  }
}
