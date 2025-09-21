#include "attention/attention.cpp"
#include "common/defines.h"
#include "getp_eval.cpp"
#include "matmul/matmul.cpp"
#include "utility/utility.cpp"
#include "utility/utility.h"
#ifndef B_TILE
#define B_TILE 32
#endif
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static void init_device_context_pp(DeviceContext &ctx, int device_id,
                                Transformer *transformer) {
  ctx.device_id = device_id;
  HIP_CHECK(hipSetDevice(device_id));

  model_config = &transformer->config;
  const int H = model_config->hidden_dim;
  const int V = model_config->vocab_size;
  const int L = model_config->n_layers;
  const int E = model_config->n_experts;
  const int D = model_config->head_dim;
  const int Hq = model_config->n_attn_heads;
  const int Hk = model_config->n_kv_heads;
  const int KV = D * Hk;
  const int S = model_config->seq_len;
  const int IM = model_config->intermediate_dim;

  // Balanced layer partitioning across devices:
  // Distribute remainder layers to the first `rem` devices so that
  // the difference between any two devices is at most 1 layer.
  const int base = L / g_num_devices;
  const int rem = L % g_num_devices;
  const int extra = (device_id < rem) ? 1 : 0;
  const int offset = device_id * base + (device_id < rem ? device_id : rem);
  ctx.stage_start = offset;
  ctx.stage_end = offset + base + extra;
  const int local_L = ctx.stage_end - ctx.stage_start;

  printf("Initializing device %d (layers [%d, %d))...\n", device_id, ctx.stage_start, ctx.stage_end);
  debug_print_gpu_memory("before allocations", device_id);

  // Activations
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb, D * Hq * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                      model_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                      model_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_e_agg, H * sizeof(float)));

  // Pre-allocate workspace for maximum expected batch size
  ctx.gpu_activations.d_gate_up_workspace = nullptr;

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q, Hq * D * sizeof(float)));
  // Stage-local KV caches (bf16)
  HIP_CHECK(
    hipMalloc(&ctx.gpu_activations.d_key_cache, (size_t)local_L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
            (size_t)local_L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_sin_vals, (D / 2) * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_token2row, S * sizeof(int)));
  {
    int *h_token2row = (int *)malloc(S * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < S; ++i)
      h_token2row[i] = i;
    HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_token2row, h_token2row,
                        S * sizeof(int), hipMemcpyHostToDevice));
    free(h_token2row);
  }

  // Batched helpers (lazily grown as needed)
  ctx.capacity_B = 1;
  ctx.gpu_activations.d_tokens = nullptr;
  ctx.gpu_activations.d_pos = nullptr;
  ctx.gpu_activations.d_inv_rms = nullptr;

  debug_print_gpu_memory("after activations", device_id);

  // Weights (small FP32)
  TransformerWeights *w = &transformer->weights;

  const int H_ = H;
  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_rms_attn_w, (size_t)local_L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_attn_w, w->rms_attn_w + (size_t)ctx.stage_start * H_,
            (size_t)local_L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_rms_ffn_w, (size_t)local_L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_ffn_w, w->rms_ffn_w + (size_t)ctx.stage_start * H_,
            (size_t)local_L * H_ * sizeof(float), hipMemcpyHostToDevice));

  const int D_ = model_config->head_dim;
  const int Hq_ = model_config->n_attn_heads;
  const int Hk_ = model_config->n_kv_heads;
  const int QKV_D = D_ * (Hq_ + 2 * Hk_);
  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_b_qkv, (size_t)local_L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_qkv, w->b_qkv + (size_t)ctx.stage_start * QKV_D,
            (size_t)local_L * QKV_D * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_o, (size_t)local_L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_o, w->b_o + (size_t)ctx.stage_start * H_,
                      (size_t)local_L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_attn_sinks, (size_t)local_L * Hq_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_attn_sinks, w->attn_sinks + (size_t)ctx.stage_start * Hq_,
            (size_t)local_L * Hq_ * sizeof(float), hipMemcpyHostToDevice));

  const int E_ = model_config->n_experts;
  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_w_router, (size_t)local_L * H_ * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_w_router, w->w_router + (size_t)ctx.stage_start * H_ * E_,
            (size_t)local_L * H_ * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
    hipMalloc(&ctx.gpu_weights_fp32.d_b_router, (size_t)local_L * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_router, w->b_router + (size_t)ctx.stage_start * E_,
            (size_t)local_L * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_out_w, H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_out_w, w->rms_out_w,
                      H_ * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights", device_id);

  // Expert biases FP32
  const int IM_ = model_config->intermediate_dim;
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp1,
                      (size_t)local_L * E_ * (2 * IM_) * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp1,
                      w->b_mlp1 + (size_t)ctx.stage_start * E_ * (size_t)(2 * IM_),
                      (size_t)local_L * E_ * (size_t)(2 * IM_) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp2,
                      (size_t)local_L * E_ * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp2,
                      w->b_mlp2 + (size_t)ctx.stage_start * E_ * (size_t)H_,
                      (size_t)local_L * E_ * H_ * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases", device_id);

  // Large BF16 weights
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;
  const int V_ = model_config->vocab_size;
  const int O_N = D_ * Hq_;

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
                      (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V_ * H_,
                           ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
                           n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_qkv_bf16,
                      (size_t)local_L * QKV_D * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv + (size_t)ctx.stage_start * QKV_D * H_, (size_t)local_L * QKV_D * H_,
                           ctx.gpu_weights_bf16.d_w_qkv_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_o_bf16,
                      (size_t)local_L * H_ * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o + (size_t)ctx.stage_start * H_ * O_N, (size_t)local_L * H_ * O_N,
                           ctx.gpu_weights_bf16.d_w_o_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp1_bf16,
                      (size_t)local_L * E_ * (2 * IM_) * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1 + (size_t)ctx.stage_start * E_ * (2 * IM_) * H_, (size_t)local_L * E_ * (2 * IM_) * H_,
                           ctx.gpu_weights_bf16.d_w_mlp1_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp2_bf16,
                      (size_t)local_L * E_ * H_ * IM_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2 + (size_t)ctx.stage_start * E_ * H_ * IM_, (size_t)local_L * E_ * H_ * IM_,
                           ctx.gpu_weights_bf16.d_w_mlp2_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_out_bf16,
                      (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V_ * H_,
                           ctx.gpu_weights_bf16.d_out_bf16, n_streams,
                           chunk_bytes);

  debug_print_gpu_memory("after large BF16 weights (model loaded)", device_id);
}

// ============================ Forward ============================
// Ensure device has capacity for B batch slots (reallocates activations &
// caches if needed)
static inline void ensure_device_capacity_pp(DeviceContext &ctx, int B) {
  HIP_CHECK(hipSetDevice(ctx.device_id));

  const bool need_realloc = B > ctx.capacity_B;

  const Config *p = model_config;
  const int H = p->hidden_dim;
  const int D = p->head_dim;
  const int Hq = p->n_attn_heads;
  const int Hk = p->n_kv_heads;
  const int KV = D * Hk;
  const int L = p->n_layers;
  const int S = p->seq_len;
  const int IM = p->intermediate_dim;
  const int V = p->vocab_size;
  const int local_L = ctx.stage_end - ctx.stage_start;

  // Free previous activations to re-alloc at batch size B
  if (need_realloc) {
  #define FREE_IF(p)                                                             \
  if ((p))                                                                     \
  HIP_CHECK(hipFree((p)))

    FREE_IF(ctx.gpu_activations.d_x);
    FREE_IF(ctx.gpu_activations.d_t);
    FREE_IF(ctx.gpu_activations.d_tb);
    FREE_IF(ctx.gpu_activations.d_router_score);
    FREE_IF(ctx.gpu_activations.d_topk_v);
    FREE_IF(ctx.gpu_activations.d_topk_i);
    FREE_IF(ctx.gpu_activations.d_gate_up);
    FREE_IF(ctx.gpu_activations.d_e_agg);
    FREE_IF(ctx.gpu_activations.d_qkv);
    FREE_IF(ctx.gpu_activations.d_q);
    FREE_IF(ctx.gpu_activations.d_key_cache);
    FREE_IF(ctx.gpu_activations.d_value_cache);
    FREE_IF(ctx.gpu_activations.d_logits);
  // mask & token2row remain shared
  #undef FREE_IF

    // Re-allocate with batch dimension
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_x, (size_t)B * H * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_t, (size_t)B * H * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb,
                        (size_t)B * D * Hq * sizeof(float)));

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score,
                        (size_t)B * p->n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                        (size_t)B * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                        (size_t)B * p->experts_per_token * sizeof(int)));

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up,
                        (size_t)B * IM * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_e_agg, (size_t)B * H * sizeof(float)));

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                        (size_t)B * (D * (Hq + 2 * Hk)) * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q,
                        (size_t)B * Hq * D * sizeof(float)));
  // Per-batch KV caches (stage-local only, bf16)
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache,
        (size_t)B * local_L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
        (size_t)B * local_L * S * KV * sizeof(bf16_t)));
    // Auxiliary buffers
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                        (size_t)B * V * sizeof(float)));

    ctx.capacity_B = B;
  }
  // Allocate per-sample inv_rms buffer
  if (need_realloc) {
    if (ctx.gpu_activations.d_inv_rms) {
      HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
    }
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_inv_rms, (size_t)B * sizeof(float)));
  } else if (!ctx.gpu_activations.d_inv_rms) {
    // First-time allocation when capacity already sufficient
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_inv_rms,
                        (size_t)ctx.capacity_B * sizeof(float)));
  }

  // Tokens and positions (host-to-device each step)
  if (need_realloc) {
    if (ctx.gpu_activations.d_tokens)
      HIP_CHECK(hipFree(ctx.gpu_activations.d_tokens));
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_tokens, (size_t)B * sizeof(int)));

    if (ctx.gpu_activations.d_pos)
      HIP_CHECK(hipFree(ctx.gpu_activations.d_pos));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_pos, (size_t)B * sizeof(int)));
  } else {
    if (!ctx.gpu_activations.d_tokens)
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tokens,
                          (size_t)ctx.capacity_B * sizeof(int)));

    if (!ctx.gpu_activations.d_pos)
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_pos,
                          (size_t)ctx.capacity_B * sizeof(int)));
  }

  // Ensure we have at least B stream
  if (!ctx.streams) {
    ctx.streams = (hipStream_t *)malloc(sizeof(hipStream_t) * B);
    ctx.n_streams = 0;
  } else if (B > ctx.n_streams) {
    hipStream_t *new_streams = (hipStream_t *)malloc(sizeof(hipStream_t) * B);
    memcpy(new_streams, ctx.streams, sizeof(hipStream_t) * ctx.n_streams);
    free(ctx.streams);
    ctx.streams = new_streams;
  }
  for (int i = ctx.n_streams; i < B; ++i) {
    HIP_CHECK(hipStreamCreateWithFlags(&ctx.streams[i], hipStreamNonBlocking));
  }
  if (ctx.n_streams < B)
    ctx.n_streams = B;
}

static float *gpu_forward_device_batch_pp(Transformer *transformer,
                                       const int *tokens, const int *pos,
                                       int batch_size, int device_id,
                                       int max_pos_in_batch,
                                       int b_base,
                                       hipStream_t stream) {
  DeviceContext &ctx = g_devices[device_id];
  HIP_CHECK(hipSetDevice(device_id));

  const Config *p = model_config;
  const int H = p->hidden_dim;
  const int D = p->head_dim;
  const int Hq = p->n_attn_heads;
  const int Hk = p->n_kv_heads;
  const int KV = D * Hk;
  const int IM = p->intermediate_dim;
  const int E = p->n_experts;
  const int S = p->seq_len;
  const int L = p->n_layers;
  const int V = p->vocab_size;
  const int QKV_D = D * (Hq + 2 * Hk);
  const int local_L = ctx.stage_end - ctx.stage_start;

  // Copy host tokens/positions into device buffers (slice)
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_tokens + b_base, tokens,
                           (size_t)batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_pos + b_base, pos,
                           (size_t)batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));

  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 gridH_warp((H + TM - 1) / TM, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Launch batched embedding kernel (only on first stage)
  if (ctx.device_id == 0) {
    dim3 gridH_batch(gridH_thread.x, batch_size, 1);
    copy_embedding_bf16_batch_kernel<<<gridH_batch, block, 0, stream>>>(
        ctx.gpu_activations.d_x + (size_t)b_base * H,
        ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
        ctx.gpu_activations.d_tokens + b_base, batch_size, H);
  }

  for (int l = ctx.stage_start; l < ctx.stage_end; ++l) {
    const int l_local = l - ctx.stage_start;
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV((QKV_D + TM - 1) / TM, 1, 1);
    // Batched QKV projection (RMSNorm + MatMul + Bias) - separate kernels
    {
      // First apply RMSNorm
      {
        dim3 gridH_batch(1, batch_size, 1);
        rmsnorm_batch_kernel<<<gridH_batch, block, 0, stream>>>(
            ctx.gpu_activations.d_t + (size_t)b_base * H,
            ctx.gpu_activations.d_x + (size_t)b_base * H,
            ctx.gpu_weights_fp32.d_rms_attn_w + l_local * H, H,
      ctx.gpu_activations.d_pos + b_base);
      }

      // Then apply MatMul + Bias
      {
    // Use kernel launch configuration matching matmul.cpp (32x32 tiles, 16x4 threads)
    dim3 gridQKV_gemm((QKV_D + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM, 1);
    dim3 blockQKV(16, 4, 1);
    matmul_bias_gemm_kernel_bf16_mfma<<<gridQKV_gemm, blockQKV, 0, stream>>>(
            ctx.gpu_activations.d_qkv + (size_t)b_base * QKV_D,
            ctx.gpu_activations.d_t + (size_t)b_base * H,
            ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l_local * QKV_D * H,
            ctx.gpu_weights_fp32.d_b_qkv + l_local * QKV_D, H, QKV_D, batch_size,
      ctx.gpu_activations.d_pos + b_base);
      }
    }

    // Scatter QKV to q / caches (batched)
  const int loff_local = (l - ctx.stage_start) * S * KV;
  // Total KV size per batch for this stage (constant across layer loop)
  const int kv_total_stage = (ctx.stage_end - ctx.stage_start) * S * KV;
    {
      dim3 grid_fused(max(Hq, Hk), batch_size, 1);
      dim3 block_fused(D / 2, 1, 1);
      fused_split_rope_scatter_qkv_batch_kernel<<<grid_fused, block_fused, 0, stream>>>(
          /*q_out*/        ctx.gpu_activations.d_q + (size_t)b_base * Hq * D,
          /*key_cache*/    ctx.gpu_activations.d_key_cache + (size_t)b_base * local_L * S * KV,
          /*value_cache*/  ctx.gpu_activations.d_value_cache + (size_t)b_base * local_L * S * KV,
          /*qkv*/          ctx.gpu_activations.d_qkv + (size_t)b_base * (Hq + 2 * Hk) * D,
      /*pos*/          ctx.gpu_activations.d_pos + b_base,
          /*Hq,Hk,D*/      Hq, Hk, D,
          /*RoPE*/         p->rope_theta, p->rope_scaling_factor, p->initial_context_length,
      /*cache*/        loff_local, kv_total_stage,
          /*B*/            batch_size);
    }

    {
      dim3 gridAttn(Hq, batch_size, 1);
      dim3 blockA(WF_SIZE);
      const bool layer_has_window = (p->sliding_window > 0) && ((l & 1) == 0);
      const int att_tokens = layer_has_window
                                 ? std::min(max_pos_in_batch + 1,
                                            p->sliding_window)
                                 : (max_pos_in_batch + 1);
      size_t shmem_size = (size_t)(att_tokens + 1) * sizeof(float);
      attention_batch_kernel<<<gridAttn, blockA, shmem_size, stream>>>(
          ctx.gpu_activations.d_tb + (size_t)b_base * Hq * D,
          ctx.gpu_activations.d_q + (size_t)b_base * Hq * D,
          ctx.gpu_activations.d_key_cache + (size_t)b_base * local_L * S * KV,
          ctx.gpu_activations.d_value_cache + (size_t)b_base * local_L * S * KV,
          ctx.gpu_weights_fp32.d_attn_sinks, l_local,
          ctx.gpu_activations.d_pos + b_base, D,
          Hq, Hk, S, layer_has_window ? p->sliding_window : 0,
          local_L * S * KV, batch_size);
    }

    // Output projection + residual (batched) - separate kernels
    {
      const int O_N = D * Hq;

      // First do MatMul + Bias: temp = tb @ W^T + b
  {
    dim3 gridO_gemm((H + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM, 1);
    dim3 blockO(16, 4, 1);
    matmul_bias_gemm_kernel_bf16_mfma<<<gridO_gemm, blockO, 0, stream>>>(
    ctx.gpu_activations.d_t + (size_t)b_base * H,
    ctx.gpu_activations.d_tb + (size_t)b_base * Hq * D,
    ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l_local * H * O_N,
    ctx.gpu_weights_fp32.d_b_o + l_local * H, O_N, H, batch_size,
    ctx.gpu_activations.d_pos + b_base);
  }

      // Then do residual add: x = x + temp
      {
        dim3 gridH_batch(gridH_thread.x, batch_size, 1);
        residual_add_batch_kernel<<<gridH_batch, block, 0, stream>>>(
            ctx.gpu_activations.d_x + (size_t)b_base * H,
            ctx.gpu_activations.d_t + (size_t)b_base * H,
            H, batch_size, ctx.gpu_activations.d_pos + b_base);
      }
    }

    // FFN (batched)
    {
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0, stream>>>(
          ctx.gpu_activations.d_t + (size_t)b_base * H,
          ctx.gpu_activations.d_x + (size_t)b_base * H,
          ctx.gpu_weights_fp32.d_rms_ffn_w + l_local * H, H,
          ctx.gpu_activations.d_pos + b_base);
    }

    {
      dim3 gridE_gemm((E + TM - 1) / TM, batch_size, 1);
      matmul_bias_gemm_kernel_float<<<gridE_gemm, block, 0, stream>>>(
          ctx.gpu_activations.d_router_score + (size_t)b_base * E,
          ctx.gpu_activations.d_t + (size_t)b_base * H,
          ctx.gpu_weights_fp32.d_w_router + (size_t)l_local * H * E,
          ctx.gpu_weights_fp32.d_b_router + l_local * E, H, E, batch_size,
          ctx.gpu_activations.d_pos + b_base);
    }

    {
      dim3 gridTopK_batch(1, batch_size, 1);
      size_t shared_mem_size = (size_t)E * sizeof(float);
      fused_topk_softmax_batch_kernel<<<gridTopK_batch, BLOCK_SIZE,
                                        shared_mem_size, stream>>>(
          ctx.gpu_activations.d_topk_v + (size_t)b_base * p->experts_per_token,
          ctx.gpu_activations.d_topk_i + (size_t)b_base * p->experts_per_token,
          ctx.gpu_activations.d_router_score + (size_t)b_base * E, E,
          p->experts_per_token, batch_size, ctx.gpu_activations.d_pos + b_base);
    }

    HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_e_agg + (size_t)b_base * H,
                             0, (size_t)batch_size * H * sizeof(float), stream));

    // Use pre-allocated workspace from DeviceContext to avoid repeated
    // malloc/free
    size_t gate_up_topk_bytes = (size_t)p->experts_per_token *
                                (size_t)batch_size * (size_t)IM * sizeof(float);
  if (!ctx.gpu_activations.d_gate_up_workspace) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace,
                          gate_up_topk_bytes));
    }
    float *d_gate_up_topk = ctx.gpu_activations.d_gate_up_workspace;
    HIP_CHECK(hipMemsetAsync(d_gate_up_topk, 0,
              (size_t)p->experts_per_token * (size_t)batch_size *
              (size_t)IM * sizeof(float),
              stream));

    // --- MLP1: per-batch per-expert, no CB
    {
      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 gridIM((IM + TM - 1) / TM, batch_size, 1);
      gridIM.z = p->experts_per_token;
      mlp1_fused_gemm_kernel<<<gridIM, block, 0, stream>>>(
          /*gate_up_topk[K,B,IM]*/ d_gate_up_topk,
          /*x[B,H]*/ ctx.gpu_activations.d_t + (size_t)b_base * H,
          /*w*/ ctx.gpu_weights_bf16.d_w_mlp1_bf16,
          /*b*/ ctx.gpu_expert_bias.g_b_mlp1,
          /*topk_i*/ ctx.gpu_activations.d_topk_i + (size_t)b_base * p->experts_per_token,
          /*layer*/ l_local,
          /*E,H,IM*/ E, H, IM,
          /*clip*/ p->swiglu_limit,
          /*B,K*/ batch_size, ctx.gpu_activations.d_pos + b_base);
    }

    // --- MLP2: per-batch per-expert, no CB
    {
      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 gridH((H + TM - 1) / TM, batch_size, 1);
      gridH.z = p->experts_per_token;
      mlp2_bias_weighted_accum_gemm_kernel<<<gridH, block, 0, stream>>>(
          /*e_agg[B,H]*/ ctx.gpu_activations.d_e_agg + (size_t)b_base * H,
          /*gate_up[K,B,IM]*/ d_gate_up_topk,
          /*w*/ ctx.gpu_weights_bf16.d_w_mlp2_bf16,
          /*b*/ ctx.gpu_expert_bias.g_b_mlp2,
          /*topk_i, topk_v*/ ctx.gpu_activations.d_topk_i + (size_t)b_base * p->experts_per_token,
          ctx.gpu_activations.d_topk_v + (size_t)b_base * p->experts_per_token,
          /*layer*/ l_local,
          /*E,IM,H,B,K*/ E, IM, H, batch_size, ctx.gpu_activations.d_pos + b_base);
    }

    // Keep workspace allocated in DeviceContext for reuse

    {
      dim3 gridH_batch(gridH_thread.x, batch_size, 1);
      residual_add_batch_kernel<<<gridH_batch, block, 0, stream>>>(
          ctx.gpu_activations.d_x + (size_t)b_base * H,
          ctx.gpu_activations.d_e_agg + (size_t)b_base * H,
          H, batch_size, ctx.gpu_activations.d_pos + b_base);
    }
  }

  // Final head
  if (ctx.device_id == g_num_devices - 1) {
    // 1) RMSNorm - separate kernel call
    {
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0, stream>>>(
          ctx.gpu_activations.d_t + (size_t)b_base * H,
          ctx.gpu_activations.d_x + (size_t)b_base * H,
          ctx.gpu_weights_fp32.d_rms_out_w, H,
      ctx.gpu_activations.d_pos + b_base);
    }

    // 2) MatMul for logits - separate GEMM version
    {
    dim3 gridV_gemm((V + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM, 1);
    dim3 blockV(16, 4, 1);
    matmul_bias_gemm_kernel_bf16_mfma<<<gridV_gemm, blockV, 0, stream>>>(
          ctx.gpu_activations.d_logits + (size_t)b_base * V,
          ctx.gpu_activations.d_t + (size_t)b_base * H,
          ctx.gpu_weights_bf16.d_out_bf16, nullptr, H, V, batch_size,
      ctx.gpu_activations.d_pos + b_base);
    }
    return ctx.gpu_activations.d_logits + (size_t)b_base * V;
  } else {
    return nullptr;
  }
}

static long long run_requests_pipeline(Transformer *transformer,
                                       Tokenizer *tokenizer, PromptCtx *ctxs,
                                       int num_ctxs) {
  long long total_tokens = 0;

  // Allocate batch helpers on host
  const int B = num_ctxs;
  std::vector<int> h_tokens(B, 0);
  std::vector<int> h_pos(B, 0);
  std::vector<char> h_active(B, 1);

  std::transform(ctxs, ctxs + B, h_tokens.begin(),
                 [](const PromptCtx &ctx) { return ctx.token; });

  // Ensure device buffers sized for B on all stages
  for (int dev = 0; dev < g_num_devices; ++dev) {
    ensure_device_capacity_pp(g_devices[dev], B);
  }
  const Config *p = model_config;
  const int V = p->vocab_size;
  // Temporary host buffer for batched logits (from last stage) - pinned for async D2H
  float *h_logits_batch = nullptr;
  HIP_CHECK(hipHostMalloc((void **)&h_logits_batch, (size_t)B * V * sizeof(float)));

  // Micro-batching configuration
  int M_env = 8; // default
  const char *mb_env = getenv("GETP_MB");
  if (mb_env) {
    M_env = atoi(mb_env);
  }
  // Heuristic: if M_env < 0, auto-select M so that micro_bs >= 4 and M <= devices
  int M = 1;
  if (M_env > 0) {
    M = std::min(M_env, B);
  } else if (M_env < 0) {
    int maxM = std::min(g_num_devices, B);
    int target_bs = 4;
    M = std::min(maxM, std::max(1, B / target_bs));
  }
  int micro_bs = (M > 0) ? (B + M - 1) / M : B;
  if (micro_bs <= 0)
    micro_bs = 1;
  // Recompute M to match micro_bs packing
  M = (B + micro_bs - 1) / micro_bs;
  if (M <= 1) {
    printf("[MB] Micro-batching disabled (GETP_MB=%d). Using M=1, micro_bs=%d, B=%d, devices=%d\n",
           M_env, micro_bs, B, g_num_devices);
  } else {
    printf("[MB] Micro-batching enabled: M=%d, micro_bs=%d, B=%d, devices=%d (GETP_MB=%d)\n",
           M, micro_bs, B, g_num_devices, M_env);
  }
  std::vector<int> mb_start(M, 0), mb_size(M, 0);
  for (int m = 0; m < M; ++m) {
    int start = m * micro_bs;
    int end = std::min(B, start + micro_bs);
    mb_start[m] = start;
    mb_size[m] = end - start;
  }

  // Create per-device compute and p2p streams, and events per micro-batch
  std::vector<hipStream_t> compute_streams(g_num_devices);
  std::vector<hipStream_t> p2p_streams(g_num_devices);
  std::vector<std::vector<hipEvent_t>> ev_compute_done(g_num_devices, std::vector<hipEvent_t>(M));
  std::vector<std::vector<hipEvent_t>> ev_copy_done(std::max(0, g_num_devices - 1), std::vector<hipEvent_t>(M));
  std::vector<hipEvent_t> ev_logits_ready(M);

  for (int dev = 0; dev < g_num_devices; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreateWithFlags(&compute_streams[dev], hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&p2p_streams[dev], hipStreamNonBlocking));
    for (int m = 0; m < M; ++m) {
      HIP_CHECK(hipEventCreateWithFlags(&ev_compute_done[dev][m], hipEventDisableTiming));
      if (dev < g_num_devices - 1) {
        // Create copy-done events on the SOURCE device; downstream stage will wait on them.
        HIP_CHECK(hipEventCreateWithFlags(&ev_copy_done[dev][m], hipEventDisableTiming));
      }
    }
  }
  // Logits ready events on last device
  if (g_num_devices > 0) {
    HIP_CHECK(hipSetDevice(g_num_devices - 1));
    for (int m = 0; m < M; ++m) {
      HIP_CHECK(hipEventCreateWithFlags(&ev_logits_ready[m], hipEventDisableTiming));
    }
  }

  int num_finished = 0;
  while (num_finished < B) {
    // Build positions and tokens for this step; mark inactive with pos=-1
    int max_pos_in_batch = 0;
    for (int i = 0; i < B; ++i) {
      if (!h_active[i]) {
        h_pos[i] = -1;
        h_tokens[i] = -1;
      } else {
        if (h_pos[i] > max_pos_in_batch)
          max_pos_in_batch = h_pos[i];
      }
    }

    // Per micro-batch max pos to size shared memory
    std::vector<int> mb_maxpos(M, 0);
    for (int m = 0; m < M; ++m) {
      int mmax = 0;
      for (int j = 0; j < mb_size[m]; ++j) {
        int posj = h_pos[mb_start[m] + j];
        if (posj > mmax)
          mmax = posj;
      }
      mb_maxpos[m] = mmax;
    }

    // GPipe-style schedule: k iterates over time steps of pipeline
    for (int k = 0; k < M + g_num_devices - 1; ++k) {
      for (int s = 0; s < g_num_devices; ++s) {
        int m = k - s;
        if (m < 0 || m >= M)
          continue;

        const int base = mb_start[m];
        const int bs = mb_size[m];
        const int mmax = mb_maxpos[m];

        // If not first stage, wait for copy from previous stage
        if (s > 0) {
          HIP_CHECK(hipSetDevice(s));
          HIP_CHECK(hipStreamWaitEvent(compute_streams[s], ev_copy_done[s - 1][m], 0));
        }

        HIP_CHECK(hipSetDevice(s));
        float *d_log = gpu_forward_device_batch_pp(transformer,
                                                h_tokens.data() + base,
                                                h_pos.data() + base,
                                                bs, s, mmax, base,
                                                compute_streams[s]);
        // Mark compute done for this stage & micro-batch
        HIP_CHECK(hipEventRecord(ev_compute_done[s][m], compute_streams[s]));
        if (s < g_num_devices - 1) {
          // Launch P2P to next stage after compute done. Record completion on DEST device to allow local wait.
          DeviceContext &src = g_devices[s];
          DeviceContext &dst = g_devices[s + 1];
          // Ensure source has finished compute
          HIP_CHECK(hipSetDevice(src.device_id));
          HIP_CHECK(hipStreamWaitEvent(p2p_streams[src.device_id], ev_compute_done[s][m], 0));

          // Start peer copy from src->dst on src stream
          HIP_CHECK(hipMemcpyPeerAsync(dst.gpu_activations.d_x + (size_t)base * p->hidden_dim,
                                       dst.device_id,
                                       src.gpu_activations.d_x + (size_t)base * p->hidden_dim,
                                       src.device_id,
                                       (size_t)bs * p->hidden_dim * sizeof(float),
                                       p2p_streams[src.device_id]));

          // Create a cross-device event by first syncing a temporary event on src stream, then making dst stream wait and record
          HIP_CHECK(hipEventRecord(ev_copy_done[s][m], p2p_streams[src.device_id]));
        } else {
          // Last stage: D2H copy logits after compute done
          HIP_CHECK(hipSetDevice(s));
          HIP_CHECK(hipStreamWaitEvent(p2p_streams[s], ev_compute_done[s][m], 0));
          HIP_CHECK(hipMemcpyAsync(&h_logits_batch[(size_t)base * V], d_log,
                                   (size_t)bs * V * sizeof(float),
                                   hipMemcpyDeviceToHost, p2p_streams[s]));
          HIP_CHECK(hipEventRecord(ev_logits_ready[m], p2p_streams[s]));
        }
      }
    }

    // Wait per micro-batch for logits and advance contexts for its samples immediately
    if (g_num_devices > 0) {
      HIP_CHECK(hipSetDevice(g_num_devices - 1));
      for (int m = 0; m < M; ++m) {
        const int base = mb_start[m];
        const int bs = mb_size[m];
        if (bs <= 0) continue;
        HIP_CHECK(hipEventSynchronize(ev_logits_ready[m]));

        // Process only samples in this micro-batch
        for (int j = 0; j < bs; ++j) {
          int i = base + j;
          if (!h_active[i])
            continue;
          PromptCtx &ctx = ctxs[i];

          // Copy logits for this sample into ctx.h_logits
          memcpy(ctx.h_logits, &h_logits_batch[(size_t)i * V],
                 (size_t)V * sizeof(float));

          ctx.pos++;
          h_pos[i] = ctx.pos;
          int next;
          if (ctx.pos < ctx.num_prompt_tokens) {
            next = ctx.prompt_tokens[ctx.pos];
          } else {
            ctx.is_context_phase = false;
            next = sample(ctx.sampler, ctx.h_logits);
            ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens] = next;
            ctx.num_generated++;
          }

          if (next == 199999 || next == 200002) {
            ctx.finished = true;
            h_active[i] = 0;
            num_finished++;
            const char *piece = decode_piece(tokenizer, ctx.token, next);
            if (piece)
              ctx.output_str += piece;
            ctx.token = next;
            continue;
          }

          const char *piece = decode_piece(tokenizer, ctx.token, next);
          if (piece)
            ctx.output_str += piece;

          ctx.token = next;
          h_tokens[i] = next;

          // Respect max steps constraint
          if (ctx.max_steps != 0 && ctx.pos + 1 >= ctx.max_steps) {
            ctx.finished = true;
            h_active[i] = 0;
            num_finished++;
          }
        }
      }
    }
  }

  for (int i = 0; i < B; ++i) {
    PromptCtx &ctx = ctxs[i];
    ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;
    total_tokens += (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
  }

  // Cleanup streams & events created for micro-batching
  for (int dev = 0; dev < g_num_devices; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    for (int m = 0; m < (int)ev_compute_done[dev].size(); ++m) {
      HIP_CHECK(hipEventDestroy(ev_compute_done[dev][m]));
      if (dev < g_num_devices - 1) {
        HIP_CHECK(hipEventDestroy(ev_copy_done[dev][m]));
      }
    }
    HIP_CHECK(hipStreamDestroy(compute_streams[dev]));
    HIP_CHECK(hipStreamDestroy(p2p_streams[dev]));
  }
  if (g_num_devices > 0) {
    HIP_CHECK(hipSetDevice(g_num_devices - 1));
    for (int m = 0; m < (int)ev_logits_ready.size(); ++m) {
      HIP_CHECK(hipEventDestroy(ev_logits_ready[m]));
    }
  }

  // Free pinned host buffer
  if (h_logits_batch) {
    HIP_CHECK(hipHostFree(h_logits_batch));
  }

  return total_tokens;
}
