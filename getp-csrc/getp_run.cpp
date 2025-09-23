#include "attention/attention.cpp"
#include "common/defines.h"
#include "getp_eval.cpp"
#include "matmul/matmul.cpp"
#include "profiler/profiler.cpp"
#include "utility/utility.cpp"
#include "utility/utility.h"
#include <algorithm>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

static Config *model_config;

static std::vector<DeviceContext> g_devices;
static int g_num_devices = 0;

static void init_device_context(DeviceContext &ctx, int device_id,
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

  printf("Initializing device %d...\n", device_id);
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
  ctx.gpu_activations.d_expert_counts = nullptr;
  ctx.gpu_activations.d_expert_offsets = nullptr;
  ctx.gpu_activations.d_expert_assignments = nullptr;
  ctx.gpu_activations.d_assignment_active_slot = nullptr;
  ctx.gpu_activations.d_active_experts = nullptr;
  ctx.gpu_activations.d_active_counts = nullptr;
  ctx.gpu_activations.d_moe_x_workspace = nullptr;
  ctx.gpu_activations.d_mlp1_workspace = nullptr;
  ctx.gpu_activations.d_mlp2_workspace = nullptr;
  ctx.gpu_activations.expert_assign_capacity = 0;
  ctx.gpu_activations.assignment_active_capacity = 0;
  ctx.gpu_activations.active_expert_capacity = 0;
  ctx.gpu_activations.gate_up_workspace_bytes = 0;
  ctx.gpu_activations.moe_x_workspace_bytes = 0;
  ctx.gpu_activations.mlp1_workspace_bytes = 0;
  ctx.gpu_activations.mlp2_workspace_bytes = 0;

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q, Hq * D * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_key_cache, L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                      L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens, sizeof(int)));

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
      hipMalloc(&ctx.gpu_weights_fp32.d_rms_attn_w, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_attn_w, w->rms_attn_w,
                      L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_rms_ffn_w, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_ffn_w, w->rms_ffn_w,
                      L * H_ * sizeof(float), hipMemcpyHostToDevice));

  const int D_ = model_config->head_dim;
  const int Hq_ = model_config->n_attn_heads;
  const int Hk_ = model_config->n_kv_heads;
  const int QKV_D = D_ * (Hq_ + 2 * Hk_);
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_qkv, w->b_qkv,
                      L * QKV_D * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_o, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_o, w->b_o,
                      L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_attn_sinks, L * Hq_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_attn_sinks, w->attn_sinks,
                      L * Hq_ * sizeof(float), hipMemcpyHostToDevice));

  const int E_ = model_config->n_experts;
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_w_router, L * H_ * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_w_router, w->w_router,
                      L * H_ * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_b_router, L * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_router, w->b_router,
                      L * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_out_w, H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_out_w, w->rms_out_w,
                      H_ * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights", device_id);

  // Expert biases FP32
  const int IM_ = model_config->intermediate_dim;
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp1,
                      (size_t)L * E_ * (2 * IM_) * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp1, w->b_mlp1,
                      (size_t)L * E_ * (2 * IM_) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp2,
                      (size_t)L * E_ * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp2, w->b_mlp2,
                      (size_t)L * E_ * H_ * sizeof(float),
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
                      (size_t)L * QKV_D * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H_,
                           ctx.gpu_weights_bf16.d_w_qkv_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_o_bf16,
                      (size_t)L * H_ * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H_ * O_N,
                           ctx.gpu_weights_bf16.d_w_o_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp1_bf16,
                      (size_t)L * E_ * (2 * IM_) * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E_ * (2 * IM_) * H_,
                           ctx.gpu_weights_bf16.d_w_mlp1_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp2_bf16,
                      (size_t)L * E_ * H_ * IM_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E_ * H_ * IM_,
                           ctx.gpu_weights_bf16.d_w_mlp2_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_out_bf16,
                      (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V_ * H_,
                           ctx.gpu_weights_bf16.d_out_bf16, n_streams,
                           chunk_bytes);

  debug_print_gpu_memory("after large BF16 weights (model loaded)", device_id);
}

// Cleanup device context
static void cleanup_device_context(DeviceContext &ctx) {
  int device_id = ctx.device_id;
  HIP_CHECK(hipSetDevice(device_id));

  HIP_CHECK(hipFree(ctx.gpu_activations.d_x));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_t));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_tb));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_router_score));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_v));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_i));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_e_agg));
  if (ctx.gpu_activations.d_gate_up_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
  if (ctx.gpu_activations.d_expert_counts)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_expert_counts));
  if (ctx.gpu_activations.d_expert_offsets)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_expert_offsets));
  if (ctx.gpu_activations.d_expert_assignments)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_expert_assignments));
  if (ctx.gpu_activations.d_assignment_active_slot)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_assignment_active_slot));
  if (ctx.gpu_activations.d_active_experts)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_active_experts));
  if (ctx.gpu_activations.d_active_counts)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_active_counts));
  if (ctx.gpu_activations.d_moe_x_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_moe_x_workspace));
  if (ctx.gpu_activations.d_mlp1_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_mlp1_workspace));
  if (ctx.gpu_activations.d_mlp2_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_mlp2_workspace));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_q));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_key_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_value_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_logits));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_next_tokens));
  if (ctx.gpu_activations.d_inv_rms)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
  if (ctx.gpu_activations.d_token2row)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_token2row));
  if (ctx.gpu_activations.d_tokens)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_tokens));
  if (ctx.gpu_activations.d_pos)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_pos));

  if (ctx.streams) {
    for (int i = 0; i < ctx.n_streams; ++i)
      HIP_CHECK(hipStreamDestroy(ctx.streams[i]));
    free(ctx.streams);
    ctx.streams = nullptr;
    ctx.n_streams = 0;
  }

  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_attn_w));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_ffn_w));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_qkv));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_o));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_attn_sinks));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_w_router));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_router));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_out_w));

  if (ctx.gpu_expert_bias.g_b_mlp1)
    HIP_CHECK(hipFree(ctx.gpu_expert_bias.g_b_mlp1));
  if (ctx.gpu_expert_bias.g_b_mlp2)
    HIP_CHECK(hipFree(ctx.gpu_expert_bias.g_b_mlp2));

  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_qkv_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_o_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_mlp1_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_mlp2_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_out_bf16));
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  model_config = &transformer->config;

  HIP_CHECK(hipGetDeviceCount(&g_num_devices));
  if (g_num_devices <= 0) {
    fprintf(stderr, "No HIP devices found!\n");
    exit(EXIT_FAILURE);
  }

  printf("Found %d HIP devices, initializing multi-GPU setup...\n",
         g_num_devices);
  g_devices.resize(g_num_devices);

// init all devices in parallel
#pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    init_device_context(g_devices[i], i, transformer);
  }

  printf("Multi-GPU initialization complete!\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
#pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    cleanup_device_context(g_devices[i]);
  }
  g_devices.clear();
  g_num_devices = 0;
}

// ============================ Forward ============================
// Ensure device has capacity for B batch slots (reallocates activations &
// caches if needed)
static inline void ensure_device_capacity(DeviceContext &ctx, int B) {
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
    FREE_IF(ctx.gpu_activations.d_gate_up_workspace);
    FREE_IF(ctx.gpu_activations.d_expert_counts);
    FREE_IF(ctx.gpu_activations.d_expert_offsets);
    FREE_IF(ctx.gpu_activations.d_expert_assignments);
    FREE_IF(ctx.gpu_activations.d_assignment_active_slot);
    FREE_IF(ctx.gpu_activations.d_active_experts);
    FREE_IF(ctx.gpu_activations.d_active_counts);
    FREE_IF(ctx.gpu_activations.d_moe_x_workspace);
    FREE_IF(ctx.gpu_activations.d_mlp1_workspace);
    FREE_IF(ctx.gpu_activations.d_mlp2_workspace);
    FREE_IF(ctx.gpu_activations.d_qkv);
    FREE_IF(ctx.gpu_activations.d_q);
    FREE_IF(ctx.gpu_activations.d_key_cache);
    FREE_IF(ctx.gpu_activations.d_value_cache);
    FREE_IF(ctx.gpu_activations.d_logits);
    FREE_IF(ctx.gpu_activations.d_next_tokens);
// token2row remains shared
#undef FREE_IF

    ctx.gpu_activations.gate_up_workspace_bytes = 0;
    ctx.gpu_activations.moe_x_workspace_bytes = 0;
    ctx.gpu_activations.mlp1_workspace_bytes = 0;
    ctx.gpu_activations.mlp2_workspace_bytes = 0;
    ctx.gpu_activations.expert_assign_capacity = 0;
    ctx.gpu_activations.assignment_active_capacity = 0;

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

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_counts,
                        (size_t)p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_offsets,
                        ((size_t)p->n_experts + 1) * sizeof(int)));

    size_t max_assignments = (size_t)B * p->experts_per_token;
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_assignments,
                        max_assignments * sizeof(int)));
    ctx.gpu_activations.expert_assign_capacity = max_assignments;

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_assignment_active_slot,
                        max_assignments * sizeof(int)));
    ctx.gpu_activations.assignment_active_capacity = max_assignments;

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_experts,
                        (size_t)p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_counts,
                        (size_t)p->n_experts * sizeof(int)));
    ctx.gpu_activations.active_expert_capacity = p->n_experts;

    size_t gate_up_bytes = max_assignments * (size_t)IM * sizeof(float);
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_gate_up_workspace, gate_up_bytes));
    ctx.gpu_activations.gate_up_workspace_bytes = gate_up_bytes;

    size_t moe_x_bytes = max_assignments * (size_t)H * sizeof(float);
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_moe_x_workspace, moe_x_bytes));
    ctx.gpu_activations.moe_x_workspace_bytes = moe_x_bytes;

    size_t mlp1_bytes = max_assignments * (size_t)(2 * IM) * sizeof(float);
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mlp1_workspace, mlp1_bytes));
    ctx.gpu_activations.mlp1_workspace_bytes = mlp1_bytes;

    size_t mlp2_bytes = max_assignments * (size_t)H * sizeof(float);
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mlp2_workspace, mlp2_bytes));
    ctx.gpu_activations.mlp2_workspace_bytes = mlp2_bytes;

    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                        (size_t)B * (D * (Hq + 2 * Hk)) * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q,
                        (size_t)B * Hq * D * sizeof(float)));

    // Per-batch KV caches
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache,
                        (size_t)B * L * S * KV * sizeof(bf16_t)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                        (size_t)B * L * S * KV * sizeof(bf16_t)));
    // Auxiliary buffers
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                        (size_t)B * V * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens, 
                        (size_t)B * sizeof(int)));

    ctx.capacity_B = B;
  } else {
    if (!ctx.gpu_activations.d_expert_counts) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_counts,
                          (size_t)p->n_experts * sizeof(int)));
    }
    if (!ctx.gpu_activations.d_expert_offsets) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_offsets,
                          ((size_t)p->n_experts + 1) * sizeof(int)));
    }

    if (!ctx.gpu_activations.d_active_experts) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_experts,
                          (size_t)p->n_experts * sizeof(int)));
      ctx.gpu_activations.active_expert_capacity = p->n_experts;
    }
    if (!ctx.gpu_activations.d_active_counts) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_counts,
                          (size_t)p->n_experts * sizeof(int)));
      ctx.gpu_activations.active_expert_capacity = p->n_experts;
    }

    size_t max_assignments = (size_t)ctx.capacity_B * p->experts_per_token;
    if (ctx.gpu_activations.expert_assign_capacity < max_assignments) {
      if (ctx.gpu_activations.d_expert_assignments) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_expert_assignments));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_expert_assignments,
                          max_assignments * sizeof(int)));
      ctx.gpu_activations.expert_assign_capacity = max_assignments;
    }

    if (ctx.gpu_activations.assignment_active_capacity < max_assignments) {
      if (ctx.gpu_activations.d_assignment_active_slot) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_assignment_active_slot));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_assignment_active_slot,
                          max_assignments * sizeof(int)));
      ctx.gpu_activations.assignment_active_capacity = max_assignments;
    }

    size_t required_gate_bytes = max_assignments * (size_t)IM * sizeof(float);
    if (ctx.gpu_activations.gate_up_workspace_bytes < required_gate_bytes) {
      if (ctx.gpu_activations.d_gate_up_workspace) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace,
                          required_gate_bytes));
      ctx.gpu_activations.gate_up_workspace_bytes = required_gate_bytes;
    }

    size_t required_moe_x_bytes = max_assignments * (size_t)H * sizeof(float);
    if (ctx.gpu_activations.moe_x_workspace_bytes < required_moe_x_bytes) {
      if (ctx.gpu_activations.d_moe_x_workspace) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_moe_x_workspace));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_moe_x_workspace,
                          required_moe_x_bytes));
      ctx.gpu_activations.moe_x_workspace_bytes = required_moe_x_bytes;
    }

    size_t required_mlp1_bytes =
        max_assignments * (size_t)(2 * IM) * sizeof(float);
    if (ctx.gpu_activations.mlp1_workspace_bytes < required_mlp1_bytes) {
      if (ctx.gpu_activations.d_mlp1_workspace) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_mlp1_workspace));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mlp1_workspace,
                          required_mlp1_bytes));
      ctx.gpu_activations.mlp1_workspace_bytes = required_mlp1_bytes;
    }

    size_t required_mlp2_bytes = max_assignments * (size_t)H * sizeof(float);
    if (ctx.gpu_activations.mlp2_workspace_bytes < required_mlp2_bytes) {
      if (ctx.gpu_activations.d_mlp2_workspace) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_mlp2_workspace));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mlp2_workspace,
                          required_mlp2_bytes));
      ctx.gpu_activations.mlp2_workspace_bytes = required_mlp2_bytes;
    }

    if (ctx.gpu_activations.active_expert_capacity < p->n_experts) {
      if (ctx.gpu_activations.d_active_experts) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_active_experts));
      }
      if (ctx.gpu_activations.d_active_counts) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_active_counts));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_experts,
                          (size_t)p->n_experts * sizeof(int)));
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_active_counts,
                          (size_t)p->n_experts * sizeof(int)));
      ctx.gpu_activations.active_expert_capacity = p->n_experts;
    }
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

static inline void setup_prompt_ctx(PromptCtx &ctx, Requests *requests, int idx,
                                    Sampler *sampler, Transformer *transformer,
                                    Tokenizer *tokenizer) {
  ctx.idx = idx;
  ctx.input_seq = get_str_req_ptr(requests, idx);
  ctx.output_tokens = get_tok_gen_ptr(requests, idx);
  ctx.max_steps = 1024;
  ctx.sampler = sampler;

  const Config &cfg = transformer->config;

  size_t alloc_tok = ctx.input_seq.length() + 3;
  if (!ctx.prompt_tokens) {
    ctx.prompt_tokens = (int *)malloc(alloc_tok * sizeof(int));
    if (!ctx.prompt_tokens) {
      fprintf(stderr, "OOM: prompt_tokens\n");
      exit(EXIT_FAILURE);
    }
  }

  ctx.num_prompt_tokens = 0;
  encode(tokenizer, ctx.input_seq.c_str(), -1, -1, ctx.prompt_tokens,
         &ctx.num_prompt_tokens, cfg.initial_context_length);
  if (ctx.num_prompt_tokens < 1) {
    fprintf(stderr, "Expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  ctx.logits_size = cfg.vocab_size;
  if (!ctx.h_logits) {
    ctx.h_logits = (float *)malloc((size_t)ctx.logits_size * sizeof(float));
    if (!ctx.h_logits) {
      fprintf(stderr, "OOM: h_logits\n");
      exit(EXIT_FAILURE);
    }
  }

  ctx.pos = 0;
  ctx.token = ctx.prompt_tokens[0];
  ctx.is_context_phase = true;
  ctx.finished = false;
  ctx.num_generated = 0;

  if (ctx.output_str.empty()) {
    const char *first_piece = decode_piece(tokenizer, 200006, ctx.token);
    if (first_piece)
      ctx.output_str += first_piece;
  }
}

static float *gpu_forward_device_batch_logits(Transformer *transformer,
                                              const int *tokens, const int *pos,
                                              int batch_size, int device_id,
                                              int max_pos_in_batch) {
  PROFILE_SCOPE("gpu_forward_device_batch");
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

  // Copy host tokens/positions into device buffers
  HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_tokens, tokens,
                      (size_t)batch_size * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_pos, pos,
                      (size_t)batch_size * sizeof(int), hipMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 gridH_warp((H + TM - 1) / TM, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Launch batched embedding kernel
  {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_batch_kernel", 0);
    dim3 gridH_batch(gridH_thread.x, batch_size, 1);
    copy_embedding_bf16_batch_kernel<<<gridH_batch, block, 0>>>(
        ctx.gpu_activations.d_x,
        ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
        ctx.gpu_activations.d_tokens, batch_size, H);
  }

  for (int l = 0; l < L; ++l) {
    dim3 gridQKV((QKV_D + TM - 1) / TM, 1, 1);
    // Batched QKV projection (RMSNorm + MatMul + Bias) - separate kernels
    {
      // First apply RMSNorm
      {
        PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
        dim3 gridH_batch(1, batch_size, 1);
        rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
            ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H,
            ctx.gpu_activations.d_pos);
      }

      // Then apply MatMul + Bias
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16_mfma", 0);
        dim3 gridQKV_gemm((QKV_D + TM_MM - 1) / TM_MM,
                          (batch_size + TN_MM - 1) / TN_MM, 1);
        dim3 blockQKV(16, 4, 1);
        matmul_bias_gemm_kernel_bf16_mfma<<<gridQKV_gemm, blockQKV>>>(
            ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_qkv_bf16 +
                (size_t)l * (size_t)QKV_D * (size_t)H,
            ctx.gpu_weights_fp32.d_b_qkv
                ? (ctx.gpu_weights_fp32.d_b_qkv + (size_t)l * QKV_D)
                : nullptr,
            H, QKV_D, batch_size, ctx.gpu_activations.d_pos);
      }
    }

    // Scatter QKV to q / caches (batched)
    const int loff = l * S * KV;
    {
      PROFILE_GPU_SCOPE("fused_split_rope_scatter_qkv_batch_kernel", 0);
      dim3 grid_fused(max(Hq, Hk), batch_size, 1);
      dim3 block_fused(D / 2, 1, 1);
      fused_split_rope_scatter_qkv_batch_kernel<<<grid_fused, block_fused, 0>>>(
          /*q_out*/ ctx.gpu_activations.d_q,
          /*key_cache*/ ctx.gpu_activations.d_key_cache,
          /*value_cache*/ ctx.gpu_activations.d_value_cache,
          /*qkv*/ ctx.gpu_activations.d_qkv,
          /*pos*/ ctx.gpu_activations.d_pos,
          /*Hq,Hk,D*/ Hq, Hk, D,
          /*RoPE*/ p->rope_theta, p->rope_scaling_factor,
          p->initial_context_length,
          /*cache*/ loff, L * S * (D * Hk),
          /*B*/ batch_size);
    }

    // Attention (batched)
    {
      PROFILE_GPU_SCOPE("attention_batch_kernel", 0);
      dim3 gridAttn(Hq, batch_size, 1);
      dim3 blockA(WF_SIZE);
      const bool layer_has_window = (p->sliding_window > 0) && ((l & 1) == 0);
      const int att_tokens =
          layer_has_window ? std::min(max_pos_in_batch + 1, p->sliding_window)
                           : (max_pos_in_batch + 1);
      size_t shmem_size = (size_t)(att_tokens + 1) * sizeof(float);
      attention_batch_kernel<<<gridAttn, blockA, shmem_size>>>(
          ctx.gpu_activations.d_tb, ctx.gpu_activations.d_q,
          ctx.gpu_activations.d_key_cache, ctx.gpu_activations.d_value_cache,
          ctx.gpu_weights_fp32.d_attn_sinks, l, ctx.gpu_activations.d_pos, D,
          Hq, Hk, S, layer_has_window ? p->sliding_window : 0, L * S * KV,
          batch_size);
    }

    // Output projection + residual (batched) - separate kernels
    {
      const int O_N = D * Hq;

      // First do MatMul + Bias: temp = tb @ W^T + b
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16_mfma", 0);
        dim3 gridO_gemm((H + TM_MM - 1) / TM_MM,
                        (batch_size + TN_MM - 1) / TN_MM, 1);
        dim3 blockO(16, 4, 1);
        matmul_bias_gemm_kernel_bf16_mfma<<<gridO_gemm, blockO>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_tb,
            ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l * H * O_N,
            ctx.gpu_weights_fp32.d_b_o + l * H, O_N, H, batch_size,
            ctx.gpu_activations.d_pos);
      }

      // Then do residual add: x = x + temp
      {
        PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
        dim3 gridH_batch(gridH_thread.x, batch_size, 1);
        residual_add_batch_kernel<<<gridH_batch, block, 0>>>(
            ctx.gpu_activations.d_x, ctx.gpu_activations.d_t, H, batch_size,
            ctx.gpu_activations.d_pos);
      }
    }

    // FFN (batched)
    {
      PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H,
          ctx.gpu_activations.d_pos);
    }

    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
      dim3 gridE_gemm((E + TM - 1) / TM, batch_size, 1);
      matmul_bias_gemm_kernel_float<<<gridE_gemm, block, 0>>>(
          ctx.gpu_activations.d_router_score, ctx.gpu_activations.d_t,
          ctx.gpu_weights_fp32.d_w_router + (size_t)l * H * E,
          ctx.gpu_weights_fp32.d_b_router + l * E, H, E, batch_size,
          ctx.gpu_activations.d_pos);
    }

    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_batch_kernel", 0);
      dim3 gridTopK_batch(1, batch_size, 1);
      size_t shared_mem_size = (size_t)E * sizeof(float);
      fused_topk_softmax_batch_kernel<<<gridTopK_batch, BLOCK_SIZE,
                                        shared_mem_size>>>(
          ctx.gpu_activations.d_topk_v, ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_router_score, E, p->experts_per_token,
          batch_size, ctx.gpu_activations.d_pos);
    }

    HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_e_agg, 0,
                             (size_t)batch_size * H * sizeof(float)));

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
                             0));

    int total_pairs = batch_size * p->experts_per_token;
    int *d_expert_counts = nullptr;
    int *d_expert_offsets = nullptr;
    int *d_assignment_batches = nullptr;
    int *d_assignment_slots = nullptr;
    std::vector<int> h_counts(E, 0);
    std::vector<int> h_offsets(E + 1, 0);
    int max_assign_per_expert = 0;
    int total_assignments = 0;
    dim3 gridCount((total_pairs + 255) / 256, 1, 1);

    if (total_pairs > 0) {
      {
        PROFILE_GPU_SCOPE("expert_counting", 0);
        HIP_CHECK(hipMalloc(&d_expert_counts, E * sizeof(int)));
        HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, E * sizeof(int)));

        count_expert_assignments_kernel<<<gridCount, 256, 0>>>(
            d_expert_counts, ctx.gpu_activations.d_topk_i,
            ctx.gpu_activations.d_pos, batch_size, p->experts_per_token, E);
        HIP_CHECK(hipMemcpy(h_counts.data(), d_expert_counts, E * sizeof(int),
                            hipMemcpyDeviceToHost));

        for (int e = 0; e < E; ++e) {
          h_offsets[e + 1] = h_offsets[e] + h_counts[e];
          if (h_counts[e] > max_assign_per_expert)
            max_assign_per_expert = h_counts[e];
        }
        total_assignments = h_offsets[E];
      }

      if (total_assignments > 0) {
        {
          PROFILE_GPU_SCOPE("expert_assignment_building", 0);
          HIP_CHECK(hipMalloc(&d_expert_offsets, (E + 1) * sizeof(int)));
          HIP_CHECK(hipMemcpy(d_expert_offsets, h_offsets.data(),
                              (E + 1) * sizeof(int), hipMemcpyHostToDevice));

          HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, E * sizeof(int)));
          HIP_CHECK(hipMalloc(&d_assignment_batches,
                              total_assignments * sizeof(int)));
          HIP_CHECK(
              hipMalloc(&d_assignment_slots, total_assignments * sizeof(int)));

          build_expert_assignments_kernel<<<gridCount, 256, 0>>>(
              ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_pos,
              d_expert_offsets, d_expert_counts, d_assignment_batches,
              d_assignment_slots, batch_size, p->experts_per_token, E);
        }
      }
    }

    if (total_assignments > 0) {
      {
        PROFILE_GPU_SCOPE("mlp1_fused_gemm", 0);
        const int max_tiles =
            (max_assign_per_expert + MLP1_TILE_TOKENS - 1) / MLP1_TILE_TOKENS;
        dim3 block_mlp1(MLP1_TILE_IM, MLP1_TILE_TOKENS, 1);
        dim3 grid_mlp1((IM + MLP1_TILE_IM - 1) / MLP1_TILE_IM, max_tiles, E);
        mlp1_fused_gemm_kernel<<<grid_mlp1, block_mlp1, 0>>>(
            d_gate_up_topk, ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_mlp1_bf16, ctx.gpu_expert_bias.g_b_mlp1,
            d_assignment_batches, d_assignment_slots, d_expert_offsets, l, E, H,
            IM, p->swiglu_limit, batch_size, ctx.gpu_activations.d_pos);
      }

      {
        PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm", 0);
        const int max_tiles =
            (max_assign_per_expert + MLP1_TILE_TOKENS - 1) / MLP1_TILE_TOKENS;
        dim3 block_mlp2(MLP2_TILE_H, MLP2_TILE_TOKENS, 1);
        dim3 grid_mlp2((H + MLP2_TILE_H - 1) / MLP2_TILE_H, max_tiles, E);
        mlp2_bias_weighted_accum_gemm_kernel<<<grid_mlp2, block_mlp2, 0>>>(
            ctx.gpu_activations.d_e_agg, d_gate_up_topk,
            ctx.gpu_weights_bf16.d_w_mlp2_bf16, ctx.gpu_expert_bias.g_b_mlp2,
            d_assignment_batches, d_assignment_slots, d_expert_offsets,
            ctx.gpu_activations.d_topk_v, l, E, IM, H, batch_size,
            ctx.gpu_activations.d_pos);
      }
    }

    {
      if (d_assignment_slots)
        HIP_CHECK(hipFree(d_assignment_slots));
      if (d_assignment_batches)
        HIP_CHECK(hipFree(d_assignment_batches));
      if (d_expert_offsets)
        HIP_CHECK(hipFree(d_expert_offsets));
      if (d_expert_counts)
        HIP_CHECK(hipFree(d_expert_counts));
    }

    // Keep workspace allocated in DeviceContext for reuse

    {
      PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
      dim3 gridH_batch(gridH_thread.x, batch_size, 1);
      residual_add_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_x, ctx.gpu_activations.d_e_agg, H, batch_size,
          ctx.gpu_activations.d_pos);
    }
  }

  // Final head
  {
    // 1) RMSNorm - separate kernel call
    {
      PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_out_w, H, ctx.gpu_activations.d_pos);
    }

    // 2) MatMul for logits - separate GEMM version
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16_mfma", 0);
      dim3 gridV_gemm((V + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM,
                      1);
      dim3 blockV(16, 4, 1);
      matmul_bias_gemm_kernel_bf16_mfma<<<gridV_gemm, blockV>>>(
          ctx.gpu_activations.d_logits, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16, nullptr, H, V, batch_size,
          ctx.gpu_activations.d_pos);
    }
  }

  return ctx.gpu_activations.d_logits;
}

static int *gpu_forward_device_batch(Transformer *transformer,
                                     const int *tokens, const int *pos,
                                     int batch_size, int device_id,
                                     int max_pos_in_batch) {
  float *d_logits = gpu_forward_device_batch_logits(
      transformer, tokens, pos, batch_size, device_id, max_pos_in_batch);

  DeviceContext &ctx = g_devices[device_id];
  HIP_CHECK(hipSetDevice(device_id));

  if (batch_size <= 0)
    return ctx.gpu_activations.d_next_tokens;

  const int vocab_size = model_config->vocab_size;
  const int threads = 256;
  dim3 grid(batch_size, 1, 1);
  dim3 block(threads, 1, 1);
  size_t shared_bytes = (size_t)threads * (sizeof(float) + sizeof(int));

  argmax_batch_kernel<<<grid, block, shared_bytes>>>(
      d_logits, ctx.gpu_activations.d_next_tokens, vocab_size, batch_size,
      ctx.gpu_activations.d_pos);
  HIP_CHECK(hipGetLastError());

  return ctx.gpu_activations.d_next_tokens;
}

static long long run_requests_on_device(Transformer *transformer,
                                        Tokenizer *tokenizer, PromptCtx *ctxs,
                                        int num_ctxs, int device_id) {
  HIP_CHECK(hipSetDevice(device_id));

  long long total_tokens = 0;

  // Process requests in batches of MAX_BATCH_SIZE
  for (int batch_start = 0; batch_start < num_ctxs;
       batch_start += MAX_BATCH_SIZE) {
    const int B = std::min(MAX_BATCH_SIZE, num_ctxs - batch_start);
    PromptCtx *batch_ctxs = ctxs + batch_start;

    // Allocate batch helpers on host
    std::vector<int> h_tokens(B, 0);
    std::vector<int> h_pos(B, 0);
    std::vector<char> h_active(B, 1);

    std::transform(batch_ctxs, batch_ctxs + B, h_tokens.begin(),
                   [](const PromptCtx &ctx) { return ctx.token; });

    // Ensure device buffers sized for B and clear KV caches
    ensure_device_capacity(g_devices[device_id], B);

    std::vector<int> h_next_tokens(B, -1);

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

      int *d_next =
          gpu_forward_device_batch(transformer, h_tokens.data(), h_pos.data(),
                                   B, device_id, max_pos_in_batch);
      HIP_CHECK(hipMemcpy(h_next_tokens.data(), d_next, (size_t)B * sizeof(int),
                          hipMemcpyDeviceToHost));

      // For each active context, advance one step
      for (int i = 0; i < B; ++i) {
        if (!h_active[i])
          continue;
        PromptCtx &ctx = batch_ctxs[i];

        ctx.pos++;
        h_pos[i] = ctx.pos;
        int next;
        if (ctx.pos < ctx.num_prompt_tokens) {
          next = ctx.prompt_tokens[ctx.pos];
        } else {
          ctx.is_context_phase = false;
          next = h_next_tokens[i];
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

    for (int i = 0; i < B; ++i) {
      PromptCtx &ctx = batch_ctxs[i];
      ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;
      total_tokens += (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
    }
  }

  return total_tokens;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_SCOPE("inference");
  if (g_num_devices == 0) {
    fprintf(stderr, "No GPUs initialized for inference!\n");
    return 0;
  }
  const int num_requests = requests->num_reqs;
  if (num_requests == 0)
    return 0;

  printf("Processing %d requests across %d GPUs...\n", num_requests,
         g_num_devices);

  long long num_token_out = 0;
  std::mutex agg_mutex;

  PromptCtx *ctxs = new PromptCtx[num_requests];

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_requests; ++i) {
    setup_prompt_ctx(ctxs[i], requests, i, sampler, transformer, tokenizer);
  }

  int ctxs_per_device = num_requests / g_num_devices;
  int remainder = num_requests % g_num_devices;

  std::vector<std::thread> workers;
  workers.reserve(g_num_devices);

  for (int dev = 0; dev < g_num_devices; ++dev) {
    workers.emplace_back([&, dev]() {
      int num_ctxs_for_device = ctxs_per_device;
      if (dev < remainder) {
        num_ctxs_for_device++;
      }

      if (num_ctxs_for_device == 0)
        return;

      int start_idx = dev * ctxs_per_device + std::min(dev, remainder);

      long long local_tokens = run_requests_on_device(
          transformer, tokenizer, &ctxs[start_idx], num_ctxs_for_device, dev);

      // Aggregate tokens
      std::lock_guard<std::mutex> lk(agg_mutex);
      num_token_out += local_tokens;
    });
  }

  for (auto &th : workers)
    th.join();

  // Sequential, ordered output & cleanup
  for (int idx = 0; idx < num_requests; ++idx) {
    safe_printf(ctxs[idx].output_str.c_str());
    safe_printf("\n");
    free_prompt_ctx_heap_buffers(ctxs[idx]);
  }
  delete[] ctxs;

  printf("Multi-GPU inference completed. Total tokens generated: %lld\n",
         num_token_out);
  return num_token_out;
}

#endif // GETP_RUN
