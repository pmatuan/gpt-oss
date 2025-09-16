#include "attention/attention.cpp"
#include "common/defines.h"
#include "getp_eval.cpp"
#include "matmul/matmul.cpp"
#include "profiler/profiler.cpp"
#include "utility/utility.cpp"
#include "utility/utility.h"
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

  ctx.stage_start = (L / g_num_devices) * device_id;
  ctx.stage_end = (device_id == g_num_devices - 1) ? L : (L / g_num_devices) * (device_id + 1);
  const int local_L = ctx.stage_end - ctx.stage_start;

  printf("Initializing device %d (layers [%d, %d))...\n", device_id, ctx.stage_start, ctx.stage_end);
  debug_print_gpu_memory("before allocations", device_id);

  // Activations
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb, D * Hq * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb2, H * sizeof(float)));

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
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_k, Hk * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_v, Hk * D * sizeof(float)));

  HIP_CHECK(
    hipMalloc(&ctx.gpu_activations.d_key_cache, (size_t)local_L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
            (size_t)local_L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att, Hq * S * sizeof(float)));
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

  if (model_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
#pragma omp parallel for
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] =
            (i - j >= model_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_mask, h_mask,
                        S * S * sizeof(float), hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    ctx.gpu_activations.d_mask = nullptr;
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

// Cleanup device context
static void cleanup_device_context(DeviceContext &ctx) {
  int device_id = ctx.device_id;
  HIP_CHECK(hipSetDevice(device_id));

  HIP_CHECK(hipFree(ctx.gpu_activations.d_x));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_t));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_tb));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_tb2));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_router_score));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_v));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_i));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_e_agg));
  if (ctx.gpu_activations.d_gate_up_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_q));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_k));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_v));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_key_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_value_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_att));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_logits));
  if (ctx.gpu_activations.d_inv_rms)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_cos_vals));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_sin_vals));
  if (ctx.gpu_activations.d_mask)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_mask));
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

  // Enable peer access between all device pairs if possible
  for (int i = 0; i < g_num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < g_num_devices; ++j) {
      if (i == j)
        continue;
      int canAccess = 0;
      hipError_t e = hipDeviceCanAccessPeer(&canAccess, i, j);
      if (e == hipSuccess && canAccess) {
        // Ignore errors if already enabled
        hipDeviceEnablePeerAccess(j, 0);
      }
    }
  }
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
  const int local_L = ctx.stage_end - ctx.stage_start;

  // Free previous activations to re-alloc at batch size B
  if (need_realloc) {
#define FREE_IF(p)                                                             \
  if ((p))                                                                     \
  HIP_CHECK(hipFree((p)))

    FREE_IF(ctx.gpu_activations.d_x);
    FREE_IF(ctx.gpu_activations.d_t);
    FREE_IF(ctx.gpu_activations.d_tb);
    FREE_IF(ctx.gpu_activations.d_tb2);
    FREE_IF(ctx.gpu_activations.d_router_score);
    FREE_IF(ctx.gpu_activations.d_topk_v);
    FREE_IF(ctx.gpu_activations.d_topk_i);
    FREE_IF(ctx.gpu_activations.d_gate_up);
    FREE_IF(ctx.gpu_activations.d_e_agg);
    FREE_IF(ctx.gpu_activations.d_qkv);
    FREE_IF(ctx.gpu_activations.d_q);
    FREE_IF(ctx.gpu_activations.d_k);
    FREE_IF(ctx.gpu_activations.d_v);
    FREE_IF(ctx.gpu_activations.d_key_cache);
    FREE_IF(ctx.gpu_activations.d_value_cache);
    FREE_IF(ctx.gpu_activations.d_att);
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
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_tb2, (size_t)B * H * sizeof(float)));

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
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_k,
                        (size_t)B * Hk * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_v,
                        (size_t)B * Hk * D * sizeof(float)));

    // Per-batch KV caches (stage-local only)
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache,
              (size_t)B * local_L * S * KV * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
              (size_t)B * local_L * S * KV * sizeof(float)));
    // Auxiliary buffers
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att,
                        (size_t)B * Hq * S * sizeof(float)));
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

static inline void setup_prompt_ctx(PromptCtx &ctx, Requests *requests, int idx,
                                    Sampler *sampler, Transformer *transformer,
                                    Tokenizer *tokenizer) {
  ctx.idx = idx;
  ctx.input_seq = get_str_req_ptr(requests, idx);
  ctx.output_tokens = get_tok_gen_ptr(requests, idx);
  ctx.max_steps = requests->max_seq_len;
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

static float *gpu_forward_device_batch(Transformer *transformer,
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
  const int local_L = ctx.stage_end - ctx.stage_start;

  // Copy host tokens/positions into device buffers
  HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_tokens, tokens,
                      (size_t)batch_size * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_pos, pos,
                      (size_t)batch_size * sizeof(int), hipMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 gridH_warp((H + TM - 1) / TM, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Launch batched embedding kernel (only on first stage)
  if (ctx.device_id == 0) {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_batch_kernel", 0);
    dim3 gridH_batch(gridH_thread.x, batch_size, 1);
    copy_embedding_bf16_batch_kernel<<<gridH_batch, block, 0>>>(
        ctx.gpu_activations.d_x,
        ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
        ctx.gpu_activations.d_tokens, batch_size, H);
  }

  for (int l = ctx.stage_start; l < ctx.stage_end; ++l) {
    const int l_local = l - ctx.stage_start;
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV((QKV_D + TM - 1) / TM, 1, 1);
    // Batched QKV projection (RMSNorm + MatMul + Bias) - separate kernels
    {
      // First apply RMSNorm
      {
        PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
        dim3 gridH_batch(1, batch_size, 1);
        rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
            ctx.gpu_weights_fp32.d_rms_attn_w + l_local * H, H,
            ctx.gpu_activations.d_pos);
      }

      // Then apply MatMul + Bias
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridQKV_gemm((QKV_D + TM - 1) / TM, (batch_size + B_TILE - 1) / B_TILE, 1);
        matmul_bias_gemm_kernel_bf16<<<gridQKV_gemm, block, 0>>>(
            ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l_local * QKV_D * H,
            ctx.gpu_weights_fp32.d_b_qkv + l_local * QKV_D, H, QKV_D, batch_size,
            ctx.gpu_activations.d_pos);
      }
    }

    // Scatter QKV to q / caches (batched)
    const int loff = l_local * S * KV;
    {
      PROFILE_GPU_SCOPE("split_qkv_scatter_to_cache_batch_kernel", 0);
      dim3 gridQKV_batch(gridQKV.x, batch_size, 1);
      split_qkv_scatter_to_cache_batch_kernel<<<gridQKV_batch, block, 0>>>(
          ctx.gpu_activations.d_q, ctx.gpu_activations.d_key_cache,
          ctx.gpu_activations.d_value_cache, ctx.gpu_activations.d_qkv, Hq, Hk,
          D, loff, ctx.gpu_activations.d_pos, batch_size, local_L * S * KV);
    }

    // Apply RoPE to q and cached k (batched)
    {
      PROFILE_GPU_SCOPE("fused_inline_rope_qkv_batch_kernel", 0);
      dim3 gridApply_batch(max(Hq, Hk), batch_size, 1);
      fused_inline_rope_qkv_batch_kernel<<<gridApply_batch, D / 2, 0>>>(
          ctx.gpu_activations.d_q, ctx.gpu_activations.d_key_cache,
          ctx.gpu_activations.d_pos, p->rope_theta, Hq, Hk, D,
          p->rope_scaling_factor, p->initial_context_length, loff, local_L * S * KV,
          batch_size);
    }

    // Attention (batched)
    {
      PROFILE_GPU_SCOPE("attention_batch_kernel", 0);
      dim3 gridAttn(Hq, batch_size, 1);
      dim3 blockA(WF_SIZE);
      size_t shmem_size = (size_t)(max_pos_in_batch + 2) * sizeof(float);
      attention_batch_kernel<<<gridAttn, blockA, shmem_size>>>(
          ctx.gpu_activations.d_tb, ctx.gpu_activations.d_q,
          ctx.gpu_activations.d_key_cache, ctx.gpu_activations.d_value_cache,
          ctx.gpu_weights_fp32.d_attn_sinks, l_local, ctx.gpu_activations.d_pos, D,
          Hq, Hk, S,
          (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.gpu_activations.d_mask
                                                  : nullptr,
          local_L * S * KV, batch_size);
    }

    // Output projection + residual (batched) - separate kernels
    {
      const int O_N = D * Hq;

      // First do MatMul + Bias: temp = tb @ W^T + b
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridO_gemm((H + TM - 1) / TM, (batch_size + B_TILE - 1) / B_TILE, 1);
        matmul_bias_gemm_kernel_bf16<<<gridO_gemm, block, 0>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_tb,
            ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l_local * H * O_N,
            ctx.gpu_weights_fp32.d_b_o + l_local * H, O_N, H, batch_size,
            ctx.gpu_activations.d_pos);
      }

      // Then do residual add: x = x + temp
      {
        PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
        dim3 gridH_batch(gridH_thread.x, batch_size, 1);
        residual_add_batch_kernel<<<gridH_batch, block, 0>>>(
            ctx.gpu_activations.d_x, ctx.gpu_activations.d_t,
            H, batch_size, ctx.gpu_activations.d_pos);
      }
    }

    // FFN (batched)
    {
      PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_ffn_w + l_local * H, H,
          ctx.gpu_activations.d_pos);
    }

    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
      dim3 gridE_gemm((E + TM - 1) / TM, batch_size, 1);
      matmul_bias_gemm_kernel_float<<<gridE_gemm, block, 0>>>(
          ctx.gpu_activations.d_router_score, ctx.gpu_activations.d_t,
          ctx.gpu_weights_fp32.d_w_router + (size_t)l_local * H * E,
          ctx.gpu_weights_fp32.d_b_router + l_local * E, H, E, batch_size,
          ctx.gpu_activations.d_pos);
    }

    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_batch_kernel", 0);
      dim3 gridTopK_batch(1, batch_size, 1);
      size_t shared_mem_size = (size_t)E * sizeof(float);
      fused_topk_softmax_batch_kernel<<<gridTopK_batch, BLOCK_SIZE,
                                        shared_mem_size>>>(
          ctx.gpu_activations.d_topk_v, ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_router_score, E,
          p->experts_per_token, batch_size, ctx.gpu_activations.d_pos);
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

    // --- MLP1: per-batch per-expert, no CB
    {
      PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 gridIM((IM + TM - 1) / TM, batch_size, 1);
      gridIM.z = p->experts_per_token;
      mlp1_fused_gemm_kernel<<<gridIM, block, 0>>>(
          /*gate_up_topk[K,B,IM]*/ d_gate_up_topk,
          /*x[B,H]*/ ctx.gpu_activations.d_t,
          /*w*/ ctx.gpu_weights_bf16.d_w_mlp1_bf16,
          /*b*/ ctx.gpu_expert_bias.g_b_mlp1,
          /*topk_i*/ ctx.gpu_activations.d_topk_i,
          /*layer*/ l_local,
          /*E,H,IM*/ E, H, IM,
          /*clip*/ p->swiglu_limit,
          /*B,K*/ batch_size, ctx.gpu_activations.d_pos);
    }

    // --- MLP2: per-batch per-expert, no CB
    {
      PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm_kernel", 0);
      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 gridH((H + TM - 1) / TM, batch_size, 1);
      gridH.z = p->experts_per_token;
      mlp2_bias_weighted_accum_gemm_kernel<<<gridH, block, 0>>>(
          /*e_agg[B,H]*/ ctx.gpu_activations.d_e_agg,
          /*gate_up[K,B,IM]*/ d_gate_up_topk,
          /*w*/ ctx.gpu_weights_bf16.d_w_mlp2_bf16,
          /*b*/ ctx.gpu_expert_bias.g_b_mlp2,
          /*topk_i, topk_v*/ ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_topk_v,
          /*layer*/ l_local,
          /*E,IM,H,B,K*/ E, IM, H, batch_size, ctx.gpu_activations.d_pos);
    }

    // Keep workspace allocated in DeviceContext for reuse

    {
      PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
      dim3 gridH_batch(gridH_thread.x, batch_size, 1);
      residual_add_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_x, ctx.gpu_activations.d_e_agg,
          H, batch_size, ctx.gpu_activations.d_pos);
    }
  }
  if (ctx.device_id < g_num_devices - 1) {
    DeviceContext &next = g_devices[ctx.device_id + 1];
    HIP_CHECK(hipMemcpyPeerAsync(next.gpu_activations.d_x,
                                 next.device_id,
                                 ctx.gpu_activations.d_x,
                                 ctx.device_id,
                                 H * batch_size * sizeof(float),
                                 0));
    HIP_CHECK(hipDeviceSynchronize());
  }
  if (ctx.device_id < g_num_devices - 1) {
    DeviceContext &next = g_devices[ctx.device_id + 1];
    HIP_CHECK(hipMemcpyPeerAsync(next.gpu_activations.d_x,
                                 next.device_id,
                                 ctx.gpu_activations.d_x,
                                 ctx.device_id,
                                 H * batch_size * sizeof(float),
                                 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // Final head
  if (ctx.device_id == g_num_devices - 1) {
    // 1) RMSNorm - separate kernel call
    {
      PROFILE_GPU_SCOPE("rmsnorm_batch_kernel", 0);
      dim3 gridH_batch(1, batch_size, 1);
      rmsnorm_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_out_w, H,
          ctx.gpu_activations.d_pos);
    }

    // 2) MatMul for logits - separate GEMM version
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
      dim3 gridV_gemm((V + TM - 1) / TM, (batch_size + B_TILE - 1) / B_TILE, 1);
      matmul_bias_gemm_kernel_bf16<<<gridV_gemm, block, 0>>>(
          ctx.gpu_activations.d_logits, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16, nullptr, H, V, batch_size,
          ctx.gpu_activations.d_pos);
    }
    return ctx.gpu_activations.d_logits;
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
    ensure_device_capacity(g_devices[dev], B);
  }
  const Config *p = model_config;
  const int V = p->vocab_size;
  // Temporary host buffer for batched logits (from last stage)
  std::vector<float> h_logits_batch((size_t)B * V);

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

    float *d_log = nullptr;
    for (int dev = 0; dev < g_num_devices; ++dev) {
      d_log = gpu_forward_device_batch(transformer, h_tokens.data(), h_pos.data(), B,
                                       dev, max_pos_in_batch);
    }
    // Last call sets current device to the last; copy logits from there
    HIP_CHECK(hipMemcpy(h_logits_batch.data(), d_log,
                        (size_t)B * V * sizeof(float), hipMemcpyDeviceToHost));

    // For each active context, advance one step
    for (int i = 0; i < B; ++i) {
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
      if (ctx.max_steps != 0 && ctx.pos >= ctx.max_steps) {
        ctx.finished = true;
        h_active[i] = 0;
        num_finished++;
      }
    }
  }

  for (int i = 0; i < B; ++i) {
    PromptCtx &ctx = ctxs[i];
    ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;
    total_tokens += (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
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
  PromptCtx *ctxs = new PromptCtx[num_requests];

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_requests; ++i) {
    setup_prompt_ctx(ctxs[i], requests, i, sampler, transformer, tokenizer);
  }

  // Single pipeline over all devices: run all requests together
  num_token_out = run_requests_pipeline(transformer, tokenizer, ctxs, num_requests);

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
