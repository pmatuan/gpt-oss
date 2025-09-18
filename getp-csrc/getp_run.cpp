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
#include <thread>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

#define PTR_B(ptr, stride, b) ((ptr) + (size_t)(b) * (size_t)(stride))

static Config *model_config;

static std::vector<DeviceContext> g_devices;
static int g_num_devices = 0;

// Forward declaration for pre-allocation in warm_up
static inline void ensure_device_capacity(DeviceContext &ctx, int capacity_B);

static void init_device_context(DeviceContext &ctx, int device_id,
                                Transformer *transformer) {
  PROFILE_SCOPE("init_device_context");
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

  // Pre-allocate workspace for requests
  ctx.gpu_activations.d_gate_up_workspace = nullptr;

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (D * (Hq + 2 * Hk)) * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_key_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                      L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));
  // next-token device buffer (single slot initially)
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens, 1 * sizeof(int)));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_sin_vals, (D / 2) * sizeof(float)));

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

  // Request helpers (lazily allocated as needed)
  ctx.capacity_B = 0;  // Will be set to 1 when first allocated
  ctx.gpu_activations.d_tokens = nullptr;
  ctx.gpu_activations.d_pos = nullptr;
  ctx.gpu_activations.d_inv_rms = nullptr;
  ctx.gpu_activations.d_token2row = nullptr;
  ctx.h_tokens_pinned = nullptr;
  ctx.h_pos_pinned = nullptr;

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
  PROFILE_SCOPE("cleanup_device_context");
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
  HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_key_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_value_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_att));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_logits));
  if (ctx.gpu_activations.d_next_tokens)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_next_tokens));
  if (ctx.gpu_activations.d_inv_rms)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_cos_vals));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_sin_vals));
  if (ctx.gpu_activations.d_mask)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_mask));
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

  if (ctx.io_streams) {
    for (int i = 0; i < ctx.capacity_B; ++i) HIP_CHECK(hipStreamDestroy(ctx.io_streams[i]));
    free(ctx.io_streams);
    ctx.io_streams = nullptr;
  }
  if (ctx.compute_done_events) {
    for (int i = 0; i < ctx.capacity_B; ++i) HIP_CHECK(hipEventDestroy(ctx.compute_done_events[i]));
    free(ctx.compute_done_events);
    ctx.compute_done_events = nullptr;
  }
  if (ctx.h_tokens_pinned) { HIP_CHECK(hipHostFree(ctx.h_tokens_pinned)); ctx.h_tokens_pinned = nullptr; }
  if (ctx.h_pos_pinned) { HIP_CHECK(hipHostFree(ctx.h_pos_pinned)); ctx.h_pos_pinned = nullptr; }

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
  PROFILE_SCOPE("warm_up");
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

  // Pre-allocate capacity to avoid mid-run KV re-allocs if requested
  int maxB = 0;
  if (const char* emax = getenv("GETP_MAX_BATCH")) { int v = atoi(emax); if (v > 0) maxB = v; }
  if (maxB > 0) {
#pragma omp parallel for num_threads(g_num_devices)
    for (int i = 0; i < g_num_devices; ++i) {
      ensure_device_capacity(g_devices[i], maxB);
      // Create per-slot IO streams and events once
      DeviceContext &ctx = g_devices[i];
      if (!ctx.io_streams) ctx.io_streams = (hipStream_t*)malloc(sizeof(hipStream_t) * ctx.capacity_B);
      if (!ctx.compute_done_events) ctx.compute_done_events = (hipEvent_t*)malloc(sizeof(hipEvent_t) * ctx.capacity_B);
      for (int b = 0; b < ctx.capacity_B; ++b) {
        HIP_CHECK(hipStreamCreateWithFlags(&ctx.io_streams[b], hipStreamNonBlocking));
        HIP_CHECK(hipEventCreateWithFlags(&ctx.compute_done_events[b], hipEventDisableTiming));
      }
    }
  }
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  PROFILE_SCOPE("finish");
#pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    cleanup_device_context(g_devices[i]);
  }
  g_devices.clear();
  g_num_devices = 0;
}

// ============================ Forward ============================
// Ensure device has capacity for single request (reallocates activations if needed)
static inline void ensure_device_capacity(DeviceContext &ctx, int capacity_B) {
  PROFILE_SCOPE("ensure_device_capacity");
  HIP_CHECK(hipSetDevice(ctx.device_id));

  if (ctx.capacity_B >= capacity_B) return;
  const int prevB = ctx.capacity_B;

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
  const int QKV_D = D * (Hq + 2*Hk);
  const int B  = capacity_B;

  // Free previous activations to re-alloc
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
    FREE_IF(ctx.gpu_activations.d_key_cache);
    FREE_IF(ctx.gpu_activations.d_value_cache);
    FREE_IF(ctx.gpu_activations.d_att);
    FREE_IF(ctx.gpu_activations.d_logits);
// mask remains shared
#undef FREE_IF

  // Re-allocate for single request
  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_x, (size_t)B*H * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_t, (size_t)B*H * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb,
                      (size_t)B*D * Hq * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score,
                      (size_t)B*p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                      (size_t)B*p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                      (size_t)B*p->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up,
                      (size_t)B*IM * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_e_agg, (size_t)B*H * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (size_t)B*QKV_D * sizeof(float)));

  // KV caches
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache,
                      (size_t)B*L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                      (size_t)B*L * S * KV * sizeof(float)));
  // Auxiliary buffers
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att,
                      (size_t)B*Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                      (size_t)B*V * sizeof(float)));
  // allocate next-tokens buffer sized by B
  if (ctx.gpu_activations.d_next_tokens)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_next_tokens));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens, (size_t)B * sizeof(int)));

  // inv_rms / tokens / pos theo B
  if (ctx.gpu_activations.d_inv_rms) HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_inv_rms, (size_t)B * sizeof(float)));
  if (ctx.gpu_activations.d_tokens) HIP_CHECK(hipFree(ctx.gpu_activations.d_tokens));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tokens, (size_t)B * sizeof(int)));
  if (ctx.gpu_activations.d_pos) HIP_CHECK(hipFree(ctx.gpu_activations.d_pos));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_pos, (size_t)B * sizeof(int)));

  // pinned host arrays for tokens/pos (graph-friendly updates)
  if (ctx.h_tokens_pinned) HIP_CHECK(hipHostFree(ctx.h_tokens_pinned));
  HIP_CHECK(hipHostMalloc((void**)&ctx.h_tokens_pinned, (size_t)B * sizeof(int)));
  if (ctx.h_pos_pinned) HIP_CHECK(hipHostFree(ctx.h_pos_pinned));
  HIP_CHECK(hipHostMalloc((void**)&ctx.h_pos_pinned, (size_t)B * sizeof(int)));

  ctx.capacity_B = capacity_B;  // capacity in samples

  // workspace cho MLP top-k theo B
  const size_t gate_up_topk_bytes = (size_t)B * p->experts_per_token * IM * sizeof(float);
  if (ctx.gpu_activations.d_gate_up_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace, gate_up_topk_bytes));

  // B stream (mỗi stream phục vụ 1 sample trong batch)
  if (!ctx.streams) ctx.streams = (hipStream_t*)malloc(sizeof(hipStream_t)*B);
  for (int i = ctx.n_streams; i < B; ++i)
    HIP_CHECK(hipStreamCreateWithFlags(&ctx.streams[i], hipStreamNonBlocking));
  ctx.n_streams = B;
  // IO streams and events sized to B
  if (!ctx.io_streams) {
    ctx.io_streams = (hipStream_t*)calloc(B, sizeof(hipStream_t));
    for (int i = 0; i < B; ++i) {
      HIP_CHECK(hipStreamCreateWithFlags(&ctx.io_streams[i], hipStreamNonBlocking));
    }
  } else if (B > prevB) {
    ctx.io_streams = (hipStream_t*)realloc(ctx.io_streams, sizeof(hipStream_t)*B);
    for (int i = prevB; i < B; ++i) {
      HIP_CHECK(hipStreamCreateWithFlags(&ctx.io_streams[i], hipStreamNonBlocking));
    }
  }
  if (!ctx.compute_done_events) {
    ctx.compute_done_events = (hipEvent_t*)calloc(B, sizeof(hipEvent_t));
    for (int i = 0; i < B; ++i) {
      HIP_CHECK(hipEventCreateWithFlags(&ctx.compute_done_events[i], hipEventDisableTiming));
    }
  } else if (B > prevB) {
    ctx.compute_done_events = (hipEvent_t*)realloc(ctx.compute_done_events, sizeof(hipEvent_t)*B);
    for (int i = prevB; i < B; ++i) {
      HIP_CHECK(hipEventCreateWithFlags(&ctx.compute_done_events[i], hipEventDisableTiming));
    }
  }
  ctx.capacity_B = B;

  // Optional IO streams and events per slot
  if (!ctx.io_streams) ctx.io_streams = (hipStream_t*)malloc(sizeof(hipStream_t)*B);
  if (!ctx.compute_done_events) ctx.compute_done_events = (hipEvent_t*)malloc(sizeof(hipEvent_t)*B);
  for (int i = 0; i < B; ++i) {
    if (!ctx.io_streams[i]) HIP_CHECK(hipStreamCreateWithFlags(&ctx.io_streams[i], hipStreamNonBlocking));
    // Create event only if not yet created; can't check pointer, create anew if default-inited
    HIP_CHECK(hipEventCreateWithFlags(&ctx.compute_done_events[i], hipEventDisableTiming));
  }
}

static inline void setup_prompt_ctx(PromptCtx &ctx, Requests *requests, int idx,
                                    Sampler *sampler, Transformer *transformer,
                                    Tokenizer *tokenizer) {
  PROFILE_SCOPE("setup_prompt_ctx");
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

static float *gpu_forward_device(Transformer *transformer,
                                 int token_host, int pos_host,
                                 int b, hipStream_t s,
                                 int device_id) {
  PROFILE_SCOPE("gpu_forward_device");
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

  // Feature toggles
  bool use_fused = false;
  if (const char* ef = getenv("GETP_USE_FUSED")) {
    if (ef[0] == '1' || ef[0] == 't' || ef[0] == 'T' || ef[0] == 'y' || ef[0] == 'Y') use_fused = true;
  }

  // Copy from pinned host staging into device buffers
  ctx.h_tokens_pinned[b] = token_host;
  ctx.h_pos_pinned[b] = pos_host;
  HIP_CHECK(hipMemcpyAsync(PTR_B(ctx.gpu_activations.d_tokens, 1, b), &ctx.h_tokens_pinned[b],
                      sizeof(int), hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(PTR_B(ctx.gpu_activations.d_pos, 1, b), &ctx.h_pos_pinned[b],
                      sizeof(int), hipMemcpyHostToDevice, s));

  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 gridH_warp((H + TM - 1) / TM, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Launch embedding kernel (per-slot launch; pass pointers already offset to b)
  {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_kernel", 0);
  dim3 gridH(gridH_thread.x, 1, 1);
  copy_embedding_bf16_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
    PTR_B(ctx.gpu_activations.d_x, H, b),
    ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
    PTR_B(ctx.gpu_activations.d_tokens, 1, b), H);
  }

  for (int l = 0; l < L; ++l) {
    dim3 gridQKV((QKV_D + TM - 1) / TM, 1, 1);
    // QKV projection (RMSNorm + MatMul + Bias)
    if (use_fused) {
      PROFILE_GPU_SCOPE("fused_rmsnorm_matmul_bias_bf16", 0);
      dim3 gridQKV_gemm((QKV_D + TM - 1) / TM, 1, 1);
      size_t shmem = (size_t)(1 + TK + LDS_PAD) * sizeof(float);
      fused_rmsnorm_matmul_bias_bf16<<<dim3(gridQKV_gemm.x, 1, 1), block, shmem, s>>>(
        PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b),
        PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_fp32.d_rms_attn_w + l * H,
        PTR_B(ctx.gpu_weights_bf16.d_w_qkv_bf16, QKV_D * H, l),
        PTR_B(ctx.gpu_weights_fp32.d_b_qkv, QKV_D, l),
        H, QKV_D, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    } else {
      // Separate RMSNorm then GEMM + Bias
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, 1, 1);
        rmsnorm_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
          ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H,
          PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridQKV_gemm((QKV_D + TM - 1) / TM, 1, 1);
        matmul_bias_gemm_kernel_bf16<<<dim3(gridQKV_gemm.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b), PTR_B(ctx.gpu_activations.d_t, H, b),
          PTR_B(ctx.gpu_weights_bf16.d_w_qkv_bf16, QKV_D * H, l), PTR_B(ctx.gpu_weights_fp32.d_b_qkv, QKV_D, l),
          H, QKV_D, PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
    }

    // Scatter QKV to caches (K and V only, Q remains in QKV buffer)
    const int loff = l * S * KV;
    {
      PROFILE_GPU_SCOPE("split_qkv_scatter_to_cache_kernel", 0);
      split_qkv_scatter_to_cache_kernel<<<dim3(gridQKV.x, 1, 1), block, 0, s>>>(
        nullptr,
        PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
        PTR_B(ctx.gpu_activations.d_value_cache, (size_t)L * S * KV, b),
        PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b), Hq, Hk,
        D, loff, PTR_B(ctx.gpu_activations.d_pos, 1, b), L * S * KV);
    }

    // Apply RoPE to q (in QKV buffer) and cached k
    {
      PROFILE_GPU_SCOPE("fused_inline_rope_qkv_kernel", 0);
      dim3 gridApply(max(Hq, Hk), 1, 1);
      fused_inline_rope_qkv_kernel<<<dim3(gridApply.x, 1, 1), D / 2, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b),
        PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
        PTR_B(ctx.gpu_activations.d_pos, 1, b), p->rope_theta, Hq, Hk, D,
        p->rope_scaling_factor, p->initial_context_length, loff, L * S * KV);
    }

    // Attention (reads Q directly from QKV buffer)
    {
      PROFILE_GPU_SCOPE("attention_kernel", 0);
  dim3 gridAttn(Hq, 1, 1);
  dim3 blockA(WF_SIZE);
  // Use a fixed upper-bound for shared memory so captured graphs remain valid
  // across varying positions. Avoids undersized shmem on graph replays.
  size_t shmem_size = (size_t)(S + 2) * sizeof(float);
      attention_kernel<<<dim3(gridAttn.x, 1, 1), blockA, shmem_size, s>>>(
        PTR_B(ctx.gpu_activations.d_tb, Hq * D, b),
        PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b),
        PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
        PTR_B(ctx.gpu_activations.d_value_cache, (size_t)L * S * KV, b),
        ctx.gpu_weights_fp32.d_attn_sinks, l, PTR_B(ctx.gpu_activations.d_pos, 1, b), D,
        Hq, Hk, S,
        (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.gpu_activations.d_mask
                            : nullptr,
        L * S * KV);
    }

    // Output projection + residual - separate kernels
    {
      const int O_N = D * Hq;

      // Fused RMSNorm (on x) isn't needed here; directly MatMul + Bias
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridO_gemm((H + TM - 1) / TM, 1, 1);
        matmul_bias_gemm_kernel_bf16<<<dim3(gridO_gemm.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_tb, Hq * D, b),
          PTR_B(ctx.gpu_weights_bf16.d_w_o_bf16, H * O_N, l),
          PTR_B(ctx.gpu_weights_fp32.d_b_o, H, l), O_N, H,
          PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }

      // Then do residual add: x = x + temp
      {
        PROFILE_GPU_SCOPE("residual_add_kernel", 0);
        dim3 gridH(gridH_thread.x, 1, 1);
        residual_add_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_x, H, b), PTR_B(ctx.gpu_activations.d_t, H, b),
          H, PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
    }

    // FFN
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, 1, 1);
      rmsnorm_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H,
        PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
      dim3 gridE_gemm((E + TM - 1) / TM, 1, 1);
      matmul_bias_gemm_kernel_float<<<dim3(gridE_gemm.x, 1, 1), block, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_router_score, E, b), PTR_B(ctx.gpu_activations.d_t, H, b),
        ctx.gpu_weights_fp32.d_w_router + (size_t)l * H * E,
        ctx.gpu_weights_fp32.d_b_router + l * E, H, E,
        PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_kernel", 0);
  dim3 gridTopK(1, 1, 1);
  size_t shared_mem_size = (size_t)E * sizeof(float);
  fused_topk_softmax_kernel<<<dim3(gridTopK.x, 1, 1), BLOCK_SIZE,
        shared_mem_size, s>>>(
    PTR_B(ctx.gpu_activations.d_topk_v, p->experts_per_token, b),
    PTR_B(ctx.gpu_activations.d_topk_i, p->experts_per_token, b),
    PTR_B(ctx.gpu_activations.d_router_score, E, b), E,
    p->experts_per_token, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

  HIP_CHECK(hipMemsetAsync(PTR_B(ctx.gpu_activations.d_e_agg, H, b), 0,
               (size_t)H * sizeof(float), s));

    // Use pre-allocated workspace from DeviceContext to avoid repeated
    // malloc/free
  const int K = p->experts_per_token;
  size_t gate_up_topk_bytes = (size_t)K * (size_t)IM * sizeof(float);
  float *d_gate_up_topk = PTR_B(ctx.gpu_activations.d_gate_up_workspace, (size_t)K * IM, b);
  HIP_CHECK(hipMemsetAsync(d_gate_up_topk, 0,
               gate_up_topk_bytes,
               s));

    // --- MLP1: per-expert
    {
      PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 gridIM((IM + TM - 1) / TM, 1, EXPERT_PER_TOKEN);
    mlp1_fused_gemm_kernel<<<dim3(gridIM.x, 1, gridIM.z), block, 0, s>>>(
      /*gate_up_topk[K,B,IM]*/ d_gate_up_topk,
      /*x[B,H]*/ PTR_B(ctx.gpu_activations.d_t, H, b),
          /*w*/ ctx.gpu_weights_bf16.d_w_mlp1_bf16,
          /*b*/ ctx.gpu_expert_bias.g_b_mlp1,
      /*topk_i*/ PTR_B(ctx.gpu_activations.d_topk_i, K, b),
          /*layer*/ l,
          /*E,H,IM*/ E, H, IM,
          /*clip*/ p->swiglu_limit,
      /*pos*/ PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    // --- MLP2: per-expert
    {
      PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm_kernel", 0);
      dim3 block(BLOCK_SIZE, 1, 1);
      dim3 gridH((H + TM - 1) / TM, 1, EXPERT_PER_TOKEN);
      mlp2_bias_weighted_accum_gemm_kernel<<<dim3(gridH.x, 1, gridH.z), block, 0, s>>>(
        /*e_agg[B,H]*/ PTR_B(ctx.gpu_activations.d_e_agg, H, b),
        /*gate_up[K,B,IM]*/ d_gate_up_topk,
        /*w*/ ctx.gpu_weights_bf16.d_w_mlp2_bf16,
        /*b*/ ctx.gpu_expert_bias.g_b_mlp2,
        /*topk_i, topk_v*/ PTR_B(ctx.gpu_activations.d_topk_i, K, b),
        PTR_B(ctx.gpu_activations.d_topk_v, K, b),
        /*layer*/ l,
        /*E,IM,H*/ E, IM, H, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    // Keep workspace allocated in DeviceContext for reuse

    {
      PROFILE_GPU_SCOPE("residual_add_kernel", 0);
      dim3 gridH(gridH_thread.x, 1, 1);
      residual_add_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_x, H, b), PTR_B(ctx.gpu_activations.d_e_agg, H, b),
        H, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }
  }

  // Final head
  {
    if (use_fused) {
      PROFILE_GPU_SCOPE("fused_rmsnorm_matmul_bias_bf16", 0);
      dim3 gridV_gemm((V + TM - 1) / TM, 1, 1);
      size_t shmem = (size_t)(1 + TK + LDS_PAD) * sizeof(float);
      fused_rmsnorm_matmul_bias_bf16<<<dim3(gridV_gemm.x, 1, 1), block, shmem, s>>>(
        PTR_B(ctx.gpu_activations.d_logits, V, b),
        PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_fp32.d_rms_out_w,
        ctx.gpu_weights_bf16.d_out_bf16,
        nullptr,
        H, V, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    } else {
      // Separate RMSNorm then GEMM
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, 1, 1);
        rmsnorm_kernel<<<dim3(gridH.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
          ctx.gpu_weights_fp32.d_rms_out_w, H,
          PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridV_gemm((V + TM - 1) / TM, 1, 1);
        matmul_bias_gemm_kernel_bf16<<<dim3(gridV_gemm.x, 1, 1), block, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_logits, V, b), PTR_B(ctx.gpu_activations.d_t, H, b),
          ctx.gpu_weights_bf16.d_out_bf16, nullptr, H, V,
          PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
    }
  }

  return PTR_B(ctx.gpu_activations.d_logits, V, b);
}

// Simple device-side argmax reduction over logits[V] -> out one int
__global__ void argmax_reduce_kernel(const float* __restrict__ logits, int V,
                                     int* __restrict__ out_idx) {
  extern __shared__ float smem[];
  float* bufv = smem;
  int* bufi = reinterpret_cast<int*>(bufv + blockDim.x);

  float bestv = -INFINITY;
  int besti = -1;
  for (int idx = threadIdx.x; idx < V; idx += blockDim.x) {
    float x = logits[idx];
    if (x > bestv) { bestv = x; besti = idx; }
  }
  bufv[threadIdx.x] = bestv;
  bufi[threadIdx.x] = besti;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (bufv[threadIdx.x + s] > bufv[threadIdx.x]) {
        bufv[threadIdx.x] = bufv[threadIdx.x + s];
        bufi[threadIdx.x] = bufi[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out_idx = bufi[0];
  }
}

// Batched variant: grid.y = B, each block reduces one sample's logits slice
__global__ void argmax_reduce_batched_kernel(const float* __restrict__ logits,
                                             int V, int* __restrict__ out_idx) {
  extern __shared__ float smem[];
  float* bufv = smem;
  int* bufi = reinterpret_cast<int*>(bufv + blockDim.x);

  const int b = blockIdx.y;
  const float* logits_b = logits + (size_t)b * V;
  int* out_b = out_idx + b;

  float bestv = -INFINITY;
  int besti = -1;
  for (int idx = threadIdx.x; idx < V; idx += blockDim.x) {
    float x = logits_b[idx];
    if (x > bestv) { bestv = x; besti = idx; }
  }
  bufv[threadIdx.x] = bestv;
  bufi[threadIdx.x] = besti;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (bufv[threadIdx.x + s] > bufv[threadIdx.x]) {
        bufv[threadIdx.x] = bufv[threadIdx.x + s];
        bufi[threadIdx.x] = bufi[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out_b = bufi[0];
  }
}

// Forward pass for a full batch: launch kernels with grid.y = B using base pointers
static void gpu_forward_device_batched(Transformer *transformer,
                                       int B, int max_active_pos,
                                       hipStream_t s,
                                       int device_id) {
  PROFILE_SCOPE("gpu_forward_device_batched");
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

  bool use_fused = false;
  if (const char* ef = getenv("GETP_USE_FUSED")) {
    if (ef[0] == '1' || ef[0] == 't' || ef[0] == 'T' || ef[0] == 'y' || ef[0] == 'Y') use_fused = true;
  }

  // Copy tokens/pos from pinned host arrays to device arrays (B ints each)
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_tokens, ctx.h_tokens_pinned,
                           (size_t)B * sizeof(int), hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_pos, ctx.h_pos_pinned,
                           (size_t)B * sizeof(int), hipMemcpyHostToDevice, s));

  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, B, 1);

  // Embedding
  {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_kernel", 0);
    dim3 gridH((H + BLOCK_SIZE - 1) / BLOCK_SIZE, B, 1);
    copy_embedding_bf16_kernel<<<gridH, block, 0, s>>>(
      ctx.gpu_activations.d_x,
      ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
      ctx.gpu_activations.d_tokens, H);
  }

  for (int l = 0; l < L; ++l) {
    dim3 gridQKV((QKV_D + TM - 1) / TM, B, 1);
    // QKV projection
    if (use_fused) {
      PROFILE_GPU_SCOPE("fused_rmsnorm_matmul_bias_bf16", 0);
      size_t shmem = (size_t)(1 + TK + LDS_PAD) * sizeof(float);
      fused_rmsnorm_matmul_bias_bf16<<<gridQKV, block, shmem, s>>>(
        ctx.gpu_activations.d_qkv,
        ctx.gpu_activations.d_x,
        ctx.gpu_weights_fp32.d_rms_attn_w + l * H,
        PTR_B(ctx.gpu_weights_bf16.d_w_qkv_bf16, QKV_D * H, l),
        PTR_B(ctx.gpu_weights_fp32.d_b_qkv, QKV_D, l),
        H, QKV_D, ctx.gpu_activations.d_pos);
    } else {
      // RMSNorm
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, B, 1);
        rmsnorm_kernel<<<gridH, block, 0, s>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H,
          ctx.gpu_activations.d_pos);
      }
      // GEMM + Bias
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        matmul_bias_gemm_kernel_bf16<<<gridQKV, block, 0, s>>>(
          ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_t,
          PTR_B(ctx.gpu_weights_bf16.d_w_qkv_bf16, QKV_D * H, l),
          PTR_B(ctx.gpu_weights_fp32.d_b_qkv, QKV_D, l),
          H, QKV_D, ctx.gpu_activations.d_pos);
      }
    }

    const int loff = l * S * KV;
    // Scatter QKV -> caches
    {
      PROFILE_GPU_SCOPE("split_qkv_scatter_to_cache_kernel", 0);
      split_qkv_scatter_to_cache_kernel<<<gridQKV, block, 0, s>>>(
        nullptr,
        ctx.gpu_activations.d_key_cache,
        ctx.gpu_activations.d_value_cache,
        ctx.gpu_activations.d_qkv, Hq, Hk,
        D, loff, ctx.gpu_activations.d_pos, L * S * KV);
    }

    // RoPE
    {
      PROFILE_GPU_SCOPE("fused_inline_rope_qkv_kernel", 0);
      dim3 gridApply(max(Hq, Hk), B, 1);
      fused_inline_rope_qkv_kernel<<<gridApply, D / 2, 0, s>>>(
        ctx.gpu_activations.d_qkv,
        ctx.gpu_activations.d_key_cache,
        ctx.gpu_activations.d_pos, p->rope_theta, Hq, Hk, D,
        p->rope_scaling_factor, p->initial_context_length, loff, L * S * KV);
    }

    // Attention
    {
      PROFILE_GPU_SCOPE("attention_kernel", 0);
      dim3 gridAttn(Hq, B, 1);
      dim3 blockA(WF_SIZE);
      // Use dynamic shared memory sized to the maximum active position in this batch
      // to improve occupancy when positions are small.
      int att_size = max(0, min(S, max_active_pos + 2));
      size_t shmem_size = (size_t)att_size * sizeof(float);
      attention_kernel<<<gridAttn, blockA, shmem_size, s>>>(
        ctx.gpu_activations.d_tb,
        ctx.gpu_activations.d_qkv,
        ctx.gpu_activations.d_key_cache,
        ctx.gpu_activations.d_value_cache,
        ctx.gpu_weights_fp32.d_attn_sinks, l,
        ctx.gpu_activations.d_pos, D, Hq, Hk, S,
        (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.gpu_activations.d_mask : nullptr,
        L * S * KV);
    }

    // Output projection + residual
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
      dim3 gridO((H + TM - 1) / TM, B, 1);
      const int O_N = D * Hq;
      matmul_bias_gemm_kernel_bf16<<<gridO, block, 0, s>>>(
        ctx.gpu_activations.d_t, ctx.gpu_activations.d_tb,
        PTR_B(ctx.gpu_weights_bf16.d_w_o_bf16, H * O_N, l),
        PTR_B(ctx.gpu_weights_fp32.d_b_o, H, l), O_N, H,
        ctx.gpu_activations.d_pos);
      {
        PROFILE_GPU_SCOPE("residual_add_kernel", 0);
        dim3 gridH((H + BLOCK_SIZE - 1) / BLOCK_SIZE, B, 1);
        residual_add_kernel<<<gridH, block, 0, s>>>(
          ctx.gpu_activations.d_x, ctx.gpu_activations.d_t,
          H, ctx.gpu_activations.d_pos);
      }
    }

    // FFN
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, B, 1);
      rmsnorm_kernel<<<gridH, block, 0, s>>>(
        ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
        ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H,
        ctx.gpu_activations.d_pos);
    }
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
      dim3 gridE((E + TM - 1) / TM, B, 1);
      matmul_bias_gemm_kernel_float<<<gridE, block, 0, s>>>(
        ctx.gpu_activations.d_router_score, ctx.gpu_activations.d_t,
        ctx.gpu_weights_fp32.d_w_router + (size_t)l * H * E,
        ctx.gpu_weights_fp32.d_b_router + l * E, H, E,
        ctx.gpu_activations.d_pos);
    }
    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_kernel", 0);
      dim3 gridTopK(1, B, 1);
      size_t shared_mem_size = (size_t)E * sizeof(float);
      fused_topk_softmax_kernel<<<gridTopK, BLOCK_SIZE, shared_mem_size, s>>>(
        ctx.gpu_activations.d_topk_v,
        ctx.gpu_activations.d_topk_i,
        ctx.gpu_activations.d_router_score, E, p->experts_per_token,
        ctx.gpu_activations.d_pos);
    }

    // Zero and reuse preallocated workspace
    HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_e_agg, 0,
                             (size_t)B * (size_t)H * sizeof(float), s));

    // MLP1
    {
      PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
      dim3 gridIM((IM + TM - 1) / TM, B, EXPERT_PER_TOKEN);
      mlp1_fused_gemm_kernel<<<gridIM, block, 0, s>>>(
        ctx.gpu_activations.d_gate_up_workspace,
        ctx.gpu_activations.d_t,
        ctx.gpu_weights_bf16.d_w_mlp1_bf16,
        ctx.gpu_expert_bias.g_b_mlp1,
        ctx.gpu_activations.d_topk_i,
        l, E, H, IM, p->swiglu_limit, ctx.gpu_activations.d_pos);
    }
    // MLP2
    {
      PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm_kernel", 0);
      dim3 gridH2((H + TM - 1) / TM, B, EXPERT_PER_TOKEN);
      mlp2_bias_weighted_accum_gemm_kernel<<<gridH2, block, 0, s>>>(
        ctx.gpu_activations.d_e_agg,
        ctx.gpu_activations.d_gate_up_workspace,
        ctx.gpu_weights_bf16.d_w_mlp2_bf16,
        ctx.gpu_expert_bias.g_b_mlp2,
        ctx.gpu_activations.d_topk_i,
        ctx.gpu_activations.d_topk_v,
        l, E, IM, H, ctx.gpu_activations.d_pos);
    }
    // Residual add
    {
      PROFILE_GPU_SCOPE("residual_add_kernel", 0);
      dim3 gridH3((H + BLOCK_SIZE - 1) / BLOCK_SIZE, B, 1);
      residual_add_kernel<<<gridH3, block, 0, s>>>(
        ctx.gpu_activations.d_x, ctx.gpu_activations.d_e_agg,
        H, ctx.gpu_activations.d_pos);
    }
  }

  // Final head
  {
    if (use_fused) {
      PROFILE_GPU_SCOPE("fused_rmsnorm_matmul_bias_bf16", 0);
      dim3 gridV((V + TM - 1) / TM, B, 1);
      size_t shmem = (size_t)(1 + TK + LDS_PAD) * sizeof(float);
      fused_rmsnorm_matmul_bias_bf16<<<gridV, block, shmem, s>>>(
        ctx.gpu_activations.d_logits,
        ctx.gpu_activations.d_x,
        ctx.gpu_weights_fp32.d_rms_out_w,
        ctx.gpu_weights_bf16.d_out_bf16,
        nullptr,
        H, V, ctx.gpu_activations.d_pos);
    } else {
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, B, 1);
        rmsnorm_kernel<<<gridH, block, 0, s>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_out_w, H,
          ctx.gpu_activations.d_pos);
      }
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
        dim3 gridV((V + TM - 1) / TM, B, 1);
        matmul_bias_gemm_kernel_bf16<<<gridV, block, 0, s>>>(
          ctx.gpu_activations.d_logits, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16, nullptr, H, V,
          ctx.gpu_activations.d_pos);
      }
    }
  }
}

static long long run_requests_on_device(Transformer *transformer,
                                        Tokenizer *tokenizer, PromptCtx *ctxs,
                                        int num_ctxs, int device_id) {
  PROFILE_SCOPE("run_requests_on_device");
  HIP_CHECK(hipSetDevice(device_id));

  long long total_tokens = 0;

  // Decide batch size B (env GETP_STATIC_BATCH, default 4)
  int B = 4;
  if (const char* envB = getenv("GETP_STATIC_BATCH")) {
    int v = atoi(envB);
    if (v > 0) B = v;
  }
  B = std::min(B, num_ctxs);
  ensure_device_capacity(g_devices[device_id], B);
  const Config *p = model_config;
  const int V = p->vocab_size;
  // Host-side staging buffer for next tokens from device per slot
  std::vector<int> next_token_host(B, -1);
  // Track how many prompt steps were enqueued per slot this iteration
  std::vector<int> prefill_steps(B, 0);

  // Static-batching scheduler over B streams
  DeviceContext &dctx = g_devices[device_id];
  std::vector<int> slot_to_ctx(B, -1);
  int next_assign = 0;
  for (int b = 0; b < B && next_assign < num_ctxs; ++b) {
    slot_to_ctx[b] = next_assign++;
  }

  while (true) {
    bool any_active = false;
    int max_active_pos = -1;
    // Build staging arrays for tokens/pos; mark inactive slots with pos=-1
    for (int b = 0; b < B; ++b) {
      int idx = slot_to_ctx[b];
      if (idx < 0) { dctx.h_pos_pinned[b] = -1; continue; }
      PromptCtx &ctx = ctxs[idx];
      if (ctx.finished) { dctx.h_pos_pinned[b] = -1; continue; }
      any_active = true;
      dctx.h_tokens_pinned[b] = ctx.token;
      dctx.h_pos_pinned[b] = ctx.pos;
      if (ctx.pos > max_active_pos) max_active_pos = ctx.pos;
    }
    if (!any_active) {
      bool has_more = false;
      for (int b = 0; b < B; ++b) if (slot_to_ctx[b] >= 0) { has_more = true; break; }
      if (!has_more) break;
    }
    // Single batched forward for all active slots on stream 0
    gpu_forward_device_batched(transformer, B, max_active_pos, dctx.streams[0], device_id);
    // Greedy sampling on device for all slots
    int threads = 256; size_t shmem = (size_t)threads * (sizeof(float) + sizeof(int));
    dim3 gridArgmax(1, B, 1);
    argmax_reduce_batched_kernel<<<gridArgmax, threads, shmem, dctx.streams[0]>>>(
      dctx.gpu_activations.d_logits, V, dctx.gpu_activations.d_next_tokens);
    // Copy back all next tokens
    HIP_CHECK(hipMemcpyAsync(next_token_host.data(), dctx.gpu_activations.d_next_tokens,
                              (size_t)B * sizeof(int), hipMemcpyDeviceToHost, dctx.streams[0]));
    HIP_CHECK(hipStreamSynchronize(dctx.streams[0]));

  // Post-process tokens (single batched stream already synchronized above)
    for (int b = 0; b < B; ++b) {
      int idx = slot_to_ctx[b];
      if (idx < 0) continue;
      PromptCtx &ctx = ctxs[idx];
      if (ctx.finished) continue;

      // If we prefetched prompt steps for this slot, skip decode bookkeeping now
      if (prefill_steps[b] > 0) {
        prefill_steps[b] = 0;
        // no token sampling/output token write during prefill
        continue;
      }

      // Advance position and determine next token
      ctx.pos++;
      int next;
      if (ctx.pos < ctx.num_prompt_tokens) {
        // still consuming prompt
        next = ctx.prompt_tokens[ctx.pos];
      } else {
        ctx.is_context_phase = false;
        const bool greedy = (ctx.sampler && ctx.sampler->temperature == 0.0f);
        if (greedy) {
          // next token produced by device-side argmax
          next = next_token_host[b];
        } else {
          // host-side sampling (temperature/top-p)
          next = sample(ctx.sampler, ctx.h_logits);
        }
        ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens] = next;
        ctx.num_generated++;
      }

      // Append decoded piece
      if (next == 199999 || next == 200002) {
        ctx.finished = true;
        const char *piece = decode_piece(tokenizer, ctx.token, next);
        if (piece) ctx.output_str += piece;
        ctx.token = next;
      } else {
        const char *piece = decode_piece(tokenizer, ctx.token, next);
        if (piece) ctx.output_str += piece;
        ctx.token = next;
        if (ctx.max_steps != 0 && ctx.pos >= ctx.max_steps) {
          ctx.finished = true;
        }
      }

      if (ctx.finished) {
        ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;
        total_tokens += (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
        // Recycle slot
        if (next_assign < num_ctxs) {
          slot_to_ctx[b] = next_assign++;
        } else {
          slot_to_ctx[b] = -1;
        }
      }
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
