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

  // Pre-allocate workspace for requests
  ctx.gpu_activations.d_gate_up_workspace = nullptr;
  // Allocate minimal workspace for single-sample path (used in warm-up)
  {
    const int IM = model_config->intermediate_dim;
    const int K  = model_config->experts_per_token;
    const size_t bytes = (size_t)K * (size_t)IM * sizeof(float);
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace, bytes));
  }

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (D * (Hq + 2 * Hk)) * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&ctx.gpu_activations.d_key_cache, L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                      L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));

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
  HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
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
// Ensure device has capacity for single request (reallocates activations if needed)
static inline void ensure_device_capacity(DeviceContext &ctx, int capacity_B) {
  HIP_CHECK(hipSetDevice(ctx.device_id));

  if (ctx.capacity_B >= capacity_B) return;
  const bool need_realloc = (ctx.capacity_B == 0);  // Only allocate once

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
    FREE_IF(ctx.gpu_activations.d_next_tokens);
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

  // KV caches (bf16)
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache,
                      (size_t)B*L * S * KV * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache,
                      (size_t)B*L * S * KV * sizeof(bf16_t)));
  // Auxiliary buffers
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att,
                      (size_t)B*Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                      (size_t)B*V * sizeof(float)));
  // next tokens buffer (B)
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
  ctx.capacity_B = B;
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

  // Copy host token/position into device buffers
  HIP_CHECK(hipMemcpyAsync(PTR_B(ctx.gpu_activations.d_tokens, 1, b), &token_host,
                      sizeof(int), hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(PTR_B(ctx.gpu_activations.d_pos, 1, b), &pos_host,
                      sizeof(int), hipMemcpyHostToDevice, s));

  // Runtime-tunable warps-per-block for warp-cooperative GEMMs
  int warps_per_block = TM;
  if (const char* envW = getenv("GETP_WARPS")) {
    int w = atoi(envW);
    if (w > 0) warps_per_block = w;
  }
  const int block_threads = WF_SIZE * warps_per_block;
  dim3 block_warp(block_threads, 1, 1);
  dim3 block_thread(BLOCK_SIZE, 1, 1);
  dim3 gridH_warp((H + warps_per_block - 1) / warps_per_block, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Launch embedding kernel
  {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_kernel", 0);
    dim3 gridH(gridH_thread.x, 1, 1);
  copy_embedding_bf16_kernel<<<gridH, block_thread, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
        PTR_B(ctx.gpu_activations.d_tokens, 1, b), H);
  }

  for (int l = 0; l < L; ++l) {
  dim3 gridQKV((QKV_D + warps_per_block - 1) / warps_per_block, 1, 1);
    // QKV projection (RMSNorm + MatMul + Bias) - separate kernels
    {
      // First apply RMSNorm
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, 1, 1);
        rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H,
        PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }

      // Then apply MatMul + Bias
  {
    PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
    dim3 gridQKV_gemm((QKV_D + warps_per_block - 1) / warps_per_block, 1, 1);
    // Single-sample (keep using existing kernel per stream)
    matmul_bias_gemm_kernel_bf16<<<gridQKV_gemm, block_warp, 0, s>>>(
    PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b), PTR_B(ctx.gpu_activations.d_t, H, b),
    PTR_B(ctx.gpu_weights_bf16.d_w_qkv_bf16, QKV_D * H, l), PTR_B(ctx.gpu_weights_fp32.d_b_qkv, QKV_D, l),
    H, QKV_D, PTR_B(ctx.gpu_activations.d_pos, 1, b));
  }
    }

    // Scatter QKV to caches (K and V only, Q remains in QKV buffer)
    const int loff = l * S * KV;
    {
      PROFILE_GPU_SCOPE("split_qkv_scatter_to_cache_kernel", 0);
      // total elements over (Q + K + V) slices
      const int total_qkv = QKV_D;
      dim3 gridSplit((total_qkv + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  split_qkv_scatter_to_cache_kernel<<<gridSplit, block_thread, 0, s>>>(
      nullptr,
      PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
      PTR_B(ctx.gpu_activations.d_value_cache, (size_t)L * S * KV, b),
      PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b), Hq, Hk, D, loff,
      PTR_B(ctx.gpu_activations.d_pos, 1, b), L * S * KV);
    }

    // Apply RoPE to q (in QKV buffer) and cached k
    {
      PROFILE_GPU_SCOPE("fused_inline_rope_qkv_kernel", 0);
      dim3 gridApply(max(Hq, Hk), 1, 1);
      fused_inline_rope_qkv_kernel<<<gridApply, D / 2, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b),
        PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
        PTR_B(ctx.gpu_activations.d_pos, 1, b), p->rope_theta, Hq, Hk, D,
        p->rope_scaling_factor, p->initial_context_length, loff, L * S * KV);
    }

    // Attention (reads Q directly from QKV buffer)
    {
  PROFILE_GPU_SCOPE("attention_kernel_bf16", 0);
      dim3 gridAttn(Hq, 1, 1);
      dim3 blockA(WF_SIZE);
      int pos_for_shmem = std::min(pos_host, S - 1);
      size_t shmem_size = (size_t)(pos_for_shmem + 2) * sizeof(float);
  attention_kernel_bf16<<<gridAttn, blockA, shmem_size, s>>>(
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

      // First do MatMul + Bias: temp = tb @ W^T + b
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
    dim3 gridO_gemm((H + warps_per_block - 1) / warps_per_block, 1, 1);
  matmul_bias_gemm_kernel_bf16<<<gridO_gemm, block_warp, 0, s>>>(
      PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_tb, Hq * D, b),
      PTR_B(ctx.gpu_weights_bf16.d_w_o_bf16, H * O_N, l),
      PTR_B(ctx.gpu_weights_fp32.d_b_o, H, l), O_N, H,
      PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }

      // Then do residual add: x = x + temp
      {
        PROFILE_GPU_SCOPE("residual_add_kernel", 0);
        dim3 gridH(gridH_thread.x, 1, 1);
        residual_add_kernel<<<gridH, block_thread, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_x, H, b), PTR_B(ctx.gpu_activations.d_t, H, b),
          H, PTR_B(ctx.gpu_activations.d_pos, 1, b));
      }
    }

    // FFN
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, 1, 1);
      rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
        PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
        ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H,
        PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
      dim3 gridE_gemm((E + warps_per_block - 1) / warps_per_block, 1, 1);
      matmul_bias_gemm_kernel_float<<<gridE_gemm, block_warp, 0, s>>>(
      PTR_B(ctx.gpu_activations.d_router_score, E, b), PTR_B(ctx.gpu_activations.d_t, H, b),
      ctx.gpu_weights_fp32.d_w_router + (size_t)l * H * E,
      ctx.gpu_weights_fp32.d_b_router + l * E, H, E,
      PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_kernel", 0);
      dim3 gridTopK(1, 1, 1);
      size_t shared_mem_size = (size_t)E * sizeof(float);
  fused_topk_softmax_kernel<<<gridTopK, BLOCK_SIZE,
                    shared_mem_size, s>>>(
        PTR_B(ctx.gpu_activations.d_topk_v, p->experts_per_token, b),
        PTR_B(ctx.gpu_activations.d_topk_i, p->experts_per_token, b),
        PTR_B(ctx.gpu_activations.d_router_score, E, b), E,
        p->experts_per_token, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    // Use pre-allocated workspace from DeviceContext to avoid repeated
    // malloc/free
  const int K = p->experts_per_token;
  size_t gate_up_topk_bytes = (size_t)K * (size_t)IM * sizeof(float);
  float *d_gate_up_topk = PTR_B(ctx.gpu_activations.d_gate_up_workspace, (size_t)K * IM, b);
  // No need to memset d_gate_up_topk: mlp1 kernel overwrites all elements.

    // --- MLP1: per-expert
    {
      PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
      dim3 block(BLOCK_SIZE, 1, 1);
      // grid.y = EXPERT_PER_TOKEN (experts), grid.z = 1 (single-sample)
      dim3 gridIM((IM + TM - 1) / TM, EXPERT_PER_TOKEN, 1);
      mlp1_fused_gemm_kernel<<<gridIM, block, 0, s>>>(
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
      // grid.y = EXPERT_PER_TOKEN (experts), grid.z = 1 (single-sample)
      dim3 gridH((H + TM - 1) / TM, EXPERT_PER_TOKEN, 1);
      mlp2_bias_weighted_accum_gemm_kernel<<<gridH, block, 0, s>>>(
        /*x[B,H] (fused residual)*/ PTR_B(ctx.gpu_activations.d_x, H, b),
        /*gate_up[K,B,IM]*/ d_gate_up_topk,
        /*w*/ ctx.gpu_weights_bf16.d_w_mlp2_bf16,
        /*b*/ ctx.gpu_expert_bias.g_b_mlp2,
        /*topk_i, topk_v*/ PTR_B(ctx.gpu_activations.d_topk_i, K, b),
        PTR_B(ctx.gpu_activations.d_topk_v, K, b),
        /*layer*/ l,
        /*E,IM,H*/ E, IM, H, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    // Keep workspace allocated in DeviceContext for reuse
    // Residual fused into MLP2 kernel via atomic adds; no separate residual add.
  }

  // Final head
  {
    // 1) RMSNorm - separate kernel call
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, 1, 1);
    rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_t, H, b), PTR_B(ctx.gpu_activations.d_x, H, b),
          ctx.gpu_weights_fp32.d_rms_out_w, H,
          PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }

    // 2) MatMul for logits - separate GEMM version
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
    dim3 gridV_gemm((V + warps_per_block - 1) / warps_per_block, 1, 1);
    matmul_bias_gemm_kernel_bf16<<<gridV_gemm, block_warp, 0, s>>>(
          PTR_B(ctx.gpu_activations.d_logits, V, b),
          PTR_B(ctx.gpu_activations.d_t, H, b), ctx.gpu_weights_bf16.d_out_bf16,
          nullptr, H, V, PTR_B(ctx.gpu_activations.d_pos, 1, b));
    }
  }

  return PTR_B(ctx.gpu_activations.d_logits, V, b);
}

// Batched forward for B samples at once (grid.z = B). Inactive slots should set pos_host[b] = -1.
static float *gpu_forward_device_batched(Transformer *transformer,
                                         const int *tokens_host_B,
                                         const int *pos_host_B,
                                         int B, hipStream_t s,
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

  // Copy host token/position arrays into device buffers
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_tokens, tokens_host_B,
                           (size_t)B * sizeof(int), hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_pos, pos_host_B,
                           (size_t)B * sizeof(int), hipMemcpyHostToDevice, s));

  // Runtime-tunable warps-per-block for warp-cooperative GEMMs
  int warps_per_block = TM;
  if (const char* envW = getenv("GETP_WARPS")) {
    int w = atoi(envW);
    if (w > 0) warps_per_block = w;
  }
  const int block_threads = WF_SIZE * warps_per_block;
  dim3 block_warp(block_threads, 1, 1);
  dim3 block_thread(BLOCK_SIZE, 1, 1);
  dim3 blockA(WF_SIZE, 1, 1);
  dim3 gridH_thread((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);
  dim3 gridH_warp((H + warps_per_block - 1) / warps_per_block, 1, B);

  // Launch embedding for all active samples
  {
    PROFILE_GPU_SCOPE("copy_embedding_bf16_kernel", 0);
    dim3 gridH((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);
  copy_embedding_bf16_kernel<<<gridH, block_thread, 0, s>>>(
        ctx.gpu_activations.d_x,
        ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
        ctx.gpu_activations.d_tokens, H);
  }

  for (int l = 0; l < L; ++l) {
  dim3 gridQKV((QKV_D + warps_per_block - 1) / warps_per_block, 1, B);
    // QKV projection (RMSNorm + MatMul + Bias)
    {
      // RMSNorm
      {
        PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
        dim3 gridH(1, 1, B);
    rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
            ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H,
            ctx.gpu_activations.d_pos);
      }

      // MatMul + Bias to produce QKV into contiguous buffer per sample
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
    dim3 gridQKV_gemm((QKV_D + warps_per_block - 1) / warps_per_block, 1, B);
    matmul_bias_gemm_kernel_bf16<<<gridQKV_gemm, block_warp, 0, s>>>(
            ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l * (size_t)QKV_D * (size_t)H,
            ctx.gpu_weights_fp32.d_b_qkv + (size_t)l * (size_t)QKV_D,
            H, QKV_D, ctx.gpu_activations.d_pos);
      }
    }

    // Scatter K/V into cache for all batch samples
    const int loff = l * S * KV;
    {
      PROFILE_GPU_SCOPE("split_qkv_scatter_to_cache_kernel", 0);
      dim3 gridSplit((QKV_D + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);
  split_qkv_scatter_to_cache_kernel<<<gridSplit, block_thread, 0, s>>>(
      nullptr, ctx.gpu_activations.d_key_cache,
      ctx.gpu_activations.d_value_cache, ctx.gpu_activations.d_qkv, Hq,
      Hk, D, loff, ctx.gpu_activations.d_pos, L * S * KV);
    }

    // Apply RoPE
    {
      PROFILE_GPU_SCOPE("fused_inline_rope_qkv_kernel", 0);
      dim3 gridApply(max(Hq, Hk), 1, B);
    fused_inline_rope_qkv_kernel<<<gridApply, D / 2, 0, s>>>(
      ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_key_cache,
      ctx.gpu_activations.d_pos, p->rope_theta, Hq, Hk, D,
      p->rope_scaling_factor, p->initial_context_length, loff,
      L * S * KV);
    }

    // Attention – launch per-sample with right-sized shared mem to avoid pos_max overhead
    {
      PROFILE_GPU_SCOPE("attention_kernel_bf16", 0);
      dim3 blockA(WF_SIZE, 1, 1);
      for (int b = 0; b < B; ++b) {
        if (pos_host_B[b] < 0) continue; // inactive slot
        int pos_b = std::min(pos_host_B[b], S - 1);
        size_t shmem_b = (size_t)(pos_b + 2) * sizeof(float);
        dim3 gridAttn(Hq, 1, 1);
        attention_kernel_bf16<<<gridAttn, blockA, shmem_b, s>>>(
            PTR_B(ctx.gpu_activations.d_tb, Hq * D, b),
            PTR_B(ctx.gpu_activations.d_qkv, QKV_D, b),
            PTR_B(ctx.gpu_activations.d_key_cache, (size_t)L * S * KV, b),
            PTR_B(ctx.gpu_activations.d_value_cache, (size_t)L * S * KV, b),
            ctx.gpu_weights_fp32.d_attn_sinks, l, PTR_B(ctx.gpu_activations.d_pos, 1, b), D,
            Hq, Hk, S,
            (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.gpu_activations.d_mask : nullptr,
            L * S * KV);
      }
    }

    // Output projection + residual
    {
      const int O_N = D * Hq;
      // MatMul + Bias into t
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16", 0);
    dim3 gridO_gemm((H + warps_per_block - 1) / warps_per_block, 1, B);
    matmul_bias_gemm_kernel_bf16<<<gridO_gemm, block_warp, 0, s>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_tb,
            ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l * (size_t)H * (size_t)O_N,
            ctx.gpu_weights_fp32.d_b_o + (size_t)l * (size_t)H, O_N, H,
            ctx.gpu_activations.d_pos);
      }

      // residual add x += t
      {
        PROFILE_GPU_SCOPE("residual_add_kernel", 0);
        dim3 gridH((H + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);
    residual_add_kernel<<<gridH, block_thread, 0, s>>>(
            ctx.gpu_activations.d_x, ctx.gpu_activations.d_t, H,
            ctx.gpu_activations.d_pos);
      }
    }

    // FFN
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, 1, B);
    rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H,
          ctx.gpu_activations.d_pos);
    }

    // Router (E)
    {
      PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_float", 0);
    dim3 gridE_gemm((E + warps_per_block - 1) / warps_per_block, 1, B);
    matmul_bias_gemm_kernel_float<<<gridE_gemm, block_warp, 0, s>>>(
          ctx.gpu_activations.d_router_score, ctx.gpu_activations.d_t,
          ctx.gpu_weights_fp32.d_w_router + (size_t)l * (size_t)H * (size_t)E,
          ctx.gpu_weights_fp32.d_b_router + (size_t)l * (size_t)E, H, E,
          ctx.gpu_activations.d_pos);
    }

    // Top-k softmax
    {
      PROFILE_GPU_SCOPE("fused_topk_softmax_kernel", 0);
      dim3 gridTopK(1, 1, B);
      size_t shared_mem_size = (size_t)E * sizeof(float);
    fused_topk_softmax_kernel<<<gridTopK, BLOCK_SIZE, shared_mem_size, s>>>(
          ctx.gpu_activations.d_topk_v, ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_router_score, E, p->experts_per_token,
          ctx.gpu_activations.d_pos);
    }

    const int K = p->experts_per_token;
    float *d_gate_up_topk = ctx.gpu_activations.d_gate_up_workspace;
    bool use_grouped = true;
    if (!use_grouped) {
      // Fallback: per-token path
      {
        PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
        dim3 gridIM((IM + TM - 1) / TM, K, B);
        mlp1_fused_gemm_kernel<<<gridIM, block_thread, 0, s>>>(
            d_gate_up_topk, ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_mlp1_bf16,
            ctx.gpu_expert_bias.g_b_mlp1,
            ctx.gpu_activations.d_topk_i,
            l, E, H, IM, p->swiglu_limit, ctx.gpu_activations.d_pos);
      }
      {
        PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm_kernel", 0);
        dim3 gridH((H + TM - 1) / TM, K, B);
        mlp2_bias_weighted_accum_gemm_kernel<<<gridH, block_thread, 0, s>>>(
            ctx.gpu_activations.d_x, d_gate_up_topk,
            ctx.gpu_weights_bf16.d_w_mlp2_bf16,
            ctx.gpu_expert_bias.g_b_mlp2,
            ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_topk_v,
            l, E, IM, H, ctx.gpu_activations.d_pos);
      }
    } else {
      // Group by expert
      PROFILE_GPU_SCOPE("moe_grouped", 0);
      // 1) counts per expert
      HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_gate_up, 0, (size_t)E * sizeof(int), s));
      int *d_counts; HIP_CHECK(hipMalloc(&d_counts, (size_t)E * sizeof(int)));
      HIP_CHECK(hipMemsetAsync(d_counts, 0, (size_t)E * sizeof(int), s));
      dim3 gridC((K + 127) / 128, 1, B);
      moe_count_assignments_kernel<<<gridC, 128, 0, s>>>(
          ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_pos, B, K, E, d_counts);
      // 2) prefix-sum on host (small E)
      std::vector<int> h_counts(E); HIP_CHECK(hipMemcpyAsync(h_counts.data(), d_counts, (size_t)E * sizeof(int), hipMemcpyDeviceToHost, s));
      HIP_CHECK(hipStreamSynchronize(s));
      std::vector<int> h_indptr(E + 1); h_indptr[0] = 0; for (int e = 0; e < E; ++e) h_indptr[e+1] = h_indptr[e] + h_counts[e];
      int Ntot = h_indptr[E];
      int *d_indptr, *d_offsets; HIP_CHECK(hipMalloc(&d_indptr, (size_t)(E + 1) * sizeof(int))); HIP_CHECK(hipMalloc(&d_offsets, (size_t)E * sizeof(int)));
      HIP_CHECK(hipMemcpyAsync(d_indptr, h_indptr.data(), (size_t)(E + 1) * sizeof(int), hipMemcpyHostToDevice, s));
      HIP_CHECK(hipMemcpyAsync(d_offsets, h_indptr.data(), (size_t)E * sizeof(int), hipMemcpyHostToDevice, s));
      // 3) fill assignment arrays
      int *d_assign_b, *d_assign_k; HIP_CHECK(hipMalloc(&d_assign_b, (size_t)Ntot * sizeof(int))); HIP_CHECK(hipMalloc(&d_assign_k, (size_t)Ntot * sizeof(int)));
      moe_fill_indices_kernel<<<gridC, 128, 0, s>>>(ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_pos,
          B, K, E, d_offsets, d_indptr, d_assign_b, d_assign_k);
      // 4) per expert GEMMs
      const int tileM = 128, tileN = 128;
      for (int e = 0; e < E; ++e) {
        int start = h_indptr[e], end = h_indptr[e+1];
        int N = end - start; if (N == 0) continue;
        // Gather GU_e[N, IM]
        float *d_GU_e; HIP_CHECK(hipMalloc(&d_GU_e, (size_t)N * IM * sizeof(float)));
        dim3 gG((N + 31) / 32, (IM + 31) / 32);
        dim3 bG(32, 32);
        moe_gather_gate_up_kernel<<<gG, bG, 0, s>>>(d_gate_up_topk, B, K, IM, d_assign_b, d_assign_k, start, N, d_GU_e);
    // Z = GU_e[N, IM] @ W2_e^T[H, IM] + b_e[H] -> [N, H]
    float *d_Z; HIP_CHECK(hipMalloc(&d_Z, (size_t)N * H * sizeof(float)));
    {
      PROFILE_GPU_SCOPE("mlp2_grouped_matmul", 0);
      // Use batched GEMM over N samples: grid.z = N, each computes one output row vector of size H
      dim3 gridH_g((H + warps_per_block - 1) / warps_per_block, 1, N);
      matmul_bias_gemm_kernel_bf16_B<<<gridH_g, block_warp, 0, s>>>(
        /*out*/ d_Z,
        /*x*/ d_GU_e,
        /*w*/ ctx.gpu_weights_bf16.d_w_mlp2_bf16 + ((size_t)l * E + e) * (size_t)H * (size_t)IM,
        /*b*/ ctx.gpu_expert_bias.g_b_mlp2 + ((size_t)l * E + e) * (size_t)H,
        /*H=n*/ IM, /*D=d*/ H, /*pos*/ nullptr);
    }
    // Scatter-accumulate into x with weights
    dim3 gS((N + 31) / 32, (H + 31) / 32);
    dim3 bS(32, 32);
    moe_scatter_mlp2_accum_kernel<<<gS, bS, 0, s>>>(/*Z*/ d_Z, N, H, d_assign_b, d_assign_k, start,
      ctx.gpu_activations.d_topk_v, K, ctx.gpu_activations.d_x, ctx.gpu_activations.d_pos);
    HIP_CHECK(hipFree(d_GU_e));
    HIP_CHECK(hipFree(d_Z));
      }
      HIP_CHECK(hipFree(d_counts)); HIP_CHECK(hipFree(d_indptr)); HIP_CHECK(hipFree(d_offsets));
      HIP_CHECK(hipFree(d_assign_b)); HIP_CHECK(hipFree(d_assign_k));
    }
    // Residual fused into MLP2 kernel; no separate residual add required.
  }

  // Final head
  {
    // RMSNorm
    {
      PROFILE_GPU_SCOPE("rmsnorm_kernel", 0);
      dim3 gridH(1, 1, B);
    rmsnorm_kernel<<<gridH, block_thread, 0, s>>>(
          ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
          ctx.gpu_weights_fp32.d_rms_out_w, H,
          ctx.gpu_activations.d_pos);
    }

    // Fused output GEMM + argmax (no intermediate logits)
    {
      PROFILE_GPU_SCOPE("out_gemm_argmax_kernel", 0);
  // One CTA per sample b for numerically stable argmax reduction
  dim3 gridV(1, 1, B);
  size_t shmem = (size_t)(TK + LDS_PAD) * sizeof(float)
       + (size_t)warps_per_block * (sizeof(float) + sizeof(int));
  out_gemm_argmax_kernel<<<gridV, block_warp, shmem, s>>>(
          ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16,
          V, H,
          ctx.gpu_activations.d_pos,
          ctx.gpu_activations.d_next_tokens);
    }
  }

  return ctx.gpu_activations.d_logits;
}

static long long run_requests_on_device(Transformer *transformer,
                                        Tokenizer *tokenizer, PromptCtx *ctxs,
                                        int num_ctxs, int device_id) {
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

  // Static-batching scheduler with single batched launches (grid.z=B)
  DeviceContext &dctx = g_devices[device_id];
  std::vector<int> slot_to_ctx(B, -1);
  int next_assign = 0;
  for (int b = 0; b < B && next_assign < num_ctxs; ++b) {
    slot_to_ctx[b] = next_assign++;
  }

  while (true) {
    // Build batched tokens/pos for this step
    std::vector<int> tokens_B(B, -1), pos_B(B, -1);
    bool any_active = false;
    for (int b = 0; b < B; ++b) {
      int idx = slot_to_ctx[b];
      if (idx < 0) continue;
      PromptCtx &ctx = ctxs[idx];
      if (ctx.finished) continue;
      any_active = true;
      tokens_B[b] = ctx.token;
      pos_B[b] = ctx.pos;
    }

    if (!any_active) {
      bool has_more = false;
      for (int b = 0; b < B; ++b) {
        if (slot_to_ctx[b] >= 0) { has_more = true; break; }
      }
      if (!has_more) break;
      // No active this cycle but more to assign; continue to recycle below
    } else {
      // Pack active slots to avoid launching on inactive ones
      std::vector<int> active_map; active_map.reserve(B);
      for (int b = 0; b < B; ++b) if (pos_B[b] >= 0) active_map.push_back(b);
      const int Bp = (int)active_map.size();
      if (Bp == 0) {
        // nothing to do this iteration (all inactive placeholders)
      } else {
        std::vector<int> tokens_active(Bp), pos_active(Bp);
        for (int i = 0; i < Bp; ++i) {
          tokens_active[i] = tokens_B[active_map[i]];
          pos_active[i]    = pos_B[active_map[i]];
        }

        // Run compact batched forward with B' active samples
        float *d_logits = gpu_forward_device_batched(
          transformer, tokens_active.data(), pos_active.data(), Bp, dctx.streams[0], device_id);

        // Copy back only B' sampled tokens
        std::vector<int> next_tokens_active(Bp, -1);
        HIP_CHECK(hipMemcpyAsync(next_tokens_active.data(), dctx.gpu_activations.d_next_tokens,
                                 (size_t)Bp * sizeof(int), hipMemcpyDeviceToHost, dctx.streams[0]));
        HIP_CHECK(hipStreamSynchronize(dctx.streams[0]));

        // Scatter back to full-batch indexing
        std::vector<int> next_tokens_B(B, -1);
        for (int i = 0; i < Bp; ++i) next_tokens_B[active_map[i]] = next_tokens_active[i];

  // Post-process per slot
      for (int b = 0; b < B; ++b) {
        int idx = slot_to_ctx[b];
        if (idx < 0) continue;
        PromptCtx &ctx = ctxs[idx];
        if (ctx.finished) continue;
        if (pos_B[b] < 0) continue; // inactive placeholder

        // Advance and pick next token (already decided on GPU when not in context phase)
        ctx.pos++;
        int next;
        if (ctx.pos < ctx.num_prompt_tokens) {
          next = ctx.prompt_tokens[ctx.pos];
        } else {
          ctx.is_context_phase = false;
          next = next_tokens_B[b];
          ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens] = next;
          ctx.num_generated++;
        }

        const char *piece = decode_piece(tokenizer, ctx.token, next);
        if (piece) ctx.output_str += piece;
        ctx.token = next;
        if (next == 199999 || next == 200002 || (ctx.max_steps != 0 && ctx.pos >= ctx.max_steps)) {
          ctx.finished = true;
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
      } // end post-process for slots
      } // end if (Bp > 0)
    } // end else any_active
    // recycle loop continues
  } // end while(true)

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
