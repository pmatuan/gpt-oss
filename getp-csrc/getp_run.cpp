#include "attention.cpp"
#include "getp_eval.cpp"
#include "matmul.cpp"
#include "utility.cpp"
#include "prompt_ctx.cpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <vector>
#include <atomic>
#include <mutex>

#ifndef GETP_RUN
#define GETP_RUN

struct GPUActivationBuffers {
  float *d_x, *d_t, *d_tb, *d_tb2;
  float *d_router_score, *d_topk_v, *d_mlp1_out;
  int *d_topk_i;
  float *d_gate_up, *d_e_agg;
  float *d_qkv, *d_q, *d_k, *d_v;
  float *d_key_cache, *d_value_cache;
  float *d_att, *d_logits, *d_mask;
  float *d_cos_vals, *d_sin_vals;
  int *d_token2row;
};

struct GPUWeightBuffersFP32 {
  float *d_rms_attn_w, *d_rms_ffn_w;
  float *d_b_qkv, *d_b_o, *d_attn_sinks;
  float *d_w_router, *d_b_router;
  float *d_rms_out_w;
};

struct GPUExpertBiasBuffers {
  float *g_b_mlp1;
  float *g_b_mlp2;
};

typedef hip_bfloat16 bf16_t;

struct GPUWeightBuffersBF16 {
  bf16_t *d_token_embedding_table_bf16;
  bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
  bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
  bf16_t *d_out_bf16;
};

// Multi-GPU device management
struct DeviceContext {
  int device_id;
  hipStream_t stream;

  GPUActivationBuffers gpu_activations;
  GPUWeightBuffersFP32 gpu_weights_fp32;
  GPUExpertBiasBuffers gpu_expert_bias;
  GPUWeightBuffersBF16 gpu_weights_bf16;
};

static Config *model_config;

// Global multi-GPU state
static std::vector<DeviceContext> g_devices;
static int g_num_devices = 0;

// ======================== Init / Finish ========================
// Initialize single device context
static void init_device_context(DeviceContext &ctx, int device_id, Transformer *transformer) {
  ctx.device_id = device_id;
  HIP_CHECK(hipSetDevice(device_id));
  HIP_CHECK(hipStreamCreateWithFlags(&ctx.stream, hipStreamNonBlocking));

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
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb2, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v, model_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i, model_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mlp1_out, 2 * IM * sizeof(float))); // kept for debug
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_e_agg, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv, (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q, Hq * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_k, Hk * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_v, Hk * D * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_key_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_value_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_sin_vals, (D / 2) * sizeof(float)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_token2row, S * sizeof(int)));
  {
    int *h_token2row = (int *)malloc(S * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < S; ++i) h_token2row[i] = i;
    HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_token2row, h_token2row, S * sizeof(int), hipMemcpyHostToDevice));
    free(h_token2row);
  }

  if (model_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] = (i - j >= model_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_mask, h_mask, S * S * sizeof(float), hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    ctx.gpu_activations.d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations", device_id);

  // Weights (small FP32)
  TransformerWeights *w = &transformer->weights;

  const int H_ = H;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_attn_w, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_attn_w, w->rms_attn_w, L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_ffn_w, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_ffn_w, w->rms_ffn_w, L * H_ * sizeof(float), hipMemcpyHostToDevice));

  const int D_ = model_config->head_dim;
  const int Hq_ = model_config->n_attn_heads;
  const int Hk_ = model_config->n_kv_heads;
  const int QKV_D = D_ * (Hq_ + 2 * Hk_);
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_qkv, w->b_qkv, L * QKV_D * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_o, L * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_o, w->b_o, L * H_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_attn_sinks, L * Hq_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_attn_sinks, w->attn_sinks, L * Hq_ * sizeof(float), hipMemcpyHostToDevice));

  const int E_ = model_config->n_experts;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_w_router, L * H_ * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_w_router, w->w_router, L * H_ * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_router, L * E_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_b_router, w->b_router, L * E_ * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_out_w, H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_fp32.d_rms_out_w, w->rms_out_w, H_ * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights", device_id);

  // Expert biases FP32
  const int IM_ = model_config->intermediate_dim;
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp1, (size_t)L * E_ * (2 * IM_) * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp1, w->b_mlp1, (size_t)L * E_ * (2 * IM_) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp2, (size_t)L * E_ * H_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(ctx.gpu_expert_bias.g_b_mlp2, w->b_mlp2, (size_t)L * E_ * H_ * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases", device_id);

  // Large BF16 weights
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;
  const int V_ = model_config->vocab_size;
  const int O_N = D_ * Hq_;

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_token_embedding_table_bf16, (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V_ * H_, ctx.gpu_weights_bf16.d_token_embedding_table_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_qkv_bf16, (size_t)L * QKV_D * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H_, ctx.gpu_weights_bf16.d_w_qkv_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_o_bf16, (size_t)L * H_ * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H_ * O_N, ctx.gpu_weights_bf16.d_w_o_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp1_bf16, (size_t)L * E_ * (2 * IM_) * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E_ * (2 * IM_) * H_, ctx.gpu_weights_bf16.d_w_mlp1_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp2_bf16, (size_t)L * E_ * H_ * IM_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E_ * H_ * IM_, ctx.gpu_weights_bf16.d_w_mlp2_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_out_bf16, (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V_ * H_, ctx.gpu_weights_bf16.d_out_bf16, n_streams, chunk_bytes);

  debug_print_gpu_memory("after large BF16 weights (model loaded)", device_id);
}

// Cleanup device context
static void cleanup_device_context(DeviceContext& ctx) {
  int device_id = ctx.device_id;
  HIP_CHECK(hipSetDevice(device_id));

  HIP_CHECK(hipFree(ctx.gpu_activations.d_x));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_t));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_tb));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_tb2));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_router_score));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_v));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_i));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_mlp1_out));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_e_agg));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_q));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_k));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_v));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_key_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_value_cache));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_att));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_logits));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_cos_vals));
  HIP_CHECK(hipFree(ctx.gpu_activations.d_sin_vals));
  if (ctx.gpu_activations.d_mask) HIP_CHECK(hipFree(ctx.gpu_activations.d_mask));
  if (ctx.gpu_activations.d_token2row) HIP_CHECK(hipFree(ctx.gpu_activations.d_token2row));

  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_attn_w));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_ffn_w));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_qkv));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_o));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_attn_sinks));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_w_router));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_b_router));
  HIP_CHECK(hipFree(ctx.gpu_weights_fp32.d_rms_out_w));

  if (ctx.gpu_expert_bias.g_b_mlp1) HIP_CHECK(hipFree(ctx.gpu_expert_bias.g_b_mlp1));
  if (ctx.gpu_expert_bias.g_b_mlp2) HIP_CHECK(hipFree(ctx.gpu_expert_bias.g_b_mlp2));

  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_qkv_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_o_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_mlp1_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_w_mlp2_bf16));
  HIP_CHECK(hipFree(ctx.gpu_weights_bf16.d_out_bf16));
}

// --------------------- Public lifecycle -------------------------
void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  model_config = &transformer->config;

  HIP_CHECK(hipGetDeviceCount(&g_num_devices));
  if (g_num_devices <= 0) {
    fprintf(stderr, "No HIP devices found!\n");
    exit(EXIT_FAILURE);
  }

  printf("Found %d HIP devices, initializing multi-GPU setup...\n", g_num_devices);
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
static inline void setup_prompt_ctx(
  PromptCtx &ctx,
  Requests *requests,
  int idx,
  Sampler *sampler,
  Transformer *transformer,
  Tokenizer *tokenizer) 
{
  ctx.idx = idx;
  ctx.input_seq = get_str_req_ptr(requests, idx);
  ctx.output_tokens = get_tok_gen_ptr(requests, idx);
  ctx.max_steps = requests->max_seq_len;
  ctx.sampler = sampler;

  const Config &cfg = transformer->config;

  size_t alloc_tok = ctx.input_seq.length() + 3;
  if (!ctx.prompt_tokens) {
      ctx.prompt_tokens = (int*)malloc(alloc_tok * sizeof(int));
      if (!ctx.prompt_tokens) { fprintf(stderr, "OOM: prompt_tokens\n"); exit(EXIT_FAILURE); }
  }

  ctx.num_prompt_tokens = 0;
  encode(tokenizer, ctx.input_seq.c_str(), -1, -1,
         ctx.prompt_tokens, &ctx.num_prompt_tokens, cfg.initial_context_length);
  if (ctx.num_prompt_tokens < 1) {
      fprintf(stderr, "Expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
  }

  ctx.logits_size = cfg.vocab_size;
  if (!ctx.h_logits) {
      ctx.h_logits = (float*)malloc((size_t)ctx.logits_size * sizeof(float));
      if (!ctx.h_logits) { fprintf(stderr, "OOM: h_logits\n"); exit(EXIT_FAILURE); }
  }

  ctx.pos = 0;
  ctx.token = ctx.prompt_tokens[0];
  ctx.is_context_phase = true;
  ctx.finished = false;
  ctx.num_generated = 0;

  if (ctx.output_str.empty()) {
      const char *first_piece = decode_piece(tokenizer, 200006, ctx.token);
      if (first_piece) ctx.output_str += first_piece;
  }

  // debug print, will be removed
  printf("PromptCtx:\n");
  printf("  idx: %d\n", ctx.idx);
  printf("  input_seq: %s\n", ctx.input_seq.c_str());
  printf("  num_prompt_tokens: %d\n", ctx.num_prompt_tokens);
  printf("  logits_size: %d\n", ctx.logits_size);
  printf("  pos: %d\n", ctx.pos);
  printf("  token: %d\n", ctx.token);
  printf("  is_context_phase: %d\n", ctx.is_context_phase);
  printf("  finished: %d\n", ctx.finished);
  printf("  num_generated: %lld\n", ctx.num_generated);
  printf("  start_time: %f\n", ctx.start_time);
  printf("  end_time: %f\n", ctx.end_time);
  printf("  user_data: %p\n", ctx.user_data);
  fflush(stdout);
}

float *gpu_forward_device(Transformer *transformer, int token, int pos, int device_id) {
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

  dim3 block = dim3(BLOCK_SIZE, 1, 1);
  dim3 gridH = get_gemv_grid_dim(H);

  // Embedding (BF16 -> FP32)
  copy_embedding_bf16_row_kernel<<<gridH, block, 0, ctx.stream>>>(
      ctx.gpu_activations.d_x,
      ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
      token, H);

  for (int l = 0; l < p->n_layers; ++l) {
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV = get_gemv_grid_dim(QKV_D);
    fused_rmsnorm_matmul_bias_kernel<bf16_t><<<gridQKV, block, 0, ctx.stream>>>(
        ctx.gpu_activations.d_qkv, ctx.gpu_activations.d_x,
        ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l * QKV_D * H,
        ctx.gpu_weights_fp32.d_b_qkv + l * QKV_D,
        ctx.gpu_weights_fp32.d_rms_attn_w + l * H, H, QKV_D);

    int loff = l * S * KV;
    split_qkv_scatter_to_cache_kernel<<<gridQKV, block, 0, ctx.stream>>>(
        ctx.gpu_activations.d_q, ctx.gpu_activations.d_key_cache,
        ctx.gpu_activations.d_value_cache,
        ctx.gpu_activations.d_qkv, Hq, Hk, D, loff,
        pos * KV);

    dim3 gridApply(Hq > Hk ? Hq : Hk);
    fused_inline_rope_qkv_kernel<<<gridApply, D / 2, 0, ctx.stream>>>(
        ctx.gpu_activations.d_q, ctx.gpu_activations.d_key_cache,
        pos, p->rope_theta, Hq, Hk, D,
        p->rope_scaling_factor, p->initial_context_length,
        loff, pos * KV);

    // Attention
    {
      dim3 grid(Hq);
      dim3 blockA(WF_SIZE);
      size_t shmem_size = (pos + 2) * sizeof(float);
      attention_kernel<<<grid, blockA, shmem_size, ctx.stream>>>(
          ctx.gpu_activations.d_tb, ctx.gpu_activations.d_q,
          ctx.gpu_activations.d_key_cache + loff,
          ctx.gpu_activations.d_value_cache + loff,
          ctx.gpu_weights_fp32.d_attn_sinks, l, pos, D, Hq, Hk, S,
          (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.gpu_activations.d_mask : nullptr);
    }

    // Output projection + residual
    const int O_N = D * Hq;
    dim3 gridO = get_gemv_grid_dim(H);
    fused_matmul_bias_residual_kernel<bf16_t><<<gridO, block, 0, ctx.stream>>>(
        ctx.gpu_activations.d_x, ctx.gpu_activations.d_tb,
        ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l * H * O_N,
        ctx.gpu_weights_fp32.d_b_o + l * H, O_N, H);

    // FFN
    rmsnorm_kernel<<<1, BLOCK_SIZE, 0, ctx.stream>>>(
        ctx.gpu_activations.d_t, ctx.gpu_activations.d_x,
        ctx.gpu_weights_fp32.d_rms_ffn_w + l * H, H);

    dim3 gridE = get_gemv_grid_dim(E);
    matmul_bias_kernel<float><<<gridE, block, 0, ctx.stream>>>(
        ctx.gpu_activations.d_router_score, ctx.gpu_activations.d_t,
        ctx.gpu_weights_fp32.d_w_router + (size_t)l * H * E,
        ctx.gpu_weights_fp32.d_b_router + l * E, H, E);

    size_t shared_mem_size = E * sizeof(float);
    fused_topk_softmax_kernel<<<1, BLOCK_SIZE, shared_mem_size, ctx.stream>>>(
        ctx.gpu_activations.d_topk_v, ctx.gpu_activations.d_topk_i,
        ctx.gpu_activations.d_router_score, E, p->experts_per_token);

    HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_e_agg, 0, H * sizeof(float), ctx.stream));

    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      dim3 gridIM = get_gemv_grid_dim(IM);
      mlp1_fused_kernel<bf16_t><<<gridIM, block, 0, ctx.stream>>>(
          ctx.gpu_activations.d_gate_up, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_w_mlp1_bf16, ctx.gpu_expert_bias.g_b_mlp1,
          ctx.gpu_activations.d_topk_i, kk, l, E, H, IM, p->swiglu_limit);

      mlp2_bias_weighted_accum_kernel<bf16_t><<<gridH, block, 0, ctx.stream>>>(
          ctx.gpu_activations.d_e_agg, ctx.gpu_activations.d_gate_up,
          ctx.gpu_weights_bf16.d_w_mlp2_bf16,
          ctx.gpu_expert_bias.g_b_mlp2,
          ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_topk_v,
          kk, l, E, IM, H);
    }

    residual_add_kernel<<<gridH, block, 0, ctx.stream>>>(ctx.gpu_activations.d_x, ctx.gpu_activations.d_e_agg, H);
  }

  // Final RMSNorm + LM head
  const int V = p->vocab_size;
  dim3 gridV = get_gemv_grid_dim(V);
  fused_rmsnorm_matmul_kernel<bf16_t><<<gridV, block, 0, ctx.stream>>>(
      ctx.gpu_activations.d_logits, ctx.gpu_activations.d_x,
      ctx.gpu_weights_bf16.d_out_bf16,
      ctx.gpu_weights_fp32.d_rms_out_w, H, V);

  return ctx.gpu_activations.d_logits;
}

static long long run_requests_on_device(Transformer *transformer, Tokenizer *tokenizer,
                                        PromptCtx *ctxs, int num_ctxs, int device_id) {
  HIP_CHECK(hipSetDevice(device_id));
  
  long long total_tokens = 0;
  
  for (int ctx_idx = 0; ctx_idx < num_ctxs; ++ctx_idx) {
    PromptCtx &ctx = ctxs[ctx_idx];
    
    const char *first_piece = decode_piece(tokenizer, 200006, ctx.token);
    if (first_piece) ctx.output_str += first_piece;

    while (!ctx.finished && (ctx.max_steps == 0 || ctx.pos < ctx.max_steps)) {
      float *d_log = gpu_forward_device(transformer, ctx.token, ctx.pos, device_id);

      HIP_CHECK(hipStreamSynchronize(g_devices[device_id].stream));
      HIP_CHECK(hipMemcpy(ctx.h_logits, d_log,
                          (size_t)ctx.logits_size * sizeof(float),
                          hipMemcpyDeviceToHost));

      ctx.pos++;
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
        break;
      }

      const char *piece = decode_piece(tokenizer, ctx.token, next);
      if (piece) ctx.output_str += piece;

      ctx.token = next;
    }

    ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;
    total_tokens += (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
  }
  
  return total_tokens;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (g_num_devices == 0) {
    fprintf(stderr, "No GPUs initialized for inference!\n");
    return 0;
  }
  const int num_requests = requests->num_reqs;
  if (num_requests == 0) return 0;

  printf("Processing %d requests across %d GPUs...\n", num_requests, g_num_devices);

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
      
      if (num_ctxs_for_device == 0) return;
      
      int start_idx = dev * ctxs_per_device + std::min(dev, remainder);
      
      long long local_tokens = run_requests_on_device(transformer, tokenizer, 
                                                      &ctxs[start_idx], num_ctxs_for_device, dev);

      // Aggregate tokens
      std::lock_guard<std::mutex> lk(agg_mutex);
      num_token_out += local_tokens;
    });
  }

  for (auto &th : workers) th.join();

  // Sequential, ordered output & cleanup
  for (int idx = 0; idx < num_requests; ++idx) {
    safe_printf(ctxs[idx].output_str.c_str());
    safe_printf("\n");
    free_prompt_ctx_heap_buffers(ctxs[idx]);
  }
  delete[] ctxs;

  printf("Multi-GPU inference completed. Total tokens generated: %lld\n", num_token_out);
  return num_token_out;
}

#endif // GETP_RUN
