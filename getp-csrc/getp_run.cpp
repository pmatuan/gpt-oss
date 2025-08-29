#include "../profiler.h"
#include "attention.cpp"
#include "getp_eval.cpp"
#include "matmul.cpp"
#include "utility.cpp"
#include "prompt_ctx.cpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

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

struct GPUWeightBuffersBF16 {
  bf16_t *d_token_embedding_table_bf16;
  bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
  bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
  bf16_t *d_out_bf16;
};

static GPUActivationBuffers gpu_activations;
static GPUWeightBuffersFP32 gpu_weights_fp32;
static GPUExpertBiasBuffers gpu_expert_bias;
static GPUWeightBuffersBF16 gpu_weights_bf16;
static Config *model_config;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  model_config = &transformer->config;
  HIP_CHECK(hipSetDevice(0));

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

  debug_print_gpu_memory("before allocations");

  // Activations
  HIP_CHECK(hipMalloc(&gpu_activations.d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_tb, D * Hq * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_tb2, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&gpu_activations.d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_topk_v,
                      model_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_topk_i,
                      model_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&gpu_activations.d_mlp1_out,
                      2 * IM * sizeof(float))); // (kept for debug options)
  HIP_CHECK(hipMalloc(&gpu_activations.d_gate_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_e_agg, H * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&gpu_activations.d_qkv, (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_q, Hq * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_k, Hk * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_v, Hk * D * sizeof(float)));

  HIP_CHECK(
      hipMalloc(&gpu_activations.d_key_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&gpu_activations.d_value_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_logits, V * sizeof(float)));

  HIP_CHECK(hipMalloc(&gpu_activations.d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&gpu_activations.d_sin_vals, (D / 2) * sizeof(float)));

  HIP_CHECK(hipMalloc(&gpu_activations.d_token2row, S * sizeof(int)));
  {
    int *h_token2row = (int *)malloc(S * sizeof(int));
    for (int i = 0; i < S; ++i)
      h_token2row[i] = i;
    HIP_CHECK(hipMemcpy(gpu_activations.d_token2row, h_token2row,
                        S * sizeof(int), hipMemcpyHostToDevice));
    free(h_token2row);
  }

  if (model_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&gpu_activations.d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] =
            (i - j >= model_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(gpu_activations.d_mask, h_mask, S * S * sizeof(float),
                        hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    gpu_activations.d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations");

  // Weights (small FP32)
  TransformerWeights *w = &transformer->weights;

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_rms_attn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_rms_attn_w, w->rms_attn_w,
                      L * H * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_rms_ffn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_rms_ffn_w, w->rms_ffn_w,
                      L * H * sizeof(float), hipMemcpyHostToDevice));

  const int QKV_D = D * (Hq + 2 * Hk);
  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_b_qkv, w->b_qkv,
                      L * QKV_D * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_b_o, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_b_o, w->b_o, L * H * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_attn_sinks, L * Hq * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_attn_sinks, w->attn_sinks,
                      L * Hq * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_w_router, L * H * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_w_router, w->w_router,
                      L * H * E * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_b_router, L * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_b_router, w->b_router,
                      L * E * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&gpu_weights_fp32.d_rms_out_w, H * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_weights_fp32.d_rms_out_w, w->rms_out_w,
                      H * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights");

  // Expert biases FP32
  HIP_CHECK(hipMalloc(&gpu_expert_bias.g_b_mlp1,
                      (size_t)L * E * (2 * IM) * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_expert_bias.g_b_mlp1, w->b_mlp1,
                      (size_t)L * E * (2 * IM) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMalloc(&gpu_expert_bias.g_b_mlp2, (size_t)L * E * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(gpu_expert_bias.g_b_mlp2, w->b_mlp2,
                      (size_t)L * E * H * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases");

  // Large BF16 weights
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;

  HIP_CHECK(hipMalloc(&gpu_weights_bf16.d_token_embedding_table_bf16,
                      (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V * H,
                           gpu_weights_bf16.d_token_embedding_table_bf16,
                           n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&gpu_weights_bf16.d_w_qkv_bf16,
                      (size_t)L * QKV_D * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H,
                           gpu_weights_bf16.d_w_qkv_bf16, n_streams,
                           chunk_bytes);

  const int O_N = D * Hq;
  HIP_CHECK(hipMalloc(&gpu_weights_bf16.d_w_o_bf16,
                      (size_t)L * H * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H * O_N,
                           gpu_weights_bf16.d_w_o_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&gpu_weights_bf16.d_w_mlp1_bf16,
                      (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E * (2 * IM) * H,
                           gpu_weights_bf16.d_w_mlp1_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&gpu_weights_bf16.d_w_mlp2_bf16,
                      (size_t)L * E * H * IM * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E * H * IM,
                           gpu_weights_bf16.d_w_mlp2_bf16, n_streams,
                           chunk_bytes);

  HIP_CHECK(
      hipMalloc(&gpu_weights_bf16.d_out_bf16, (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V * H, gpu_weights_bf16.d_out_bf16,
                           n_streams, chunk_bytes);

  debug_print_gpu_memory("after large BF16 weights (model loaded)");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  HIP_CHECK(hipFree(gpu_activations.d_x));
  HIP_CHECK(hipFree(gpu_activations.d_t));
  HIP_CHECK(hipFree(gpu_activations.d_tb));
  HIP_CHECK(hipFree(gpu_activations.d_tb2));
  HIP_CHECK(hipFree(gpu_activations.d_router_score));
  HIP_CHECK(hipFree(gpu_activations.d_topk_v));
  HIP_CHECK(hipFree(gpu_activations.d_topk_i));
  HIP_CHECK(hipFree(gpu_activations.d_mlp1_out));
  HIP_CHECK(hipFree(gpu_activations.d_gate_up));
  HIP_CHECK(hipFree(gpu_activations.d_e_agg));
  HIP_CHECK(hipFree(gpu_activations.d_qkv));
  HIP_CHECK(hipFree(gpu_activations.d_q));
  HIP_CHECK(hipFree(gpu_activations.d_k));
  HIP_CHECK(hipFree(gpu_activations.d_v));
  HIP_CHECK(hipFree(gpu_activations.d_key_cache));
  HIP_CHECK(hipFree(gpu_activations.d_value_cache));
  HIP_CHECK(hipFree(gpu_activations.d_att));
  HIP_CHECK(hipFree(gpu_activations.d_logits));
  HIP_CHECK(hipFree(gpu_activations.d_cos_vals));
  HIP_CHECK(hipFree(gpu_activations.d_sin_vals));
  if (gpu_activations.d_mask)
    HIP_CHECK(hipFree(gpu_activations.d_mask));
  if (gpu_activations.d_token2row)
    HIP_CHECK(hipFree(gpu_activations.d_token2row));

  HIP_CHECK(hipFree(gpu_weights_fp32.d_rms_attn_w));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_rms_ffn_w));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_b_qkv));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_b_o));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_attn_sinks));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_w_router));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_b_router));
  HIP_CHECK(hipFree(gpu_weights_fp32.d_rms_out_w));

  if (gpu_expert_bias.g_b_mlp1)
    HIP_CHECK(hipFree(gpu_expert_bias.g_b_mlp1));
  if (gpu_expert_bias.g_b_mlp2)
    HIP_CHECK(hipFree(gpu_expert_bias.g_b_mlp2));

  HIP_CHECK(hipFree(gpu_weights_bf16.d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(gpu_weights_bf16.d_w_qkv_bf16));
  HIP_CHECK(hipFree(gpu_weights_bf16.d_w_o_bf16));
  HIP_CHECK(hipFree(gpu_weights_bf16.d_w_mlp1_bf16));
  HIP_CHECK(hipFree(gpu_weights_bf16.d_w_mlp2_bf16));
  HIP_CHECK(hipFree(gpu_weights_bf16.d_out_bf16));
}

// ============================ Forward ============================
float *gpu_forward(Transformer *transformer, int token, int pos) {
  PROFILE_FUNCTION();
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
  PROFILE_KERNEL_LAUNCH("copy_embedding_bf16_row_kernel",
                        copy_embedding_bf16_row_kernel<<<gridH, block>>>(
                            gpu_activations.d_x,
                            gpu_weights_bf16.d_token_embedding_table_bf16,
                            token, H));

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attn)
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                          rmsnorm_kernel<<<1, BLOCK_SIZE>>>(
                              gpu_activations.d_t, gpu_activations.d_x,
                              gpu_weights_fp32.d_rms_attn_w + l * H, H));

    // (A) QKV = W_qkv@t + b_qkv  (fused) - Using MFMA optimized kernel
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV = get_gemv_grid_dim(QKV_D);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bias_kernel",
        matmul_bias_kernel<bf16_t><<<gridQKV, block>>>(
            gpu_activations.d_qkv, gpu_activations.d_t,
            gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l * QKV_D * H,
            gpu_weights_fp32.d_b_qkv + l * QKV_D, H, QKV_D));

    int loff = l * S * KV;
    PROFILE_KERNEL_LAUNCH("split_qkv_scatter_to_cache_kernel",
                          split_qkv_scatter_to_cache_kernel<<<gridQKV, block>>>(
                              gpu_activations.d_q, gpu_activations.d_key_cache,
                              gpu_activations.d_value_cache,
                              gpu_activations.d_qkv, Hq, Hk, D, loff,
                              pos * KV));

    dim3 gridApply(max(Hq, Hk));
    PROFILE_KERNEL_LAUNCH("fused_inline_rope_qkv_kernel",
                          fused_inline_rope_qkv_kernel<<<gridApply, D / 2>>>(
                              gpu_activations.d_q, gpu_activations.d_key_cache,
                              pos, p->rope_theta, Hq, Hk, D,
                              p->rope_scaling_factor, p->initial_context_length,
                              loff, pos * KV));

    // --- Attention ---
    {
      dim3 grid(Hq);
      dim3 block(WF_SIZE); // 64 threads per block for optimal warp utilization

      // Shared memory for attention scores: pos + 2 elements (pos+1 for sink)
      size_t shmem_size = (pos + 2) * sizeof(float);

      PROFILE_KERNEL_LAUNCH(
          "attention_kernel",
          attention_kernel<<<grid, block, shmem_size>>>(
              gpu_activations.d_tb, gpu_activations.d_q,
              gpu_activations.d_key_cache +
                  loff, // [S, KV] FP32 for this layer - no conversion overhead
              gpu_activations.d_value_cache +
                  loff, // [S, KV] FP32 for this layer - no conversion overhead
              gpu_weights_fp32.d_attn_sinks, l, pos, D, Hq, Hk, S,
              (p->sliding_window > 0 && (l % 2 == 0)) ? gpu_activations.d_mask
                                                      : nullptr));
    }

    // Output projection + bias + residual (bias fused in add) - Using MFMA
    // optimized kernel
    const int O_N = D * Hq;
    dim3 gridO = get_gemv_grid_dim(H);
    PROFILE_KERNEL_LAUNCH(
        "matmul_kernel", matmul_kernel<bf16_t>
        <<<gridO, block>>>(gpu_activations.d_tb2, gpu_activations.d_tb,
                           gpu_weights_bf16.d_w_o_bf16 + (size_t)l * H * O_N,
                           O_N, H));
    PROFILE_KERNEL_LAUNCH("add_bias_residual_inplace_kernel",
                          add_bias_residual_inplace_kernel<<<gridO, block>>>(
                              gpu_activations.d_x, gpu_activations.d_tb2,
                              gpu_weights_fp32.d_b_o + l * H, H));

    // FFN RMSNorm
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                          rmsnorm_kernel<<<1, BLOCK_SIZE>>>(
                              gpu_activations.d_t, gpu_activations.d_x,
                              gpu_weights_fp32.d_rms_ffn_w + l * H, H));

    // Router: scores = W_router@t + b (FP32, use standard kernel)
    dim3 gridE = get_gemv_grid_dim(E);
    PROFILE_KERNEL_LAUNCH(
        "matmul_kernel", matmul_kernel<float>
        <<<gridE, block>>>(gpu_activations.d_router_score, gpu_activations.d_t,
                           gpu_weights_fp32.d_w_router + (size_t)l * H * E, H,
                           E));
    PROFILE_KERNEL_LAUNCH("add_bias_kernel_router",
                          add_bias_kernel<<<gridE, block>>>(
                              gpu_activations.d_router_score,
                              gpu_weights_fp32.d_b_router + l * E, E));

    // Top-k on device
    size_t shared_mem_size = E * sizeof(float);
    PROFILE_KERNEL_LAUNCH(
        "topk_kernel_1token",
        topk_kernel_1token<<<1, BLOCK_SIZE, shared_mem_size>>>(
            gpu_activations.d_topk_v, gpu_activations.d_topk_i,
            gpu_activations.d_router_score, E, p->experts_per_token));
    PROFILE_KERNEL_LAUNCH("softmax_kernel",
                          softmax_kernel<<<1, 1>>>(gpu_activations.d_topk_v,
                                                   p->experts_per_token));

    // Zero e_agg
    HIP_CHECK(hipMemsetAsync(gpu_activations.d_e_agg, 0, H * sizeof(float), 0));

    // (B)+(C): For each k in topk (remain on device): MLP1 fused, then MLP2
    // fused accumulate
    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      // MLP1 fused: gate_up (IM)
      dim3 gridIM = get_gemv_grid_dim(IM);
      PROFILE_KERNEL_LAUNCH(
          "mlp1_fused_kernel",
          mlp1_fused_kernel<bf16_t><<<gridIM, block>>>(
              gpu_activations.d_gate_up, gpu_activations.d_t,
              gpu_weights_bf16.d_w_mlp1_bf16, gpu_expert_bias.g_b_mlp1,
              gpu_activations.d_topk_i, kk, l, E, H, IM, p->swiglu_limit));

      // MLP2 + bias + weighted accumulate to e_agg
      PROFILE_KERNEL_LAUNCH(
          "mlp2_bias_weighted_accum_kernel",
          mlp2_bias_weighted_accum_kernel<bf16_t>
          <<<gridH, block>>>(gpu_activations.d_e_agg, gpu_activations.d_gate_up,
                             gpu_weights_bf16.d_w_mlp2_bf16,
                             gpu_expert_bias.g_b_mlp2, gpu_activations.d_topk_i,
                             gpu_activations.d_topk_v, kk, l, E, IM, H));
    }

    // Residual add (x += e_agg)
    PROFILE_KERNEL_LAUNCH("residual_add_kernel",
                          residual_add_kernel<<<gridH, block>>>(
                              gpu_activations.d_x, gpu_activations.d_e_agg, H));
  }

  // Final RMSNorm
  PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                        rmsnorm_kernel<<<1, BLOCK_SIZE>>>(
                            gpu_activations.d_t, gpu_activations.d_x,
                            gpu_weights_fp32.d_rms_out_w, H));
  HIP_CHECK(hipMemcpy(gpu_activations.d_x, gpu_activations.d_t,
                      H * sizeof(float), hipMemcpyDeviceToDevice));

  // LM head - Using MFMA optimized kernel
  const int V = p->vocab_size;
  dim3 gridV = get_gemv_grid_dim(V);
  PROFILE_KERNEL_LAUNCH("matmul_kernel", matmul_kernel<bf16_t>
                        <<<gridV, block>>>(gpu_activations.d_logits,
                                           gpu_activations.d_x,
                                           gpu_weights_bf16.d_out_bf16, H, V));

  return gpu_activations.d_logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               PromptCtx &ctx) {
  PROFILE_FUNCTION();
  const Config &cfg = transformer->config;

  size_t alloc_tok = ctx.input_seq.length() + 3;
  ctx.prompt_tokens = (int *)malloc(alloc_tok * sizeof(int));
  if (!ctx.prompt_tokens) {
    fprintf(stderr, "OOM: prompt_tokens\n");
    exit(EXIT_FAILURE);
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
    ctx.h_logits = (float *)malloc(ctx.logits_size * sizeof(float));
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

  {
    const char *first_piece = decode_piece(tokenizer, 200006, ctx.token);
    ctx.output_str += first_piece;
  }

  while (!ctx.finished && (ctx.max_steps == 0 || ctx.pos < ctx.max_steps)) {
    float *d_log = gpu_forward(transformer, ctx.token, ctx.pos);

    HIP_CHECK(hipMemcpy(ctx.h_logits, d_log,
                        (size_t)ctx.logits_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    ctx.pos++;
    int next;
    if (ctx.pos < ctx.num_prompt_tokens) {
      next = ctx.prompt_tokens[ctx.pos];
    } else {
      // generation phase
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
    ctx.output_str += piece;

    ctx.token = next;
  }

  ctx.output_tokens[ctx.pos - ctx.num_prompt_tokens + 1] = -1;

  return (long long)(ctx.pos - ctx.num_prompt_tokens + 1);
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();
  long long num_token_out = 0;
  PromptCtx *ctxs = new PromptCtx[requests->num_reqs];
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    PromptCtx &ctx = ctxs[idx];
    ctx.idx = idx;
    ctx.input_seq = get_str_req_ptr(requests, idx);
    ctx.output_tokens = get_tok_gen_ptr(requests, idx);
    ctx.max_steps = requests->max_seq_len;
    ctx.sampler = sampler;
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, ctx);
  }

  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    safe_printf(ctxs[idx].output_str.c_str());
    safe_printf("\n");
    free_prompt_ctx_heap_buffers(ctxs[idx]);
  }
  delete[] ctxs;
  return num_token_out;
}

#endif // GETP_RUN
