#include "../profiler.h"
#include "getp_eval.cpp"
#include "matmul.cpp"
#include "attention.cpp"
#include "utility.cpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <future>
#include <atomic>

#ifndef GETP_RUN
#define GETP_RUN

// Multi-GPU device management
struct DeviceContext {
  int device_id;
  hipStream_t stream;
  
  // Device-specific buffers (same as original single GPU)
  float *d_x, *d_t, *d_tb, *d_tb2;
  float *d_router_score, *d_topk_v, *d_mlp1_out;
  int *d_topk_i;
  float *d_gate_up, *d_e_agg;
  float *d_w_mlp2, *g_b_mlp1, *g_b_mlp2;
  float *d_qkv, *d_q, *d_k, *d_v;
  float *d_key_cache, *d_value_cache;
  float *d_att, *d_logits, *d_mask;
  float *d_cos_vals, *d_sin_vals;
  
  // Small weights (FP32)
  float *d_rms_attn_w, *d_rms_ffn_w;
  float *d_b_qkv, *d_b_o, *d_attn_sinks;
  float *d_w_router, *d_b_router;
  float *d_rms_out_w;
  
  // Expert biases
  float *d_b_mlp1, *d_b_mlp2;
  
  // Large weights (BF16)
  bf16_t *d_token_embedding_table_bf16;
  bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
  bf16_t *d_w_mlp1, *d_w_mlp1_bf16, *d_w_mlp2_bf16;
  bf16_t *d_out_bf16;
};

// Global multi-GPU state
static std::vector<DeviceContext> g_devices;
static int g_num_devices = 0;
static Config *h_config = nullptr;
static std::atomic<int> g_device_round_robin{0};

// Thread-local device assignment
thread_local int tl_assigned_device = -1;

// ---------------- GPU Buffer Structures ----------------
struct GPUActivationBuffers {
  float *d_x, *d_t, *d_tb, *d_tb2;
  float *d_router_score, *d_topk_v, *d_mlp1_out;
  int *d_topk_i;
  float *d_gate_up, *d_e_agg;
  float *d_qkv, *d_q, *d_k, *d_v;
  float *d_key_cache, *d_value_cache;
  float *d_att, *d_logits, *d_mask;
  float *d_cos_vals, *d_sin_vals;
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

struct ModelConfig {
  Config *h_config;
};

// Global instances
static GPUActivationBuffers gpu_activations;
static GPUWeightBuffersFP32 gpu_weights_fp32;
static GPUExpertBiasBuffers gpu_expert_bias = {nullptr, nullptr};
static GPUWeightBuffersBF16 gpu_weights_bf16;
static ModelConfig model_config = {nullptr};

// Device assignment for load balancing
static int get_next_device() {
  return g_device_round_robin.fetch_add(1) % g_num_devices;
}

static int get_thread_device() {
  if (tl_assigned_device == -1) {
    tl_assigned_device = get_next_device();
    HIP_CHECK(hipSetDevice(tl_assigned_device));
  }
  return tl_assigned_device;
}

// ======================== Init / Finish ========================
// Initialize single device context
static void init_device_context(DeviceContext& ctx, int device_id, 
                                Transformer* transformer) {
  ctx.device_id = device_id;
  HIP_CHECK_DEVICE(hipSetDevice(device_id), device_id);
  HIP_CHECK_DEVICE(hipStreamCreateWithFlags(&ctx.stream, hipStreamNonBlocking), device_id);
  
  const Config* p = &transformer->config;
  const int H = p->hidden_dim;
  const int V = p->vocab_size;
  const int L = p->n_layers;
  const int E = p->n_experts;
  const int D = p->head_dim;
  const int Hq = p->n_attn_heads;
  const int Hk = p->n_kv_heads;
  const int KV = D * Hk;
  const int S = p->seq_len;
  const int IM = p->intermediate_dim;

  printf("Initializing device %d...\n", device_id);
  debug_print_gpu_memory("before allocations");

  // Activations
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_x, H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_t, H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_tb, D * Hq * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_tb2, H * sizeof(float)), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_router_score, E * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_topk_v, p->experts_per_token * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_topk_i, p->experts_per_token * sizeof(int)), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_mlp1_out, 2 * IM * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_gate_up, IM * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_e_agg, H * sizeof(float)), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_qkv, (D * (Hq + 2 * Hk)) * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_q, Hq * D * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_k, Hk * D * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_v, Hk * D * sizeof(float)), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_key_cache, L * S * KV * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_value_cache, L * S * KV * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_att, Hq * S * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_logits, V * sizeof(float)), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_cos_vals, (D / 2) * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_sin_vals, (D / 2) * sizeof(float)), device_id);

  if (p->sliding_window > 0) {
    HIP_CHECK_DEVICE(hipMalloc(&ctx.d_mask, S * S * sizeof(float)), device_id);
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] =
            (i - j >= p->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_mask, h_mask, S * S * sizeof(float),
                        hipMemcpyHostToDevice, ctx.stream), device_id);
    free(h_mask);
  } else {
    ctx.d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations");

  // Copy weights to this device
  TransformerWeights *w = &transformer->weights;

  // Small FP32 weights
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_rms_attn_w, L * H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_rms_attn_w, w->rms_attn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_rms_ffn_w, L * H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_rms_ffn_w, w->rms_ffn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  const int QKV_D = D * (Hq + 2 * Hk);
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_b_qkv, L * QKV_D * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_b_qkv, w->b_qkv, L * QKV_D * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_b_o, L * H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_b_o, w->b_o, L * H * sizeof(float), 
                     hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_attn_sinks, L * Hq * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_attn_sinks, w->attn_sinks, L * Hq * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_w_router, L * H * E * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_w_router, w->w_router, L * H * E * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_b_router, L * E * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_b_router, w->b_router, L * E * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_rms_out_w, H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_rms_out_w, w->rms_out_w, H * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  debug_print_gpu_memory("after small FP32 weights");

  // Expert biases FP32
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_b_mlp1, (size_t)L * E * (2 * IM) * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_b_mlp1, w->b_mlp1,
                      (size_t)L * E * (2 * IM) * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);
  
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_b_mlp2, (size_t)L * E * H * sizeof(float)), device_id);
  HIP_CHECK_DEVICE(hipMemcpyAsync(ctx.d_b_mlp2, w->b_mlp2, (size_t)L * E * H * sizeof(float),
                      hipMemcpyHostToDevice, ctx.stream), device_id);

  debug_print_gpu_memory("after expert biases");

  // Large BF16 weights - using async pipelined loading for performance
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;
  
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_token_embedding_table_bf16, (size_t)V * H * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V * H,
                                 ctx.d_token_embedding_table_bf16, ctx.stream, device_id, n_streams, chunk_bytes);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_w_qkv_bf16, (size_t)L * QKV_D * H * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H, ctx.d_w_qkv_bf16,
                                 ctx.stream, device_id, n_streams, chunk_bytes);

  const int O_N = D * Hq;
  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_w_o_bf16, (size_t)L * H * O_N * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H * O_N, ctx.d_w_o_bf16,
                                 ctx.stream, device_id, n_streams, chunk_bytes);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_w_mlp1_bf16, (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E * (2 * IM) * H,
                                 ctx.d_w_mlp1_bf16, ctx.stream, device_id, n_streams, chunk_bytes);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_w_mlp2_bf16, (size_t)L * E * H * IM * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E * H * IM, ctx.d_w_mlp2_bf16,
                                 ctx.stream, device_id, n_streams, chunk_bytes);

  HIP_CHECK_DEVICE(hipMalloc(&ctx.d_out_bf16, (size_t)V * H * sizeof(bf16_t)), device_id);
  copy_fp32_to_bf16_device(w->out, (size_t)V * H, ctx.d_out_bf16,
                                 ctx.stream, device_id, n_streams, chunk_bytes);

  // Wait for all async operations to complete
  HIP_CHECK_DEVICE(hipStreamSynchronize(ctx.stream), device_id);
  
  debug_print_gpu_memory("after large BF16 weights (model loaded)");
}

// Cleanup device context
static void cleanup_device_context(DeviceContext& ctx) {
  int device_id = ctx.device_id;
  HIP_CHECK_DEVICE(hipSetDevice(device_id), device_id);
  
  // Free activations
  HIP_CHECK_DEVICE(hipFree(ctx.d_x), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_t), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_tb), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_tb2), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_router_score), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_topk_v), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_topk_i), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_mlp1_out), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_gate_up), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_e_agg), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_qkv), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_q), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_k), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_v), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_key_cache), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_value_cache), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_att), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_logits), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_cos_vals), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_sin_vals), device_id);
  if (ctx.d_mask)
    HIP_CHECK_DEVICE(hipFree(ctx.d_mask), device_id);

  // Free weights
  HIP_CHECK_DEVICE(hipFree(ctx.d_rms_attn_w), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_rms_ffn_w), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_b_qkv), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_b_o), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_attn_sinks), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_w_router), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_b_router), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_rms_out_w), device_id);

  if (ctx.d_b_mlp1)
    HIP_CHECK_DEVICE(hipFree(ctx.d_b_mlp1), device_id);
  if (ctx.d_b_mlp2)
    HIP_CHECK_DEVICE(hipFree(ctx.d_b_mlp2), device_id);

  HIP_CHECK_DEVICE(hipFree(ctx.d_token_embedding_table_bf16), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_w_qkv_bf16), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_w_o_bf16), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_w_mlp1_bf16), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_w_mlp2_bf16), device_id);
  HIP_CHECK_DEVICE(hipFree(ctx.d_out_bf16), device_id);
  
  HIP_CHECK_DEVICE(hipStreamDestroy(ctx.stream), device_id);
}

// Multi-GPU initialization
void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  h_config = &transformer->config;
  // Ensure model_config is set before any forward pass
  model_config.h_config = &transformer->config;
  
  // Detect available GPUs
  HIP_CHECK(hipGetDeviceCount(&g_num_devices));
  if (g_num_devices <= 0) {
    fprintf(stderr, "No HIP devices found!\n");
    exit(EXIT_FAILURE);
  }
  
  printf("Found %d HIP devices, initializing multi-GPU setup...\n", g_num_devices);
  
  g_devices.resize(g_num_devices);
  
  // Initialize all devices in parallel using OpenMP
  #pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    init_device_context(g_devices[i], i, transformer);
  }
  
  printf("Multi-GPU initialization complete!\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Cleanup all devices in parallel
  #pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    cleanup_device_context(g_devices[i]);
  }
  
  g_devices.clear();
  g_num_devices = 0;
}

// ============================ Forward ============================
// Multi-GPU forward pass
float *gpu_forward_device(Transformer *transformer, int token, int pos, int device_id) {
  DeviceContext& ctx = g_devices[device_id];
  HIP_CHECK_DEVICE(hipSetDevice(device_id), device_id);

  const Config *p = &transformer->config;
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
                        copy_embedding_bf16_row_kernel<<<gridH, block, 0, ctx.stream>>>(
                            ctx.d_x, ctx.d_token_embedding_table_bf16, token, H));

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attn)
    PROFILE_KERNEL_LAUNCH(
        "rmsnorm_kernel",
        rmsnorm_kernel<<<1, BLOCK_SIZE, 0, ctx.stream>>>(ctx.d_t, ctx.d_x, ctx.d_rms_attn_w + l * H, H));

    // (A) QKV = W_qkv@t + b_qkv  (fused) - Using MFMA optimized kernel
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV = get_gemv_grid_dim(QKV_D);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bias_kernel", matmul_bias_kernel<bf16_t>
        <<<gridQKV, block, 0, ctx.stream>>>(ctx.d_qkv, ctx.d_t, ctx.d_w_qkv_bf16 + (size_t)l * QKV_D * H,
                             ctx.d_b_qkv + l * QKV_D, H, QKV_D));

    int loff = l * S * KV;
    PROFILE_KERNEL_LAUNCH(
        "split_qkv_scatter_to_cache_kernel",
        split_qkv_scatter_to_cache_kernel<<<gridQKV, block, 0, ctx.stream>>>(
            ctx.d_q, ctx.d_key_cache, ctx.d_value_cache, ctx.d_qkv, Hq, Hk, D, loff, pos * KV));

    dim3 gridApply(max(Hq, Hk));
    PROFILE_KERNEL_LAUNCH("fused_inline_rope_qkv_kernel",
                          fused_inline_rope_qkv_kernel<<<gridApply, D / 2, 0, ctx.stream>>>(
                              ctx.d_q, ctx.d_key_cache, pos, p->rope_theta,
                              Hq, Hk, D, p->rope_scaling_factor, p->initial_context_length,
                              loff, pos * KV));

    // --- Attention ---
    {
      dim3 grid(Hq);
      dim3 block(WF_SIZE); // 64 threads per block for optimal warp utilization
      
      // Shared memory for attention scores: pos + 2 elements (pos+1 for sink)
      size_t shmem_size = (pos + 2) * sizeof(float);

      PROFILE_KERNEL_LAUNCH(
          "attention_kernel",
          attention_kernel<<<grid, block, shmem_size, ctx.stream>>>(
              ctx.d_tb, ctx.d_q,
              ctx.d_key_cache + loff,   // [S, KV] FP32 for this layer - no conversion overhead
              ctx.d_value_cache + loff, // [S, KV] FP32 for this layer - no conversion overhead
              ctx.d_attn_sinks, l, pos, D, Hq, Hk, S,
              (p->sliding_window > 0 && (l % 2 == 0)) ? ctx.d_mask : nullptr));
    }

    // Output projection + bias + residual (bias fused in add) - Using MFMA
    // optimized kernel
    const int O_N = D * Hq;
    dim3 gridO = get_gemv_grid_dim(H);
    PROFILE_KERNEL_LAUNCH("matmul_kernel", matmul_kernel<bf16_t>
                          <<<gridO, block, 0, ctx.stream>>>(ctx.d_tb2, ctx.d_tb,
                                             ctx.d_w_o_bf16 + (size_t)l * H * O_N,
                                             O_N, H));
    PROFILE_KERNEL_LAUNCH("add_bias_residual_inplace_kernel",
                          add_bias_residual_inplace_kernel<<<gridO, block, 0, ctx.stream>>>(ctx.d_x, ctx.d_tb2, ctx.d_b_o + l * H, H));

    // FFN RMSNorm
    PROFILE_KERNEL_LAUNCH(
        "rmsnorm_kernel",
        rmsnorm_kernel<<<1, BLOCK_SIZE, 0, ctx.stream>>>(ctx.d_t, ctx.d_x, ctx.d_rms_ffn_w + l * H, H));

    // Router: scores = W_router@t + b (FP32, use standard kernel)
    dim3 gridE = get_gemv_grid_dim(E);
    PROFILE_KERNEL_LAUNCH("matmul_kernel", matmul_kernel<float>
              <<<gridE, block, 0, ctx.stream>>>(ctx.d_router_score, ctx.d_t,
                       ctx.d_w_router + (size_t)l * H * E, H,
                       E));
    PROFILE_KERNEL_LAUNCH("add_bias_kernel_router",
                          add_bias_kernel<<<gridE, block, 0, ctx.stream>>>(
                              ctx.d_router_score, ctx.d_b_router + l * E, E));

    // Top-k on device
    size_t shared_mem_size = E * sizeof(float);
    PROFILE_KERNEL_LAUNCH(
        "topk_kernel_1token",
        topk_kernel_1token<<<1, BLOCK_SIZE, shared_mem_size, ctx.stream>>>(
            ctx.d_topk_v, ctx.d_topk_i, ctx.d_router_score, E, p->experts_per_token));

    PROFILE_KERNEL_LAUNCH(
        "softmax_kernel",
        softmax_kernel<<<1, 1, 0, ctx.stream>>>(ctx.d_topk_v, p->experts_per_token));

    // Zero e_agg
  HIP_CHECK(hipMemsetAsync(ctx.d_e_agg, 0, H * sizeof(float), ctx.stream));

    // (B)+(C): For each k in topk (remain on device): MLP1 fused, then MLP2
    // fused accumulate
    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      // MLP1 fused: gate_up (IM)
      dim3 gridIM = get_gemv_grid_dim(IM);
      PROFILE_KERNEL_LAUNCH("mlp1_fused_kernel", mlp1_fused_kernel<bf16_t>
                            <<<gridIM, block, 0, ctx.stream>>>(ctx.d_gate_up, ctx.d_t, ctx.d_w_mlp1_bf16, ctx.d_b_mlp1, ctx.d_topk_i, kk, l, E, H,
                                                IM, p->swiglu_limit));
      // MLP2 + bias + weighted accumulate to e_agg
      PROFILE_KERNEL_LAUNCH(
          "mlp2_bias_weighted_accum_kernel",
          mlp2_bias_weighted_accum_kernel<bf16_t>
          <<<gridH, block, 0, ctx.stream>>>(ctx.d_e_agg, ctx.d_gate_up, ctx.d_w_mlp2_bf16, ctx.d_b_mlp2,
                             ctx.d_topk_i, ctx.d_topk_v, kk, l, E, IM, H));
    }

    // Residual add (x += e_agg)
    PROFILE_KERNEL_LAUNCH(
        "residual_add_kernel",
        residual_add_kernel<<<gridH, block, 0, ctx.stream>>>(ctx.d_x, ctx.d_e_agg, H));
  }
  
  // Final RMSNorm
  PROFILE_KERNEL_LAUNCH("rmsnorm_kernel", rmsnorm_kernel<<<1, BLOCK_SIZE, 0, ctx.stream>>>(
                                              ctx.d_t, ctx.d_x, ctx.d_rms_out_w, H));
  HIP_CHECK(hipMemcpyAsync(ctx.d_x, ctx.d_t, H * sizeof(float), hipMemcpyDeviceToDevice, ctx.stream));

  // LM head - Using MFMA optimized kernel
  const int V = p->vocab_size;
  dim3 gridV = get_gemv_grid_dim(V);
  PROFILE_KERNEL_LAUNCH("matmul_kernel", matmul_kernel<bf16_t>
                        <<<gridV, block, 0, ctx.stream>>>(ctx.d_logits, ctx.d_x, ctx.d_out_bf16, H, V));

  return ctx.d_logits;
}

// ================= Greedy / Sampling Loop =================
long long simple_getp_generate_multigpu(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps, int device_id = -1) {
  // Auto-assign device if not specified
  if (device_id == -1) {
    device_id = get_thread_device();
  } else {
    HIP_CHECK_DEVICE(hipSetDevice(device_id), device_id);
  }

  const char *empty_prompt = "";
  if (!input_seq)
    input_seq = empty_prompt;

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "Expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  int next;
  int token = prompt_tokens[0];
  int pos = 0;

  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {
    float *d_log = gpu_forward_device(transformer, token, pos, device_id);

    float *h_logits = (float *)malloc(transformer->config.vocab_size * sizeof(float));
    HIP_CHECK_DEVICE(hipMemcpyAsync(h_logits, d_log,
              transformer->config.vocab_size * sizeof(float),
              hipMemcpyDeviceToHost, g_devices[device_id].stream), device_id);
    HIP_CHECK_DEVICE(hipStreamSynchronize(g_devices[device_id].stream), device_id);
                      
    pos++;
    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      next = sample(sampler, h_logits);
      output_tokens[pos - num_prompt_tokens] = next;
    }

    if (next == 199999 || next == 200002)
      break;

    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);

    token = next;
    free(h_logits);
  }
  printf("\n");
  output_tokens[pos - num_prompt_tokens + 1] = -1;
  free(prompt_tokens);
  return pos - num_prompt_tokens + 1;
}

// Worker function for multi-threaded request processing
struct WorkerTask {
  Transformer *transformer;
  Tokenizer *tokenizer;
  Sampler *sampler;
  const char *input_seq;
  int *output_tokens;
  int max_seq_len;
  int device_id;
  long long result;
};

void process_request_worker(WorkerTask* task) {
  try {
    task->result = simple_getp_generate_multigpu(
        task->transformer, task->tokenizer, task->sampler,
        task->input_seq, task->output_tokens, task->max_seq_len, task->device_id);
  } catch (const std::exception& e) {
    fprintf(stderr, "Worker thread exception on device %d: %s\n", task->device_id, e.what());
    task->result = 0;
  } catch (...) {
    fprintf(stderr, "Unknown worker thread exception on device %d\n", task->device_id);
    task->result = 0;
  }
}

// Backward compatibility wrapper
float *gpu_forward(Transformer *transformer, int token, int pos) {
  return gpu_forward_device(transformer, token, pos, 0);
}

// Multi-GPU inference with load balancing
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  if (g_num_devices == 0) {
    fprintf(stderr, "No GPUs initialized for inference!\n");
    return 0;
  }
  
  const int num_requests = requests->num_reqs;
  if (num_requests == 0) {
    return 0;
  }
  printf("Processing %d requests across %d GPUs...\n", num_requests, g_num_devices);                                   

  long long total_tokens = 0;

  if (num_requests == 1) {
    // Single request - use first GPU directly
    const char *input_seq = get_str_req_ptr(requests, 0);
    int *output_tokens = get_tok_gen_ptr(requests, 0);
    total_tokens = simple_getp_generate_multigpu(transformer, tokenizer, sampler,
                                                input_seq, output_tokens, 
                                                requests->max_seq_len, 0);
  } else if (num_requests <= g_num_devices) {
    // Fewer or equal requests than GPUs - assign one GPU per request
    std::vector<std::thread> workers;
    std::vector<WorkerTask> tasks(num_requests);
    
    for (int i = 0; i < num_requests; ++i) {
      tasks[i] = {
          transformer, tokenizer, sampler,
          get_str_req_ptr(requests, i),
          get_tok_gen_ptr(requests, i),
          requests->max_seq_len,
          i % g_num_devices,  // Round-robin device assignment
          0
      };
      
      workers.emplace_back(process_request_worker, &tasks[i]);
    }
    
    // Wait for all workers to complete
    for (auto& worker : workers) {
      worker.join();
    }
    
    // Collect results
    for (int i = 0; i < num_requests; ++i) {
      total_tokens += tasks[i].result;
    }
  } else {
    // More requests than GPUs - use thread pool with work stealing
    std::atomic<int> request_counter{0};
    std::vector<std::thread> workers;
    std::mutex output_mutex;
    
    auto worker_func = [&](int worker_id) {
      int device_id = worker_id % g_num_devices;
      long long worker_tokens = 0;
      
      while (true) {
        int req_idx = request_counter.fetch_add(1);
        if (req_idx >= num_requests) break;
        
        const char *input_seq = get_str_req_ptr(requests, req_idx);
        int *output_tokens = get_tok_gen_ptr(requests, req_idx);
        
        long long tokens = simple_getp_generate_multigpu(
            transformer, tokenizer, sampler, input_seq, 
            output_tokens, requests->max_seq_len, device_id);
        
        worker_tokens += tokens;
      }
      
      std::lock_guard<std::mutex> lock(output_mutex);
      total_tokens += worker_tokens;
    };
    
    // Create one worker per GPU (or capped by number of requests)
    int num_workers = std::min(g_num_devices, num_requests);
    for (int i = 0; i < num_workers; ++i) {
      workers.emplace_back(worker_func, i);
    }
    
    // Wait for all workers
    for (auto& worker : workers) {
      worker.join();
    }
  }
  
  printf("Multi-GPU inference completed. Total tokens generated: %lld\n", total_tokens);
  return total_tokens;
}

#endif // GETP_RUN
