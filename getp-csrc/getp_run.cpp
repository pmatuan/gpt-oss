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
  const int IM = model_config->intermediate_dim;

  printf("Initializing device %d...\n", device_id);
  debug_print_gpu_memory("before allocations", device_id);

  // Activations
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_x, H * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_t, H * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb, D * Hq * sizeof(bf16_t)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                      model_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                      model_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_e_agg, H * sizeof(float)));

  // Pre-allocate workspace for maximum expected batch size
  ctx.gpu_activations.d_gate_up_workspace = nullptr;
  ctx.gpu_activations.gate_up_workspace_bytes = 0;

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (D * (Hq + 2 * Hk)) * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q, Hq * D * sizeof(bf16_t)));

  ctx.gpu_activations.d_key_cache = nullptr;
  ctx.gpu_activations.d_value_cache = nullptr;
  ctx.gpu_activations.kv_seq_capacity = 0;
  ctx.gpu_activations.kv_window_capacity = 0;
  ctx.gpu_activations.kv_seq_limit = 0;
  ctx.gpu_activations.kv_batch_stride = 0;
  ctx.h_kv_layer_offsets.clear();
  ctx.h_kv_layer_capacity.clear();
  ctx.d_kv_layer_offsets = nullptr;
  ctx.d_kv_layer_capacity = nullptr;
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, V * sizeof(float)));

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

  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;

  // Expert biases (converted to BF16 on device)
  const int IM_ = model_config->intermediate_dim;
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp1,
                      (size_t)L * E_ * (2 * IM_) * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->b_mlp1, (size_t)L * E_ * (2 * IM_),
                           ctx.gpu_expert_bias.g_b_mlp1, n_streams,
                           chunk_bytes);
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp2,
                      (size_t)L * E_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->b_mlp2, (size_t)L * E_ * H_,
                           ctx.gpu_expert_bias.g_b_mlp2, n_streams,
                           chunk_bytes);

  debug_print_gpu_memory("after expert biases", device_id);

  // Large BF16 weights
  const int V_ = model_config->vocab_size;
  const int O_N = D_ * Hq_;

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
                      (size_t)V_ * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V_ * H_,
                           ctx.gpu_weights_bf16.d_token_embedding_table_bf16,
                           n_streams, chunk_bytes);

  const size_t qkv_stride = matmul_packed_elems(QKV_D, H_);
  ctx.stride_w_qkv_bf16 = qkv_stride;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_qkv_bf16,
                      (size_t)L * qkv_stride * sizeof(bf16_t)));
  std::vector<bf16_t> packed_matrix(qkv_stride);
  for (int l = 0; l < L; ++l) {
    const float *layer_src = w->w_qkv + (size_t)l * QKV_D * H_;
    pack_fp32_to_bf16_matmul(layer_src, QKV_D, H_, packed_matrix.data());
    HIP_CHECK(
        hipMemcpy(ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l * qkv_stride,
                  packed_matrix.data(), qkv_stride * sizeof(bf16_t),
                  hipMemcpyHostToDevice));
  }

  const size_t w_o_stride = matmul_packed_elems(H_, O_N);
  ctx.stride_w_o_bf16 = w_o_stride;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_o_bf16,
                      (size_t)L * w_o_stride * sizeof(bf16_t)));
  packed_matrix.resize(w_o_stride);
  for (int l = 0; l < L; ++l) {
    const float *layer_src = w->w_o + (size_t)l * H_ * O_N;
    pack_fp32_to_bf16_matmul(layer_src, H_, O_N, packed_matrix.data());
    HIP_CHECK(
        hipMemcpy(ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l * w_o_stride,
                  packed_matrix.data(), w_o_stride * sizeof(bf16_t),
                  hipMemcpyHostToDevice));
  }

  const size_t mlp1_stride = matmul_packed_elems(2 * IM_, H_);
  ctx.stride_w_mlp1_bf16 = mlp1_stride;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp1_bf16,
                      (size_t)L * E_ * mlp1_stride * sizeof(bf16_t)));
  packed_matrix.resize(mlp1_stride);
  for (int l = 0; l < L; ++l) {
    for (int e = 0; e < E_; ++e) {
      const size_t offset =
          ((size_t)l * E_ + (size_t)e) * (size_t)(2 * IM_) * (size_t)H_;
      const float *matrix_src = w->w_mlp1 + offset;
      pack_fp32_to_bf16_matmul(matrix_src, 2 * IM_, H_, packed_matrix.data());
      const size_t dst_index = ((size_t)l * E_ + (size_t)e) * mlp1_stride;
      HIP_CHECK(hipMemcpy(ctx.gpu_weights_bf16.d_w_mlp1_bf16 + dst_index,
                          packed_matrix.data(), mlp1_stride * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
    }
  }

  const size_t mlp2_stride = matmul_packed_elems(H_, IM_);
  ctx.stride_w_mlp2_bf16 = mlp2_stride;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp2_bf16,
                      (size_t)L * E_ * mlp2_stride * sizeof(bf16_t)));
  packed_matrix.resize(mlp2_stride);
  for (int l = 0; l < L; ++l) {
    for (int e = 0; e < E_; ++e) {
      const size_t offset =
          ((size_t)l * E_ + (size_t)e) * (size_t)H_ * (size_t)IM_;
      const float *matrix_src = w->w_mlp2 + offset;
      pack_fp32_to_bf16_matmul(matrix_src, H_, IM_, packed_matrix.data());
      const size_t dst_index = ((size_t)l * E_ + (size_t)e) * mlp2_stride;
      HIP_CHECK(hipMemcpy(ctx.gpu_weights_bf16.d_w_mlp2_bf16 + dst_index,
                          packed_matrix.data(), mlp2_stride * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
    }
  }

  const size_t out_stride = matmul_packed_elems(V_, H_);
  ctx.stride_w_out_bf16 = out_stride;
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_bf16.d_out_bf16, out_stride * sizeof(bf16_t)));
  packed_matrix.resize(out_stride);
  pack_fp32_to_bf16_matmul(w->out, V_, H_, packed_matrix.data());
  HIP_CHECK(hipMemcpy(ctx.gpu_weights_bf16.d_out_bf16, packed_matrix.data(),
                      out_stride * sizeof(bf16_t), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after large BF16 weights (model loaded)", device_id);
}

// Cleanup device context
static void cleanup_device_context(DeviceContext &ctx) {
  int device_id = ctx.device_id;
  HIP_CHECK(hipSetDevice(device_id));

  if (ctx.gpu_activations.d_x)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_x));
  if (ctx.gpu_activations.d_t)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_t));
  if (ctx.gpu_activations.d_tb)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_tb));
  if (ctx.gpu_activations.d_router_score)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_router_score));
  if (ctx.gpu_activations.d_topk_v)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_v));
  if (ctx.gpu_activations.d_topk_i)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_topk_i));
  if (ctx.gpu_activations.d_e_agg)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_e_agg));
  if (ctx.gpu_activations.d_gate_up_workspace)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
  if (ctx.gpu_activations.d_qkv)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_qkv));
  if (ctx.gpu_activations.d_q)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_q));
  if (ctx.gpu_activations.d_key_cache)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_key_cache));
  if (ctx.gpu_activations.d_value_cache)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_value_cache));
  if (ctx.d_kv_layer_offsets)
    HIP_CHECK(hipFree(ctx.d_kv_layer_offsets));
  if (ctx.d_kv_layer_capacity)
    HIP_CHECK(hipFree(ctx.d_kv_layer_capacity));
  ctx.h_kv_layer_offsets.clear();
  ctx.h_kv_layer_capacity.clear();
  ctx.gpu_activations.kv_seq_capacity = 0;
  ctx.gpu_activations.kv_window_capacity = 0;
  ctx.gpu_activations.kv_batch_stride = 0;
  if (ctx.gpu_activations.d_logits)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_logits));
  if (ctx.gpu_activations.d_next_tokens)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_next_tokens));
  if (ctx.gpu_activations.d_inv_rms)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_inv_rms));
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
static int ensure_kv_cache_capacity(DeviceContext &ctx, int required_seq);

// Ensure device has capacity for B batch slots (reallocates activations &
// caches if needed)
static inline void ensure_device_capacity(DeviceContext &ctx, int B,
                                          int max_seq_hint = 0) {
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
    auto free_if = [](auto *&ptr) {
      if (ptr) {
        HIP_CHECK(hipFree(ptr));
        ptr = nullptr;
      }
    };

    free_if(ctx.gpu_activations.d_x);
    free_if(ctx.gpu_activations.d_t);
    free_if(ctx.gpu_activations.d_tb);
    free_if(ctx.gpu_activations.d_router_score);
    free_if(ctx.gpu_activations.d_topk_v);
    free_if(ctx.gpu_activations.d_topk_i);
    free_if(ctx.gpu_activations.d_e_agg);
    free_if(ctx.gpu_activations.d_gate_up_workspace);
    free_if(ctx.gpu_activations.d_qkv);
    free_if(ctx.gpu_activations.d_q);
    free_if(ctx.gpu_activations.d_key_cache);
    free_if(ctx.gpu_activations.d_value_cache);
    free_if(ctx.gpu_activations.d_logits);
    free_if(ctx.gpu_activations.d_next_tokens);

    ctx.gpu_activations.gate_up_workspace_bytes = 0;
    ctx.gpu_activations.kv_seq_capacity = 0;
    ctx.gpu_activations.kv_window_capacity = 0;
    ctx.gpu_activations.kv_seq_limit = 0;
    ctx.gpu_activations.kv_batch_stride = 0;
    free_if(ctx.d_kv_layer_offsets);
    free_if(ctx.d_kv_layer_capacity);
    ctx.h_kv_layer_offsets.clear();
    ctx.h_kv_layer_capacity.clear();

    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_x, (size_t)B * H * sizeof(bf16_t)));
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_t, (size_t)B * H * sizeof(bf16_t)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb,
                        (size_t)B * D * Hq * sizeof(bf16_t)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score,
                        (size_t)B * p->n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                        (size_t)B * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                        (size_t)B * p->experts_per_token * sizeof(int)));
    HIP_CHECK(
        hipMalloc(&ctx.gpu_activations.d_e_agg, (size_t)B * H * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                        (size_t)B * (D * (Hq + 2 * Hk)) * sizeof(bf16_t)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q,
                        (size_t)B * Hq * D * sizeof(bf16_t)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                        (size_t)B * V * sizeof(float)));
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens,
                        (size_t)B * sizeof(int)));

    ctx.capacity_B = B;
  } else {
    if (!ctx.gpu_activations.d_x) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_x,
                          (size_t)ctx.capacity_B * H * sizeof(bf16_t)));
    }
    if (!ctx.gpu_activations.d_t) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_t,
                          (size_t)ctx.capacity_B * H * sizeof(bf16_t)));
    }
    if (!ctx.gpu_activations.d_tb) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb,
                          (size_t)ctx.capacity_B * D * Hq * sizeof(bf16_t)));
    }
    if (!ctx.gpu_activations.d_router_score) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score,
                          (size_t)ctx.capacity_B * p->n_experts * sizeof(float)));
    }
    if (!ctx.gpu_activations.d_topk_v) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                          (size_t)ctx.capacity_B * p->experts_per_token * sizeof(float)));
    }
    if (!ctx.gpu_activations.d_topk_i) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                          (size_t)ctx.capacity_B * p->experts_per_token * sizeof(int)));
    }
    if (!ctx.gpu_activations.d_e_agg) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_e_agg,
                          (size_t)ctx.capacity_B * H * sizeof(float)));
    }
    if (!ctx.gpu_activations.d_qkv) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                          (size_t)ctx.capacity_B * (D * (Hq + 2 * Hk)) * sizeof(bf16_t)));
    }
    if (!ctx.gpu_activations.d_q) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q,
                          (size_t)ctx.capacity_B * Hq * D * sizeof(bf16_t)));
    }
    if (!ctx.gpu_activations.d_logits) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits,
                          (size_t)ctx.capacity_B * V * sizeof(float)));
    }
    if (!ctx.gpu_activations.d_next_tokens) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens,
                          (size_t)ctx.capacity_B * sizeof(int)));
    }
  }

  size_t max_assignments = (size_t)ctx.capacity_B * p->experts_per_token;
  size_t required_gate_bytes =
      max_assignments * (size_t)IM * sizeof(bf16_t);
  if (ctx.gpu_activations.gate_up_workspace_bytes < required_gate_bytes) {
    if (ctx.gpu_activations.d_gate_up_workspace) {
      HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
    }
    if (required_gate_bytes > 0) {
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace,
                          required_gate_bytes));
    }
    ctx.gpu_activations.gate_up_workspace_bytes = required_gate_bytes;
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
  int seq_hint = max_seq_hint > 0 ? max_seq_hint : p->seq_len;
  seq_hint = std::max(1, std::min(seq_hint, p->seq_len));

  const bool need_kv_alloc =
      !ctx.gpu_activations.d_key_cache || !ctx.gpu_activations.d_value_cache ||
      ctx.gpu_activations.kv_seq_capacity == 0;

  if (need_kv_alloc) {
    auto kv_bytes_for_seq = [&](int seq) -> size_t {
      seq = std::max(1, std::min(seq, p->seq_len));
      const bool has_window = (p->sliding_window > 0);
      const int even_layers = has_window ? (p->n_layers + 1) / 2 : 0;
      const int odd_layers = p->n_layers - even_layers;
      const int window_tokens = has_window ? std::min(seq, p->sliding_window)
                                           : seq;
      const size_t tokens_per_seq =
          (size_t)odd_layers * seq + (size_t)even_layers * window_tokens;
      const size_t elems_per_seq = tokens_per_seq * (size_t)KV;
      return elems_per_seq * (size_t)ctx.capacity_B * sizeof(bf16_t);
    };

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

    constexpr size_t kSafetyMargin = 1024ULL << 20; // 1024 MiB
    size_t available_for_kv =
        free_bytes > kSafetyMargin ? free_bytes - kSafetyMargin : 0;

    int target_seq = seq_hint;
    size_t key_bytes = kv_bytes_for_seq(target_seq);

    while (target_seq > 1 &&
           (key_bytes == 0 || 2 * key_bytes > available_for_kv)) {
      const int next_seq = std::max(1, target_seq / 2);
      if (next_seq == target_seq)
        break;
      target_seq = next_seq;
      key_bytes = kv_bytes_for_seq(target_seq);
    }

    if (key_bytes == 0 || 2 * key_bytes > available_for_kv) {
      double need_gib = (2.0 * (double)key_bytes) / (1024.0 * 1024.0 * 1024.0);
      double avail_gib = (double)available_for_kv /
                         (1024.0 * 1024.0 * 1024.0);
      fprintf(stderr,
              "Unable to allocate KV cache for batch %d: need %.2f GiB, have %.2f GiB\n",
              ctx.capacity_B, need_gib, avail_gib);
      exit(EXIT_FAILURE);
    }

    ctx.gpu_activations.kv_seq_limit = target_seq;
    int actual_seq = ensure_kv_cache_capacity(ctx, target_seq);
    ctx.gpu_activations.kv_seq_limit = actual_seq;
  }
}

static int ensure_kv_cache_capacity(DeviceContext &ctx, int required_seq) {
  const Config *p = model_config;
  if (!p)
    return ctx.gpu_activations.kv_seq_capacity;

  int seq_limit_hint = ctx.gpu_activations.kv_seq_limit;
  if (seq_limit_hint > 0 && required_seq > seq_limit_hint)
    required_seq = seq_limit_hint;

  required_seq = std::max(1, std::min(required_seq, p->seq_len));

  const bool has_window = (p->sliding_window > 0);
  const int KV = p->head_dim * p->n_kv_heads;
  const int L = p->n_layers;

  int current_full = ctx.gpu_activations.kv_seq_capacity;
  if (current_full >= required_seq && ctx.gpu_activations.d_key_cache &&
      ctx.gpu_activations.d_value_cache) {
    return current_full;
  }

  int target_full = current_full > 0 ? current_full : 1;
  while (target_full < required_seq) {
    int grown = target_full * 2;
    if (grown < required_seq)
      grown = required_seq;
    if (seq_limit_hint > 0 && grown > seq_limit_hint)
      grown = seq_limit_hint;
    if (grown > p->seq_len)
      grown = p->seq_len;
    if (grown == target_full)
      break;
    target_full = grown;
  }

  if (target_full <= current_full && ctx.gpu_activations.d_key_cache &&
      ctx.gpu_activations.d_value_cache) {
    if (seq_limit_hint > 0) {
      ctx.gpu_activations.kv_seq_limit =
          std::min(seq_limit_hint, ctx.gpu_activations.kv_seq_capacity);
    } else {
      ctx.gpu_activations.kv_seq_limit = ctx.gpu_activations.kv_seq_capacity;
    }
    return ctx.gpu_activations.kv_seq_capacity;
  }

  std::vector<uint32_t> old_offsets = ctx.h_kv_layer_offsets;
  std::vector<int> old_capacity = ctx.h_kv_layer_capacity;
  const uint32_t old_batch_stride = ctx.gpu_activations.kv_batch_stride;
  bf16_t *old_key_ptr = ctx.gpu_activations.d_key_cache;
  bf16_t *old_value_ptr = ctx.gpu_activations.d_value_cache;

  int attempt_full = target_full;
  int fallback_full = current_full;

  while (attempt_full > fallback_full) {
    const int window_limit = has_window ? p->sliding_window : attempt_full;
    const int attempt_window = has_window ? std::min(attempt_full, window_limit)
                                          : attempt_full;

    std::vector<uint32_t> new_offsets(L + 1, 0);
    std::vector<int> new_capacity(L, 0);

    size_t accum = 0;
    for (int l = 0; l < L; ++l) {
      const bool layer_has_window = has_window && ((l & 1) == 0);
      const int layer_cap = layer_has_window ? attempt_window : attempt_full;
      new_offsets[l] = static_cast<uint32_t>(accum);
      new_capacity[l] = layer_cap;
      accum += static_cast<size_t>(layer_cap) * KV;
    }
    new_offsets[L] = static_cast<uint32_t>(accum);

    const size_t batch_stride_new = accum;
    const size_t total_elems = (size_t)ctx.capacity_B * batch_stride_new;
    const size_t total_bytes = total_elems * sizeof(bf16_t);

    bf16_t *new_key = nullptr;
    hipError_t err_key = hipMalloc(&new_key, total_bytes);
    if (err_key == hipErrorOutOfMemory) {
      if (attempt_full == fallback_full + 1) {
        attempt_full = fallback_full;
      } else {
        attempt_full = std::max(fallback_full + 1, attempt_full / 2);
      }
      continue;
    } else if (err_key != hipSuccess) {
      HIP_CHECK(err_key);
    }
    hipError_t err_mem_key = hipMemsetAsync(new_key, 0, total_bytes, 0);
    if (err_mem_key != hipSuccess) {
      HIP_CHECK(hipFree(new_key));
      HIP_CHECK(err_mem_key);
    }

    bf16_t *new_value = nullptr;
    hipError_t err_value = hipMalloc(&new_value, total_bytes);
    if (err_value == hipErrorOutOfMemory) {
      HIP_CHECK(hipFree(new_key));
      if (attempt_full == fallback_full + 1) {
        attempt_full = fallback_full;
      } else {
        attempt_full = std::max(fallback_full + 1, attempt_full / 2);
      }
      continue;
    } else if (err_value != hipSuccess) {
      HIP_CHECK(hipFree(new_key));
      HIP_CHECK(err_value);
    }
    hipError_t err_mem_value = hipMemsetAsync(new_value, 0, total_bytes, 0);
    if (err_mem_value != hipSuccess) {
      HIP_CHECK(hipFree(new_key));
      HIP_CHECK(hipFree(new_value));
      HIP_CHECK(err_mem_value);
    }

    if (old_key_ptr && !old_offsets.empty()) {
      const size_t old_layers = old_capacity.size();
      for (int l = 0; l < std::min(L, static_cast<int>(old_layers)); ++l) {
        const int copy_tokens = std::min(old_capacity[l], new_capacity[l]);
        if (copy_tokens <= 0)
          continue;
        const size_t bytes = static_cast<size_t>(copy_tokens) * KV *
                             sizeof(bf16_t);
        if (bytes == 0)
          continue;
        const size_t src_layer_offset = old_offsets[l];
        const size_t dst_layer_offset = new_offsets[l];
        for (int b = 0; b < ctx.capacity_B; ++b) {
          const bf16_t *src_k = old_key_ptr +
                                (size_t)b * old_batch_stride + src_layer_offset;
          bf16_t *dst_k = new_key + (size_t)b * batch_stride_new +
                           dst_layer_offset;
          HIP_CHECK(hipMemcpy(dst_k, src_k, bytes, hipMemcpyDeviceToDevice));
        }
      }
    }
    if (old_value_ptr && !old_offsets.empty()) {
      const size_t old_layers = old_capacity.size();
      for (int l = 0; l < std::min(L, static_cast<int>(old_layers)); ++l) {
        const int copy_tokens = std::min(old_capacity[l], new_capacity[l]);
        if (copy_tokens <= 0)
          continue;
        const size_t bytes = static_cast<size_t>(copy_tokens) * KV *
                             sizeof(bf16_t);
        if (bytes == 0)
          continue;
        const size_t src_layer_offset = old_offsets[l];
        const size_t dst_layer_offset = new_offsets[l];
        for (int b = 0; b < ctx.capacity_B; ++b) {
          const bf16_t *src_v = old_value_ptr +
                                (size_t)b * old_batch_stride + src_layer_offset;
          bf16_t *dst_v = new_value + (size_t)b * batch_stride_new +
                           dst_layer_offset;
          HIP_CHECK(hipMemcpy(dst_v, src_v, bytes, hipMemcpyDeviceToDevice));
        }
      }
    }

    if (old_key_ptr)
      HIP_CHECK(hipFree(old_key_ptr));
    if (old_value_ptr)
      HIP_CHECK(hipFree(old_value_ptr));

    if (ctx.d_kv_layer_offsets)
      HIP_CHECK(hipFree(ctx.d_kv_layer_offsets));
    if (ctx.d_kv_layer_capacity)
      HIP_CHECK(hipFree(ctx.d_kv_layer_capacity));

    ctx.h_kv_layer_offsets = new_offsets;
    ctx.h_kv_layer_capacity = new_capacity;

    const size_t offsets_bytes = ctx.h_kv_layer_offsets.size() *
                                 sizeof(uint32_t);
    const size_t capacity_bytes = ctx.h_kv_layer_capacity.size() *
                                  sizeof(int);
    HIP_CHECK(hipMalloc(&ctx.d_kv_layer_offsets, offsets_bytes));
    HIP_CHECK(hipMemcpy(ctx.d_kv_layer_offsets,
                        ctx.h_kv_layer_offsets.data(), offsets_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&ctx.d_kv_layer_capacity, capacity_bytes));
    HIP_CHECK(hipMemcpy(ctx.d_kv_layer_capacity,
                        ctx.h_kv_layer_capacity.data(), capacity_bytes,
                        hipMemcpyHostToDevice));

    ctx.gpu_activations.d_key_cache = new_key;
    ctx.gpu_activations.d_value_cache = new_value;
    ctx.gpu_activations.kv_seq_capacity = attempt_full;
    ctx.gpu_activations.kv_window_capacity = attempt_window;
    ctx.gpu_activations.kv_batch_stride =
        static_cast<uint32_t>(batch_stride_new);
    ctx.gpu_activations.kv_seq_limit = attempt_full;

    if (attempt_full < target_full) {
      printf("[DEVICE] %d reduced KV cache to %d tokens due to allocation limits\n",
             ctx.device_id, attempt_full);
    }

    return attempt_full;
  }

  if (ctx.gpu_activations.d_key_cache && ctx.gpu_activations.d_value_cache) {
    ctx.gpu_activations.kv_seq_limit =
        ctx.gpu_activations.kv_seq_capacity;
    return ctx.gpu_activations.kv_seq_capacity;
  }

  fprintf(stderr,
          "Failed to allocate KV cache for device %d (batch=%d, seq=%d)\n",
          ctx.device_id, ctx.capacity_B, required_seq);
  exit(EXIT_FAILURE);
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

static int *gpu_forward_device_batch(Transformer *transformer,
                                     const int *tokens, const int *pos,
                                     int batch_size, int device_id,
                                     int max_pos_in_batch) {
  PROFILE_SCOPE("gpu_forward_device_batch");
  DeviceContext &ctx = g_devices[device_id];
  HIP_CHECK(hipSetDevice(device_id));

  if (batch_size <= 0) {
    return ctx.gpu_activations.d_next_tokens;
  }

  const Config *p = model_config;
  const int H = p->hidden_dim;
  const int D = p->head_dim;
  const int Hq = p->n_attn_heads;
  const int Hk = p->n_kv_heads;
  const int KV = D * Hk;
  const int IM = p->intermediate_dim;
  const int E = p->n_experts;
  const int L = p->n_layers;
  const int V = p->vocab_size;
  const int QKV_D = D * (Hq + 2 * Hk);

  (void)ensure_kv_cache_capacity(ctx, max_pos_in_batch + 1);
  const uint32_t kv_batch_stride = ctx.gpu_activations.kv_batch_stride;

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
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16_mfma_qkv", 0);
        dim3 gridQKV_gemm((QKV_D + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM, 1);
        dim3 blockQKV(16, 4, 1);
        matmul_bias_gemm_kernel_bf16_mfma<<<gridQKV_gemm, blockQKV>>>(
            ctx.gpu_activations.d_qkv,
            ctx.gpu_activations.d_t,
            ctx.gpu_weights_bf16.d_w_qkv_bf16 + (size_t)l * ctx.stride_w_qkv_bf16,
            ctx.gpu_weights_fp32.d_b_qkv + (size_t)l * QKV_D,
            H, QKV_D, batch_size,
            ctx.gpu_activations.d_pos);
      }
    }

    // Scatter QKV to q / caches (batched)
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
          l,
          ctx.d_kv_layer_offsets, ctx.d_kv_layer_capacity,
          kv_batch_stride,
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
          Hq, Hk, ctx.d_kv_layer_offsets, ctx.d_kv_layer_capacity,
          layer_has_window ? p->sliding_window : 0, kv_batch_stride,
          batch_size);
    }

    // Output projection + residual (batched) - separate kernels
    {
      const int O_N = D * Hq;

      // First do MatMul + Bias: temp = tb @ W^T + b
      {
        PROFILE_GPU_SCOPE("matmul_bias_gemm_kernel_bf16_mfma_att", 0);
        dim3 gridO_gemm((H + TM_MM - 1) / TM_MM, (batch_size + TN_MM - 1) / TN_MM, 1);
        dim3 blockO(16, 4, 1);
        matmul_bias_gemm_kernel_bf16_mfma<<<gridO_gemm, blockO>>>(
            ctx.gpu_activations.d_t, ctx.gpu_activations.d_tb,
            ctx.gpu_weights_bf16.d_w_o_bf16 + (size_t)l * ctx.stride_w_o_bf16,
            ctx.gpu_weights_fp32.d_b_o + l * H, O_N, H, batch_size,
            ctx.gpu_activations.d_pos);
      }

      // Then do residual add: x = x + temp
      {
        PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
        dim3 gridH_batch(gridH_thread.x, batch_size, 1);
        residual_add_batch_kernel_bf16<<<gridH_batch, block, 0>>>(
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
                                (size_t)batch_size * (size_t)IM *
                                sizeof(bf16_t);
    if (ctx.gpu_activations.gate_up_workspace_bytes < gate_up_topk_bytes) {
      if (ctx.gpu_activations.d_gate_up_workspace) {
        HIP_CHECK(hipFree(ctx.gpu_activations.d_gate_up_workspace));
      }
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace,
                          gate_up_topk_bytes));
      ctx.gpu_activations.gate_up_workspace_bytes = gate_up_topk_bytes;
    }
    bf16_t *d_gate_up_topk = ctx.gpu_activations.d_gate_up_workspace;
    HIP_CHECK(hipMemsetAsync(d_gate_up_topk, 0, gate_up_topk_bytes, 0));

    int total_pairs = batch_size * p->experts_per_token;
    int *d_expert_counts = nullptr;
    int *d_expert_offsets = nullptr;
    uint16_t *d_assignment_batches = nullptr;
    uint8_t *d_assignment_slots = nullptr;
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
                              total_assignments * sizeof(uint16_t)));
          HIP_CHECK(
              hipMalloc(&d_assignment_slots, total_assignments * sizeof(uint8_t)));

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
            (max_assign_per_expert + MLP_TILE_TOKENS - 1) / MLP_TILE_TOKENS;
        dim3 block_mlp1(MLP_THREAD_X, MLP_THREAD_Y, 1);
        dim3 grid_mlp1((2 * IM + MLP_TILE_COLS - 1) / MLP_TILE_COLS,
                       max_tiles, E);
        mlp1_fused_gemm_kernel<<<grid_mlp1, block_mlp1, 0>>>(
          d_gate_up_topk, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_w_mlp1_bf16, ctx.stride_w_mlp1_bf16,
          ctx.gpu_expert_bias.g_b_mlp1, d_assignment_batches,
          d_assignment_slots, d_expert_offsets, l, E, H, IM, p->swiglu_limit,
          batch_size, ctx.gpu_activations.d_pos);
      }

      {
        PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm", 0);
        const int max_tiles =
            (max_assign_per_expert + MLP_TILE_TOKENS - 1) / MLP_TILE_TOKENS;
        dim3 block_mlp2(MLP_THREAD_X, MLP_THREAD_Y, 1);
        dim3 grid_mlp2((H + MLP_TILE_COLS - 1) / MLP_TILE_COLS,
                       max_tiles, E);
        mlp2_bias_weighted_accum_gemm_kernel<<<grid_mlp2, block_mlp2, 0>>>(
          ctx.gpu_activations.d_e_agg, d_gate_up_topk,
          ctx.gpu_weights_bf16.d_w_mlp2_bf16, ctx.stride_w_mlp2_bf16,
          ctx.gpu_expert_bias.g_b_mlp2, d_assignment_batches,
          d_assignment_slots, d_expert_offsets, ctx.gpu_activations.d_topk_v, l,
          E, IM, H, batch_size, ctx.gpu_activations.d_pos);
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

    // 2) MatMul for logits - separate GEMM version (no bias)
    {
      PROFILE_GPU_SCOPE("matmul_gemm_kernel_bf16_mfma", 0);
      dim3 gridV_gemm(
          (V + MATMUL_LOGITS_TILE_COLS - 1) / MATMUL_LOGITS_TILE_COLS,
          (batch_size + MATMUL_LOGITS_TILE_ROWS - 1) / MATMUL_LOGITS_TILE_ROWS,
          1);
      dim3 blockV(16, 4, 1);
      matmul_gemm_kernel_bf16_mfma<<<gridV_gemm, blockV>>>(
          ctx.gpu_activations.d_logits, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16, H, V, batch_size,
          ctx.gpu_activations.d_pos);
    }
  }

  dim3 grid_argmax(batch_size, 1, 1);
  size_t shared_bytes = (size_t)BLOCK_SIZE * (sizeof(float) + sizeof(int));

  argmax_batch_kernel<<<grid_argmax, block, shared_bytes>>>(
      ctx.gpu_activations.d_logits, ctx.gpu_activations.d_next_tokens,
      V, batch_size, ctx.gpu_activations.d_pos);
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

    int max_seq_hint = 0;
    const int seq_cap = transformer->config.seq_len;
    for (int i = 0; i < B; ++i) {
      const PromptCtx &ctx = batch_ctxs[i];
      int limit = ctx.max_steps > 0 ? ctx.max_steps : seq_cap;
      limit = std::max(limit, ctx.num_prompt_tokens);
      if (limit > seq_cap)
        limit = seq_cap;
      if (limit > max_seq_hint)
        max_seq_hint = limit;
    }

    // Ensure device buffers sized for B and reserve KV cache up front
    ensure_device_capacity(g_devices[device_id], B, max_seq_hint);

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
