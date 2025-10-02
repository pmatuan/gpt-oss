#include "attention/attention.cpp"
#include "common/defines.h"
#include "getp_eval.cpp"
#include "expert.cpp"
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

static void build_rope_tables(const Config *cfg,
                              std::vector<float> &inv_freq_out,
                              float &concentration_out) {
  const int D = cfg->head_dim;
  const int half = D >> 1;
  inv_freq_out.assign(half, 0.0f);
  concentration_out = 1.0f;

  if (half == 0)
    return;

  const float theta = cfg->rope_theta;
  const float scaling = cfg->rope_scaling_factor;
  const float initial_context = static_cast<float>(cfg->initial_context_length);
  const float ntk_beta = 32.0f;
  const float ntk_alpha = 1.0f;
  const float two_pi = 6.28318530717958647692f;
  const float log_theta = logf(theta);

  float low = 0.0f;
  float high = 0.0f;

  if (scaling > 1.0f) {
    concentration_out = 0.1f * logf(scaling) + 1.0f;
    const float denom = log_theta;
    if (denom != 0.0f) {
      low = half * logf(initial_context / (ntk_beta * two_pi)) / denom;
      high = half * logf(initial_context / (ntk_alpha * two_pi)) / denom;
    }
  }

  for (int i = 0; i < half; ++i) {
    const float exponent = (2.0f * static_cast<float>(i)) /
                           static_cast<float>(D);
    const float freq = powf(theta, exponent);
    const float inv_base = freq != 0.0f ? 1.0f / freq : 0.0f;
    float inv = inv_base;

    if (scaling > 1.0f) {
      const float interpolation = inv_base / scaling;
      const float extrapolation = inv_base;
      const float denom = high - low;
      float ramp = denom != 0.0f ? (static_cast<float>(i) - low) / denom : 0.0f;
      ramp = fmaxf(0.0f, fminf(1.0f, ramp));
      const float mask = 1.0f - ramp;
      inv = interpolation * (1.0f - mask) + extrapolation * mask;
    }

    inv_freq_out[i] = inv;
  }
}

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
  const int B = MAX_BATCH_SIZE;  // Allocate for max batch size upfront

  printf("Initializing device %d for batch size %d...\n", device_id, B);
  debug_print_gpu_memory("before allocations", device_id);

  // Activations - allocate for MAX_BATCH_SIZE
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_x, (size_t)B * H * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_t, (size_t)B * H * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tb, (size_t)B * D * Hq * sizeof(bf16_t)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_router_score, (size_t)B * E * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_v,
                      (size_t)B * model_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_topk_i,
                      (size_t)B * model_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_e_agg, (size_t)B * H * sizeof(float)));

  // Pre-allocate workspace for maximum expected batch size
  const size_t max_assignment_pairs =
      (size_t)B * (size_t)model_config->experts_per_token;
  size_t required_gate_bytes =
      max_assignment_pairs * (size_t)IM * sizeof(bf16_t);
  if (required_gate_bytes > 0) {
    HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_gate_up_workspace, required_gate_bytes));
    ctx.gpu_activations.gate_up_workspace_bytes = required_gate_bytes;
  } else {
    ctx.gpu_activations.d_gate_up_workspace = nullptr;
    ctx.gpu_activations.gate_up_workspace_bytes = 0;
  }

  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_qkv,
                      (size_t)B * (D * (Hq + 2 * Hk)) * sizeof(bf16_t)));
  ctx.gpu_activations.d_rope_inv_freq = nullptr;
  ctx.gpu_activations.rope_concentration = 1.0f;

  {
    std::vector<float> rope_inv_freq;
    float rope_concentration = 1.0f;
    build_rope_tables(model_config, rope_inv_freq, rope_concentration);
    if (!rope_inv_freq.empty()) {
      const size_t bytes = rope_inv_freq.size() * sizeof(float);
      HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_rope_inv_freq, bytes));
      HIP_CHECK(hipMemcpy(ctx.gpu_activations.d_rope_inv_freq,
                          rope_inv_freq.data(), bytes, hipMemcpyHostToDevice));
    }
    ctx.gpu_activations.rope_concentration = rope_concentration;
  }

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
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_logits, (size_t)B * V * sizeof(float)));

  // Allocate batch helpers upfront for MAX_BATCH_SIZE
  ctx.capacity_B = B;
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_tokens, (size_t)B * sizeof(int)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_pos, (size_t)B * sizeof(int)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_inv_rms, (size_t)B * sizeof(float)));
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_next_tokens, (size_t)B * sizeof(int)));

  // Allocate pinned host buffers reused across batch iterations
  HostPinnedBatchBuffers &host_ws = ctx.host_pinned_batch;
  host_ws.batch_capacity = B;
  host_ws.expert_capacity = E;
  if (B > 0) {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&host_ws.tokens),
                            (size_t)B * sizeof(int)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&host_ws.pos),
                            (size_t)B * sizeof(int)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&host_ws.next_tokens),
                            (size_t)B * sizeof(int)));
  }
  if (E > 0) {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&host_ws.expert_counts),
                            (size_t)E * sizeof(int)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&host_ws.expert_offsets),
                            (size_t)(E + 1) * sizeof(int)));
  }

  // Persistent device workspace for expert routing
  DeviceExpertWorkspace &expert_ws = ctx.expert_workspace;
  expert_ws.expert_capacity = E;
  expert_ws.assignment_capacity = max_assignment_pairs;
  if (E > 0) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&expert_ws.d_expert_counts),
                        (size_t)E * sizeof(int)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&expert_ws.d_expert_offsets),
                        (size_t)(E + 1) * sizeof(int)));
  }
  if (max_assignment_pairs > 0) {
    HIP_CHECK(
        hipMalloc(reinterpret_cast<void **>(&expert_ws.d_assignment_batches),
                  max_assignment_pairs * sizeof(uint16_t)));
    HIP_CHECK(
        hipMalloc(reinterpret_cast<void **>(&expert_ws.d_assignment_slots),
                  max_assignment_pairs * sizeof(uint8_t)));
  }

  debug_print_gpu_memory("after activations", device_id);

  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;

  // Weights (converted to BF16 on device)
  TransformerWeights *w = &transformer->weights;

  const int H_ = H;
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_rms_attn_w, L * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->rms_attn_w, (size_t)L * H_,
                           ctx.gpu_weights_fp32.d_rms_attn_w, n_streams,
                           chunk_bytes);

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_rms_ffn_w, L * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->rms_ffn_w, (size_t)L * H_,
                           ctx.gpu_weights_fp32.d_rms_ffn_w, n_streams,
                           chunk_bytes);

  const int D_ = model_config->head_dim;
  const int Hq_ = model_config->n_attn_heads;
  const int Hk_ = model_config->n_kv_heads;
  const int QKV_D = D_ * (Hq_ + 2 * Hk_);
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_b_qkv, L * QKV_D * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->b_qkv, (size_t)L * QKV_D,
                           ctx.gpu_weights_fp32.d_b_qkv, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_b_o, L * H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->b_o, (size_t)L * H_, ctx.gpu_weights_fp32.d_b_o,
                           n_streams, chunk_bytes);

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_attn_sinks, L * Hq_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->attn_sinks, (size_t)L * Hq_,
                           ctx.gpu_weights_fp32.d_attn_sinks, n_streams,
                           chunk_bytes);

  const int E_ = model_config->n_experts;
  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_w_router, L * H_ * E_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_router, (size_t)L * H_ * E_,
                           ctx.gpu_weights_fp32.d_w_router, n_streams,
                           chunk_bytes);

  HIP_CHECK(
      hipMalloc(&ctx.gpu_weights_fp32.d_b_router, L * E_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->b_router, (size_t)L * E_,
                           ctx.gpu_weights_fp32.d_b_router, n_streams,
                           chunk_bytes);

  HIP_CHECK(hipMalloc(&ctx.gpu_weights_fp32.d_rms_out_w,
                      H_ * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->rms_out_w, (size_t)H_,
                           ctx.gpu_weights_fp32.d_rms_out_w, n_streams,
                           chunk_bytes);

  debug_print_gpu_memory("after small BF16 weights", device_id);
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

  debug_print_gpu_memory("after large BF16 weights", device_id);

  // Initialize KV cache with a conservative sequence length during warmup
  int seq_hint =
      std::min(model_config->seq_len, model_config->initial_context_length);
  ensure_kv_cache_capacity(ctx, seq_hint);

  debug_print_gpu_memory("after KV cache allocation (model fully loaded)", device_id);
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
  if (ctx.gpu_activations.d_rope_inv_freq)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_rope_inv_freq));
  if (ctx.gpu_activations.d_tokens)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_tokens));
  if (ctx.gpu_activations.d_pos)
    HIP_CHECK(hipFree(ctx.gpu_activations.d_pos));

  if (ctx.expert_workspace.d_assignment_slots)
    HIP_CHECK(hipFree(ctx.expert_workspace.d_assignment_slots));
  if (ctx.expert_workspace.d_assignment_batches)
    HIP_CHECK(hipFree(ctx.expert_workspace.d_assignment_batches));
  if (ctx.expert_workspace.d_expert_offsets)
    HIP_CHECK(hipFree(ctx.expert_workspace.d_expert_offsets));
  if (ctx.expert_workspace.d_expert_counts)
    HIP_CHECK(hipFree(ctx.expert_workspace.d_expert_counts));
  ctx.expert_workspace = DeviceExpertWorkspace{};

  if (ctx.host_pinned_batch.tokens)
    HIP_CHECK(hipHostFree(ctx.host_pinned_batch.tokens));
  if (ctx.host_pinned_batch.pos)
    HIP_CHECK(hipHostFree(ctx.host_pinned_batch.pos));
  if (ctx.host_pinned_batch.next_tokens)
    HIP_CHECK(hipHostFree(ctx.host_pinned_batch.next_tokens));
  if (ctx.host_pinned_batch.expert_counts)
    HIP_CHECK(hipHostFree(ctx.host_pinned_batch.expert_counts));
  if (ctx.host_pinned_batch.expert_offsets)
    HIP_CHECK(hipHostFree(ctx.host_pinned_batch.expert_offsets));
  ctx.host_pinned_batch = HostPinnedBatchBuffers{};

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

  if (use_expert_parallelism) {
    cleanup_device_context_ep(ctx);
  }
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

  if (g_num_devices > 1 && (model_config->n_experts == 128 || std::getenv("GETP_FORCE_EP"))) {
    use_expert_parallelism = true;
    printf("Using expert parallelism with %d devices and %d experts\n", g_num_devices, model_config->n_experts);
  } else {
    use_expert_parallelism = false;
    printf("Not using expert parallelism (n_experts=%d, n_devices=%d)\n", model_config->n_experts, g_num_devices);
  }

  if (use_expert_parallelism) {
    warm_up_ep(transformer);
  } else {
    // Single GPU case
    // init all devices in parallel
    #pragma omp parallel for num_threads(g_num_devices)
    for (int i = 0; i < g_num_devices; ++i) {
      init_device_context(g_devices[i], i, transformer);
    }
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
}

static int *gpu_forward_device_batch(Transformer *transformer,
                                     const int *tokens, const int *pos,
                                     int batch_size, int device_id,
                                     int max_pos_in_batch) {
  PROFILE_SCOPE("gpu_forward_device_batch");
  DeviceContext &ctx = g_devices[device_id];
  HIP_CHECK(hipSetDevice(device_id));

  if (max_pos_in_batch >= 0) {
    ensure_kv_cache_capacity(ctx, max_pos_in_batch + 1);
  }

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
  const int K = p->experts_per_token;
  const int QKV_D = D * (Hq + 2 * Hk);

  HostPinnedBatchBuffers &host_ws = ctx.host_pinned_batch;
  DeviceExpertWorkspace &expert_ws = ctx.expert_workspace;
  if (batch_size > static_cast<int>(host_ws.batch_capacity)) {
    fprintf(stderr,
            "Batch size %d exceeds pinned host capacity %zu on device %d\n",
            batch_size, host_ws.batch_capacity, device_id);
    exit(EXIT_FAILURE);
  }
  if (E > static_cast<int>(host_ws.expert_capacity) ||
      E > static_cast<int>(expert_ws.expert_capacity)) {
    fprintf(stderr,
            "Expert count %d exceeds pre-allocated capacity on device %d\n",
            E, device_id);
    exit(EXIT_FAILURE);
  }
  const size_t required_assignments =
      (size_t)batch_size * (size_t)p->experts_per_token;
  if (required_assignments > expert_ws.assignment_capacity) {
    fprintf(stderr,
            "Assignment capacity %zu insufficient for %zu pairs on device %d\n",
            expert_ws.assignment_capacity, required_assignments, device_id);
    exit(EXIT_FAILURE);
  }

  const uint32_t kv_batch_stride = ctx.gpu_activations.kv_batch_stride;

  // Copy host tokens/positions into device buffers
  const size_t token_bytes = (size_t)batch_size * sizeof(int);
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_tokens, tokens, token_bytes,
                           hipMemcpyHostToDevice, 0));
  HIP_CHECK(hipMemcpyAsync(ctx.gpu_activations.d_pos, pos, token_bytes,
                           hipMemcpyHostToDevice, 0));

  dim3 block(BLOCK_SIZE, 1, 1);
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
        PROFILE_GPU_SCOPE("matmul_bias_qkv_kernel", 0);
        dim3 gridQKV_gemm(
            (QKV_D + MATMUL_QKV_BLOCK_COLS - 1) / MATMUL_QKV_BLOCK_COLS,
            (batch_size + MATMUL_QKV_BLOCK_ROWS - 1) / MATMUL_QKV_BLOCK_ROWS,
            1);
        dim3 blockQKV(WF_SIZE, MATMUL_QKV_WAVES_PER_BLOCK, 1);
        matmul_bias_qkv_kernel<<<gridQKV_gemm, blockQKV>>>(
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
      dim3 grid_fused(Hk, batch_size, 1);
      dim3 block_fused(WF_SIZE, 1, 1);
      fused_split_rope_scatter_qkv_batch_kernel<<<grid_fused, block_fused, 0>>>(
          /*key_cache*/ ctx.gpu_activations.d_key_cache,
          /*value_cache*/ ctx.gpu_activations.d_value_cache,
          /*qkv*/ ctx.gpu_activations.d_qkv,
          /*pos*/ ctx.gpu_activations.d_pos,
          /*Hq,Hk,D*/ Hq, Hk, D,
          /*RoPE*/ ctx.gpu_activations.d_rope_inv_freq,
          ctx.gpu_activations.rope_concentration,
          l,
          ctx.d_kv_layer_offsets, ctx.d_kv_layer_capacity,
          kv_batch_stride,
          /*B*/ batch_size);
    }

    {
      const int kv_mul = Hq / Hk;
      dim3 gridAttn(Hk, batch_size, 1);
      dim3 blockA(ATTN_THREADS_PER_BLOCK);
      const int acc_stride = ATTN_THREADS_PER_BLOCK + 1;
      const size_t shmem_size =
          (size_t)kv_mul *
          (size_t)(ATTN_FLASH_TILE + 1 + D + acc_stride) *
          sizeof(float);
      const bool layer_has_window = (l & 1) == 0;
      if (layer_has_window) {
        PROFILE_GPU_SCOPE("attention_flashdecode_mqa_even", 0);
        attention_flashdecode_mqa_even<<<gridAttn, blockA, shmem_size>>>(
            ctx.gpu_activations.d_tb, ctx.gpu_activations.d_qkv,
            ctx.gpu_activations.d_key_cache, ctx.gpu_activations.d_value_cache,
            ctx.gpu_weights_fp32.d_attn_sinks, l, ctx.gpu_activations.d_pos, D,
            Hq, Hk, ctx.gpu_activations.d_rope_inv_freq,
            ctx.gpu_activations.rope_concentration, ctx.d_kv_layer_offsets,
            ctx.d_kv_layer_capacity,
            p->sliding_window, kv_batch_stride, batch_size);
      } else {
        PROFILE_GPU_SCOPE("attention_flashdecode_mqa_odd", 0);
        attention_flashdecode_mqa_odd<<<gridAttn, blockA, shmem_size>>>(
            ctx.gpu_activations.d_tb, ctx.gpu_activations.d_qkv,
            ctx.gpu_activations.d_key_cache, ctx.gpu_activations.d_value_cache,
            ctx.gpu_weights_fp32.d_attn_sinks, l, ctx.gpu_activations.d_pos, D,
            Hq, Hk, ctx.gpu_activations.d_rope_inv_freq,
            ctx.gpu_activations.rope_concentration, ctx.d_kv_layer_offsets,
            ctx.d_kv_layer_capacity, kv_batch_stride, batch_size);
      }
    }

    // Output projection + residual (batched) - separate kernels
    {
      const int O_N = D * Hq;

      // First do MatMul + Bias: temp = tb @ W^T + b
      {
        PROFILE_GPU_SCOPE("matmul_bias_att_kernel", 0);
        dim3 gridO_gemm(
            (H + MATMUL_ATT_BLOCK_COLS - 1) / MATMUL_ATT_BLOCK_COLS,
            (batch_size + MATMUL_ATT_BLOCK_ROWS - 1) / MATMUL_ATT_BLOCK_ROWS,
            1);
        dim3 blockO(WF_SIZE, MATMUL_ATT_WAVES_PER_BLOCK, 1);
        matmul_bias_att_kernel<<<gridO_gemm, blockO>>>(
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
      PROFILE_GPU_SCOPE("matmul_router_kernel", 0);
      dim3 gridE_gemm((E + TM - 1) / TM, batch_size, 1);
      matmul_router_kernel<<<gridE_gemm, block, 0>>>(
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

    // Zero accumulation buffer before MoE compute
    HIP_CHECK(hipMemsetAsync(ctx.gpu_activations.d_e_agg, 0,
          (size_t)batch_size * H * sizeof(float)));

    // Device-side routing and packing per owner, then bulk P2P (multi-GPU)
    if (use_expert_parallelism) {
      gpu_forward_device_batch_ep(ctx, p->swiglu_limit, H, IM, E, L, K, batch_size, l);
    } else {
      // Baseline local MoE compute (no routing)
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

      const int total_pairs = batch_size * p->experts_per_token;
      int max_assign_per_expert = std::min(total_pairs, batch_size);
      dim3 gridCount((total_pairs + 255) / 256, 1, 1);

      int *d_expert_counts = expert_ws.d_expert_counts;
      int *d_expert_offsets = expert_ws.d_expert_offsets;
      uint16_t *d_assignment_batches = expert_ws.d_assignment_batches;
      uint8_t *d_assignment_slots = expert_ws.d_assignment_slots;

      if (total_pairs > 0 && E > 0) {
        {
          PROFILE_GPU_SCOPE("expert_counting", 0);
          HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, (size_t)E * sizeof(int), 0));

          count_expert_assignments_kernel<<<gridCount, 256, 0>>>(
              d_expert_counts, ctx.gpu_activations.d_topk_i,
              ctx.gpu_activations.d_pos, batch_size, p->experts_per_token, E);
          HIP_CHECK(hipGetLastError());
        }
        {
          PROFILE_GPU_SCOPE("exclusive_scan_small_kernel", 0);
          int threads_scan = 1;
          while (threads_scan < E && threads_scan < 1024)
            threads_scan <<= 1;
          threads_scan = std::max(threads_scan, 32);
          if (threads_scan < E) {
            fprintf(stderr,
                    "exclusive_scan_small_kernel requires blockDim >= E (E=%d)\n",
                    E);
            exit(EXIT_FAILURE);
          }
          size_t shared_scan_bytes = (size_t)threads_scan * sizeof(int);
          exclusive_scan_small_kernel<<<1, threads_scan, shared_scan_bytes>>>(
              d_expert_counts, d_expert_offsets, E);
          HIP_CHECK(hipGetLastError());
        }

        {
          PROFILE_GPU_SCOPE("expert_assignment_building", 0);
          HIP_CHECK(hipMemsetAsync(d_expert_counts, 0, (size_t)E * sizeof(int),
                                   0));

          build_expert_assignments_kernel<<<gridCount, 256, 0>>>(
              ctx.gpu_activations.d_topk_i, ctx.gpu_activations.d_pos,
              d_expert_offsets, d_expert_counts, d_assignment_batches,
              d_assignment_slots, batch_size, p->experts_per_token, E);
          HIP_CHECK(hipGetLastError());
        }

        {
          PROFILE_GPU_SCOPE("mlp1_fused_gemm", 0);
          const int max_tiles =
              (max_assign_per_expert + MATMUL_MLP1_BLOCK_ROWS - 1) /
              MATMUL_MLP1_BLOCK_ROWS;
          dim3 block_mlp1(WF_SIZE, MATMUL_MLP1_WAVES_PER_BLOCK, 1);
          dim3 grid_mlp1((2 * IM + MATMUL_MLP1_BLOCK_COLS - 1) /
                            MATMUL_MLP1_BLOCK_COLS,
                        max_tiles, E);
          mlp1_kernel<<<grid_mlp1, block_mlp1, 0>>>(
              d_gate_up_topk, ctx.gpu_activations.d_t,
              ctx.gpu_weights_bf16.d_w_mlp1_bf16, ctx.stride_w_mlp1_bf16,
              ctx.gpu_expert_bias.g_b_mlp1, d_assignment_batches,
              d_assignment_slots, d_expert_offsets, l, E, H, IM, p->swiglu_limit,
              batch_size, ctx.gpu_activations.d_pos);
        }

        {
          PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm", 0);
          const int max_tiles =
              (max_assign_per_expert + MATMUL_MLP2_BLOCK_ROWS - 1) /
              MATMUL_MLP2_BLOCK_ROWS;
          dim3 block_mlp2(WF_SIZE, MATMUL_MLP2_WAVES_PER_BLOCK, 1);
          dim3 grid_mlp2((H + MATMUL_MLP2_BLOCK_COLS - 1) /
                            MATMUL_MLP2_BLOCK_COLS,
                        max_tiles, E);
          mlp2_kernel<<<grid_mlp2, block_mlp2, 0>>>(
              ctx.gpu_activations.d_e_agg, d_gate_up_topk,
              ctx.gpu_weights_bf16.d_w_mlp2_bf16, ctx.stride_w_mlp2_bf16,
              ctx.gpu_expert_bias.g_b_mlp2, d_assignment_batches,
              d_assignment_slots, d_expert_offsets, ctx.gpu_activations.d_topk_v, l,
              E, IM, H, batch_size, ctx.gpu_activations.d_pos);
        }
      } // End of if total_pairs > 0 && E > 0

    } // End of local MoE compute
    {
      PROFILE_GPU_SCOPE("residual_add_batch_kernel", 0);
      dim3 gridH_batch(gridH_thread.x, batch_size, 1);
      residual_add_batch_kernel<<<gridH_batch, block, 0>>>(
          ctx.gpu_activations.d_x, ctx.gpu_activations.d_e_agg, H, batch_size,
          ctx.gpu_activations.d_pos);
    }
  // Close per-layer loop
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
      PROFILE_GPU_SCOPE("matmul_logits_kernel", 0);
      dim3 gridV_gemm(
          (V + MATMUL_LOGITS_BLOCK_COLS - 1) / MATMUL_LOGITS_BLOCK_COLS,
          (batch_size + MATMUL_LOGITS_BLOCK_ROWS - 1) / MATMUL_LOGITS_BLOCK_ROWS,
          1);
      dim3 blockV(WF_SIZE, MATMUL_LOGITS_WAVES_PER_BLOCK, 1);
      matmul_logits_kernel<<<gridV_gemm, blockV>>>(
          ctx.gpu_activations.d_logits, ctx.gpu_activations.d_t,
          ctx.gpu_weights_bf16.d_out_bf16, H, V, batch_size,
          ctx.gpu_activations.d_pos);
    }
  }

  {
    PROFILE_GPU_SCOPE("argmax_batch_kernel", 0);
    dim3 grid_argmax(batch_size, 1, 1);
    size_t shared_bytes = (size_t)BLOCK_SIZE * (sizeof(float) + sizeof(int));

    argmax_batch_kernel<<<grid_argmax, block, shared_bytes>>>(
        ctx.gpu_activations.d_logits, ctx.gpu_activations.d_next_tokens,
        V, batch_size, ctx.gpu_activations.d_pos);
    HIP_CHECK(hipGetLastError());
  }

  return ctx.gpu_activations.d_next_tokens;
}

static long long run_requests_on_device(Transformer *transformer,
                                        Tokenizer *tokenizer, PromptCtx *ctxs,
                                        int num_ctxs, int device_id) {
  HIP_CHECK(hipSetDevice(device_id));
  DeviceContext &device_ctx = g_devices[device_id];

  long long total_tokens = 0;
  const int BATCH = use_expert_parallelism ? MAX_BATCH_SIZE_120B : MAX_BATCH_SIZE;

  // Process requests in batches of BATCH
  for (int batch_start = 0; batch_start < num_ctxs;
       batch_start += BATCH) {
    const int B = std::min(BATCH, num_ctxs - batch_start);
    PromptCtx *batch_ctxs = ctxs + batch_start;

    HostPinnedBatchBuffers &host_ws = device_ctx.host_pinned_batch;
    if (B > static_cast<int>(host_ws.batch_capacity)) {
      fprintf(stderr,
              "Batch size %d exceeds pinned host capacity %zu on device %d\n",
              B, host_ws.batch_capacity, device_id);
      exit(EXIT_FAILURE);
    }
    int *h_tokens = host_ws.tokens;
    int *h_pos = host_ws.pos;
    int *h_next_tokens = host_ws.next_tokens;
    std::vector<char> h_active(B, 1);

    for (int i = 0; i < B; ++i) {
      const PromptCtx &ctx = batch_ctxs[i];
      h_tokens[i] = ctx.token;
      h_pos[i] = ctx.pos;
      h_next_tokens[i] = -1;
    }

    [[maybe_unused]] int max_seq_hint = 0;
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
          gpu_forward_device_batch(transformer, h_tokens, h_pos, B, device_id,
                                   max_pos_in_batch);
      HIP_CHECK(hipMemcpyAsync(h_next_tokens, d_next, (size_t)B * sizeof(int),
                               hipMemcpyDeviceToHost, 0));
      HIP_CHECK(hipStreamSynchronize(0));

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
          ctx.token = next;
          continue;
        }

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

  // Sequential cleanup, preserve request order
  for (int idx = 0; idx < num_requests; ++idx) {
    free_prompt_ctx_heap_buffers(ctxs[idx]);
  }
  delete[] ctxs;

  printf("Multi-GPU inference completed. Total tokens generated: %lld\n",
         num_token_out);
  return num_token_out;
}

#endif // GETP_RUN
