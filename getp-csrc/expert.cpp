#include "common/defines.h"
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

#define EP_OWNER_RULE(e, ndev) ((int)((e) % (ndev)))

static std::vector<int> g_ep_owner_map; // size = L * E, row-major [l*E + e]
// For each layer and device, build local expert id mapping: -1 if not local else [0..E_local-1]
static std::vector<std::vector<int>> g_ep_local_id; // [L][E] -> local id or -1
// For convenience: flattened e2lid per owner per layer for upload [ndev, L, E]
static std::vector<int> g_e2lid_allowners; // size = ndev*L*E
static std::vector<int> g_E_local_per_owner_layer; // size = ndev*L

static inline int ep_owner_of(int l, int e, int E) {
  return g_ep_owner_map.empty() ? 0 : g_ep_owner_map[(size_t)l * (size_t)E + (size_t)e];
}

// Helper to allocate host arrays of device pointers lazily
template <typename T>
static void ensure_host_ptr_array(T ***arr, int ndev) {
  if (!*arr) {
    *arr = (T **)malloc(sizeof(T *) * ndev);
    for (int i = 0; i < ndev; ++i) (*arr)[i] = nullptr;
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

static void init_device_context_ep(DeviceContext &ctx, int device_id,
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
  HIP_CHECK(hipMalloc(&ctx.gpu_activations.d_q, (size_t)B * Hq * D * sizeof(bf16_t)));

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

  // Expert biases FP32 (sharded per-device by local experts)
  const int IM_ = model_config->intermediate_dim;
  // Count local experts per layer and set ctx.E_local to max across layers (for strides)
  int E_local_max = 0;
  for (int l = 0; l < L; ++l) {
    int cnt = 0;
    for (int e = 0; e < E_; ++e) if (ep_owner_of(l, e, E_) == device_id) cnt++;
    if (cnt > E_local_max) E_local_max = cnt;
  }
  ctx.E_local = E_local_max;

  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp1,
                      (size_t)L * E_ * (2 * IM_) * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&ctx.gpu_expert_bias.g_b_mlp2,
                      (size_t)L * E_ * H_ * sizeof(bf16_t)));

  for (int l = 0; l < L; ++l) {
    int lid = 0;
    for (int e = 0; e < E_; ++e) {
      if (ep_owner_of(l, e, E_) != device_id) continue;
      // copy biases for expert e at layer l to local slot lid
      const float* src_b1 = w->b_mlp1 + ((size_t)l * E_ + (size_t)e) * (size_t)(2 * IM_);
      bf16_t* dst_b1 = ctx.gpu_expert_bias.g_b_mlp1 + ((size_t)l * ctx.E_local + (size_t)lid) * (size_t)(2 * IM_);
      copy_fp32_to_bf16_device(src_b1, (size_t)(2 * IM_), dst_b1, n_streams, chunk_bytes);

      const float* src_b2 = w->b_mlp2 + ((size_t)l * E_ + (size_t)e) * (size_t)H_;
      bf16_t* dst_b2 = ctx.gpu_expert_bias.g_b_mlp2 + ((size_t)l * ctx.E_local + (size_t)lid) * (size_t)H_;
      copy_fp32_to_bf16_device(src_b2, (size_t)H_, dst_b2, n_streams, chunk_bytes);
      lid++;
    }
    // Zero-fill any unused local slots to keep indices safe
    for (; lid < ctx.E_local; ++lid) {
      HIP_CHECK(hipMemset(ctx.gpu_expert_bias.g_b_mlp1 + ((size_t)l * ctx.E_local + (size_t)lid) * (size_t)(2 * IM_), 0, (size_t)(2 * IM_) * sizeof(bf16_t)));
      HIP_CHECK(hipMemset(ctx.gpu_expert_bias.g_b_mlp2 + ((size_t)l * ctx.E_local + (size_t)lid) * (size_t)H_, 0, (size_t)H_ * sizeof(bf16_t)));
    }
  }
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
                      (size_t)L * ctx.E_local * mlp1_stride * sizeof(bf16_t)));
  packed_matrix.resize(mlp1_stride);
  for (int l = 0; l < L; ++l) {
    int lid = 0;
    for (int e = 0; e < E_; ++e) {
      if (ep_owner_of(l, e, E_) != device_id) continue;
      const size_t src_off = ((size_t)l * E_ + (size_t)e) * (size_t)(2 * IM_) * (size_t)H_;
      const float *matrix_src = w->w_mlp1 + src_off;
      pack_fp32_to_bf16_matmul(matrix_src, 2 * IM_, H_, packed_matrix.data());
      const size_t dst_index = ((size_t)l * ctx.E_local + (size_t)lid) * mlp1_stride;
      HIP_CHECK(hipMemcpy(ctx.gpu_weights_bf16.d_w_mlp1_bf16 + dst_index,
                          packed_matrix.data(), mlp1_stride * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
      lid++;
    }
  }

  const size_t mlp2_stride = matmul_packed_elems(H_, IM_);
  ctx.stride_w_mlp2_bf16 = mlp2_stride;
  HIP_CHECK(hipMalloc(&ctx.gpu_weights_bf16.d_w_mlp2_bf16,
                      (size_t)L * ctx.E_local * mlp2_stride * sizeof(bf16_t)));
  packed_matrix.resize(mlp2_stride);
  for (int l = 0; l < L; ++l) {
    int lid = 0;
    for (int e = 0; e < E_; ++e) {
      if (ep_owner_of(l, e, E_) != device_id) continue;
      const size_t src_off = ((size_t)l * E_ + (size_t)e) * (size_t)H_ * (size_t)IM_;
      const float *matrix_src = w->w_mlp2 + src_off;
      pack_fp32_to_bf16_matmul(matrix_src, H_, IM_, packed_matrix.data());
      const size_t dst_index = ((size_t)l * ctx.E_local + (size_t)lid) * mlp2_stride;
      HIP_CHECK(hipMemcpy(ctx.gpu_weights_bf16.d_w_mlp2_bf16 + dst_index,
                          packed_matrix.data(), mlp2_stride * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
      lid++;
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

  // Initialize KV cache with optimal sequence length during warmup
  int seq_hint = model_config->seq_len;
  ensure_kv_cache_capacity(ctx, seq_hint);

  debug_print_gpu_memory("after KV cache allocation (model fully loaded)", device_id);

  // Create dedicated streams
  HIP_CHECK(hipStreamCreateWithFlags(&ctx.compute_stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&ctx.pack_stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&ctx.mlp_stream, hipStreamNonBlocking));
  ctx.comm_streams = (hipStream_t*)malloc(sizeof(hipStream_t) * g_num_devices);
  for (int i = 0; i < g_num_devices; ++i) {
    HIP_CHECK(hipStreamCreateWithFlags(&ctx.comm_streams[i], hipStreamNonBlocking));
  }

  if (g_num_devices > 1) {
    const int K = model_config->experts_per_token;
    const size_t B_owner_peer_cap = (size_t)MAX_BATCH_SIZE;

    // Tạo mảng host chứa device pointers nếu chưa có
    ensure_host_ptr_array(&ctx.send_x_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.send_pos_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.send_topk_v_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.send_assignment_batches_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.send_assignment_slots_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.send_expert_offsets_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_owner_B_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_b2local_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_local2b_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_expert_counts_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_expert_writes_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_expert_offsets_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_pos_peer, g_num_devices);
    ensure_host_ptr_array(&ctx.d_topk_v_peer, g_num_devices);

    ensure_host_ptr_array(&ctx.recv_x_from_home, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_pos_from_home, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_topk_v_from_home, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_assignment_batches, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_assignment_slots, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_expert_offsets, g_num_devices);
    ensure_host_ptr_array(&ctx.partial_owner_per_home, g_num_devices);
    ensure_host_ptr_array(&ctx.recv_partial_home, g_num_devices);
    ensure_host_ptr_array(&ctx.gate_up_owner_per_home, g_num_devices);

    auto hip_alloc = [&](void **ptr, size_t bytes) {
      HIP_CHECK(hipMalloc(ptr, bytes));
    };

    for (int peer = 0; peer < g_num_devices; ++peer) {
      // E_local lớn nhất của 'peer' này trên mọi layer (để sizing offsets/counts)
      int E_local_max = 0;
      for (int l = 0; l < L; ++l) {
        int v = g_E_local_per_owner_layer[(size_t)peer * (size_t)L + (size_t)l];
        if (v > E_local_max) E_local_max = v;
      }

      // HOME → OWNER
      hip_alloc((void**)&ctx.send_x_peer[peer],                  (size_t)B * (size_t)H * sizeof(bf16_t));
      hip_alloc((void**)&ctx.send_pos_peer[peer],                (size_t)B * sizeof(int));
      hip_alloc((void**)&ctx.send_topk_v_peer[peer],             (size_t)B * (size_t)K * sizeof(float));
      hip_alloc((void**)&ctx.send_assignment_batches_peer[peer], (size_t)B * (size_t)K * sizeof(uint16_t));
      hip_alloc((void**)&ctx.send_assignment_slots_peer[peer],   (size_t)B * (size_t)K * sizeof(uint8_t));
      hip_alloc((void**)&ctx.send_expert_offsets_peer[peer],     (size_t)(E_local_max + 1) * sizeof(int));
      hip_alloc((void**)&ctx.d_owner_B_peer[peer],               sizeof(int));
      hip_alloc((void**)&ctx.d_b2local_peer[peer],               (size_t)B * sizeof(int));
      hip_alloc((void**)&ctx.d_local2b_peer[peer],               (size_t)B * sizeof(int));
      hip_alloc((void**)&ctx.d_expert_counts_peer[peer],         (size_t)E_local_max * sizeof(int));
      hip_alloc((void**)&ctx.d_expert_writes_peer[peer],         (size_t)E_local_max * sizeof(int));
      hip_alloc((void**)&ctx.d_expert_offsets_peer[peer],        (size_t)(E_local_max + 1) * sizeof(int));
      hip_alloc((void**)&ctx.d_pos_peer[peer],                   (size_t)B * sizeof(int));
      hip_alloc((void**)&ctx.d_topk_v_peer[peer],                (size_t)B * (size_t)K * sizeof(float));

      // OWNER (this ctx) nhận từ HOME=peer
      hip_alloc((void**)&ctx.recv_x_from_home[peer],             B_owner_peer_cap * (size_t)H * sizeof(bf16_t));
      hip_alloc((void**)&ctx.recv_pos_from_home[peer],           B_owner_peer_cap * sizeof(int));
      hip_alloc((void**)&ctx.recv_topk_v_from_home[peer],        B_owner_peer_cap * (size_t)K * sizeof(float));
      hip_alloc((void**)&ctx.recv_assignment_batches[peer],      B_owner_peer_cap * (size_t)K * sizeof(uint16_t));
      hip_alloc((void**)&ctx.recv_assignment_slots[peer],        B_owner_peer_cap * (size_t)K * sizeof(uint8_t));
      hip_alloc((void**)&ctx.recv_expert_offsets[peer],          (size_t)(E_local_max + 1) * sizeof(int));
      hip_alloc((void**)&ctx.partial_owner_per_home[peer],       B_owner_peer_cap * (size_t)H * sizeof(float));
      hip_alloc((void**)&ctx.recv_partial_home[peer],            B_owner_peer_cap * (size_t)H * sizeof(float));
      hip_alloc((void**)&ctx.gate_up_owner_per_home[peer],       (size_t)K * B_owner_peer_cap * (size_t)IM * sizeof(bf16_t));
    }
  }
}

static void warm_up_ep(Transformer *transformer) {
  for (int i = 0; i < g_num_devices; ++i) {
    for (int j = 0; j < g_num_devices; ++j) {
      if (i != j) {
        int can = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
        if (can) { 
          HIP_CHECK(hipSetDevice(i)); 
          hipError_t e = hipDeviceEnablePeerAccess(j, 0);
          if (e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
        }
      }
    }
  }

  const int L = model_config->n_layers;
  const int E = model_config->n_experts;
  g_ep_owner_map.resize((size_t)L * (size_t)E);
  for (int l = 0; l < L; ++l) {
    for (int e = 0; e < E; ++e) {
      g_ep_owner_map[(size_t)l * (size_t)E + (size_t)e] = EP_OWNER_RULE(e, g_num_devices);
    }
  }

  // Build e2lid for all owners per layer and record E_local per owner per layer
  g_e2lid_allowners.assign((size_t)g_num_devices * (size_t)L * (size_t)E, -1);
  g_E_local_per_owner_layer.assign((size_t)g_num_devices * (size_t)L, 0);
  for (int o = 0; o < g_num_devices; ++o) {
    for (int l = 0; l < L; ++l) {
      int lid = 0;
      for (int e = 0; e < E; ++e) {
        if (g_ep_owner_map[(size_t)l * (size_t)E + (size_t)e] == o) {
          g_e2lid_allowners[((size_t)o * L + (size_t)l) * (size_t)E + (size_t)e] = lid++;
        }
      }
      g_E_local_per_owner_layer[(size_t)o * (size_t)L + (size_t)l] = lid;
    }
  }

  // init all devices in parallel
  #pragma omp parallel for num_threads(g_num_devices)
  for (int i = 0; i < g_num_devices; ++i) {
    init_device_context_ep(g_devices[i], i, transformer);
    // Upload routing maps
    HIP_CHECK(hipSetDevice(i));
    DeviceContext &ctx = g_devices[i];
    const size_t map_elems = (size_t)g_num_devices * (size_t)L * (size_t)E;
    const size_t map_bytes = map_elems * sizeof(int);
    HIP_CHECK(hipMalloc(&ctx.d_e2lid_allowners, map_bytes));
    HIP_CHECK(hipMemcpy(ctx.d_e2lid_allowners, g_e2lid_allowners.data(), map_bytes, hipMemcpyHostToDevice));
    const size_t eloc_elems = (size_t)g_num_devices * (size_t)L;
    const size_t eloc_bytes = eloc_elems * sizeof(int);
    HIP_CHECK(hipMalloc(&ctx.d_E_local_per_owner_layer, eloc_bytes));
    HIP_CHECK(hipMemcpy(ctx.d_E_local_per_owner_layer, g_E_local_per_owner_layer.data(), eloc_bytes, hipMemcpyHostToDevice));
  }
}

// Cleanup device context
static void cleanup_device_context_ep(DeviceContext &ctx) {
  if (ctx.compute_stream) HIP_CHECK(hipStreamDestroy(ctx.compute_stream));
  if (ctx.pack_stream) HIP_CHECK(hipStreamDestroy(ctx.pack_stream));
  if (ctx.mlp_stream) HIP_CHECK(hipStreamDestroy(ctx.mlp_stream));
  if (ctx.comm_streams) {
    for (int i = 0; i < g_num_devices; ++i) HIP_CHECK(hipStreamDestroy(ctx.comm_streams[i]));
    free(ctx.comm_streams);
    ctx.comm_streams = nullptr;
  }

  // Free routing uploads and ring buffers
  if (ctx.d_e2lid_allowners) HIP_CHECK(hipFree(ctx.d_e2lid_allowners));
  if (ctx.d_E_local_per_owner_layer) HIP_CHECK(hipFree(ctx.d_E_local_per_owner_layer));

  auto free_per_peer_ptrs = [&](auto **&arr){
    if (!arr) return;
    for (int i = 0; i < g_num_devices; ++i) {
      if (arr[i]) HIP_CHECK(hipFree(arr[i]));
    }
    free(arr); arr = nullptr;
  };
  auto free_per_peer_host = [&](auto *&arr){ if (arr) { free(arr); arr = nullptr; } };
  free_per_peer_ptrs(ctx.send_x_peer);
  free_per_peer_ptrs(ctx.send_pos_peer);
  free_per_peer_ptrs(ctx.send_topk_v_peer);
  free_per_peer_ptrs(ctx.send_assignment_batches_peer);
  free_per_peer_ptrs(ctx.send_assignment_slots_peer);
  free_per_peer_ptrs(ctx.send_expert_offsets_peer);
  free_per_peer_ptrs(ctx.d_owner_B_peer);
  free_per_peer_ptrs(ctx.recv_x_from_home);
  free_per_peer_ptrs(ctx.recv_pos_from_home);
  free_per_peer_ptrs(ctx.recv_topk_v_from_home);
  free_per_peer_ptrs(ctx.recv_assignment_batches);
  free_per_peer_ptrs(ctx.recv_assignment_slots);
  free_per_peer_ptrs(ctx.recv_expert_offsets);
  free_per_peer_ptrs(ctx.partial_owner_per_home);
  free_per_peer_ptrs(ctx.recv_partial_home);
  free_per_peer_ptrs(ctx.gate_up_owner_per_home);
  free_per_peer_ptrs(ctx.d_b2local_peer);
  free_per_peer_ptrs(ctx.d_local2b_peer);
  free_per_peer_ptrs(ctx.d_expert_counts_peer);
  free_per_peer_ptrs(ctx.d_expert_offsets_peer);
  free_per_peer_ptrs(ctx.d_expert_writes_peer);
  free_per_peer_ptrs(ctx.d_pos_peer);
  free_per_peer_ptrs(ctx.d_topk_v_peer);
}

static void gpu_forward_device_batch_ep(DeviceContext &ctx, const float swiglu_limit, const int H, const int IM, const int E, const int L, const int K, int batch_size, int layer) {
  // For each owner, count -> scan -> pack -> bulk copy
  dim3 blockB(256, 1, 1);
  dim3 gridB((batch_size + 255) / 256, 1, 1);
  for (int owner = 0; owner < g_num_devices; ++owner) {
    // Get E_local for this owner at layer l
    const int E_local_layer = g_E_local_per_owner_layer[(size_t)owner * (size_t)L + (size_t)layer];
    if (E_local_layer == 0) continue;
    // Pointers to home-side scratch/buffers
    int *d_counts = ctx.d_expert_counts_peer[owner];
    int *d_offsets = ctx.send_expert_offsets_peer[owner];
    int *d_writes = ctx.d_expert_writes_peer[owner];
    int *d_b2local = ctx.d_b2local_peer[owner];
    int *d_local2b = ctx.d_local2b_peer[owner];
    int *d_owner_B = ctx.d_owner_B_peer[owner];

    // Reset buffers
    HIP_CHECK(hipMemsetAsync(d_counts, 0, (size_t)E_local_layer * sizeof(int), ctx.pack_stream));
    HIP_CHECK(hipMemsetAsync(d_writes, 0, (size_t)E_local_layer * sizeof(int), ctx.pack_stream));
    HIP_CHECK(hipMemsetAsync(d_b2local, 0xFF, (size_t)batch_size * sizeof(int), ctx.pack_stream)); // -1
    HIP_CHECK(hipMemsetAsync(d_owner_B, 0, sizeof(int), ctx.pack_stream));

    // owner-specific e2lid map base on device
    const int *d_e2lid_owner_l = ctx.d_e2lid_allowners + (((size_t)owner * L + (size_t)layer) * (size_t)E);

    {
      // route_count
      PROFILE_GPU_SCOPE("route_count_owner_kernel", 0);
      route_count_owner_kernel<<<gridB, blockB, 0, ctx.pack_stream>>>(
          d_counts,
          ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_pos,
          d_e2lid_owner_l,
          batch_size,
          K,
          E);
    }
    {
      PROFILE_GPU_SCOPE("exclusive_scan_small_kernel", 0);
      // exclusive scan counts -> offsets
      const int shared_scan = std::max(32, E_local_layer) * (int)sizeof(int);
      exclusive_scan_small_kernel<<<1, E_local_layer, shared_scan, ctx.pack_stream>>>(
          d_counts,
          d_offsets,
          E_local_layer);
    }

    // Total assignments = offsets[E_local]
    int h_total_assign = 0;
    HIP_CHECK(hipMemcpyAsync(&h_total_assign, d_offsets + E_local_layer, sizeof(int), hipMemcpyDeviceToHost, ctx.pack_stream));
    HIP_CHECK(hipStreamSynchronize(ctx.pack_stream));
    if (h_total_assign == 0) continue;

    {
      PROFILE_GPU_SCOPE("route_pack_owner_kernel", 0);
      // route_pack -> assignment arrays and b2local/local2b
      route_pack_owner_kernel<<<gridB, blockB, 0, ctx.pack_stream>>>(
          d_b2local,
          d_local2b,
          d_owner_B,
          d_offsets,
          d_writes,
          ctx.send_assignment_batches_peer[owner],
          ctx.send_assignment_slots_peer[owner],
          ctx.gpu_activations.d_topk_i,
          ctx.gpu_activations.d_pos,
          d_e2lid_owner_l,
          batch_size,
          K,
          E);
    }

    // Read B_local
    int h_B_local = 0;
    HIP_CHECK(hipMemcpyAsync(&h_B_local, d_owner_B, sizeof(int), hipMemcpyDeviceToHost, ctx.pack_stream));
    HIP_CHECK(hipStreamSynchronize(ctx.pack_stream));
    if (h_B_local == 0) continue;

    // Pack rows and meta to contiguous send buffers
    {
      PROFILE_GPU_SCOPE("pack_rows_owner_kernel", 0);
      dim3 gridPack((H + BLOCK_SIZE - 1) / BLOCK_SIZE, h_B_local, 1);
      dim3 blockH(BLOCK_SIZE, 1, 1);
      pack_rows_owner_kernel<<<gridPack, blockH, 0, ctx.pack_stream>>>(
          ctx.send_x_peer[owner],
          ctx.gpu_activations.d_t,
          d_b2local,
          batch_size,
          H);
    }
    {
      PROFILE_GPU_SCOPE("pack_meta_owner_kernel", 0);
      dim3 gridMeta((batch_size + 255) / 256, 1, 1);
      pack_meta_owner_kernel<<<gridMeta, 256, 0, ctx.pack_stream>>>(
          ctx.send_pos_peer[owner],
          ctx.send_topk_v_peer[owner],
          d_b2local,
          ctx.gpu_activations.d_pos,
          ctx.gpu_activations.d_topk_v,
          batch_size,
          K);
    }

    // Ensure pack completes before starting P2P copies.
    // Note: cross-device stream-wait is not supported; use CPU sync on the pack event.
    hipEvent_t pack_done_evt;
    HIP_CHECK(hipEventCreateWithFlags(&pack_done_evt, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(pack_done_evt, ctx.pack_stream));
    HIP_CHECK(hipEventSynchronize(pack_done_evt));
    HIP_CHECK(hipEventDestroy(pack_done_evt));

    // Enqueue copies on the DESTINATION (owner) device's comm stream for this home.
    HIP_CHECK(hipSetDevice(owner));
    hipStream_t ocomm = g_devices[owner].comm_streams[ctx.device_id];
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_x_from_home[ctx.device_id], owner,
              ctx.send_x_peer[owner], ctx.device_id,
              (size_t)h_B_local * H * sizeof(bf16_t), ocomm));
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_pos_from_home[ctx.device_id], owner,
              ctx.send_pos_peer[owner], ctx.device_id,
              (size_t)h_B_local * sizeof(int), ocomm));
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_topk_v_from_home[ctx.device_id], owner,
              ctx.send_topk_v_peer[owner], ctx.device_id,
              (size_t)h_B_local * K * sizeof(float), ocomm));
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_assignment_batches[ctx.device_id], owner,
              ctx.send_assignment_batches_peer[owner], ctx.device_id,
              (size_t)h_total_assign * sizeof(uint16_t), ocomm));
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_assignment_slots[ctx.device_id], owner,
              ctx.send_assignment_slots_peer[owner], ctx.device_id,
              (size_t)h_total_assign * sizeof(uint8_t), ocomm));
    HIP_CHECK(hipMemcpyPeerAsync(g_devices[owner].recv_expert_offsets[ctx.device_id], owner,
              d_offsets, ctx.device_id,
              (size_t)(E_local_layer + 1) * sizeof(int), ocomm));

    // Ensure copies have completed on the owner before launching owner-side MLP.
    HIP_CHECK(hipStreamSynchronize(ocomm));

    // Launch MLP on owner after the copies complete (enqueue on owner's mlp_stream)
    {
      PROFILE_GPU_SCOPE("mlp1_fused_gemm_kernel", 0);
      // For simplicity, reuse counts buffer by copying from offsets diff
      int *d_owner_offsets = g_devices[owner].recv_expert_offsets[ctx.device_id];
      // Clear owner-side buffers before compute to avoid stale data
      HIP_CHECK(hipMemsetAsync(g_devices[owner].partial_owner_per_home[ctx.device_id], 0, (size_t)h_B_local * H * sizeof(float), g_devices[owner].mlp_stream));
      // Optional: clear gate_up buffer region (mlp1 will overwrite assigned entries)
      HIP_CHECK(hipMemsetAsync(g_devices[owner].gate_up_owner_per_home[ctx.device_id], 0, (size_t)K * (size_t)h_B_local * (size_t)IM * sizeof(bf16_t), g_devices[owner].mlp_stream));

      const int max_tiles = (h_B_local + MATMUL_MLP1_BLOCK_ROWS_120B - 1) / MATMUL_MLP1_BLOCK_ROWS_120B;
      dim3 block_mlp1(WF_SIZE, MATMUL_MLP1_WAVES_PER_BLOCK_120B, 1);
      dim3 grid_mlp1((2 * IM + MATMUL_MLP1_BLOCK_COLS_120B - 1) / MATMUL_MLP1_BLOCK_COLS_120B, max_tiles, E_local_layer);
      mlp1_fused_gemm_kernel<<<grid_mlp1, block_mlp1, 0, g_devices[owner].mlp_stream>>>(
        g_devices[owner].gate_up_owner_per_home[ctx.device_id], 
        g_devices[owner].recv_x_from_home[ctx.device_id],
        g_devices[owner].gpu_weights_bf16.d_w_mlp1_bf16, 
        g_devices[owner].stride_w_mlp1_bf16,
        g_devices[owner].gpu_expert_bias.g_b_mlp1, 
        g_devices[owner].recv_assignment_batches[ctx.device_id],
        g_devices[owner].recv_assignment_slots[ctx.device_id], 
        d_owner_offsets, 
        layer, 
        g_devices[owner].E_local, 
        H, IM, swiglu_limit,
        h_B_local,
        g_devices[owner].recv_pos_from_home[ctx.device_id]);
    }
    {
      PROFILE_GPU_SCOPE("mlp2_bias_weighted_accum_gemm_kernel", 0);
      const int max_tiles = (h_B_local + MATMUL_MLP2_BLOCK_ROWS_120B - 1) / MATMUL_MLP2_BLOCK_ROWS_120B;
      dim3 block_mlp2(WF_SIZE, MATMUL_MLP2_WAVES_PER_BLOCK_120B, 1);
      dim3 grid_mlp2((H + MATMUL_MLP2_BLOCK_COLS_120B - 1) / MATMUL_MLP2_BLOCK_COLS_120B,
                    max_tiles, E_local_layer);
      mlp2_bias_weighted_accum_gemm_kernel<<<grid_mlp2, block_mlp2, 0, g_devices[owner].mlp_stream>>>(
        g_devices[owner].partial_owner_per_home[ctx.device_id], 
        g_devices[owner].gate_up_owner_per_home[ctx.device_id],
        g_devices[owner].gpu_weights_bf16.d_w_mlp2_bf16, 
        g_devices[owner].stride_w_mlp2_bf16,
        g_devices[owner].gpu_expert_bias.g_b_mlp2, 
        g_devices[owner].recv_assignment_batches[ctx.device_id],
        g_devices[owner].recv_assignment_slots[ctx.device_id], 
        g_devices[owner].recv_expert_offsets[ctx.device_id], 
        g_devices[owner].recv_topk_v_from_home[ctx.device_id], 
        layer,
        g_devices[owner].E_local, 
        IM, H, h_B_local, 
        g_devices[owner].recv_pos_from_home[ctx.device_id]);
    }
    HIP_CHECK(hipStreamSynchronize(g_devices[owner].mlp_stream));

    // Copy partials back to home on same comm stream and enqueue accumulate on home.comm_stream[owner]
    HIP_CHECK(hipSetDevice(ctx.device_id));
    HIP_CHECK(hipMemcpyPeerAsync(ctx.recv_partial_home[owner], ctx.device_id,
                                  g_devices[owner].partial_owner_per_home[ctx.device_id], owner,
                                  (size_t)h_B_local * H * sizeof(float), ctx.comm_streams[owner]));
    {
      PROFILE_GPU_SCOPE("accumulate_partials_kernel", 0);
      // Accumulate on home on comm stream to preserve order
      dim3 blockH(BLOCK_SIZE, 1, 1);
      dim3 gridAcc((H + BLOCK_SIZE - 1) / BLOCK_SIZE, h_B_local, 1);
      accumulate_partials_kernel<<<gridAcc, blockH, 0, ctx.comm_streams[owner]>>>(
          ctx.gpu_activations.d_e_agg,
          ctx.recv_partial_home[owner],
          ctx.d_local2b_peer[owner], // local2b gives batch_ids order by lb index
          H,
          h_B_local);
      HIP_CHECK(hipStreamSynchronize(ctx.comm_streams[owner]));
    }
  }
}
