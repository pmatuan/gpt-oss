#include "../profiler.h"
#include "getp_eval.cpp"
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef GETP_RUN
#define GETP_RUN

#define TK 128
#define TM 4
#define BLOCK_SIZE 256
#define LDS_PAD 8
#define WF_SIZE 64

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = call;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef rocwmma::bfloat16_t bf16_t;

static inline void debug_print_gpu_memory(const char *tag) {
  size_t free_b = 0, total_b = 0;
  hipError_t err = hipMemGetInfo(&free_b, &total_b);
  if (err != hipSuccess) {
    fprintf(stderr, "HIP hipMemGetInfo failed: %s\n", hipGetErrorString(err));
    return;
  }
  double free_gib = (double)free_b / (1024.0 * 1024.0 * 1024.0);
  double total_gib = (double)total_b / (1024.0 * 1024.0 * 1024.0);
  double used_gib = total_gib - free_gib;
  printf("[HIP] %s: HBM free %.2f GiB / total %.2f GiB (used %.2f GiB)\n", tag,
         free_gib, total_gib, used_gib);
  fflush(stdout);
}

inline dim3 get_gemv_grid_dim(int d) { return dim3((d + TM - 1) / TM, 1, 1); }

// ---------------- Global GPU Buffers ----------------
static float *d_x, *d_t, *d_tb, *d_tb2;
static float *d_router_score, *d_topk_v, *d_mlp1_out;
static int *d_topk_i;
static float *d_gate_up, *d_e_agg;
static float *d_qkv, *d_q, *d_k, *d_v;
static float *d_key_cache, *d_value_cache;
static float *d_att, *d_logits, *d_mask;
static float *d_cos_vals, *d_sin_vals;
static int *d_token2row;

// Small FP32 weights
static float *d_rms_attn_w, *d_rms_ffn_w;
static float *d_b_qkv, *d_b_o, *d_attn_sinks;
static float *d_w_router, *d_b_router;
static float *d_rms_out_w;

// Expert biases FP32
static float *g_b_mlp1 = nullptr;
static float *g_b_mlp2 = nullptr;

// Large BF16 weights
static bf16_t *d_token_embedding_table_bf16;
static bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
static bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
static bf16_t *d_out_bf16;

static Config *h_config = nullptr;

// ======================= Compute Kernels =======================

__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight,
                               int size) {
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    float v = x[i];
    sum += v * v;
  }
  #pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    sum += __shfl_down(sum, off, WF_SIZE);
  }
  __shared__ float warp_sums[BLOCK_SIZE / WF_SIZE];
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();

  float total = 0.f;
  if (tid < BLOCK_SIZE / WF_SIZE) total = warp_sums[tid];

  if (wid == 0) {
    float t = (tid < BLOCK_SIZE / WF_SIZE) ? total : 0.f;
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      t += __shfl_down(t, off, WF_SIZE);
    }
    if (lane == 0) warp_sums[0] = t;
  }
  __syncthreads();

  const float inv = rsqrtf(warp_sums[0] / (float)size + 1e-5f);
  for (int i = tid; i < size; i += blockDim.x) {
    o[i] = weight[i] * (x[i] * inv);
  }
}

__global__ void softmax_kernel(float *x, int size) {
  if (threadIdx.x == 0) {
    double max_val = (double)x[0];
    for (int i = 1; i < size; i++) max_val = fmax(max_val, (double)x[i]);
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      float ev = expf((float)((double)x[i] - max_val));
      x[i] = ev;
      sum += (double)ev;
    }
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < size; i++) x[i] = (float)((double)x[i] * inv_sum);
  }
}

__global__ void copy_embedding_bf16_row_kernel(float *dst, const bf16_t *src,
                                               int token, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    dst[i] = static_cast<float>(src[(size_t)token * hidden_dim + i]);
  }
}

__global__ void split_qkv_kernel(float *q, float *k, float *v, const float *qkv,
                                 int n_attn_heads, int n_kv_heads,
                                 int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int q_size = n_attn_heads * head_dim;
  int kv_size = n_kv_heads * head_dim;
  int total = q_size + 2 * kv_size;
  if (idx >= total) return;
  if (idx < q_size) q[idx] = qkv[idx];
  else if (idx < q_size + kv_size) k[idx - q_size] = qkv[idx];
  else v[idx - q_size - kv_size] = qkv[idx];
}

__global__ void apply_rotary_emb_kernel(float *x, const float *cosv,
                                        const float *sinv, int n_heads,
                                        int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim >> 1;
  if (h < n_heads && i < half) {
    float x1 = x[h * head_dim + i];
    float x2 = x[h * head_dim + half + i];
    float c = cosv[i], s = sinv[i];
    x[h * head_dim + i] = x1 * c - x2 * s;
    x[h * head_dim + half + i] = x2 * c + x1 * s;
  }
}

__global__ void compute_cos_sin_kernel(float *cosv, float *sinv, int pos,
                                       float rope_theta, int head_dim,
                                       float scaling_factor,
                                       float initial_context_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim >> 1;
  if (i >= d_half) return;
  float freq = powf(rope_theta, (float)(2 * i) / (float)head_dim);
  float inv_freq;
  float concentration = 1.0f;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(rope_theta);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(rope_theta);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    ramp = fmaxf(0.0f, fminf(1.0f, ramp));
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }
  float val = pos * inv_freq;
  cosv[i] = cosf(val) * concentration;
  sinv[i] = sinf(val) * concentration;
}

__global__ void add_bias_kernel(float *y, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) y[i] += b[i];
}

__global__ void residual_add_kernel(float *x, const float *residual, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) x[i] += residual[i];
}

__global__ void add_bias_residual_inplace_kernel(float *x, const float *y,
                                                 const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) x[i] = x[i] + (y[i] + b[i]);
}

// --------- Device-only TopK (kept, used as step C foundation) ----------
__global__ void topk_kernel_1token(float *topk_values, int *topk_indices,
                                   float *router_score, int num_experts,
                                   int experts_per_token) {
  extern __shared__ float smem[];
  float *scores = smem;
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  for (int i = tid; i < num_experts; i += blockDim.x) {
    scores[i] = router_score[i];
  }
  __syncthreads();

  __shared__ float warp_vals[BLOCK_SIZE / WF_SIZE];
  __shared__ int   warp_idxs[BLOCK_SIZE / WF_SIZE];

  for (int k = 0; k < experts_per_token; k++) {
    float local_best = -INFINITY;
    int   local_idx  = -1;
    for (int i = tid; i < num_experts; i += blockDim.x) {
      float v = scores[i];
      if (v > local_best) { local_best = v; local_idx = i; }
    }
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      float oVal = __shfl_down(local_best, off, WF_SIZE);
      int   oIdx = __shfl_down(local_idx,  off, WF_SIZE);
      if (oVal > local_best) { local_best = oVal; local_idx = oIdx; }
    }
    if (lane == 0) { warp_vals[wid] = local_best; warp_idxs[wid] = local_idx; }
    __syncthreads();

    if (wid == 0) {
      float bVal = (lane < BLOCK_SIZE / WF_SIZE) ? warp_vals[lane] : -INFINITY;
      int   bIdx = (lane < BLOCK_SIZE / WF_SIZE) ? warp_idxs[lane] : -1;
      #pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
        float oVal = __shfl_down(bVal, off, WF_SIZE);
        int   oIdx = __shfl_down(bIdx, off, WF_SIZE);
        if (oVal > bVal) { bVal = oVal; bIdx = oIdx; }
      }
      if (lane == 0) {
        topk_values[k] = bVal;
        topk_indices[k] = bIdx;
        if (bIdx >= 0) scores[bIdx] = -INFINITY;
      }
    }
    __syncthreads();
  }
}

// ---------------- GEMV kernels -----------------

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_kernel(float *__restrict__ y, const float *__restrict__ x,
                              const T *__restrict__ w, int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w + (size_t)row * n + k_base;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float x_val = lds_x[k];
      if constexpr (std::is_same_v<T, bf16_t>) {
        bf16_t w_val = w_row[k];
        acc = fmaf(static_cast<float>(w_val), x_val, acc);
      } else {
        float w_val = w_row[k];
        acc = fmaf(w_val, x_val, acc);
      }
    }
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) acc += __shfl_down(acc, off, WF_SIZE);
    if (lane == 0) acc_all += acc;
    __syncthreads();
  }
  if (lane == 0) y[row] = acc_all;
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_bias_kernel(float *__restrict__ y, const float *__restrict__ x,
                                   const T *__restrict__ w, const float *__restrict__ b,
                                   int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w + (size_t)row * n + k_base;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float x_val = lds_x[k];
      if constexpr (std::is_same_v<T, bf16_t>) {
        bf16_t w_val = w_row[k];
        acc = fmaf(static_cast<float>(w_val), x_val, acc);
      } else {
        float w_val = w_row[k];
        acc = fmaf(w_val, x_val, acc);
      }
    }
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) acc += __shfl_down(acc, off, WF_SIZE);
    if (lane == 0) acc_all += acc;
    __syncthreads();
  }
  if (lane == 0) y[row] = acc_all + b[row];
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void mlp1_fused_kernel(float *__restrict__ gate_up,
                                  const float *__restrict__ x,
                                  const T *__restrict__ w_mlp1_all,
                                  const float *__restrict__ b_mlp1_all,
                                  const int *__restrict__ topk_i,
                                  int k_index, int l_layer, int E,
                                  int H, int IM, float swiglu_limit) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int i = blockIdx.x * TM + wid; // output index in [0..IM)
  if (wid >= TM || i >= IM) return;

  // read expert id from device
  const int e = topk_i[k_index];

  float acc_g_all = 0.0f;
  float acc_u_all = 0.0f;

  // Offsets
  const size_t base_rows = ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
  const size_t off_gate = (base_rows + (size_t)(2 * i + 0)) * (size_t)H;
  const size_t off_up   = (base_rows + (size_t)(2 * i + 1)) * (size_t)H;

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);
    for (int k = lane; k < k_size; k += WF_SIZE) lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ wg = w_mlp1_all + off_gate + k_base;
    const T *__restrict__ wu = w_mlp1_all + off_up + k_base;

    float acc_g = 0.0f, acc_u = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float xv = lds_x[k];
      float wgk = std::is_same_v<T, bf16_t> ? (float)wg[k] : (float)((const float *)wg)[k];
      float wuk = std::is_same_v<T, bf16_t> ? (float)wu[k] : (float)((const float *)wu)[k];
      acc_g = fmaf(wgk, xv, acc_g);
      acc_u = fmaf(wuk, xv, acc_u);
    }
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc_g += __shfl_down(acc_g, off, WF_SIZE);
      acc_u += __shfl_down(acc_u, off, WF_SIZE);
    }
    if (lane == 0) { acc_g_all += acc_g; acc_u_all += acc_u; }
    __syncthreads();
  }

  if (lane == 0) {
    // Add bias
    const size_t bbase = ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
    float g = acc_g_all + b_mlp1_all[bbase + (size_t)(2 * i + 0)];
    float u = acc_u_all + b_mlp1_all[bbase + (size_t)(2 * i + 1)];

    // Clamp (match previous kernel)
    if (g > swiglu_limit) g = swiglu_limit;
    if (u > swiglu_limit) u = swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;

    // SiLU approx then * (u + 1)
    const float alpha = 1.702f;
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);

    gate_up[i] = g;
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void mlp2_bias_weighted_accum_kernel(float *__restrict__ e_agg,
                                                const float *__restrict__ gate_up,
                                                const T *__restrict__ w_mlp2_all,
                                                const float *__restrict__ b_mlp2_all,
                                                const int *__restrict__ topk_i,
                                                const float *__restrict__ topk_v,
                                                int k_index, int l_layer, int E,
                                                int IM, int H) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid; // output index in [0..H)
  if (wid >= TM || row >= H) return;

  const int e = topk_i[k_index];
  const float weight = topk_v[k_index];

  float acc_all = 0.0f;

  const size_t base = ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)H * (size_t)IM;
  const T *__restrict__ w_row_base = w_mlp2_all + (size_t)row * (size_t)IM + base;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);
    for (int k = lane; k < k_size; k += WF_SIZE) lds_x[k] = gate_up[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w_row_base + k_base;
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float x_val = lds_x[k];
      if constexpr (std::is_same_v<T, bf16_t>) {
        bf16_t w_val = w_row[k];
        acc = fmaf((float)w_val, x_val, acc);
      } else {
        float w_val = ((const float *)w_row)[k];
        acc = fmaf(w_val, x_val, acc);
      }
    }
    #pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) acc += __shfl_down(acc, off, WF_SIZE);
    if (lane == 0) acc_all += acc;
    __syncthreads();
  }

  if (lane == 0) {
    const size_t bbase = ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)H;
    float y = acc_all + b_mlp2_all[bbase + (size_t)row];
    e_agg[row] += y * weight;
  }
}

// =================== FP32->BF16 Async Host->Device ====================
static void copy_fp32_to_bf16_device(const float *h_src, size_t count,
                                     bf16_t *d_dst, int n_streams = 4,
                                     size_t chunk_bytes = 64ULL * 1024 * 1024) {
  if (count == 0) return;

  const size_t chunk_elems = chunk_bytes / sizeof(bf16_t);
  const size_t actual_chunk_elems = (chunk_elems > count) ? count : chunk_elems;

  hipStream_t *streams = nullptr;
  hipEvent_t *events = nullptr;
  bf16_t **pinned_chunks = nullptr;
  bool async_success = true;

  streams = (hipStream_t *)malloc(n_streams * sizeof(hipStream_t));
  if (!streams) async_success = false;
  if (async_success) {
    events = (hipEvent_t *)malloc(n_streams * sizeof(hipEvent_t));
    if (!events) async_success = false;
  }
  if (async_success) {
    pinned_chunks = (bf16_t **)malloc(n_streams * sizeof(bf16_t *));
    if (!pinned_chunks) async_success = false;
    else { for (int i = 0; i < n_streams; i++) pinned_chunks[i] = nullptr; }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
      if (err != hipSuccess) { async_success = false; break; }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipEventCreateWithFlags(&events[i], hipEventDisableTiming);
      if (err != hipSuccess) { async_success = false; break; }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipHostMalloc((void **)&pinned_chunks[i],
                                     actual_chunk_elems * sizeof(bf16_t));
      if (err != hipSuccess) { async_success = false; break; }
    }
  }

  if (!async_success) {
    if (pinned_chunks) {
      for (int i = 0; i < n_streams; i++) { if (pinned_chunks[i]) { (void)hipHostFree(pinned_chunks[i]); } }
      free(pinned_chunks);
    }
    if (events) {
      for (int i = 0; i < n_streams; i++) { (void)hipEventDestroy(events[i]); }
      free(events);
    }
    if (streams) {
      for (int i = 0; i < n_streams; i++) { (void)hipStreamDestroy(streams[i]); }
      free(streams);
    }

    const size_t SYNC_CHUNK_ELEMS = 8 * 1024 * 1024;
    bf16_t *h_chunk = nullptr;
    HIP_CHECK(hipHostMalloc((void **)&h_chunk, SYNC_CHUNK_ELEMS * sizeof(bf16_t)));
    size_t done = 0;
    while (done < count) {
      size_t todo = (count - done > SYNC_CHUNK_ELEMS) ? SYNC_CHUNK_ELEMS : (count - done);
      for (size_t i = 0; i < todo; ++i) h_chunk[i] = rocwmma::bfloat16_t(h_src[done + i]);
      HIP_CHECK(hipMemcpy(d_dst + done, h_chunk, todo * sizeof(bf16_t), hipMemcpyHostToDevice));
      done += todo;
    }
    HIP_CHECK(hipHostFree(h_chunk));
    return;
  }

  size_t done = 0;
  int stream_idx = 0;
  bool *buffer_ready = (bool *)calloc(n_streams, sizeof(bool));
  for (int i = 0; i < n_streams; i++) buffer_ready[i] = true;

  while (done < count) {
    size_t todo = (count - done > actual_chunk_elems) ? actual_chunk_elems : (count - done);
    if (!buffer_ready[stream_idx]) {
      HIP_CHECK(hipEventSynchronize(events[stream_idx]));
      buffer_ready[stream_idx] = true;
    }
    bf16_t *chunk = pinned_chunks[stream_idx];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < todo; ++i) chunk[i] = rocwmma::bfloat16_t(h_src[done + i]);

    HIP_CHECK(hipMemcpyAsync(d_dst + done, chunk, todo * sizeof(bf16_t),
                             hipMemcpyHostToDevice, streams[stream_idx]));
    HIP_CHECK(hipEventRecord(events[stream_idx], streams[stream_idx]));
    buffer_ready[stream_idx] = false;

    done += todo; stream_idx = (stream_idx + 1) % n_streams;
  }
  for (int i = 0; i < n_streams; i++) HIP_CHECK(hipEventSynchronize(events[i]));

  free(buffer_ready);
  for (int i = 0; i < n_streams; i++) {
    HIP_CHECK(hipHostFree(pinned_chunks[i]));
    HIP_CHECK(hipEventDestroy(events[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  free(pinned_chunks); free(events); free(streams);
}

template<int TILE_T>
__global__ void paged_attention_fused_kernel(
    float* __restrict__ tb,
    const float* __restrict__ q,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ token2row,
    const float* __restrict__ attn_sinks_layer,
    const float* __restrict__ mask,
    int Hq, int Hk, int D, int KV,
    int S, int pos) {

  extern __shared__ unsigned char smem_raw[];
  double* s_logits = reinterpret_cast<double*>(smem_raw);
  double* s_warp   = s_logits + TILE_T;
  float*  s_q      = reinterpret_cast<float*>(s_warp + blockDim.x);

  const int h   = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= Hq) return;

  const int kv_mul = Hq / Hk;
  const int base   = (h / kv_mul) * D;
  const double scale = 1.0 / sqrt((double)D);

  const float* __restrict__ qh = q + (size_t)h * D;
  const float* __restrict__ K  = key_cache;
  const float* __restrict__ V  = value_cache;

  const int T_real = pos + 1;

  for (int i = tid; i < D; i += blockDim.x) s_q[i] = qh[i];
  __syncthreads();

  double m_run = -INFINITY;
  double s_run = 0.0;

  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int warp_count = (blockDim.x + WF_SIZE - 1) / WF_SIZE;

  double num_acc = 0.0;

  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    double tile_max = -INFINITY;

    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      int row = token2row[t];
      if (row < 0) row = 0; else if (row >= S) row = S - 1;

      float partf = 0.0f;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)row * KV + base + i];
        partf = fmaf(s_q[i], k, partf);
      }
      #pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1) partf += __shfl_down(partf, off, WF_SIZE);
      if (lane == 0) s_warp[wid] = (double)partf;
      __syncthreads();

      if (wid == 0) {
        double sum = (lane < warp_count) ? s_warp[lane] : 0.0;
        #pragma unroll
        for (int off = WF_SIZE >> 1; off > 0; off >>= 1) sum += __shfl_down(sum, off, WF_SIZE);
        if (lane == 0) {
          double logit = sum * scale;
          if (mask) logit += (double)mask[(size_t)pos * S + t];
          if (!isfinite(logit)) logit = -1e30;
          s_logits[tt] = logit;
          tile_max = fmax(tile_max, logit);
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      double m_new = fmax(m_run, tile_max);
      double scale_old = isfinite(m_run) ? exp(m_run - m_new) : 0.0;
      s_run *= scale_old;
      s_warp[0] = m_new;
      s_warp[1] = scale_old;
      s_warp[2] = exp(tile_max - m_new);
    }
    __syncthreads();
    const double m_new = s_warp[0];
    const double num_scale = s_warp[1];
    const double factor = s_warp[2];
    num_acc *= num_scale;

    if (tid < D) {
      for (int tt = 0; tt < tile; ++tt) {
        if (tid == 0) {
          double w_local = exp(s_logits[tt] - (double)tile_max);
          s_logits[tt] = w_local * factor;
          s_run += s_logits[tt];
        }
        __syncthreads();
        const int t = t0 + tt;
        int row = token2row[t];
        if (row < 0) row = 0; else if (row >= S) row = S - 1;
        const float v = V[(size_t)row * KV + base + tid];
        num_acc += s_logits[tt] * (double)v;
        __syncthreads();
      }
    } else {
      for (int tt = 0; tt < tile; ++tt) {
        if (tid == 0) {
          double w_local = exp(s_logits[tt] - (double)tile_max);
          s_logits[tt] = w_local * factor;
          s_run += s_logits[tt];
        }
        __syncthreads();
      }
    }
    __syncthreads();
    if (tid == 0) m_run = m_new;
    __syncthreads();
  }

  const double sinkVal = (double)attn_sinks_layer[h];
  if (tid == 0) {
    double m_new = fmax(m_run, sinkVal);
    double scale_old = isfinite(m_run) ? exp(m_run - m_new) : 0.0;
    s_run = s_run * scale_old + exp(sinkVal - m_new);
    s_warp[0] = 1.0 / s_run;
    s_warp[1] = scale_old;
  }
  __syncthreads();
  double inv = s_warp[0];
  if (!isfinite(inv)) inv = 0.0;
  const double final_scale = s_warp[1];
  num_acc *= final_scale;
  if (tid < D) tb[(size_t)h * D + tid] = (float)(num_acc * inv);
}

// ======================== Init / Finish ========================
void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  h_config = &transformer->config;
  HIP_CHECK(hipSetDevice(0));

  const int H = h_config->hidden_dim;
  const int V = h_config->vocab_size;
  const int L = h_config->n_layers;
  const int E = h_config->n_experts;
  const int D = h_config->head_dim;
  const int Hq = h_config->n_attn_heads;
  const int Hk = h_config->n_kv_heads;
  const int KV = D * Hk;
  const int S = h_config->seq_len;
  const int IM = h_config->intermediate_dim;

  debug_print_gpu_memory("before allocations");

  // Activations
  HIP_CHECK(hipMalloc(&d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb, D * Hq * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb2, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_v, h_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_i, h_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&d_mlp1_out, 2 * IM * sizeof(float))); // (kept for debug options)
  HIP_CHECK(hipMalloc(&d_gate_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_e_agg, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv, (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, Hq * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, Hk * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, Hk * D * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_key_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_value_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, V * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sin_vals, (D / 2) * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_token2row, S * sizeof(int)));
  {
    int *h_token2row = (int *)malloc(S * sizeof(int));
    for (int i = 0; i < S; ++i) h_token2row[i] = i;
    HIP_CHECK(hipMemcpy(d_token2row, h_token2row, S * sizeof(int), hipMemcpyHostToDevice));
    free(h_token2row);
  }

  if (h_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] = (i - j >= h_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(d_mask, h_mask, S * S * sizeof(float), hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations");

  // Weights (small FP32)
  TransformerWeights *w = &transformer->weights;

  HIP_CHECK(hipMalloc(&d_rms_attn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_attn_w, w->rms_attn_w, L * H * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_ffn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_ffn_w, w->rms_ffn_w, L * H * sizeof(float), hipMemcpyHostToDevice));

  const int QKV_D = D * (Hq + 2 * Hk);
  HIP_CHECK(hipMalloc(&d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_qkv, w->b_qkv, L * QKV_D * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_o, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_o, w->b_o, L * H * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_attn_sinks, L * Hq * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_attn_sinks, w->attn_sinks, L * Hq * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_w_router, L * H * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w_router, w->w_router, L * H * E * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_router, L * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_router, w->b_router, L * E * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_out_w, H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_out_w, w->rms_out_w, H * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights");

  // Expert biases FP32
  HIP_CHECK(hipMalloc(&g_b_mlp1, (size_t)L * E * (2 * IM) * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp1, w->b_mlp1, (size_t)L * E * (2 * IM) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&g_b_mlp2, (size_t)L * E * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp2, w->b_mlp2, (size_t)L * E * H * sizeof(float), hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases");

  // Large BF16 weights
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024;

  HIP_CHECK(hipMalloc(&d_token_embedding_table_bf16, (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V * H, d_token_embedding_table_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&d_w_qkv_bf16, (size_t)L * QKV_D * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H, d_w_qkv_bf16, n_streams, chunk_bytes);

  const int O_N = D * Hq;
  HIP_CHECK(hipMalloc(&d_w_o_bf16, (size_t)L * H * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H * O_N, d_w_o_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&d_w_mlp1_bf16, (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E * (2 * IM) * H, d_w_mlp1_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&d_w_mlp2_bf16, (size_t)L * E * H * IM * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E * H * IM, d_w_mlp2_bf16, n_streams, chunk_bytes);

  HIP_CHECK(hipMalloc(&d_out_bf16, (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V * H, d_out_bf16, n_streams, chunk_bytes);

  debug_print_gpu_memory("after large BF16 weights (model loaded)");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_t));
  HIP_CHECK(hipFree(d_tb));
  HIP_CHECK(hipFree(d_tb2));
  HIP_CHECK(hipFree(d_router_score));
  HIP_CHECK(hipFree(d_topk_v));
  HIP_CHECK(hipFree(d_topk_i));
  HIP_CHECK(hipFree(d_mlp1_out));
  HIP_CHECK(hipFree(d_gate_up));
  HIP_CHECK(hipFree(d_e_agg));
  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_key_cache));
  HIP_CHECK(hipFree(d_value_cache));
  HIP_CHECK(hipFree(d_att));
  HIP_CHECK(hipFree(d_logits));
  HIP_CHECK(hipFree(d_cos_vals));
  HIP_CHECK(hipFree(d_sin_vals));
  if (d_mask) HIP_CHECK(hipFree(d_mask));
  if (d_token2row) HIP_CHECK(hipFree(d_token2row));

  HIP_CHECK(hipFree(d_rms_attn_w));
  HIP_CHECK(hipFree(d_rms_ffn_w));
  HIP_CHECK(hipFree(d_b_qkv));
  HIP_CHECK(hipFree(d_b_o));
  HIP_CHECK(hipFree(d_attn_sinks));
  HIP_CHECK(hipFree(d_w_router));
  HIP_CHECK(hipFree(d_b_router));
  HIP_CHECK(hipFree(d_rms_out_w));

  if (g_b_mlp1) HIP_CHECK(hipFree(g_b_mlp1));
  if (g_b_mlp2) HIP_CHECK(hipFree(g_b_mlp2));

  HIP_CHECK(hipFree(d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(d_w_qkv_bf16));
  HIP_CHECK(hipFree(d_w_o_bf16));
  HIP_CHECK(hipFree(d_w_mlp1_bf16));
  HIP_CHECK(hipFree(d_w_mlp2_bf16));
  HIP_CHECK(hipFree(d_out_bf16));
}

// ============================ Forward ============================
float *gpu_forward(Transformer *transformer, int token, int pos) {
  PROFILE_FUNCTION();
  const Config *p = h_config;
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
                        copy_embedding_bf16_row_kernel<<<gridH, block>>>(d_x, d_token_embedding_table_bf16, token, H));
  HIP_CHECK(hipStreamSynchronize(0));

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attn)
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                          rmsnorm_kernel<<<1, BLOCK_SIZE>>>(d_t, d_x, d_rms_attn_w + l * H, H));
    HIP_CHECK(hipStreamSynchronize(0));

    // (A) QKV = W_qkv@t + b_qkv  (fused)
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV = get_gemv_grid_dim(QKV_D);
    PROFILE_KERNEL_LAUNCH("matmul_bias_kernel",
                          matmul_bias_kernel<bf16_t><<<gridQKV, block>>>(d_qkv, d_t,
                            d_w_qkv_bf16 + (size_t)l * QKV_D * H,
                            d_b_qkv + l * QKV_D, H, QKV_D));
    HIP_CHECK(hipStreamSynchronize(0));

    PROFILE_KERNEL_LAUNCH("split_qkv_kernel",
                          split_qkv_kernel<<<gridQKV, block>>>(d_q, d_k, d_v, d_qkv, Hq, Hk, D));
    HIP_CHECK(hipStreamSynchronize(0));

    int loff = l * S * KV;
    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(d_value_cache + loff + pos * KV, d_v, KV * sizeof(float), hipMemcpyDeviceToDevice));

    dim3 gridR(((D / 2) + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH("compute_cos_sin_kernel",
                          compute_cos_sin_kernel<<<gridR, block>>>(d_cos_vals, d_sin_vals, pos, p->rope_theta, D,
                                             p->rope_scaling_factor, p->initial_context_length));
    HIP_CHECK(hipStreamSynchronize(0));

    dim3 gridApplyQ(Hq), gridApplyK(Hk);
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel",
                          apply_rotary_emb_kernel<<<gridApplyQ, D / 2>>>(d_q, d_cos_vals, d_sin_vals, Hq, D));
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel",
                          apply_rotary_emb_kernel<<<gridApplyK, D / 2>>>(d_k, d_cos_vals, d_sin_vals, Hk, D));
    HIP_CHECK(hipStreamSynchronize(0));

    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float), hipMemcpyDeviceToDevice));

    // Attention
    const int TILE_T = 256;
    dim3 grid(Hq);
    int blockAttn = (D <= 256) ? (((D + WF_SIZE - 1) / WF_SIZE) * WF_SIZE) : 256;
    if (blockAttn < WF_SIZE) blockAttn = WF_SIZE;
    dim3 blockA(blockAttn);
    size_t shmem = (size_t)TILE_T * sizeof(double)
                 + (size_t)blockAttn * sizeof(double)
                 + (size_t)D * sizeof(float);
    PROFILE_KERNEL_LAUNCH("paged_attention_fused_kernel",
                          paged_attention_fused_kernel<TILE_T><<<grid, blockA, shmem>>>(
        d_tb, d_q, d_key_cache + loff, d_value_cache + loff, d_token2row,
        d_attn_sinks + l * Hq, (p->sliding_window > 0 && (l % 2 == 0)) ? d_mask : nullptr,
        Hq, Hk, D, KV, S, pos));
    HIP_CHECK(hipStreamSynchronize(0));

    // Output projection + bias + residual (bias fused in add)
    const int O_N = D * Hq;
    dim3 gridO = get_gemv_grid_dim(H);
    PROFILE_KERNEL_LAUNCH("matmul_kernel",
                          matmul_kernel<bf16_t><<<gridO, block>>>(d_tb2, d_tb, d_w_o_bf16 + (size_t)l * H * O_N, O_N, H));
    PROFILE_KERNEL_LAUNCH("add_bias_residual_inplace_kernel",
                          add_bias_residual_inplace_kernel<<<gridO, block>>>(d_x, d_tb2, d_b_o + l * H, H));
    HIP_CHECK(hipStreamSynchronize(0));

    // FFN RMSNorm
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                          rmsnorm_kernel<<<1, BLOCK_SIZE>>>(d_t, d_x, d_rms_ffn_w + l * H, H));
    HIP_CHECK(hipStreamSynchronize(0));

    // Router: scores = W_router@t + b
    dim3 gridE = get_gemv_grid_dim(E);
    PROFILE_KERNEL_LAUNCH("matmul_kernel",
                          matmul_kernel<float><<<gridE, block>>>(d_router_score, d_t, d_w_router + (size_t)l * H * E, H, E));
    PROFILE_KERNEL_LAUNCH("add_bias_kernel_router",
                          add_bias_kernel<<<gridE, block>>>(d_router_score, d_b_router + l * E, E));
    HIP_CHECK(hipStreamSynchronize(0));

    // Top-k on device
    size_t shared_mem_size = E * sizeof(float);
    PROFILE_KERNEL_LAUNCH("topk_kernel_1token",
                          topk_kernel_1token<<<1, BLOCK_SIZE, shared_mem_size>>>(d_topk_v, d_topk_i, d_router_score, E, p->experts_per_token));
    PROFILE_KERNEL_LAUNCH("softmax_kernel",
                          softmax_kernel<<<1, 1>>>(d_topk_v, p->experts_per_token));
    HIP_CHECK(hipStreamSynchronize(0));

    // Zero e_agg
    HIP_CHECK(hipMemsetAsync(d_e_agg, 0, H * sizeof(float), 0));

    // (B)+(C): For each k in topk (remain on device): MLP1 fused, then MLP2 fused accumulate
    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      // MLP1 fused: gate_up (IM)
      dim3 gridIM = get_gemv_grid_dim(IM);
      PROFILE_KERNEL_LAUNCH("mlp1_fused_kernel",
        mlp1_fused_kernel<bf16_t><<<gridIM, block>>>(d_gate_up, d_t,
          d_w_mlp1_bf16, g_b_mlp1, d_topk_i, kk, l, E, H, IM, p->swiglu_limit));
      HIP_CHECK(hipStreamSynchronize(0));

      // MLP2 + bias + weighted accumulate to e_agg
      PROFILE_KERNEL_LAUNCH("mlp2_bias_weighted_accum_kernel",
        mlp2_bias_weighted_accum_kernel<bf16_t><<<gridH, block>>>(d_e_agg, d_gate_up,
          d_w_mlp2_bf16, g_b_mlp2, d_topk_i, d_topk_v, kk, l, E, IM, H));
      HIP_CHECK(hipStreamSynchronize(0));
    }

    // Residual add (x += e_agg)
    PROFILE_KERNEL_LAUNCH("residual_add_kernel",
                          residual_add_kernel<<<gridH, block>>>(d_x, d_e_agg, H));
    HIP_CHECK(hipStreamSynchronize(0));
  }

  // Final RMSNorm
  PROFILE_KERNEL_LAUNCH("rmsnorm_kernel",
                        rmsnorm_kernel<<<1, BLOCK_SIZE>>>(d_t, d_x, d_rms_out_w, H));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(d_x, d_t, H * sizeof(float), hipMemcpyDeviceToDevice));

  // LM head
  const int V = p->vocab_size;
  dim3 gridV = get_gemv_grid_dim(V);
  PROFILE_KERNEL_LAUNCH("matmul_kernel",
                        matmul_kernel<bf16_t><<<gridV, block>>>(d_logits, d_x, d_out_bf16, H, V));
  HIP_CHECK(hipStreamSynchronize(0));

  return d_logits;
}

// ================= Greedy / Sampling Loop =================
long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  PROFILE_FUNCTION();
  const char *empty_prompt = "";
  if (!input_seq) input_seq = empty_prompt;

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
    float *d_log = gpu_forward(transformer, token, pos);

    float *h_logits = (float *)malloc(transformer->config.vocab_size * sizeof(float));
    HIP_CHECK(hipMemcpy(h_logits, d_log, transformer->config.vocab_size * sizeof(float), hipMemcpyDeviceToHost));

    pos++;
    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      next = sample(sampler, h_logits);
      output_tokens[pos - num_prompt_tokens] = next;
    }

    if (next == 199999 || next == 200002) break;

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

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out += simple_getp_generate(transformer, tokenizer, sampler,
                                          input_seq, output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
