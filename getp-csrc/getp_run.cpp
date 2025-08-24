#include "../profiler.h"
#include "getp_eval.cpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
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

typedef __hip_bfloat16 bf16_t;

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

static inline bf16_t f32_to_bf16(float f) {
  return __hip_bfloat16(f);
}

__device__ __forceinline__ float bf16_to_f32(bf16_t val) {
  return __bfloat162float(val);
}

inline dim3 get_gemv_grid_dim(int d) { return dim3((d + TM - 1) / TM, 1, 1); }

// Global GPU Buffers
// Activations / temporaries (FP32)
static float *d_x, *d_t, *d_tb, *d_tb2;
static float *d_router_score, *d_topk_v, *d_mlp1_out;
static int *d_topk_i;
static float *d_gate, *d_up, *d_gate_up, *d_e_agg;
static float *d_qkv, *d_q, *d_k, *d_v;
static float *d_key_cache, *d_value_cache;
static float *d_att, *d_logits, *d_mask;
static float *d_cos_vals, *d_sin_vals;

// Small weights (keep FP32)
static float *d_rms_attn_w, *d_rms_ffn_w;
static float *d_b_qkv, *d_b_o, *d_attn_sinks;
static float *d_w_router, *d_b_router;
static float *d_rms_out_w;

// Expert biases on device (FP32)
static float *g_b_mlp1 = nullptr;
static float *g_b_mlp2 = nullptr;

// Large weights (store BF16)
static bf16_t *d_token_embedding_table_bf16;
static bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
static bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
static bf16_t *d_out_bf16;

// Config pointer
static Config *h_config = nullptr;

// Kernels (FP32 activations)
__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight,
                               int size) {
  // Block-wide reduction for sum of squares
  extern __shared__ double s_rms[];
  double sum = 0.0;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = x[i];
    sum += (double)v * (double)v;
  }
  s_rms[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_rms[threadIdx.x] += s_rms[threadIdx.x + s];
    __syncthreads();
  }
  double inv = rsqrt(s_rms[0] / (double)size + 1e-5);

  // Apply scale
  // TODO: can be optimize
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    o[i] = weight[i] * (float)(inv * (double)x[i]);
  }
}

__global__ void softmax_kernel(float *x, int size) {
  // Single-threaded sequential implementation to match CPU exactly
  if (threadIdx.x == 0) {
    // Find max value (sequential like CPU)
    double max_val = (double)x[0];
    for (int i = 1; i < size; i++) {
      double v = (double)x[i];
      if (v > max_val) {
        max_val = v;
      }
    }
    
    // Exp and sum (sequential like CPU)
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      float ev = expf((float)((double)x[i] - max_val));
      x[i] = ev;
      sum += (double)ev;
    }
    
    // Normalize (sequential like CPU)
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < size; i++) {
      x[i] = (float)((double)x[i] * inv_sum);
    }
  }
}

// Writes att[h*S + pos+1] = attn_sinks_layer[h] for all h in [0, Hq)
__global__ void write_sinks_kernel(float* __restrict__ att,
                                   const float* __restrict__ attn_sinks_layer,
                                   int S, int pos, int Hq) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h < Hq) {
    att[(size_t)h * S + (pos + 1)] = attn_sinks_layer[h];
  }
}

// att is [Hq, S] in row-major per head (stride S). For each head, softmax over t in [0, T)
__global__ void softmax_heads_kernel(float* __restrict__ att,
                                     int S, int T, int Hq) {
  extern __shared__ double sdata[]; // use for reductions
  int h = blockIdx.x;               // one block per head
  if (h >= Hq) return;

  float* x = att + (size_t)h * S;

  // 1) compute max in double with deterministic tree reduction
  double local_max = -INFINITY;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    local_max = fmax(local_max, (double)x[t]);
  }
  
  // Store local max to shared memory
  sdata[threadIdx.x] = local_max;
  __syncthreads();
  
  // Binary tree reduction for max
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }
  
  // Broadcast max value
  double max_val = sdata[0];
  __syncthreads();

  // 2) write back (x[t] = expf((float)((double)x[t] - max))) and accumulate double sum
  double local_sum = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    float ev = expf((float)((double)x[t] - max_val));
    x[t] = ev;
    local_sum += (double)ev;
  }
  
  // Store local sum to shared memory
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  
  // Binary tree reduction for sum
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  // Broadcast sum value
  double sum_val = sdata[0];
  __syncthreads();

  // 3) normalize: x[t] *= (float)(1.0 / sum)
  double inv_sum = 1.0 / sum_val;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    x[t] = (float)((double)x[t] * inv_sum);
  }
}


__global__ void add_bias_kernel(float *y, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    y[i] += b[i];
}

__global__ void copy_embedding_bf16_row_kernel(float *dst, const bf16_t *src,
                                               int token, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    dst[i] = bf16_to_f32(src[(size_t)token * hidden_dim + i]);
  }
}

__global__ void split_qkv_kernel(float *q, float *k, float *v, const float *qkv,
                                 int n_attn_heads, int n_kv_heads,
                                 int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int q_size = n_attn_heads * head_dim;
  int kv_size = n_kv_heads * head_dim;
  int total = q_size + 2 * kv_size;
  if (idx >= total)
    return;
  if (idx < q_size)
    q[idx] = qkv[idx];
  else if (idx < q_size + kv_size)
    k[idx - q_size] = qkv[idx];
  else
    v[idx - q_size - kv_size] = qkv[idx];
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
  if (i >= d_half)
    return;
  float freq = powf(rope_theta, (float)(2 * i) / (float)head_dim);
  float inv_freq;
  float concentration = 1.0f;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = d_half *
                logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                logf(rope_theta);
    float high = d_half *
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(rope_theta);
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

__global__ void attention_scores_kernel(
    float *att,
    const float *q,
    const float *key_cache,
    int h, int pos,
    int head_dim, int kv_dim,
    int n_attn_heads, int n_kv_heads,
    int seq_len, const float *mask) {

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t > pos || t >= seq_len) return;

  const int kv_mul = n_attn_heads / n_kv_heads;
  const float *k = key_cache + (size_t)t * kv_dim + (h / kv_mul) * head_dim;

  double acc = 0.0;
  const float *qh = q + (size_t)h * head_dim;
  #pragma unroll
  for (int i = 0; i < head_dim; ++i) {
    acc += (double)qh[i] * (double)k[i];
  }

  const double scale = 1.0 / sqrt((double)head_dim);
  double score = acc * scale;

  if (mask) {
    score += (double)mask[(size_t)pos * seq_len + t];
  }

  att[(size_t)h * seq_len + t] = (float)score;
}

__global__ void attention_values_kernel(
    float *tb,
    const float *att,
    const float *value_cache,
    int h, int pos,
    int head_dim, int kv_dim,
    int n_attn_heads, int n_kv_heads,
    int seq_len) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= head_dim) return;

  const int kv_mul = n_attn_heads / n_kv_heads;
  const int base_v = (h / kv_mul) * head_dim;

  double acc = 0.0;
  const float *att_h = att + (size_t)h * seq_len;

  for (int t = 0; t <= pos; ++t) {
    const float *v = value_cache + (size_t)t * kv_dim + base_v;
    acc += (double)att_h[t] * (double)v[i];
  }

  tb[(size_t)h * head_dim + i] = (float)acc;
}

// Computes tb[h, i] = sum_{t=0..T-1} att[h, t] * value_cache[t, base_v + i]
// for all heads h in [0, Hq) and i in [0, D).
template<int TILE_T>
__global__ void attn_values_heads_kernel(
    float* __restrict__ tb,           // [Hq, D]
    const float* __restrict__ att,    // [Hq, S]
    const float* __restrict__ value_cache, // [S, KV]
    int Hq, int Hk, int D, int KV,    // KV = D * Hk
    int S, int T                      // T = pos + 2
) {
  extern __shared__ double s_att[];   // size = TILE_T

  const int h   = blockIdx.x;         // head id
  const int tid = threadIdx.x;
  if (h >= Hq) return;

  // kv_mul = n_attn_heads / n_kv_heads
  const int kv_mul = Hq / Hk;
  const int base_v = (h / kv_mul) * D;

  // i-dimension scheduling: each thread handles one i
  int i_start = blockIdx.y * blockDim.x + tid;

  // Accumulator in double
  double acc = 0.0;

  // Tile over time dimension
  for (int t0 = 0; t0 < T; t0 += TILE_T) {
    const int tile = min(TILE_T, T - t0);

    // 1) Load att[h, t] into shared memory as double
    for (int t = tid; t < tile; t += blockDim.x) {
      s_att[t] = (double)att[(size_t)h * S + (t0 + t)];
    }
    __syncthreads();

    // 2) Accumulate products for the assigned i
    if (i_start < D) {
      // base pointer for this i over time
      const float* v_ptr = value_cache + (size_t)(t0) * KV + base_v + i_start;
      for (int t = 0; t < tile; ++t) {
        // value_cache layout: [t, KV]
        // element: value_cache[t, base_v + i]
        double vv = (double)v_ptr[t * (size_t)KV];
        acc += s_att[t] * vv;
      }
    }
    __syncthreads();
  }

  // 3) Write back
  if (i_start < D) {
    tb[(size_t)h * D + i_start] = (float)acc;
  }
}

// Fused attention kernel: scores + sink + softmax + values in one pass
template<int TILE_T>
__global__ void attention_fused_kernel(
    float* __restrict__ tb,                 // [Hq, D] out
    const float* __restrict__ q,            // [Hq, D]
    const float* __restrict__ key_cache,    // [S, KV] for layer l
    const float* __restrict__ value_cache,  // [S, KV] for layer l
    const float* __restrict__ attn_sinks_layer, // [Hq]
    const float* __restrict__ mask,         // [S, S] or nullptr
    int Hq, int Hk, int D, int KV,          // KV = D * Hk
    int S, int pos                          // T_real = pos + 1
) {
  extern __shared__ double s_tile[]; // used for per-tile score/weight work
  const int h   = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= Hq) return;

  const int kv_mul = Hq / Hk;
  const int base   = (h / kv_mul) * D;
  const double scale = 1.0 / sqrt((double)D);

  // Pointers
  const float* qh = q + (size_t)h * D;    // [D]
  const float* K  = key_cache;            // [S, KV]
  const float* V  = value_cache;          // [S, KV]

  const int T_real = pos + 1;             // indices t ∈ [0, T_real)
  
  // Shared memory layout: first TILE_T for logits/weights, then blockDim.x for reductions
  double* s_logits = s_tile;
  double* s_reduce = s_tile + TILE_T;
  
  // 1) Find global max over all logits and sink
  double gmax = -INFINITY;

  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    
    // Compute logits for this tile
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      
      // Compute Q·K dot product with all threads participating
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)t * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      
      // Store logit and update local max
      if (tid == 0) {
        s_logits[tt] = logit;
        gmax = fmax(gmax, logit);
      }
      __syncthreads();
    }
  }
  
  // Include sink in global max
  const double sinkVal = (double)attn_sinks_layer[h];
  if (tid == 0) {
    gmax = fmax(gmax, sinkVal);
    s_reduce[0] = gmax; // Broadcast gmax
  }
  __syncthreads();
  gmax = s_reduce[0];

  // 2) Second pass: compute sumExp and numerator accumulation
  double sumExp = 0.0;
  
  // Each thread handles one output dimension i
  const int i0 = tid;
  double num_acc = 0.0;

  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);

    // Recompute logits for this tile (same as pass 1)
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      
      // Compute Q·K dot product
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)t * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      
      // Compute unnormalized weight and accumulate sumExp
      if (tid == 0) {
        double w = exp(logit - gmax);
        s_logits[tt] = w;  // Store weight as double
        sumExp += w;
      }
      __syncthreads();
    }

    // Accumulate numerator for owned output dimension
    if (i0 < D) {
      for (int tt = 0; tt < tile; ++tt) {
        const int t = t0 + tt;
        const double w = s_logits[tt];  // Double precision weight
        const float v = V[(size_t)t * KV + base + i0];
        num_acc += w * (double)v;
      }
    }
    __syncthreads();
  }

  // Add sink contribution to denominator (but not numerator)
  if (tid == 0) {
    const double sinkExp = exp(sinkVal - gmax);
    sumExp += sinkExp;
    s_reduce[0] = 1.0 / sumExp; // Store inverse for broadcast
  }
  __syncthreads();
  const double inv = s_reduce[0];

  // Write final result
  if (i0 < D) {
    tb[(size_t)h * D + i0] = (float)(num_acc * inv);
  }
}


__global__ void residual_add_kernel(float *x, const float *residual, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    x[i] += residual[i];
}

// Fused matmul with bias and residual add: W(d,n)[bf16] @ x(n)[f32] + b(d)[f32] + residual(d)[f32] -> output(d)[f32]
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_bf16_wxf32_yf32_bias_residual(float * __restrict__ output,
                                                     const float * __restrict__ x,
                                                     const bf16_t * __restrict__ w,
                                                     const float * __restrict__ b,
                                                     const float * __restrict__ residual,
                                                     int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;               // wave id in block: 0..TM-1
  const int row  = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  double acc_all = 0.0;  // Use double precision like CPU

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) {
      lds_x[k] = x[k_base + k];
    }
    __syncthreads();

    const bf16_t* __restrict__ w_row = w + (size_t)row * n + k_base;

    double acc = 0.0;  // Use double precision like CPU
    
    // Direct bf16 operations without casting
    for (int k = lane; k < k_size; k += WF_SIZE) {
      bf16_t bf16_val = w_row[k];
      double wx = (double)bf16_to_f32(bf16_val);
      double xval = (double)lds_x[k];
      acc += wx * xval;  // Double precision accumulation
    }

    // Wave reduction (width = 64) with double precision
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, WF_SIZE);
    }
    if (lane == 0) acc_all += acc;

    __syncthreads();
  }

  if (lane == 0) output[row] = (float)acc_all + b[row] + residual[row];  // Add bias and residual
}

// Pair struct for GPU sorting (matching CPU implementation)
typedef struct {
  float value;
  int index;
} GPUPair;

// Comparison function for GPU qsort (descending order)
__device__ int gpu_compare_desc(const void *a, const void *b) {
  const GPUPair *pa = (const GPUPair *)a;
  const GPUPair *pb = (const GPUPair *)b;
  if (pb->value > pa->value) return 1;
  if (pb->value < pa->value) return -1;
  return 0;
}

// GPU sorting function using bubble sort (since qsort not available on device)
__device__ void gpu_sort_pairs(GPUPair *pairs, int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (pairs[j].value < pairs[j + 1].value) {
        GPUPair temp = pairs[j];
        pairs[j] = pairs[j + 1];
        pairs[j + 1] = temp;
      }
    }
  }
}

__global__ void topk_kernel_1token(float *topk_values, int *topk_indices,
                                   const float *router_score, int num_experts,
                                   int experts_per_token) {
  // Allocate shared memory for pairs (assuming small number of experts)
  extern __shared__ GPUPair pairs[];
  
  // Copy router scores to pairs array
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    pairs[i].value = router_score[i];
    pairs[i].index = i;
  }
  __syncthreads();
  
  // Sort pairs in descending order (only thread 0 does the sorting)
  if (threadIdx.x == 0) {
    gpu_sort_pairs(pairs, num_experts);
  }
  __syncthreads();
  
  // Extract top-k results
  for (int i = threadIdx.x; i < experts_per_token; i += blockDim.x) {
    topk_values[i] = pairs[i].value;
    topk_indices[i] = pairs[i].index;
  }
}

__global__ void split_gate_up_kernel(float *gate, float *up,
                                     const float *mlp1_out,
                                     int intermediate_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < intermediate_dim) {
    gate[i] = mlp1_out[2 * i];
    up[i] = mlp1_out[2 * i + 1];
  }
}

__global__ void swiglu_kernel(float *gate_up, const float *gate,
                              const float *up, int intermediate_dim,
                              float swiglu_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < intermediate_dim) {
    float g = gate[i];
    float u = up[i];
    // Clamping to match CPU behavior:
    // gate: only upper bound (positive direction)
    if (g > swiglu_limit) g = swiglu_limit;
    // up: both directions (±limit)
    if (u > swiglu_limit) u = swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;
    const float alpha = 1.702f; // SiLU approx
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);
    gate_up[i] = g;
  }
}

__global__ void weighted_sum_kernel(float *e_agg, const float *expert_out,
                                    float weight, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim)
    e_agg[i] += expert_out[i] * weight;
}


__global__ void memset_zero_kernel(float *ptr, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    ptr[i] = 0.0f;
}

// Matmul: W(d,n)[bf16] @ x(n)[f32] -> y(d)[f32]; FP32 accumulate
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_bf16_wxf32_yf32(float * __restrict__ y,
                                       const float * __restrict__ x,
                                       const bf16_t * __restrict__ w,
                                       int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;               // wave id in block: 0..TM-1
  const int row  = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  double acc_all = 0.0;  // Use double precision like CPU

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) {
      lds_x[k] = x[k_base + k];
    }
    __syncthreads();

    const bf16_t* __restrict__ w_row = w + (size_t)row * n + k_base;

    double acc = 0.0;  // Use double precision like CPU
    
    // Direct bf16 operations without casting
    for (int k = lane; k < k_size; k += WF_SIZE) {
      bf16_t bf16_val = w_row[k];
      double wx = (double)bf16_to_f32(bf16_val);
      double xval = (double)lds_x[k];
      acc += wx * xval;  // Double precision accumulation
    }

    // Wave reduction (width = 64) with double precision
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, WF_SIZE);
    }
    if (lane == 0) acc_all += acc;

    __syncthreads();
  }

  if (lane == 0) y[row] = (float)acc_all;  // Convert back to float for output
}

// Fused matmul with bias: W(d,n)[bf16] @ x(n)[f32] + b(d)[f32] -> y(d)[f32]
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_bf16_wxf32_yf32_bias(float * __restrict__ y,
                                            const float * __restrict__ x,
                                            const bf16_t * __restrict__ w,
                                            const float * __restrict__ b,
                                            int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;               // wave id in block: 0..TM-1
  const int row  = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  double acc_all = 0.0;  // Use double precision like CPU

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) {
      lds_x[k] = x[k_base + k];
    }
    __syncthreads();

    const bf16_t* __restrict__ w_row = w + (size_t)row * n + k_base;

    double acc = 0.0;  // Use double precision like CPU
    
    // Direct bf16 operations without casting
    for (int k = lane; k < k_size; k += WF_SIZE) {
      bf16_t bf16_val = w_row[k];
      double wx = (double)bf16_to_f32(bf16_val);
      double xval = (double)lds_x[k];
      acc += wx * xval;  // Double precision accumulation
    }

    // Wave reduction (width = 64) with double precision
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, WF_SIZE);
    }
    if (lane == 0) acc_all += acc;

    __syncthreads();
  }

  if (lane == 0) y[row] = (float)acc_all + b[row];  // Add bias
}

// FP32 matmul: W(d,n)[f32] @ x(n)[f32] -> y(d)[f32]
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_f32_wxf32_yf32(float * __restrict__ y,
                                      const float * __restrict__ x,
                                      const float * __restrict__ w,
                                      int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  double acc_all = 0.0;  // Use double precision like CPU

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) {
      lds_x[k] = x[k_base + k];
    }
    __syncthreads();

    const float* __restrict__ w_row = w + (size_t)row * n + k_base;

    double acc = 0.0;  // Use double precision like CPU
    int k = lane;

    // Simplified loop for double precision accumulation
    for (; k < k_size; k += WF_SIZE) {
      double wval = (double)w_row[k];
      double xval = (double)lds_x[k];
      acc += wval * xval;  // Double precision accumulation
    }

    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, WF_SIZE);
    }
    if (lane == 0) acc_all += acc;

    __syncthreads();
  }

  if (lane == 0) y[row] = (float)acc_all;  // Convert back to float for output
}

// Fused FP32 matmul with bias: W(d,n)[f32] @ x(n)[f32] + b(d)[f32] -> y(d)[f32]
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void matmul_f32_wxf32_yf32_bias(float * __restrict__ y,
                                           const float * __restrict__ x,
                                           const float * __restrict__ w,
                                           const float * __restrict__ b,
                                           int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d) return;

  double acc_all = 0.0;  // Use double precision like CPU

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE) {
      lds_x[k] = x[k_base + k];
    }
    __syncthreads();

    const float* __restrict__ w_row = w + (size_t)row * n + k_base;

    double acc = 0.0;  // Use double precision like CPU
    int k = lane;

    // Simplified loop for double precision accumulation
    for (; k < k_size; k += WF_SIZE) {
      double wval = (double)w_row[k];
      double xval = (double)lds_x[k];
      acc += wval * xval;  // Double precision accumulation
    }

    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, WF_SIZE);
    }
    if (lane == 0) acc_all += acc;

    __syncthreads();
  }

  if (lane == 0) y[row] = (float)acc_all + b[row];  // Add bias
}


// Async pipelined FP32->BF16 converter with overlapped CPU conversion and H2D transfers
static void copy_fp32_to_bf16_device_async(const float *h_src, size_t count,
                                           bf16_t *d_dst, int n_streams = 4,
                                           size_t chunk_bytes = 64ULL * 1024 * 1024) {
  if (count == 0) return;
  
  // Calculate chunk size in elements
  const size_t chunk_elems = chunk_bytes / sizeof(bf16_t);
  const size_t actual_chunk_elems = (chunk_elems > count) ? count : chunk_elems;
  
  // Try to allocate pinned buffers and streams
  hipStream_t *streams = nullptr;
  hipEvent_t *events = nullptr;
  bf16_t **pinned_chunks = nullptr;
  bool async_success = true;
  
  // Allocate stream array
  streams = (hipStream_t*)malloc(n_streams * sizeof(hipStream_t));
  if (!streams) async_success = false;
  
  // Allocate event array
  if (async_success) {
    events = (hipEvent_t*)malloc(n_streams * sizeof(hipEvent_t));
    if (!events) async_success = false;
  }
  
  // Allocate pinned buffer array
  if (async_success) {
    pinned_chunks = (bf16_t**)malloc(n_streams * sizeof(bf16_t*));
    if (!pinned_chunks) async_success = false;
    else {
      for (int i = 0; i < n_streams; i++) pinned_chunks[i] = nullptr;
    }
  }
  
  // Create streams
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  
  // Create events
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipEventCreateWithFlags(&events[i], hipEventDisableTiming);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  
  // Allocate pinned host buffers
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipHostMalloc((void**)&pinned_chunks[i], 
                                     actual_chunk_elems * sizeof(bf16_t));
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  
  // If async allocation failed, fallback to synchronous
  if (!async_success) {
    // Cleanup any partial allocations
    if (pinned_chunks) {
      for (int i = 0; i < n_streams; i++) {
        if (pinned_chunks[i]) {
          (void)hipHostFree(pinned_chunks[i]); // Ignore return value in cleanup
        }
      }
      free(pinned_chunks);
    }
    if (events) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipEventDestroy(events[i]); // May fail silently if not created
      }
      free(events);
    }
    if (streams) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipStreamDestroy(streams[i]); // May fail silently if not created
      }
      free(streams);
    }
    
    // Fallback to synchronous copy
    const size_t SYNC_CHUNK_ELEMS = 8 * 1024 * 1024; // 8M elems (~16MB) per chunk
    bf16_t *h_chunk = nullptr;
    HIP_CHECK(hipHostMalloc((void **)&h_chunk, SYNC_CHUNK_ELEMS * sizeof(bf16_t)));
    size_t done = 0;
    while (done < count) {
      size_t todo = (count - done > SYNC_CHUNK_ELEMS) ? SYNC_CHUNK_ELEMS : (count - done);
      // convert
      for (size_t i = 0; i < todo; ++i)
        h_chunk[i] = f32_to_bf16(h_src[done + i]);
      // copy
      HIP_CHECK(hipMemcpy(d_dst + done, h_chunk, todo * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
      done += todo;
    }
    HIP_CHECK(hipHostFree(h_chunk));
    return;
  }
  
  // Async pipeline
  size_t done = 0;
  int stream_idx = 0;
  bool *buffer_ready = (bool*)calloc(n_streams, sizeof(bool));
  
  // Initialize all buffers as ready
  for (int i = 0; i < n_streams; i++) {
    buffer_ready[i] = true;
  }
  
  while (done < count) {
    size_t todo = (count - done > actual_chunk_elems) ? actual_chunk_elems : (count - done);
    
    // Wait for this buffer slot to be available
    if (!buffer_ready[stream_idx]) {
      HIP_CHECK(hipEventSynchronize(events[stream_idx]));
      buffer_ready[stream_idx] = true;
    }
    
    // Convert FP32 to BF16 on CPU (parallelized with OpenMP)
    bf16_t *chunk = pinned_chunks[stream_idx];
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < todo; ++i) {
      chunk[i] = f32_to_bf16(h_src[done + i]);
    }
    
    // Launch async H2D copy
    HIP_CHECK(hipMemcpyAsync(d_dst + done, chunk, todo * sizeof(bf16_t),
                             hipMemcpyHostToDevice, streams[stream_idx]));
    
    // Record event to track completion
    HIP_CHECK(hipEventRecord(events[stream_idx], streams[stream_idx]));
    buffer_ready[stream_idx] = false;
    
    done += todo;
    stream_idx = (stream_idx + 1) % n_streams;
  }
  
  // Wait for all transfers to complete
  for (int i = 0; i < n_streams; i++) {
    HIP_CHECK(hipEventSynchronize(events[i]));
  }
  
  // Cleanup resources
  free(buffer_ready);
  
  for (int i = 0; i < n_streams; i++) {
    HIP_CHECK(hipHostFree(pinned_chunks[i]));
    HIP_CHECK(hipEventDestroy(events[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  
  free(pinned_chunks);
  free(events);
  free(streams);
}

static void copy_fp32_to_bf16_device(const float *h_src, size_t count,
                                     bf16_t *d_dst) {
  const size_t CHUNK_ELEMS =
      (size_t)8 * 1024 * 1024; // 8M elems (~16MB) per chunk
  bf16_t *h_chunk = nullptr;
  HIP_CHECK(hipHostMalloc((void **)&h_chunk, CHUNK_ELEMS * sizeof(bf16_t)));
  size_t done = 0;
  while (done < count) {
    size_t todo = (count - done > CHUNK_ELEMS) ? CHUNK_ELEMS : (count - done);
    // convert
    for (size_t i = 0; i < todo; ++i)
      h_chunk[i] = f32_to_bf16(h_src[done + i]);
    // copy
    HIP_CHECK(hipMemcpy(d_dst + done, h_chunk, todo * sizeof(bf16_t),
                        hipMemcpyHostToDevice));
    done += todo;
  }
  HIP_CHECK(hipHostFree(h_chunk));
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  PROFILE_FUNCTION();
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

  // ---------------- Activations ----------------
  HIP_CHECK(hipMalloc(&d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb, D * Hq * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb2, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_v, h_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_i, h_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&d_mlp1_out, 2 * IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up, IM * sizeof(float)));
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

  if (h_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] =
            (i - j >= h_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(d_mask, h_mask, S * S * sizeof(float),
                        hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations");

  // ---------------- Weights ----------------
  TransformerWeights *w = &transformer->weights;

  // Small FP32 weights
  HIP_CHECK(hipMalloc(&d_rms_attn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_attn_w, w->rms_attn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_ffn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_ffn_w, w->rms_ffn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice));

  const int QKV_D = D * (Hq + 2 * Hk);
  HIP_CHECK(hipMalloc(&d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_qkv, w->b_qkv, L * QKV_D * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_o, L * H * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_b_o, w->b_o, L * H * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_attn_sinks, L * Hq * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_attn_sinks, w->attn_sinks, L * Hq * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_w_router, L * H * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w_router, w->w_router, L * H * E * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_router, L * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_router, w->b_router, L * E * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_out_w, H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_out_w, w->rms_out_w, H * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights");

  // Expert biases FP32
  HIP_CHECK(hipMalloc(&g_b_mlp1, (size_t)L * E * (2 * IM) * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp1, w->b_mlp1,
                      (size_t)L * E * (2 * IM) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&g_b_mlp2, (size_t)L * E * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp2, w->b_mlp2, (size_t)L * E * H * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases");

  // Large BF16 weights - using async pipelined loading for performance
  const int n_streams = 4;
  const size_t chunk_bytes = 64ULL * 1024 * 1024; // 64 MiB per chunk
  
  HIP_CHECK(
      hipMalloc(&d_token_embedding_table_bf16, (size_t)V * H * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("token_embedding_table_async_load");
    copy_fp32_to_bf16_device_async(w->token_embedding_table, (size_t)V * H,
                                   d_token_embedding_table_bf16, n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_w_qkv_bf16, (size_t)L * QKV_D * H * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("w_qkv_async_load");
    copy_fp32_to_bf16_device_async(w->w_qkv, (size_t)L * QKV_D * H, d_w_qkv_bf16,
                                   n_streams, chunk_bytes);
  }

  const int O_N = D * Hq;
  HIP_CHECK(hipMalloc(&d_w_o_bf16, (size_t)L * H * O_N * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("w_o_async_load");
    copy_fp32_to_bf16_device_async(w->w_o, (size_t)L * H * O_N, d_w_o_bf16,
                                   n_streams, chunk_bytes);
  }

  HIP_CHECK(
      hipMalloc(&d_w_mlp1_bf16, (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("w_mlp1_async_load");
    copy_fp32_to_bf16_device_async(w->w_mlp1, (size_t)L * E * (2 * IM) * H,
                                   d_w_mlp1_bf16, n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_w_mlp2_bf16, (size_t)L * E * H * IM * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("w_mlp2_async_load");
    copy_fp32_to_bf16_device_async(w->w_mlp2, (size_t)L * E * H * IM, d_w_mlp2_bf16,
                                   n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_out_bf16, (size_t)V * H * sizeof(bf16_t)));
  {
    PROFILE_SCOPE("out_async_load");
    copy_fp32_to_bf16_device_async(w->out, (size_t)V * H, d_out_bf16,
                                   n_streams, chunk_bytes);
  }

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
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
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
  if (d_mask)
    HIP_CHECK(hipFree(d_mask));

  HIP_CHECK(hipFree(d_rms_attn_w));
  HIP_CHECK(hipFree(d_rms_ffn_w));
  HIP_CHECK(hipFree(d_b_qkv));
  HIP_CHECK(hipFree(d_b_o));
  HIP_CHECK(hipFree(d_attn_sinks));
  HIP_CHECK(hipFree(d_w_router));
  HIP_CHECK(hipFree(d_b_router));
  HIP_CHECK(hipFree(d_rms_out_w));

  if (g_b_mlp1)
    HIP_CHECK(hipFree(g_b_mlp1));
  if (g_b_mlp2)
    HIP_CHECK(hipFree(g_b_mlp2));

  HIP_CHECK(hipFree(d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(d_w_qkv_bf16));
  HIP_CHECK(hipFree(d_w_o_bf16));
  HIP_CHECK(hipFree(d_w_mlp1_bf16));
  HIP_CHECK(hipFree(d_w_mlp2_bf16));
  HIP_CHECK(hipFree(d_out_bf16));
}

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

  // Use optimized launch configurations
  dim3 block = dim3(BLOCK_SIZE, 1, 1);
  dim3 gridH = get_gemv_grid_dim(H);

  // x <- embedding[token] (BF16 -> FP32)
  PROFILE_KERNEL_LAUNCH("copy_embedding_bf16_row_kernel",
                        copy_embedding_bf16_row_kernel<<<gridH, block>>>(
                            d_x, d_token_embedding_table_bf16, token, H));

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attn)
    PROFILE_KERNEL_LAUNCH(
        "rmsnorm_kernel(attn)",
        rmsnorm_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
            d_t, d_x, d_rms_attn_w + l * H, H));

    // QKV with fused bias
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV = get_gemv_grid_dim(QKV_D);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bf16_wxf32_yf32_bias(QKV)",
        matmul_bf16_wxf32_yf32_bias<<<gridQKV, block>>>(
            d_qkv, d_t, d_w_qkv_bf16 + (size_t)l * QKV_D * H, d_b_qkv + l * QKV_D, H, QKV_D));

    PROFILE_KERNEL_LAUNCH(
        "split_qkv_kernel",
        split_qkv_kernel<<<gridQKV, block>>>(d_q, d_k, d_v, d_qkv, Hq, Hk, D));

    int loff = l * S * KV;
    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float),
                        hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(d_value_cache + loff + pos * KV, d_v,
                        KV * sizeof(float), hipMemcpyDeviceToDevice));

    dim3 gridR(((D / 2) + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH("compute_cos_sin_kernel",
                          compute_cos_sin_kernel<<<gridR, block>>>(
                              d_cos_vals, d_sin_vals, pos, p->rope_theta, D,
                              p->rope_scaling_factor,
                              p->initial_context_length));

    dim3 gridApplyQ(Hq), gridApplyK(Hk);
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel(Q)",
                          apply_rotary_emb_kernel<<<gridApplyQ, D / 2>>>(
                              d_q, d_cos_vals, d_sin_vals, Hq, D));
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel(K)",
                          apply_rotary_emb_kernel<<<gridApplyK, D / 2>>>(
                              d_k, d_cos_vals, d_sin_vals, Hk, D));

    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float),
                        hipMemcpyDeviceToDevice));

    // Fused attention kernel: scores + sink + softmax + values in one pass
    const int T_real = pos + 1;
    const int TILE_T = 256;
    dim3 grid(Hq);
    dim3 block(256);
    // Shared memory: TILE_T doubles for logits/weights + blockDim.x doubles for reductions
    size_t shmem = (TILE_T + block.x) * sizeof(double);
    
    PROFILE_KERNEL_LAUNCH(
        "attention_fused_kernel",
        attention_fused_kernel<TILE_T><<<grid, block, shmem>>>(
            d_tb,                      // [Hq, D]
            d_q,                       // [Hq, D]
            d_key_cache + loff,        // [S, KV] base for this layer
            d_value_cache + loff,      // [S, KV]
            d_attn_sinks + l * Hq,     // [Hq]
            (p->sliding_window > 0 && (l % 2 == 0)) ? d_mask : nullptr,
            Hq, Hk, D, KV, S, pos));

    const int O_N = D * Hq;
    dim3 gridO = get_gemv_grid_dim(H);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bf16_wxf32_yf32_bias_residual(W_o)",
        matmul_bf16_wxf32_yf32_bias_residual<<<gridO, block>>>(
            d_x, d_tb, d_w_o_bf16 + (size_t)l * H * O_N, d_b_o + l * H, d_x, O_N, H));

    // FFN RMSNorm
    PROFILE_KERNEL_LAUNCH(
        "rmsnorm_kernel(ffn)",
        rmsnorm_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
            d_t, d_x, d_rms_ffn_w + l * H, H));

    // Router FP32 with fused bias: d_router_score(E) = W_router(E,H) @ d_t(H) + b_router
    dim3 gridE = get_gemv_grid_dim(E);
    PROFILE_KERNEL_LAUNCH(
        "matmul_f32_wxf32_yf32_bias(router)",
        matmul_f32_wxf32_yf32_bias<<<gridE, block>>>(
            d_router_score, d_t, d_w_router + (size_t)l * H * E, d_b_router + l * E, H, E));

    // Top-k experts
    size_t shared_mem_size = E * sizeof(GPUPair);
    PROFILE_KERNEL_LAUNCH("topk_kernel_1token",
                          topk_kernel_1token<<<1, BLOCK_SIZE, shared_mem_size>>>(d_topk_v, d_topk_i,
                                                       d_router_score, E,
                                                       p->experts_per_token));
    PROFILE_KERNEL_LAUNCH(
        "softmax_kernel(topk)",
        softmax_kernel<<<1, 1>>>(
            d_topk_v, p->experts_per_token));

    // Zero-initialize expert aggregation buffer
    HIP_CHECK(hipMemsetAsync(d_e_agg, 0, H * sizeof(float), 0));

    // Read topk to host (small)
    float *h_topk_v = (float *)malloc(p->experts_per_token * sizeof(float));
    int *h_topk_i = (int *)malloc(p->experts_per_token * sizeof(int));
    HIP_CHECK(hipMemcpy(h_topk_v, d_topk_v,
                        p->experts_per_token * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_topk_i, d_topk_i, p->experts_per_token * sizeof(int),
                        hipMemcpyDeviceToHost));

    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      int e = h_topk_i[kk];
      float ew = h_topk_v[kk];
      // MLP1 with fused bias: (2IM x H) @ d_t + b_mlp1
      size_t off = ((size_t)l * E + e) * (size_t)(2 * IM) * (size_t)H;
      size_t b1off = ((size_t)l * E + e) * (size_t)(2 * IM);
      dim3 gridM1 = get_gemv_grid_dim(2 * IM);
      PROFILE_KERNEL_LAUNCH(
          "matmul_bf16_wxf32_yf32_bias(MLP1)",
          matmul_bf16_wxf32_yf32_bias<<<gridM1, block>>>(
              d_mlp1_out, d_t, d_w_mlp1_bf16 + off, g_b_mlp1 + b1off, H, 2 * IM));

      // Split & SwiGLU
      dim3 gridIM((IM + block.x - 1) / block.x);
      PROFILE_KERNEL_LAUNCH("split_gate_up_kernel",
                            split_gate_up_kernel<<<gridIM, block>>>(
                                d_gate, d_up, d_mlp1_out, IM));
      PROFILE_KERNEL_LAUNCH("swiglu_kernel",
                            swiglu_kernel<<<gridIM, block>>>(
                                d_gate_up, d_gate, d_up, IM, p->swiglu_limit));

      // MLP2 with fused bias: (H x IM) @ gate_up + b_mlp2
      off = ((size_t)l * E + e) * (size_t)H * (size_t)IM;
      size_t b2off = ((size_t)l * E + e) * (size_t)H;
      PROFILE_KERNEL_LAUNCH(
          "matmul_bf16_wxf32_yf32_bias(MLP2)",
          matmul_bf16_wxf32_yf32_bias<<<get_gemv_grid_dim(H), block>>>(
              d_tb2, d_gate_up, d_w_mlp2_bf16 + off, g_b_mlp2 + b2off, IM, H));

      // Weighted sum into e_agg
      PROFILE_KERNEL_LAUNCH(
          "weighted_sum_kernel",
          weighted_sum_kernel<<<gridH, block>>>(d_e_agg, d_tb2, ew, H));
    }
    free(h_topk_v);
    free(h_topk_i);

    // Residual
    PROFILE_KERNEL_LAUNCH(
        "residual_add_kernel(ffn)",
        residual_add_kernel<<<gridH, block>>>(d_x, d_e_agg, H));
  }

  // Final RMSNorm
  PROFILE_KERNEL_LAUNCH(
      "rmsnorm_kernel(final)",
      rmsnorm_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
          d_t, d_x, d_rms_out_w, H));
  HIP_CHECK(hipMemcpy(d_x, d_t, H * sizeof(float), hipMemcpyDeviceToDevice));

  // LM head: (V x H) @ x -> logits(V)
  const int V = p->vocab_size;
  dim3 gridV = get_gemv_grid_dim(V);
  PROFILE_KERNEL_LAUNCH("matmul_bf16_wxf32_yf32(OUT)",
                        matmul_bf16_wxf32_yf32<<<gridV, block>>>(
                            d_logits, d_x, d_out_bf16, H, V));

  return d_logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  PROFILE_FUNCTION();
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

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {
    float *d_log = gpu_forward(transformer, token, pos);

    float *h_logits =
        (float *)malloc(transformer->config.vocab_size * sizeof(float));
    HIP_CHECK(hipMemcpy(h_logits, d_log,
                        transformer->config.vocab_size * sizeof(float),
                        hipMemcpyDeviceToHost));

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

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
