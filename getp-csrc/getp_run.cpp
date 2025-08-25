#include "../profiler.h"
#include "getp_eval.cpp"
#include "../attention/attention.hpp"
#include "../attention/attention.cpp"
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
static int *d_token2row; // token -> physical row mapping for paged attention
static int *d_nan_flag;  // debug flag for NaN/Inf detection

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

  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    o[i] = weight[i] * (float)(inv * (double)x[i]);
  }
}

__global__ void softmax_kernel(float *x, int size) {
  if (threadIdx.x == 0) {
    double max_val = (double)x[0];
    for (int i = 1; i < size; i++) {
      double v = (double)x[i];
      if (v > max_val) {
        max_val = v;
      }
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      float ev = expf((float)((double)x[i] - max_val));
      x[i] = ev;
      sum += (double)ev;
    }
    
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < size; i++) {
      x[i] = (float)((double)x[i] * inv_sum);
    }
  }
}

// Debug kernel: sets flag to 1 if any x[i] is NaN/Inf
__global__ void check_nans_kernel(const float *x, int n, int *flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = x[idx];
    if (!(isfinite((double)v))) atomicExch(flag, 1);
  }
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

__global__ void topk_kernel_1token(float *topk_values, int *topk_indices,
                                   const float *router_score,
                                   int num_experts, int experts_per_token) {
    extern __shared__ float smem_scores[];
    int tid = threadIdx.x;

    for (int i = tid; i < num_experts; i += blockDim.x) {
        smem_scores[i] = router_score[i];
    }
    __syncthreads();

    if (tid == 0) {
        for (int k = 0; k < experts_per_token; k++) {
            float max_val = -INFINITY;
            int max_idx = -1;

            for (int j = k; j < num_experts; j++) {
                float v = smem_scores[j];
                if (v > max_val) {
                    max_val = v;
                    max_idx = j;
                }
            }

            float tmp = smem_scores[k];
            smem_scores[k] = smem_scores[max_idx];
            smem_scores[max_idx] = tmp;

            topk_values[k] = smem_scores[k];
            topk_indices[k] = max_idx;
        }
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
static void copy_fp32_to_bf16_device(const float *h_src, size_t count,
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

   // Allocate token2row and initialize to identity (no paging indirection by default)
  HIP_CHECK(hipMalloc(&d_token2row, S * sizeof(int)));
  {
    int *h_token2row = (int *)malloc(S * sizeof(int));
    for (int i = 0; i < S; ++i) h_token2row[i] = i;
    HIP_CHECK(hipMemcpy(d_token2row, h_token2row, S * sizeof(int), hipMemcpyHostToDevice));
    free(h_token2row);
  }
  // Allocate NaN flag
  HIP_CHECK(hipMalloc(&d_nan_flag, sizeof(int)));

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
    copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V * H,
                                   d_token_embedding_table_bf16, n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_w_qkv_bf16, (size_t)L * QKV_D * H * sizeof(bf16_t)));
  {
    copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H, d_w_qkv_bf16,
                                   n_streams, chunk_bytes);
  }

  const int O_N = D * Hq;
  HIP_CHECK(hipMalloc(&d_w_o_bf16, (size_t)L * H * O_N * sizeof(bf16_t)));
  {
    copy_fp32_to_bf16_device(w->w_o, (size_t)L * H * O_N, d_w_o_bf16,
                                   n_streams, chunk_bytes);
  }

  HIP_CHECK(
      hipMalloc(&d_w_mlp1_bf16, (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)));
  {
    copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E * (2 * IM) * H,
                                   d_w_mlp1_bf16, n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_w_mlp2_bf16, (size_t)L * E * H * IM * sizeof(bf16_t)));
  {
    copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E * H * IM, d_w_mlp2_bf16,
                                   n_streams, chunk_bytes);
  }

  HIP_CHECK(hipMalloc(&d_out_bf16, (size_t)V * H * sizeof(bf16_t)));
  {
    copy_fp32_to_bf16_device(w->out, (size_t)V * H, d_out_bf16,
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
  if (d_token2row)
    HIP_CHECK(hipFree(d_token2row));
  if (d_nan_flag)
    HIP_CHECK(hipFree(d_nan_flag));

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

    // // Fused attention kernel: scores + sink + softmax + values in one pass
    // const int T_real = pos + 1;
    // const int TILE_T = 256;
    // dim3 grid(Hq);
    // dim3 block(256);
    // // Shared memory: TILE_T doubles for logits/weights + blockDim.x doubles for reductions
    // size_t shmem = (TILE_T + block.x) * sizeof(double);
    
    // PROFILE_KERNEL_LAUNCH(
    //     "attention_fused_kernel",
    //     attention_fused_kernel<TILE_T><<<grid, block, shmem>>>(
    //         d_tb,                      // [Hq, D]
    //         d_q,                       // [Hq, D]
    //         d_key_cache + loff,        // [S, KV] base for this layer
    //         d_value_cache + loff,      // [S, KV]
    //         d_attn_sinks + l * Hq,     // [Hq]
    //         (p->sliding_window > 0 && (l % 2 == 0)) ? d_mask : nullptr,
    //         Hq, Hk, D, KV, S, pos));

    // Fused attention kernel: scores + sink + softmax + values in one pass
    const int T_real = pos + 1;
    const int TILE_T = 256;
    dim3 grid(Hq);
    dim3 block(256);
    // Shared memory: TILE_T doubles for logits/weights + blockDim.x doubles for reductions
    size_t shmem = (TILE_T + block.x) * sizeof(double);
    
    PROFILE_KERNEL_LAUNCH(
    "paged_attention_fused_kernel",
    paged_attention_fused_kernel<TILE_T><<<grid, block, shmem>>>(
      d_tb,                      // [Hq, D]
      d_q,                       // [Hq, D]
      d_key_cache + loff,        // [Rows, KV] base for this layer
      d_value_cache + loff,      // [Rows, KV]
      d_token2row,               // [S]
      d_attn_sinks + l * Hq,     // [Hq]
      (p->sliding_window > 0 && (l % 2 == 0)) ? d_mask : nullptr,
      Hq, Hk, D, KV, S, pos));

    // Debug: check for NaNs/Infs in attention output tb
    int h_flag = 0;
    HIP_CHECK(hipMemset(d_nan_flag, 0, sizeof(int)));
    {
      int n = D * Hq;
      dim3 g((n + 255) / 256);
      dim3 b(256);
      PROFILE_KERNEL_LAUNCH("check_nans_kernel(tb)",
                            check_nans_kernel<<<g, b>>>(d_tb, n, d_nan_flag));
      HIP_CHECK(hipMemcpy(&h_flag, d_nan_flag, sizeof(int), hipMemcpyDeviceToHost));
      if (h_flag) {
        fprintf(stderr, "[WARN] NaN/Inf detected in attention output at layer %d, pos %d\n", l, pos);
      }
    }

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
    size_t shared_mem_size = E * sizeof(float);

    PROFILE_KERNEL_LAUNCH("topk_kernel_1token",
        topk_kernel_1token<<<1, BLOCK_SIZE, shared_mem_size>>>(
            d_topk_v,
            d_topk_i,
            d_router_score,
            E,
            p->experts_per_token
        )
    );

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