#include "../common/defines.h"
#include "utility.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline void debug_print_gpu_memory(const char *tag, int device_id) {
  size_t free_b = 0, total_b = 0;
  hipError_t err = hipMemGetInfo(&free_b, &total_b);
  if (err != hipSuccess) {
    fprintf(stderr, "HIP hipMemGetInfo failed: %s\n", hipGetErrorString(err));
    return;
  }
  double free_gib = (double)free_b / (1024.0 * 1024.0 * 1024.0);
  double total_gib = (double)total_b / (1024.0 * 1024.0 * 1024.0);
  double used_gib = total_gib - free_gib;
  printf("[DEVICE] %d [HIP] %s: HBM free %.2f GiB / total %.2f GiB (used %.2f "
         "GiB)\n",
         device_id, tag, free_gib, total_gib, used_gib);
  fflush(stdout);
}

__device__ __forceinline__ short f32_to_bf16_bits_short(float f) {
  union { uint32_t u; float f; } v; v.f = f;
  return (short)(v.u >> 16);
}

__device__ __forceinline__ s16x4 pack4_bf16_from_f32_guard(
    const float* base_f32, int k_off, int k_rem, bool row_valid) {
  s16x4 v = {0,0,0,0};
  if (!row_valid) return v;
  #pragma unroll
  for (int i=0;i<4;i++) {
    if (i < k_rem) v[i] = f32_to_bf16_bits_short(base_f32[k_off + i]);
  }
  return v;
}

__device__ __forceinline__ s16x4 pack4_bf16_from_bf16_guard(
    const bf16_t* base_bf16, int k_off, int k_rem, bool col_valid) {
  s16x4 v = {0,0,0,0};
  if (!col_valid) return v;
  #pragma unroll
  for (int i=0;i<4;i++) {
    if (i < k_rem) v[i] = base_bf16[k_off + i].data;
  }
  return v;
}

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1) {
  union { uint32_t u; float f; } a, b;
  a.u = (u & 0x0000FFFFu) << 16; // lower bf16 -> fp32
  b.u = (u & 0xFFFF0000u);       // upper bf16 -> fp32
  f0 = a.f;
  f1 = b.f;
}

__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u) {
  float4 r;
  bf16pair_to_float2(u.x, r.x, r.y);
  bf16pair_to_float2(u.y, r.z, r.w);
  return r;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    v += __shfl_down(v, off, WF_SIZE);
  }
  return v;
}

__global__ void copy_embedding_bf16_batch_kernel(float *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 int batch_size,
                                                 int hidden_dim) {
  int batch_idx = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx < batch_size && i < hidden_dim && tokens[batch_idx] >= 0) {
    int token = tokens[batch_idx];
    dst[(size_t)batch_idx * hidden_dim + i] =
        static_cast<float>(src[(size_t)token * hidden_dim + i]);
  }
}

// Batched: grid.y is batch, uses d_pos to compute offset per sample
__global__ void split_qkv_scatter_to_cache_batch_kernel(
    float *q, float *key_cache, float *value_cache, const float *qkv,
    int n_attn_heads, int n_kv_heads, int head_dim, int layer_offset,
    const int *pos, int batch_size, int kv_stride) {
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int pos_b = pos[b];
  if (pos_b < 0)
    return;

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_size = n_attn_heads * head_dim;
  const int kv_size = n_kv_heads * head_dim;
  const int total = q_size + 2 * kv_size;
  if (idx >= total)
    return;

  float *q_b = q + (size_t)b * q_size;
  float *kcache_b = key_cache + (size_t)b * kv_stride;
  float *vcache_b = value_cache + (size_t)b * kv_stride;
  const float *qkv_b = qkv + (size_t)b * total;

  const int pos_offset = pos_b * kv_size;
  if (idx < q_size) {
    q_b[idx] = qkv_b[idx];
  } else if (idx < q_size + kv_size) {
    int k_idx = idx - q_size;
    kcache_b[layer_offset + pos_offset + k_idx] = qkv_b[idx];
  } else {
    int v_idx = idx - q_size - kv_size;
    vcache_b[layer_offset + pos_offset + v_idx] = qkv_b[idx];
  }
}

// Batched variant reading pos[b] and scattering per batch
__global__ void fused_inline_rope_qkv_batch_kernel(
    float *q, float *k_cache, const int *pos, float theta, int Hq, int Hk,
    int D, float rope_scaling_factor, int initial_context_length, int loff,
    int kv_total_size, int batch_size) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int half = D >> 1;
  if (i >= half)
    return;

  const int pos_b = pos[b];
  if (pos_b < 0)
    return;

  float freq = powf(theta, (float)(2 * i) / (float)D);
  float inv_freq;
  float concentration = 1.0f;

  if (rope_scaling_factor > 1.0f) {
    concentration = 0.1f * logf(rope_scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                logf(theta);
    float high = half *
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(theta);
    float interpolation = 1.0f / (rope_scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    ramp = fmaxf(0.0f, fminf(1.0f, ramp));
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }

  float val = pos_b * inv_freq;
  float c = cosf(val) * concentration;
  float s = sinf(val) * concentration;

  // Apply to Q for this batch
  if (h < Hq) {
    float *q_b = q + (size_t)b * Hq * D;
    float x1 = q_b[h * D + i];
    float x2 = q_b[h * D + half + i];
    q_b[h * D + i] = x1 * c - x2 * s;
    q_b[h * D + half + i] = x2 * c + x1 * s;
  }

  if (h < Hk) {
    const int KV = Hk * D;
    const int cache_idx =
        (size_t)b * kv_total_size + loff + pos_b * KV + h * D;
    float x1 = k_cache[cache_idx + i];
    float x2 = k_cache[cache_idx + half + i];
    k_cache[cache_idx + i] = x1 * c - x2 * s;
    k_cache[cache_idx + half + i] = x2 * c + x1 * s;
  }
}

__global__ void residual_add_batch_kernel(float *x, const float *residual,
                                          int size,
                                          int batch_size,
                                          const int *pos) {
  const int b = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;
  if (pos && pos[b] < 0)
    return;
  if (i < size) {
    x[(size_t)b * size + i] += residual[(size_t)b * size + i];
  }
}

__global__ void rmsnorm_batch_kernel(float *o, const float *x,
                                     const float *weight, int size,
                                     const int *pos) {
  const int b = blockIdx.y;

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  if (pos && pos[b] < 0)
    return;

  const float *x_b = x + (size_t)b * size;
  float *o_b = o + (size_t)b * size;

  // Vectorized sum of squares using float4
  float sum = 0.0f;
  const int size4 = size >> 2;
  
  // Process float4 chunks
  for (int i = tid; i < size4; i += blockDim.x) {
    float4 v = reinterpret_cast<const float4*>(x_b)[i];
    sum = fmaf(v.x, v.x, sum);
    sum = fmaf(v.y, v.y, sum);
    sum = fmaf(v.z, v.z, sum);
    sum = fmaf(v.w, v.w, sum);
  }
  
  // Handle remaining elements
  for (int i = (size4 << 2) + tid; i < size; i += blockDim.x) {
    float v = x_b[i];
    sum = fmaf(v, v, sum);
  }

  // Warp reduction
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    sum += __shfl_down(sum, off, WF_SIZE);
  }
  
  __shared__ float warp_sums[BLOCK_SIZE / WF_SIZE];
  if (lane == 0)
    warp_sums[wid] = sum;
  __syncthreads();

  // Block reduction
  float total = 0.0f;
  if (tid < BLOCK_SIZE / WF_SIZE)
    total = warp_sums[tid];

  if (wid == 0) {
    float t = (tid < BLOCK_SIZE / WF_SIZE) ? total : 0.0f;
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      t += __shfl_down(t, off, WF_SIZE);
    }
    if (lane == 0)
      warp_sums[0] = t;
  }
  __syncthreads();

  // Use rsqrt directly as in Python reference
  const float mean_sq = warp_sums[0] / (float)size;
  const float inv_rms = rsqrtf(mean_sq + 1e-5f);
  
  // Vectorized output computation
  for (int i = tid; i < size4; i += blockDim.x) {
    float4 v = reinterpret_cast<const float4*>(x_b)[i];
    float4 w = reinterpret_cast<const float4*>(weight)[i];
    float4 result;
    result.x = w.x * (v.x * inv_rms);
    result.y = w.y * (v.y * inv_rms);
    result.z = w.z * (v.z * inv_rms);
    result.w = w.w * (v.w * inv_rms);
    reinterpret_cast<float4*>(o_b)[i] = result;
  }
  
  // Handle remaining elements
  for (int i = (size4 << 2) + tid; i < size; i += blockDim.x) {
    o_b[i] = weight[i] * (x_b[i] * inv_rms);
  }
}

/**
 * Compute inv RMS per sample: inv_rms = rsqrt(mean(x^2)+eps)
 * Optimized with vectorization
 */
 __global__ void compute_inv_rms_batch_kernel(
  float* __restrict__ out_inv,
  const float* __restrict__ x,
  int H, int batch_size) {

const int b = blockIdx.y;
if (b >= batch_size) return;

if (threadIdx.x == 0) out_inv[b] = 0.0f; return;

const float* xb = x + (size_t)b * H;
const int H4 = H >> 2;

// Vectorized sum of squares
float sum = 0.0f;
for (int i = threadIdx.x; i < H4; i += blockDim.x) {
  float4 v = reinterpret_cast<const float4*>(xb)[i];
  sum = fmaf(v.x, v.x, sum);
  sum = fmaf(v.y, v.y, sum);
  sum = fmaf(v.z, v.z, sum);
  sum = fmaf(v.w, v.w, sum);
}
// Handle remainder
for (int i = (H4 << 2) + threadIdx.x; i < H; i += blockDim.x) {
  float v = xb[i];
  sum = fmaf(v, v, sum);
}

sum = warp_reduce_sum(sum);

__shared__ float warp_sums[1024 / WF_SIZE];
const int lane = threadIdx.x & (WF_SIZE - 1);
const int wid  = threadIdx.x >> 6;

if (lane == 0) warp_sums[wid] = sum;
__syncthreads();

float total = 0.0f;
if (wid == 0) {
  const int num_warps = blockDim.x / WF_SIZE;
  total = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
  total = warp_reduce_sum(total);
  if (lane == 0) {
    float mean_sq = total / (float)H;
    out_inv[b] = rsqrtf(mean_sq + 1e-5f);
  }
}
}

// Batched Top-K + Softmax kernel
__global__ void fused_topk_softmax_batch_kernel(
    float *topk_values, int *topk_indices, float *router_score,
    int E, int K, int batch_size, const int *pos) {
  extern __shared__ float smem[];
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;
  if (pos && pos[b] < 0)
    return;

  float *scores = smem;
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  float *router_score_b = router_score + (size_t)b * E;
  float *topk_values_b = topk_values + (size_t)b * K;
  int *topk_indices_b = topk_indices + (size_t)b * K;

  for (int i = tid; i < E; i += blockDim.x) {
    scores[i] = router_score_b[i];
  }
  __syncthreads();

  __shared__ float warp_vals[BLOCK_SIZE / WF_SIZE];
  __shared__ int warp_idxs[BLOCK_SIZE / WF_SIZE];

  // Step 1: Top-K selection
  for (int k = 0; k < K; k++) {
    float local_best = -INFINITY;
    int local_idx = -1;
    for (int i = tid; i < E; i += blockDim.x) {
      float v = scores[i];
      if (v > local_best) {
        local_best = v;
        local_idx = i;
      }
    }
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      float oVal = __shfl_down(local_best, off, WF_SIZE);
      int oIdx = __shfl_down(local_idx, off, WF_SIZE);
      if (oVal > local_best) {
        local_best = oVal;
        local_idx = oIdx;
      }
    }
    if (lane == 0) {
      warp_vals[wid] = local_best;
      warp_idxs[wid] = local_idx;
    }
    __syncthreads();
    if (wid == 0) {
      float bVal = (lane < BLOCK_SIZE / WF_SIZE) ? warp_vals[lane] : -INFINITY;
      int bIdx = (lane < BLOCK_SIZE / WF_SIZE) ? warp_idxs[lane] : -1;
#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
        float oVal = __shfl_down(bVal, off, WF_SIZE);
        int oIdx = __shfl_down(bIdx, off, WF_SIZE);
        if (oVal > bVal) {
          bVal = oVal;
          bIdx = oIdx;
        }
      }
      if (lane == 0) {
        topk_values_b[k] = bVal;
        topk_indices_b[k] = bIdx;
        if (bIdx >= 0)
          scores[bIdx] = -INFINITY;
      }
    }
    __syncthreads();
  }

  // Step 2: Fused Softmax on top-k values with fast intrinsics
  if (tid == 0) {
    float max_val = topk_values_b[0];
    for (int i = 1; i < K; i++)
      max_val = fmaxf(max_val, topk_values_b[i]);

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      float ev = __expf(topk_values_b[i] - max_val); // Use fast exp
      topk_values_b[i] = ev;
      sum += ev;
    }

    float inv_sum = __frcp_rn(sum); // Use fast reciprocal
    for (int i = 0; i < K; i++)
      topk_values_b[i] *= inv_sum;
  }
}

void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                               int n_streams, size_t chunk_bytes) {
  if (n == 0)
    return;

  const size_t chunk_elems = chunk_bytes / sizeof(bf16_t);
  const size_t actual_chunk_elems = (chunk_elems > n) ? n : chunk_elems;

  hipStream_t *streams = nullptr;
  hipEvent_t *events = nullptr;
  bf16_t **pinned_chunks = nullptr;
  bool async_success = true;

  streams = (hipStream_t *)malloc(n_streams * sizeof(hipStream_t));
  if (!streams)
    async_success = false;
  if (async_success) {
    events = (hipEvent_t *)malloc(n_streams * sizeof(hipEvent_t));
    if (!events)
      async_success = false;
  }
  if (async_success) {
    pinned_chunks = (bf16_t **)malloc(n_streams * sizeof(bf16_t *));
    if (!pinned_chunks)
      async_success = false;
    else {
      for (int i = 0; i < n_streams; i++)
        pinned_chunks[i] = nullptr;
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err =
          hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err =
          hipEventCreateWithFlags(&events[i], hipEventDisableTiming);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipHostMalloc((void **)&pinned_chunks[i],
                                     actual_chunk_elems * sizeof(bf16_t));
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }

  if (!async_success) {
    if (pinned_chunks) {
      for (int i = 0; i < n_streams; i++) {
        if (pinned_chunks[i]) {
          (void)hipHostFree(pinned_chunks[i]);
        }
      }
      free(pinned_chunks);
    }
    if (events) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipEventDestroy(events[i]);
      }
      free(events);
    }
    if (streams) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipStreamDestroy(streams[i]);
      }
      free(streams);
    }

    const size_t SYNC_CHUNK_ELEMS = 8 * 1024 * 1024;
    bf16_t *h_chunk = nullptr;
    HIP_CHECK(
        hipHostMalloc((void **)&h_chunk, SYNC_CHUNK_ELEMS * sizeof(bf16_t)));
    size_t done = 0;
    while (done < n) {
      size_t todo =
          (n - done > SYNC_CHUNK_ELEMS) ? SYNC_CHUNK_ELEMS : (n - done);
      for (size_t i = 0; i < todo; ++i)
        h_chunk[i] = hip_bfloat16(src[done + i]);
      HIP_CHECK(hipMemcpy(dst + done, h_chunk, todo * sizeof(bf16_t),
                          hipMemcpyHostToDevice));
      done += todo;
    }
    HIP_CHECK(hipHostFree(h_chunk));
    return;
  }

  size_t done = 0;
  int stream_idx = 0;
  bool *buffer_ready = (bool *)calloc(n_streams, sizeof(bool));
  for (int i = 0; i < n_streams; i++)
    buffer_ready[i] = true;

  while (done < n) {
    size_t todo = (n - done > actual_chunk_elems) ? actual_chunk_elems
                                                      : (n - done);
    if (!buffer_ready[stream_idx]) {
      HIP_CHECK(hipEventSynchronize(events[stream_idx]));
      buffer_ready[stream_idx] = true;
    }
    bf16_t *chunk = pinned_chunks[stream_idx];
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < todo; ++i)
      chunk[i] = hip_bfloat16(src[done + i]);

    HIP_CHECK(hipMemcpyAsync(dst + done, chunk, todo * sizeof(bf16_t),
                             hipMemcpyHostToDevice, streams[stream_idx]));
    HIP_CHECK(hipEventRecord(events[stream_idx], streams[stream_idx]));
    buffer_ready[stream_idx] = false;

    done += todo;
    stream_idx = (stream_idx + 1) % n_streams;
  }
  for (int i = 0; i < n_streams; i++)
    HIP_CHECK(hipEventSynchronize(events[i]));

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
