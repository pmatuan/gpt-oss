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

// ===== MoE bucketing kernels =====
__global__ void moe_count_assignments_kernel(const int *topk_i, const int *pos,
                                             int B, int K, int E,
                                             int *counts) {
  int b = blockIdx.z;
  if (b >= B || (pos && pos[b] < 0)) return;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  int e = topk_i[(size_t)b * K + k];
  if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
}

__global__ void moe_fill_indices_kernel(const int *topk_i, const int *pos,
                                        int B, int K, int E, int *offsets,
                                        const int *indptr, int *assign_b,
                                        int *assign_k) {
  int b = blockIdx.z;
  if (b >= B || (pos && pos[b] < 0)) return;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  int e = topk_i[(size_t)b * K + k];
  if (e < 0 || e >= E) return;
  int idx = atomicAdd(&offsets[e], 1);
  int out = indptr[e] + idx;
  assign_b[out] = b;
  assign_k[out] = k;
}

__global__ void moe_gather_gate_up_kernel(const float *gate_up, int B, int K,
                                          int IM, const int *assign_b,
                                          const int *assign_k, int start,
                                          int N, float *GU_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || m >= IM) return;
  int b = assign_b[start + i];
  int k = assign_k[start + i];
  const float *src = gate_up + ((size_t)b * K + (size_t)k) * IM;
  GU_out[(size_t)i * IM + m] = src[m];
}

__global__ void moe_scatter_mlp2_accum_kernel(const float *Z, int N, int H,
                                              const int *assign_b,
                                              const int *assign_k, int start,
                                              const float *topk_v, int K,
                                              float *x, const int *pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || h >= H) return;
  int b = assign_b[start + i];
  if (pos && pos[b] < 0) return;
  int k = assign_k[start + i];
  float w = topk_v[(size_t)b * K + k];
  float contrib = Z[(size_t)i * H + h] * w;
  atomicAdd(&x[(size_t)b * H + h], contrib);
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

__global__ void copy_embedding_bf16_kernel(float *dst, const bf16_t *src,
                                           const int *tokens,
                                           int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.z;

  if (i < hidden_dim) {
    int token = tokens[b];
    if (token >= 0) {
      float *dst_b = dst + (size_t)b * hidden_dim;
      dst_b[i] = static_cast<float>(src[(size_t)token * hidden_dim + i]);
    }
  }
}

// Optimized to scatter directly to cache without intermediate q buffer for single sample
__global__ void split_qkv_scatter_to_cache_kernel(
  float *q_temp, bf16_t *key_cache, bf16_t *value_cache, const float *qkv,
    int n_attn_heads, int n_kv_heads, int head_dim, int layer_offset,
    const int *pos, int kv_total_size) {
  const int b = blockIdx.z;
  const int pos_current = pos[b];
  if (pos_current < 0)
    return;

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_size = n_attn_heads * head_dim;
  const int kv_size = n_kv_heads * head_dim;
  const int total = q_size + 2 * kv_size;
  if (idx >= total)
    return;

  bf16_t *kcache = key_cache + (size_t)b * kv_total_size;
  bf16_t *vcache = value_cache + (size_t)b * kv_total_size;
  const float *qkv_data = qkv + (size_t)b * total;

  const int pos_offset = pos_current * kv_size;
  if (idx < q_size) {
    // Q left in-place in qkv buffer for later kernels
  } else if (idx < q_size + kv_size) {
    int k_idx = idx - q_size;
  // fp32 -> bf16
  kcache[layer_offset + pos_offset + k_idx] = static_cast<bf16_t>(qkv_data[idx]);
  } else {
    int v_idx = idx - q_size - kv_size;
  vcache[layer_offset + pos_offset + v_idx] = static_cast<bf16_t>(qkv_data[idx]);
  }
}

// Optimized to work with in-place QKV buffer for single sample
__global__ void fused_inline_rope_qkv_kernel(
  float *qkv, bf16_t *k_cache, const int *pos, float theta, int Hq, int Hk,
    int D, float rope_scaling_factor, int initial_context_length, int loff,
    int kv_total_size) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;
  const int b = blockIdx.z;

  const int half = D >> 1;
  if (i >= half)
    return;

  const int pos_current = pos[b];
  if (pos_current < 0)
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

  float val = pos_current * inv_freq;
  float c = cosf(val) * concentration;
  float s = sinf(val) * concentration;

  // Apply to Q in the QKV buffer for this sample b
  if (h < Hq) {
    const int q_size = Hq * D;
    const int kv_size = Hk * D;
    const int total = q_size + 2 * kv_size;
    float *qkv_data = qkv + (size_t)b * total;
    float x1 = qkv_data[h * D + i];
    float x2 = qkv_data[h * D + half + i];
    qkv_data[h * D + i] = x1 * c - x2 * s;
    qkv_data[h * D + half + i] = x2 * c + x1 * s;
  }

  if (h < Hk) {
    const int KV = Hk * D;
  bf16_t *k_b = k_cache + (size_t)b * kv_total_size;
    const int cache_idx = loff + pos_current * KV + h * D;
  // load bf16 -> fp32
  float x1 = static_cast<float>(k_b[cache_idx + i]);
  float x2 = static_cast<float>(k_b[cache_idx + half + i]);
  float y1 = x1 * c - x2 * s;
  float y2 = x2 * c + x1 * s;
  // store fp32 -> bf16
  k_b[cache_idx + i] = static_cast<bf16_t>(y1);
  k_b[cache_idx + half + i] = static_cast<bf16_t>(y2);
  }
}

__global__ void residual_add_kernel(float *x, const float *residual,
                                    int size, const int *pos) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.z;
  if (pos && pos[b] < 0) return;
  if (i < size) {
    float *xb = x + (size_t)b * size;
    const float *rb = residual + (size_t)b * size;
    xb[i] += rb[i];
  }
}

__global__ void rmsnorm_kernel(float *o, const float *x,
                               const float *weight, int size,
                               const int *pos) {
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  const int b = blockIdx.z;
  if (pos && pos[b] < 0) return;

  const float *x_data = x + (size_t)b * size;
  float *o_data = o + (size_t)b * size;

  // Vectorized sum of squares using float4
  float sum = 0.0f;
  const int size4 = size >> 2;
  
  // Process float4 chunks
  for (int i = tid; i < size4; i += blockDim.x) {
    float4 v = reinterpret_cast<const float4*>(x_data)[i];
    sum = fmaf(v.x, v.x, sum);
    sum = fmaf(v.y, v.y, sum);
    sum = fmaf(v.z, v.z, sum);
    sum = fmaf(v.w, v.w, sum);
  }
  
  // Handle remaining elements
  for (int i = (size4 << 2) + tid; i < size; i += blockDim.x) {
    float v = x_data[i];
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
    float4 v = reinterpret_cast<const float4*>(x_data)[i];
    float4 w = reinterpret_cast<const float4*>(weight)[i];
    float4 result;
    result.x = w.x * (v.x * inv_rms);
    result.y = w.y * (v.y * inv_rms);
    result.z = w.z * (v.z * inv_rms);
    result.w = w.w * (v.w * inv_rms);
    reinterpret_cast<float4*>(o_data)[i] = result;
  }
  
  // Handle remaining elements
  for (int i = (size4 << 2) + tid; i < size; i += blockDim.x) {
    o_data[i] = weight[i] * (x_data[i] * inv_rms);
  }
}

/**
 * Compute inv RMS per sample: inv_rms = rsqrt(mean(x^2)+eps)
 * Optimized with vectorization
 */
 __global__ void compute_inv_rms_kernel(
  float* __restrict__ out_inv,
  const float* __restrict__ x,
  int H) {

if (threadIdx.x == 0) out_inv[0] = 0.0f;

const float* x_data = x;
const int H4 = H >> 2;

// Vectorized sum of squares
float sum = 0.0f;
for (int i = threadIdx.x; i < H4; i += blockDim.x) {
  float4 v = reinterpret_cast<const float4*>(x_data)[i];
  sum = fmaf(v.x, v.x, sum);
  sum = fmaf(v.y, v.y, sum);
  sum = fmaf(v.z, v.z, sum);
  sum = fmaf(v.w, v.w, sum);
}
// Handle remainder
for (int i = (H4 << 2) + threadIdx.x; i < H; i += blockDim.x) {
  float v = x_data[i];
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
    out_inv[0] = rsqrtf(mean_sq + 1e-5f);
  }
}
}

// Top-K + Softmax kernel
__global__ void fused_topk_softmax_kernel(
    float *topk_values, int *topk_indices, float *router_score,
    int E, int K, const int *pos) {
  extern __shared__ float smem[];
  const int b = blockIdx.z;
  if (pos && pos[b] < 0) return;

  float *scores = smem;
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  float *router_score_data = router_score + (size_t)b * E;
  float *topk_values_data = topk_values + (size_t)b * K;
  int *topk_indices_data = topk_indices + (size_t)b * K;

  for (int i = tid; i < E; i += blockDim.x) {
    scores[i] = router_score_data[i];
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
        topk_values_data[k] = bVal;
        topk_indices_data[k] = bIdx;
        if (bIdx >= 0)
          scores[bIdx] = -INFINITY;
      }
    }
    __syncthreads();
  }

  // Step 2: Fused Softmax on top-k values with fast intrinsics
  if (tid == 0) {
    float max_val = topk_values_data[0];
    for (int i = 1; i < K; i++)
      max_val = fmaxf(max_val, topk_values_data[i]);

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      float ev = __expf(topk_values_data[i] - max_val); // Use fast exp
      topk_values_data[i] = ev;
      sum += ev;
    }

    float inv_sum = __frcp_rn(sum); // Use fast reciprocal
    for (int i = 0; i < K; i++)
      topk_values_data[i] *= inv_sum;
  }
}

// Greedy argmax over vocabulary per batch sample b.
// Grid: (ceil(V / BLOCK_SIZE), 1, B). One block per vocab tile; reduce via shared memory
__global__ void argmax_logits_kernel(const float *logits, int V,
                                     const int *pos, int *next_tokens) {
  const int b = blockIdx.z;
  if (pos && pos[b] < 0) return;

  // Each block scans a tile of vocab; we then use atomicMax on integer-packed value to select global best.
  // Represent (max_val, idx) as pair; use atomic on 64-bit encoded values.
  extern __shared__ unsigned long long sdata[]; // store encoded (val, idx) per thread or per warp

  const int tid = threadIdx.x;
  const int base = blockIdx.x * blockDim.x + tid;

  // Helper to pack float value and index into 64-bit for atomic compare: order by value then index
  auto pack = [](float val, int idx) {
    // Convert float to ordered 32-bit (flip sign bit to get increasing order)
    unsigned int vbits = __float_as_uint(val);
    unsigned int sign = vbits >> 31;
    vbits ^= (sign ? 0xffffffffu : 0x80000000u);
    return (unsigned long long)vbits << 32 | (unsigned int)idx;
  };

  // Use global scratch in next_tokens as accumulator not possible; use block 0 to init per-b value via first block
  // We'll use first block (x==0) to initialize a per-b global best using a separate buffer would be better,
  // but to keep changes minimal we do two-phase within block 0 only

  // Compute local best within this block tile
  float best_val = -INFINITY;
  int best_idx = -1;
  for (int i = base; i < V; i += gridDim.x * blockDim.x) {
    float v = logits[(size_t)b * V + i];
    if (i < V && v > best_val) { best_val = v; best_idx = i; }
  }

  // Reduce within block to find block best
  __shared__ float svals[BLOCK_SIZE];
  __shared__ int sidx[BLOCK_SIZE];
  svals[tid] = best_val;
  sidx[tid] = best_idx;
  __syncthreads();
  // parallel reduction
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (svals[tid + stride] > svals[tid]) {
        svals[tid] = svals[tid + stride];
        sidx[tid] = sidx[tid + stride];
      }
    }
    __syncthreads();
  }

  // Write block best into shared-memory array sdata[blockIdx.x] by thread 0
  if (tid == 0) {
    // reuse sdata as temporary per-block storage; only valid within this kernel launch
    sdata[blockIdx.x] = pack(svals[0], sidx[0]);
  }
  __syncthreads();

  // Block 0 does a second-level reduction across blocks to choose final best for batch b
  if (blockIdx.x == 0) {
    // Ensure all blocks have written sdata. We can't sync across blocks generally; however,
    // we rely on cooperative launch ordering within same grid not guaranteed. To keep correctness,
    // we enforce gridDim.x == 1 for now when calling this kernel. We'll call with a single block over V using BLOCK_SIZE stride.
    // So do nothing here; final best is already at svals[0]/sidx[0].
  }

  // With the constraint gridDim.x == 1, thread 0 writes out next token
  if (blockIdx.x == 0 && tid == 0) {
    next_tokens[b] = sidx[0];
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
