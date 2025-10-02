#include "../common/defines.h"
#include "utility.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstring>

void debug_print_gpu_memory(const char *tag, int device_id) {
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

__global__ void copy_embedding_bf16_batch_kernel(bf16_t *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 int batch_size,
                                                 int hidden_dim) {
  int batch_idx = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx < batch_size && i < hidden_dim && tokens[batch_idx] >= 0) {
    int token = tokens[batch_idx];
    dst[(size_t)batch_idx * hidden_dim + i] =
        src[(size_t)token * hidden_dim + i];
  }
}

__global__ void fused_split_rope_scatter_qkv_batch_kernel(
    bf16_t* __restrict__ key_cache,
    bf16_t* __restrict__ value_cache,
    const bf16_t* __restrict__ qkv,    // [B, Hq*D + 2*Hk*D]
    const int* __restrict__ pos,       // [B]
    // model params
    int Hq, int Hk, int D,
    // RoPE params
    const float* __restrict__ rope_inv_freq,
    float rope_concentration,
    // cache params
    int layer_idx,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity,
    uint32_t kv_batch_stride,
    int batch_size)
{

    const int h = blockIdx.x;           // head idx
    const int b = blockIdx.y;           // batch idx
    const int lane = threadIdx.x & (WF_SIZE - 1);
    const int half = D >> 1;

    if (b >= batch_size || half <= 0 || !rope_inv_freq) return;

    if (h >= Hk) return;

    const int pos_b = pos[b];
    if (pos_b < 0) return;

    // sizes
    const int q_size  = Hq * D;
    const int kv_size = Hk * D;
    const int KV      = kv_size;

    const int cap = layer_capacity[layer_idx];
    if (cap <= 0) return;
    const uint32_t layer_base = layer_offsets[layer_idx];
    const int slot = pos_b % cap;

    // base pointers per batch
    bf16_t* __restrict__ kcache_b = key_cache   + (size_t)b * kv_batch_stride;
    bf16_t* __restrict__ vcache_b = value_cache + (size_t)b * kv_batch_stride;
    const bf16_t* __restrict__ qkv_b = qkv + (size_t)b * (q_size + 2 * kv_size);

    const float concentration = rope_concentration;

    const size_t kc_idx_base = (size_t)layer_base + (size_t)slot * KV +
                               (size_t)h * D;
    bf16_t* __restrict__ kcache_head = kcache_b + kc_idx_base;
    bf16_t* __restrict__ vcache_head = vcache_b + kc_idx_base;
    const int k_off_qkv = q_size + h * D;           // K starts after Q
    const int v_off_qkv = q_size + kv_size + h * D; // V starts after K

    for (int i = lane; i < half; i += WF_SIZE) {
        const float inv_freq = rope_inv_freq[i];
        const float ang = pos_b * inv_freq;
        float s_val, c_val;
        sincosf(ang, &s_val, &c_val);
        c_val *= concentration;
        s_val *= concentration;

        // ---- K: read from qkv, apply RoPE, scatter to key_cache
        const float k1 = static_cast<float>(qkv_b[k_off_qkv + i]);
        const float k2 = static_cast<float>(qkv_b[k_off_qkv + half + i]);
        const float rk1 = fmaf(-k2, s_val, k1 * c_val);
        const float rk2 = fmaf(k1, s_val, k2 * c_val);
        kcache_head[i]        = hip_bfloat16(rk1);
        kcache_head[half + i] = hip_bfloat16(rk2);

        // ---- V: read from qkv, direct scatter (no RoPE)
        const float v1 = static_cast<float>(qkv_b[v_off_qkv + i]);
        const float v2 = static_cast<float>(qkv_b[v_off_qkv + half + i]);
        vcache_head[i]        = hip_bfloat16(v1);
        vcache_head[half + i] = hip_bfloat16(v2);
    }

    if ((D & 1) && lane == 0) {
        const int tail = D - 1;
        kcache_head[tail] = qkv_b[k_off_qkv + tail];
        vcache_head[tail] = qkv_b[v_off_qkv + tail];
    }
}

__global__ void residual_add_batch_kernel(bf16_t *x, const float *residual,
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
    const size_t offset = (size_t)b * size + i;
    float acc = static_cast<float>(x[offset]) + residual[offset];
    x[offset] = hip_bfloat16(acc);
  }
}

__global__ void residual_add_batch_kernel_bf16(bf16_t *x,
                                               const bf16_t *residual,
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
    const size_t offset = (size_t)b * size + i;
    const float acc = static_cast<float>(x[offset]) +
                      static_cast<float>(residual[offset]);
    x[offset] = hip_bfloat16(acc);
  }
}

__global__ void rmsnorm_batch_kernel(bf16_t *o, const bf16_t *x,
                                     const bf16_t *weight, int size,
                                     const int *pos) {
  const int b = blockIdx.y;

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  if (pos && pos[b] < 0)
    return;

  const bf16_t *x_b = x + (size_t)b * size;
  bf16_t *o_b = o + (size_t)b * size;

  const int pair_elems = size >> 1;
  const int tail_start = pair_elems << 1;
  const uint32_t *x_pairs = reinterpret_cast<const uint32_t *>(x_b);
  const uint32_t *w_pairs = reinterpret_cast<const uint32_t *>(weight);

  float sum = 0.0f;
  for (int idx = tid; idx < pair_elems; idx += blockDim.x) {
    float v0, v1;
    bf16pair_to_float2(x_pairs[idx], v0, v1);
    sum = fmaf(v0, v0, sum);
    sum = fmaf(v1, v1, sum);
  }
  for (int idx = tail_start + tid; idx < size; idx += blockDim.x) {
    float v = float(x_b[idx]);
    sum = fmaf(v, v, sum);
  }

#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    sum += __shfl_down(sum, off, WF_SIZE);
  }

  __shared__ float warp_sums[BLOCK_SIZE / WF_SIZE];
  if (lane == 0)
    warp_sums[wid] = sum;
  __syncthreads();

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

  const float mean_sq = warp_sums[0] / (float)size;
  const float inv_rms = rsqrtf(mean_sq + 1e-5f);

  for (int idx = tid; idx < pair_elems; idx += blockDim.x) {
    float v0, v1;
    bf16pair_to_float2(x_pairs[idx], v0, v1);
    float w0, w1;
    bf16pair_to_float2(w_pairs[idx], w0, w1);

    const float base = inv_rms;
    const int out_idx = idx << 1;
    o_b[out_idx] = bf16_t(w0 * (v0 * base));
    o_b[out_idx + 1] = bf16_t(w1 * (v1 * base));
  }

  for (int idx = tail_start + tid; idx < size; idx += blockDim.x) {
    float v = float(x_b[idx]);
    float w = float(weight[idx]);
    o_b[idx] = bf16_t(w * (v * inv_rms));
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

__global__ void argmax_batch_kernel(const float *logits, int *out_indices,
                                    int vocab_size, int batch_size,
                                    const int *pos) {
  const int b = blockIdx.x;
  if (b >= batch_size)
    return;
  if (pos && pos[b] < 0) {
    if (threadIdx.x == 0)
      out_indices[b] = -1;
    return;
  }

  const float *row = logits + (size_t)b * vocab_size;
  const int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = -1;
  for (int idx = tid; idx < vocab_size; idx += blockDim.x) {
    float val = row[idx];
    if (val > local_max) {
      local_max = val;
      local_idx = idx;
    }
  }

  extern __shared__ unsigned char smem_argmax[];
  float *s_vals = reinterpret_cast<float *>(smem_argmax);
  int *s_idx = reinterpret_cast<int *>(s_vals + blockDim.x);
  s_vals[tid] = local_max;
  s_idx[tid] = local_idx;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      float other_val = s_vals[tid + offset];
      int other_idx = s_idx[tid + offset];
      if (other_val > s_vals[tid] ||
          (other_val == s_vals[tid] && other_idx < s_idx[tid])) {
        s_vals[tid] = other_val;
        s_idx[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_indices[b] = s_idx[0];
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

size_t matmul_packed_elems(int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    return 0;
  const size_t tiles_cols =
      ((size_t)rows + MATMUL_TILE_COLS - 1) / MATMUL_TILE_COLS;
  const size_t tiles_k =
      ((size_t)cols + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  return tiles_cols * tiles_k * (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
}

void pack_fp32_to_bf16_matmul(const float *src, int rows, int cols,
                              bf16_t *dst) {
  if (!src || !dst || rows <= 0 || cols <= 0)
    return;

  const size_t tiles_cols =
      ((size_t)rows + MATMUL_TILE_COLS - 1) / MATMUL_TILE_COLS;
  const size_t tiles_k =
      ((size_t)cols + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;

  std::vector<bf16_t> tile(tile_elems, bf16_t(0.0f));

  for (size_t tile_col = 0; tile_col < tiles_cols; ++tile_col) {
    const int col_base = (int)(tile_col * MATMUL_TILE_COLS);
    const int col_block = std::min(rows - col_base, MATMUL_TILE_COLS);

    for (size_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
      const int k_base = (int)(tile_k * MATMUL_TILE_K);
      const int k_block = std::min(cols - k_base, MATMUL_TILE_K);

      std::fill(tile.begin(), tile.end(), bf16_t(0.0f));

      if (col_block > 0 && k_block > 0) {
        for (int group = 0; group < MATMUL_TILE_K; group += MATMUL_CHUNK_K) {
          const int k_off = k_base + group;
          const int remaining = k_block - group;
          if (remaining <= 0)
            break;
          const int actual_chunk = remaining > MATMUL_CHUNK_K ? MATMUL_CHUNK_K
                                                               : remaining;
          const size_t group_base =
              (size_t)(group / MATMUL_CHUNK_K) * group_stride;

          for (int col = 0; col < col_block; ++col) {
            const float *src_row =
                src + (size_t)(col_base + col) * cols + k_off;
            const size_t dst_base =
                group_base + (size_t)col * MATMUL_CHUNK_K;

            for (int i = 0; i < actual_chunk; ++i) {
              tile[dst_base + i] = hip_bfloat16(src_row[i]);
            }
          }
        }
      }

      const size_t tile_offset =
          (tile_col * tiles_k + tile_k) * tile_elems;
      std::memcpy(dst + tile_offset, tile.data(),
                  tile_elems * sizeof(bf16_t));
    }
  }
}

// Accumulate compact partials [cnt, H] into dest [B, H] using batch_ids[cnt]
__global__ void accumulate_partials_kernel(
    float* __restrict__ dest,
    const float* __restrict__ src,
    const int* __restrict__ batch_ids,
    int H,
    int cnt) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  if (i >= cnt || h >= H) return;
  const int b = batch_ids[i];
  if (b < 0) return;
  const float val = src[(size_t)i * (size_t)H + h];
  if (val != 0.0f) {
    atomicAdd(dest + (size_t)b * (size_t)H + h, val);
  }
}

__global__ void zero_partial_rows_kernel(
    float* __restrict__ dst,
    int H,
    int rows) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= rows || h >= H) return;
  dst[(size_t)b * (size_t)H + h] = 0.0f;
}

// ================= Device-side bucketization & packing (MoE routing) ================
__global__ void route_count_owner_kernel(
    int* __restrict__ expert_counts,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ e2lid_owner_l,
    int B,
    int K,
    int E) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  if (pos[b] < 0) return;
  for (int k = 0; k < K; ++k) {
    const int e = topk_i[(size_t)b * K + k];
    if ((unsigned)e >= (unsigned)E) continue;
    const int lid = e2lid_owner_l[e];
    if (lid >= 0) atomicAdd(expert_counts + lid, 1);
  }
}

__global__ void exclusive_scan_small_kernel(
    const int* __restrict__ counts,
    int* __restrict__ offsets,
    int n) {
  // Single block scan using shared memory; n is small (<= E_local <= E)
  extern __shared__ int s[];
  int tid = threadIdx.x;
  if (tid < n) s[tid] = counts[tid];
  if (tid == 0) offsets[0] = 0;
  __syncthreads();
  // Inclusive scan on s -> then shift right to exclusive in offsets
  for (int stride = 1; stride < n; stride <<= 1) {
    int val = 0;
    if (tid >= stride && tid < n) val = s[tid - stride];
    __syncthreads();
    if (tid < n) s[tid] += val;
    __syncthreads();
  }
  if (tid < n) offsets[tid + 1] = s[tid];
}

__global__ void route_pack_owner_kernel(
    int* __restrict__ b2local,
    int* __restrict__ local2b,
    int* __restrict__ owner_B,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_writes,
    uint16_t* __restrict__ assignment_batches,
    uint8_t* __restrict__ assignment_slots,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ e2lid_owner_l,
    int B,
    int K,
    int E) {
  // One thread per token-row; uses atomics for local batch map and per-expert writes.
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  if (pos[b] < 0) return;

  // Get or assign local batch id
  int lb = atomicCAS(&b2local[b], -1, -2); // try reserve
  if (lb == -1) {
    // We are the first to claim; assign new local id
    int new_id = atomicAdd(owner_B, 1);
    local2b[new_id] = b;
    __threadfence();
    b2local[b] = new_id;
    lb = new_id;
  } else {
    // If someone else already processing, spin-wait until resolved if needed
    if (lb == -2) {
      // wait until set to real id
      int v;
      do { v = b2local[b]; } while (v == -2);
      lb = v;
    }
  }

  // Write assignments
  for (int k = 0; k < K; ++k) {
    const int e = topk_i[(size_t)b * K + k];
    if ((unsigned)e >= (unsigned)E) continue;
    const int lid = e2lid_owner_l[e];
    if (lid < 0) continue;
    const int off = atomicAdd(expert_writes + lid, 1);
    const int idx = expert_offsets[lid] + off;
    assignment_batches[idx] = lb;
    assignment_slots[idx] = k;
  }
}

__global__ void pack_rows_owner_kernel(
    bf16_t* __restrict__ dst,
    const bf16_t* __restrict__ src,
    const int* __restrict__ b2local,
    int B,
    int H) {
  const int b = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || i >= H) return;
  const int lb = b2local[b];
  if (lb < 0) return;
  dst[(size_t)lb * H + i] = src[(size_t)b * H + i];
}

__global__ void pack_meta_owner_kernel(
    int* __restrict__ pos_owner,
    float* __restrict__ topk_v_owner,
    const int* __restrict__ local2b,
    int owner_B,
    const int* __restrict__ pos,
    const float* __restrict__ topk_v,
    int K) {
  const int lb = blockIdx.x * blockDim.x + threadIdx.x;
  if (lb >= owner_B) return;
  const int b = local2b[lb];
  if (b < 0) return;
  pos_owner[lb] = pos[b];
  for (int k = 0; k < K; ++k) {
    topk_v_owner[(size_t)lb * K + k] = topk_v[(size_t)b * K + k];
  }
}

__global__ void reset_owner_mappings_kernel(
    int* __restrict__ b2local,
    int* __restrict__ local2b,
    int prev_owner_B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= prev_owner_B) return;
  const int b = local2b[idx];
  if (b >= 0) {
    b2local[b] = -1;
  }
  local2b[idx] = -1;
}

// Matmul Helper Functions
__device__ __forceinline__ s16x4 load_bf16x4(const uint16_t* src, int valid_elems) {
  s16x4 out = {0, 0, 0, 0};
  if (!src || valid_elems <= 0) {
    return out;
  }

  if (valid_elems >= MATMUL_CHUNK_K) {
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    const uint32_t packed0 = src32[0];
    const uint32_t packed1 = src32[1];
    out[0] = static_cast<short>(packed0 & 0xFFFF);
    out[1] = static_cast<short>(packed0 >> 16);
    out[2] = static_cast<short>(packed1 & 0xFFFF);
    out[3] = static_cast<short>(packed1 >> 16);
    return out;
  }

#pragma unroll
  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    out[i] = (i < valid_elems) ? static_cast<short>(src[i]) : 0;
  }
  return out;
}


// Expert Assignment Kernels
__global__ void count_expert_assignments_kernel(
    int* __restrict__ counts,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int batch_size,
    int experts_per_token,
    int E)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * experts_per_token;
  if (idx >= total) return;

  const int batch_idx = idx / experts_per_token;
  if (pos && pos[batch_idx] < 0) return;

  const int expert = topk_i[idx];
  if (expert < 0 || expert >= E) return;

  atomicAdd(counts + expert, 1);
}

__global__ void build_expert_assignments_kernel(
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_counters,
    uint16_t* __restrict__ assignment_batches,
    uint8_t* __restrict__ assignment_slots,
    int batch_size,
    int experts_per_token,
    int E)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * experts_per_token;
  if (idx >= total) return;

  const int batch_idx = idx / experts_per_token;
  if (pos && pos[batch_idx] < 0) return;

  const int expert = topk_i[idx];
  if (expert < 0 || expert >= E) return;

  const int slot = idx % experts_per_token;
  const int offset = atomicAdd(expert_counters + expert, 1);
  const int write_idx = expert_offsets[expert] + offset;

  assignment_batches[write_idx] = static_cast<uint16_t>(batch_idx);
  assignment_slots[write_idx] = static_cast<uint8_t>(slot);
}

// Activation Functions
__device__ __forceinline__ float clamp_with_limit(float v, float limit) {
  if (limit <= 0.0f)
    return v;
  const float inv_range = 1.0f / (2.0f * limit);
  const float normalized = __saturatef((v + limit) * inv_range);
  return normalized * (2.0f * limit) - limit;
}

__device__ __forceinline__ float swiglu_fused(float gate, float up,
                                              float limit) {
  const float gate_val = clamp_with_limit(gate, limit);
  const float up_val = clamp_with_limit(up, limit);
  const float alpha = 1.702f;
  const float gate_act = gate_val *
      __saturatef(0.5f + 0.5f * tanhf(alpha * gate_val * 0.5f));
  return gate_act * (up_val + 1.0f);
}
