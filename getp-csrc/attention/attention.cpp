#include "../common/defines.h"
#include "../utility/utility.h"
#include "attention.h"
#include <math.h>

__device__ __forceinline__ float block_reduce_max(float thread_val,
                                                  float *warp_buffer,
                                                  int lane, int warp_id,
                                                  int num_warps) {
  for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
    thread_val = fmaxf(thread_val, __shfl_down(thread_val, offset, WF_SIZE));
  }

  if (lane == 0)
    warp_buffer[warp_id] = thread_val;
  __syncthreads();

  float block_val = -INFINITY;
  if (warp_id == 0) {
    block_val = (lane < num_warps) ? warp_buffer[lane] : -INFINITY;
    for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
      block_val = fmaxf(block_val, __shfl_down(block_val, offset, WF_SIZE));
    }
    if (lane == 0)
      warp_buffer[0] = block_val;
  }
  __syncthreads();

  return warp_buffer[0];
}

__device__ __forceinline__ float block_reduce_sum(float thread_val,
                                                  float *warp_buffer,
                                                  int lane, int warp_id,
                                                  int num_warps) {
  for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
    thread_val += __shfl_down(thread_val, offset, WF_SIZE);
  }

  if (lane == 0)
    warp_buffer[warp_id] = thread_val;
  __syncthreads();

  float block_val = 0.0f;
  if (warp_id == 0) {
    block_val = (lane < num_warps) ? warp_buffer[lane] : 0.0f;
    for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
      block_val += __shfl_down(block_val, offset, WF_SIZE);
    }
    if (lane == 0)
      warp_buffer[0] = block_val;
  }
  __syncthreads();

  return warp_buffer[0];
}

template <bool HAS_WINDOW>
__device__ void attention_batch_kernel_impl(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const float *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size, float *__restrict__ shared_mem,
    float *__restrict__ warp_buffer) {
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int head = blockIdx.x;
  if (head >= Hq)
    return;

  const int thread = threadIdx.x;
  const int lane = thread & (WF_SIZE - 1);
  const int warp_id = thread / WF_SIZE;
  const int num_threads = blockDim.x;
  const int num_warps = (num_threads + WF_SIZE - 1) / WF_SIZE;

  const int pos_b = pos[b];
  if (pos_b < 0)
    return;

  const int cap = layer_capacity[layer_idx];
  if (cap <= 0)
    return;

  if constexpr (HAS_WINDOW) {
    if (sliding_window <= 0)
      return;
  }

  const int kv_dim = Hk * D;
  const int kv_mul = Hq / Hk;
  const int kv_head = head / kv_mul;
  const int kv_head_offset = kv_head * D;
  const float rsqrt_D = rsqrtf(static_cast<float>(D));

  const bf16_t *__restrict__ q_b = q + (size_t)b * Hq * D;
  const uint32_t layer_base = layer_offsets[layer_idx];
  const bf16_t *__restrict__ k_layer =
      k_cache + (size_t)b * kv_batch_stride + layer_base;
  const bf16_t *__restrict__ v_layer =
      v_cache + (size_t)b * kv_batch_stride + layer_base;
  bf16_t *__restrict__ out_b = out_tb + (size_t)b * Hq * D;

  int start_t = 0;
  if constexpr (HAS_WINDOW) {
    start_t = pos_b + 1 - sliding_window;
    if (start_t < 0)
      start_t = 0;
  }
  const int att_tokens = pos_b - start_t + 1;
  if (att_tokens <= 0)
    return;

  const int att_size = att_tokens + 1;
  const int att_aligned = (att_size + 3) & ~3;
  float *__restrict__ s_att = shared_mem;
  float *__restrict__ s_q = shared_mem + att_aligned;

  const bf16_t *__restrict__ q_head_ptr = q_b + head * D;
  const uint32_t *__restrict__ q_head_u32 =
      reinterpret_cast<const uint32_t *>(q_head_ptr);
  const int D2 = D >> 1;

  for (int idx = thread; idx < D2; idx += num_threads) {
    float q0, q1;
    bf16pair_to_float2(q_head_u32[idx], q0, q1);
    const int base = idx << 1;
    s_q[base] = q0;
    s_q[base + 1] = q1;
  }
  if ((D & 1) && thread == 0) {
    s_q[D - 1] = static_cast<float>(q_head_ptr[D - 1]);
  }
  __syncthreads();

  const float *__restrict__ q_cache = s_q;
  float thread_max = -INFINITY;

  const int D4 = D >> 2;
  for (int local_t = thread; local_t < att_tokens; local_t += num_threads) {
    const int t = start_t + local_t;
    const int slot = t % cap;
    const size_t slot_offset = (size_t)slot * kv_dim + kv_head_offset;
    const bf16_t *__restrict__ k_ptr = k_layer + slot_offset;
    const uint2 *__restrict__ k_ptr_u2 =
        reinterpret_cast<const uint2 *>(k_ptr);

    float score = 0.0f;
#pragma unroll 4
    for (int d4 = 0; d4 < D4; ++d4) {
      float4 kv = bf16quad_to_float4(k_ptr_u2[d4]);
      const int base = d4 << 2;
      score = fmaf(q_cache[base + 0], kv.x, score);
      score = fmaf(q_cache[base + 1], kv.y, score);
      score = fmaf(q_cache[base + 2], kv.z, score);
      score = fmaf(q_cache[base + 3], kv.w, score);
    }
    for (int d = D4 << 2; d < D; ++d) {
      score = fmaf(q_cache[d], static_cast<float>(k_ptr[d]), score);
    }

    score *= rsqrt_D;
    s_att[local_t] = score;
    thread_max = fmaxf(thread_max, score);
  }
  __syncthreads();

  const float sink_score = attn_sinks[layer_idx * Hq + head];
  if (thread == 0)
    s_att[att_tokens] = sink_score;
  thread_max = fmaxf(thread_max, sink_score);
  __syncthreads();

  const float maxv = block_reduce_max(thread_max, warp_buffer, lane, warp_id,
                                      num_warps);

  float thread_sum = 0.0f;
  for (int idx = thread; idx < att_size; idx += num_threads) {
    float v = __expf(s_att[idx] - maxv);
    s_att[idx] = v;
    thread_sum += v;
  }
  __syncthreads();

  const float sum = block_reduce_sum(thread_sum, warp_buffer, lane, warp_id,
                                     num_warps);
  const float inv_sum = __frcp_rn(sum);

  for (int idx = thread; idx < att_size; idx += num_threads)
    s_att[idx] *= inv_sum;
  __syncthreads();

  const int threads_per_dim = min(num_threads, D);
  const size_t out_head_offset = (size_t)head * D;
  if (thread < threads_per_dim) {
    const int dims_per_thread = (D + threads_per_dim - 1) / threads_per_dim;
    const int start_dim = thread * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, D);

    for (int d = start_dim; d < end_dim; ++d) {
      float result = 0.0f;
      int local_t = 0;
      constexpr int TILE = 8;

      for (; local_t <= att_tokens - TILE; local_t += TILE) {
#pragma unroll
        for (int ti = 0; ti < TILE; ++ti) {
          const int t = start_t + local_t + ti;
          const int slot = t % cap;
          const size_t slot_offset = (size_t)slot * kv_dim + kv_head_offset;
          const bf16_t *__restrict__ v_ptr = v_layer + slot_offset;
          result = fmaf(s_att[local_t + ti],
                        static_cast<float>(v_ptr[d]), result);
        }
      }

      for (; local_t < att_tokens; ++local_t) {
        const int t = start_t + local_t;
        const int slot = t % cap;
        const size_t slot_offset = (size_t)slot * kv_dim + kv_head_offset;
        const bf16_t *__restrict__ v_ptr = v_layer + slot_offset;
        result = fmaf(s_att[local_t], static_cast<float>(v_ptr[d]), result);
      }

      out_b[out_head_offset + d] = hip_bfloat16(result);
    }
  }
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4)
__global__ void attention_batch_kernel_even(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const float *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size) {
  extern __shared__ float shared_mem[];
  __shared__ float warp_buffer[ATTN_WARPS_PER_BLOCK];

  attention_batch_kernel_impl<true>(out_tb, q, k_cache, v_cache, attn_sinks,
                                    layer_idx, pos, D, Hq, Hk, layer_offsets,
                                    layer_capacity, sliding_window,
                                    kv_batch_stride, batch_size, shared_mem,
                                    warp_buffer);
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4)
__global__ void attention_batch_kernel_odd(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const float *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, uint32_t kv_batch_stride,
    int batch_size) {
  extern __shared__ float shared_mem[];
  __shared__ float warp_buffer[ATTN_WARPS_PER_BLOCK];

  attention_batch_kernel_impl<false>(out_tb, q, k_cache, v_cache, attn_sinks,
                                     layer_idx, pos, D, Hq, Hk, layer_offsets,
                                     layer_capacity, /*sliding_window*/ 0,
                                     kv_batch_stride, batch_size, shared_mem,
                                     warp_buffer);
}
