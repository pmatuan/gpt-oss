#include "attention.h"
#include "../common/defines.h"
#include "../utility/utility.h"
#include <math.h>

__device__ __forceinline__ float block_reduce_max(float thread_val,
                                                  float *warp_buffer, int lane,
                                                  int warp_id, int num_warps) {
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
                                                  float *warp_buffer, int lane,
                                                  int warp_id, int num_warps) {
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
__device__ __forceinline__ void attention_flashdecode_mqa_kernel(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const bf16_t *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size, float *__restrict__ smem_dyn,
    float *__restrict__ warp_buffer) {

  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int kv_head = blockIdx.x;
  if (kv_head >= Hk)
    return;

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid / WF_SIZE;
  const int nwarp = max(1, (blockDim.x + WF_SIZE - 1) / WF_SIZE);

  const int pos_b = pos[b];
  if (pos_b < 0)
    return;

  const int cap = layer_capacity[layer_idx];
  if (cap <= 0)
    return;

  const int kv_mul = 8;
  if (kv_mul > ATTN_FLASH_MAX_KV_MUL)
    return;
  const int kv_dim = Hk * D;
  const int kv_head_offset = kv_head * D;
  const float rsqrt_D = rsqrtf(static_cast<float>(D));

  const uint32_t layer_base = layer_offsets[layer_idx];
  const bf16_t *__restrict__ k_layer =
      k_cache + (size_t)b * kv_batch_stride + layer_base;
  const bf16_t *__restrict__ v_layer =
      v_cache + (size_t)b * kv_batch_stride + layer_base;
  const bf16_t *__restrict__ q_b = q + (size_t)b * Hq * D;
  bf16_t *__restrict__ out_b = out_tb + (size_t)b * Hq * D;

  const int d_owner = tid % D;
  const int workers_this_dim = ((blockDim.x - 1 - d_owner) / D) + 1;
  const int worker_rank = tid / D;

  int start_t = 0;
  if constexpr (HAS_WINDOW) {
    start_t = pos_b + 1 - sliding_window;
    if (start_t < 0)
      start_t = 0;
  }
  const int att_tokens = pos_b - start_t + 1;
  if (att_tokens <= 0)
    return;

  const int score_stride = ATTN_FLASH_TILE + 1;
  float *s_scores = smem_dyn;
  float *s_q = s_scores + kv_mul * score_stride;
  float *s_accum = s_q + kv_mul * D;

  __shared__ float s_m[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_l[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_scale[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_tmp[ATTN_FLASH_MAX_KV_MUL];

  if (tid == 0) {
    for (int qh = 0; qh < kv_mul; ++qh) {
      s_m[qh] = -INFINITY;
      s_l[qh] = 0.0f;
    }
  }
  __syncthreads();

  const int D2 = D >> 1;
  for (int qh = 0; qh < kv_mul; ++qh) {
    const int q_head = kv_head * kv_mul + qh;
    const bf16_t *__restrict__ q_head_ptr = q_b + (size_t)q_head * D;
    const uint32_t *__restrict__ q_u32 =
        reinterpret_cast<const uint32_t *>(q_head_ptr);

    for (int i = tid; i < D2; i += blockDim.x) {
      float a, b;
      bf16pair_to_float2(q_u32[i], a, b);
      const int base = qh * D + (i << 1);
      s_q[base + 0] = a;
      s_q[base + 1] = b;
    }
    if ((D & 1) && tid == 0) {
      s_q[qh * D + (D - 1)] = static_cast<float>(q_head_ptr[D - 1]);
    }
  }
  __syncthreads();

  float acc[ATTN_FLASH_MAX_KV_MUL];
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
    acc[qh] = 0.0f;

  int done = 0;
  while (done < att_tokens) {
    const int tile_len = min(ATTN_FLASH_TILE, att_tokens - done);
    const int tile_base_t = start_t + done;

    float thread_max[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
      thread_max[qh] = -INFINITY;

    for (int local_t = wid; local_t < tile_len; local_t += nwarp) {
      const int t = tile_base_t + local_t;
      const int slot = t % cap;
      const size_t slot_off = (size_t)slot * kv_dim + kv_head_offset;
      const bf16_t *__restrict__ k_ptr = k_layer + slot_off;
      const uint2 *__restrict__ k_u2 = reinterpret_cast<const uint2 *>(k_ptr);

      float sc[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
        sc[qh] = 0.0f;

      const int D4 = D >> 2;
#pragma unroll 4
      for (int d4 = lane; d4 < D4; d4 += WF_SIZE) {
        const float4 kv = bf16quad_to_float4(k_u2[d4]);
        const int base = d4 << 2;
#pragma unroll
        for (int qh = 0; qh < kv_mul; ++qh) {
          const float *qv = s_q + qh * D + base;
          sc[qh] = fmaf(qv[0], kv.x, sc[qh]);
          sc[qh] = fmaf(qv[1], kv.y, sc[qh]);
          sc[qh] = fmaf(qv[2], kv.z, sc[qh]);
          sc[qh] = fmaf(qv[3], kv.w, sc[qh]);
        }
      }
      for (int d = (D4 << 2) + lane; d < D; d += WF_SIZE) {
        const float kd = static_cast<float>(k_ptr[d]);
#pragma unroll
        for (int qh = 0; qh < kv_mul; ++qh) {
          sc[qh] = fmaf(s_q[qh * D + d], kd, sc[qh]);
        }
      }
#pragma unroll
      for (int qh = 0; qh < kv_mul; ++qh) {
        float sum = sc[qh];
        for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
          sum += __shfl_down(sum, offset, WF_SIZE);
        }
        if (lane == 0) {
          const float scaled = sum * rsqrt_D;
          s_scores[qh * score_stride + local_t] = scaled;
          thread_max[qh] = fmaxf(thread_max[qh], scaled);
        }
      }
    }
    __syncthreads();

    for (int qh = 0; qh < kv_mul; ++qh) {
      const float mx =
          block_reduce_max(thread_max[qh], warp_buffer, lane, wid, nwarp);
      if (tid == 0) {
        if (mx > s_m[qh]) {
          const float scale = __expf(s_m[qh] - mx);
          s_l[qh] *= scale;
          s_m[qh] = mx;
          s_scale[qh] = scale;
        } else {
          s_scale[qh] = 1.0f;
        }
      }
      __syncthreads();
      acc[qh] *= s_scale[qh];
      __syncthreads();
    }

    for (int qh = 0; qh < kv_mul; ++qh) {
      float thread_sum = 0.0f;
      for (int local_t = tid; local_t < tile_len; local_t += blockDim.x) {
        float p = __expf(s_scores[qh * score_stride + local_t] - s_m[qh]);
        s_scores[qh * score_stride + local_t] = p;
        thread_sum += p;
      }
      __syncthreads();
      const float sum =
          block_reduce_sum(thread_sum, warp_buffer, lane, wid, nwarp);
      if (tid == 0)
        s_l[qh] += sum;
      __syncthreads();
    }

    for (int local_t = worker_rank; local_t < tile_len;
         local_t += workers_this_dim) {
      const int t = tile_base_t + local_t;
      const int slot = t % cap;
      const size_t slot_off = (size_t)slot * kv_dim + kv_head_offset;
      const bf16_t *__restrict__ v_ptr = v_layer + slot_off;
      const float v_d = static_cast<float>(v_ptr[d_owner]);
#pragma unroll
      for (int qh = 0; qh < kv_mul; ++qh) {
        const float p = s_scores[qh * score_stride + local_t];
        acc[qh] = fmaf(p, v_d, acc[qh]);
      }
    }
    __syncthreads();

    done += tile_len;
  }

  if (tid == 0) {
    for (int qh = 0; qh < kv_mul; ++qh) {
      const int q_head = kv_head * kv_mul + qh;
      const float sink = static_cast<float>(attn_sinks[layer_idx * Hq + q_head]);
      if (sink > s_m[qh]) {
        const float scale = __expf(s_m[qh] - sink);
        s_l[qh] *= scale;
        s_m[qh] = sink;
        s_scale[qh] = scale;
      } else {
        s_scale[qh] = 1.0f;
      }
      s_l[qh] += __expf(sink - s_m[qh]);
    }
  }
  __syncthreads();

  for (int qh = 0; qh < kv_mul; ++qh) {
    acc[qh] *= s_scale[qh];
    s_accum[qh * ATTN_THREADS_PER_BLOCK + tid] = acc[qh];
  }
  __syncthreads();

  if (worker_rank == 0) {
#pragma unroll
    for (int qh = 0; qh < kv_mul; ++qh) {
      float total = 0.0f;
      for (int w = 0; w < workers_this_dim; ++w) {
        const int peer_tid = d_owner + w * D;
        if (peer_tid >= blockDim.x)
          break;
        total += s_accum[qh * ATTN_THREADS_PER_BLOCK + peer_tid];
      }
      const float inv_l = __frcp_rn(s_l[qh]);
      const float val = total * inv_l;
      const int q_head = kv_head * kv_mul + qh;
      out_b[(size_t)q_head * D + d_owner] = hip_bfloat16(val);
    }
  }
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4) __global__
    void attention_flashdecode_mqa_even(
        bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
        const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
        const bf16_t *__restrict__ attn_sinks, int layer_idx,
        const int *__restrict__ pos, int D, int Hq, int Hk,
        const uint32_t *__restrict__ layer_offsets,
        const int *__restrict__ layer_capacity, int sliding_window,
        uint32_t kv_batch_stride, int batch_size) {
  extern __shared__ float smem[];
  __shared__ float warp_buffer[ATTN_WARPS_PER_BLOCK];
  attention_flashdecode_mqa_kernel<true>(
      out_tb, q, k_cache, v_cache, attn_sinks, layer_idx, pos, D, Hq, Hk,
      layer_offsets, layer_capacity, sliding_window, kv_batch_stride,
      batch_size, smem, warp_buffer);
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4) __global__
    void attention_flashdecode_mqa_odd(
        bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
        const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
        const bf16_t *__restrict__ attn_sinks, int layer_idx,
        const int *__restrict__ pos, int D, int Hq, int Hk,
        const uint32_t *__restrict__ layer_offsets,
        const int *__restrict__ layer_capacity, uint32_t kv_batch_stride,
        int batch_size) {
  extern __shared__ float smem[];
  __shared__ float warp_buffer[ATTN_WARPS_PER_BLOCK];
  attention_flashdecode_mqa_kernel<false>(
      out_tb, q, k_cache, v_cache, attn_sinks, layer_idx, pos, D, Hq, Hk,
      layer_offsets, layer_capacity, /*sliding_window*/ 0, kv_batch_stride,
      batch_size, smem, warp_buffer);
}
