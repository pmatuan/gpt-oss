#include "attention.h"
#include "../common/defines.h"
#include "../utility/utility.h"
#include <math.h>

__device__ __forceinline__ void block_reduce_max_multi(
    float (&vals)[ATTN_FLASH_MAX_KV_MUL],
    float (&warp_storage)[ATTN_FLASH_MAX_KV_MUL][ATTN_WARPS_PER_BLOCK],
    float (&block_storage)[ATTN_FLASH_MAX_KV_MUL], int lane, int warp_id,
    int num_warps) {
#pragma unroll
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
    float v = vals[qh];
#pragma unroll
    for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
      v = fmaxf(v, __shfl_down(v, offset, WF_SIZE));
    }
    vals[qh] = v;
  }

  if (lane == 0) {
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      warp_storage[qh][warp_id] = vals[qh];
    }
  }
  __syncthreads();

  if (warp_id == 0) {
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      float v = (lane < num_warps) ? warp_storage[qh][lane] : -INFINITY;
#pragma unroll
      for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down(v, offset, WF_SIZE));
      }
      if (lane == 0)
        block_storage[qh] = v;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void block_reduce_sum_multi(
    float (&vals)[ATTN_FLASH_MAX_KV_MUL],
    float (&warp_storage)[ATTN_FLASH_MAX_KV_MUL][ATTN_WARPS_PER_BLOCK],
    float (&block_storage)[ATTN_FLASH_MAX_KV_MUL], int lane, int warp_id,
    int num_warps) {
#pragma unroll
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
    float v = vals[qh];
#pragma unroll
    for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
      v += __shfl_down(v, offset, WF_SIZE);
    }
    vals[qh] = v;
  }

  if (lane == 0) {
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      warp_storage[qh][warp_id] = vals[qh];
    }
  }
  __syncthreads();

  if (warp_id == 0) {
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      float v = (lane < num_warps) ? warp_storage[qh][lane] : 0.0f;
#pragma unroll
      for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, WF_SIZE);
      }
      if (lane == 0)
        block_storage[qh] = v;
    }
  }
  __syncthreads();
}

template <bool HAS_WINDOW>
__device__ __forceinline__ void flash_decoding_body(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ qkv,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const bf16_t *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos,
    const float *__restrict__ rope_inv_freq, float rope_concentration,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size, float *__restrict__ smem_dyn) {

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

  const int kv_dim = Hk * D;
  const int kv_head_offset = kv_head * D;
  const float rsqrt_D = rsqrtf(static_cast<float>(D));

  const uint32_t layer_base = layer_offsets[layer_idx];
  const bf16_t *__restrict__ k_layer =
      k_cache + (size_t)b * kv_batch_stride + layer_base;
  const bf16_t *__restrict__ v_layer =
      v_cache + (size_t)b * kv_batch_stride + layer_base;
  const size_t q_stride = (size_t)Hq * D;
  const size_t kv_stride = (size_t)Hk * D;
  const size_t qkv_stride = q_stride + 2 * kv_stride;
  const bf16_t *__restrict__ qkv_b = qkv + (size_t)b * qkv_stride;
  bf16_t *__restrict__ out_b = out_tb + (size_t)b * Hq * D;

  const int d_owner = tid % D;
  const int workers_this_dim = ((blockDim.x - 1 - d_owner) / D) + 1;
  const int worker_rank = tid / D;
  const int subgroup = lane >> 4;
  const int lane16 = lane & 15;
  const int groups_per_warp = WF_SIZE / 16;

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
  float *s_q = s_scores + ATTN_FLASH_MAX_KV_MUL * score_stride;
  const int acc_stride = ATTN_THREADS_PER_BLOCK + 1;
  float *s_accum = s_q + ATTN_FLASH_MAX_KV_MUL * D;

  __shared__ float s_m[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_l[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_scale[ATTN_FLASH_MAX_KV_MUL];
  __shared__ float s_warp_red[ATTN_FLASH_MAX_KV_MUL][ATTN_WARPS_PER_BLOCK];
  __shared__ float s_red_tmp[ATTN_FLASH_MAX_KV_MUL];

  if (tid == 0) {
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      s_m[qh] = -INFINITY;
      s_l[qh] = 0.0f;
    }
  }
  __syncthreads();

  const int D2 = D >> 1;
  const float position_f = static_cast<float>(pos_b);
  const float concentration = rope_concentration;
  if (wid == 0 && rope_inv_freq != nullptr) {
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      const int q_head = kv_head * ATTN_FLASH_MAX_KV_MUL + qh;
      const bf16_t *__restrict__ q_head_ptr = qkv_b + (size_t)q_head * D;
      for (int i = lane; i < D2; i += WF_SIZE) {
        const float x1 = static_cast<float>(q_head_ptr[i]);
        const float x2 = static_cast<float>(q_head_ptr[D2 + i]);
        const float inv = rope_inv_freq[i];
        float s_val, c_val;
        sincosf(position_f * inv, &s_val, &c_val);
        c_val *= concentration;
        s_val *= concentration;
        const float y1 = fmaf(-x2, s_val, x1 * c_val);
        const float y2 = fmaf(x1, s_val, x2 * c_val);
        s_q[qh * D + i] = y1;
        s_q[qh * D + D2 + i] = y2;
      }
      if ((D & 1) && lane == 0) {
        s_q[qh * D + (D - 1)] = static_cast<float>(q_head_ptr[D - 1]);
      }
    }
  } else if (wid == 0) {
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      const int q_head = kv_head * ATTN_FLASH_MAX_KV_MUL + qh;
      const bf16_t *__restrict__ q_head_ptr = qkv_b + (size_t)q_head * D;
      for (int i = lane; i < D; i += WF_SIZE) {
        s_q[qh * D + i] = static_cast<float>(q_head_ptr[i]);
      }
    }
  }
  __syncthreads();

  const float *q_heads[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
    q_heads[qh] = s_q + qh * D;

  float acc[ATTN_FLASH_MAX_KV_MUL];
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
    acc[qh] = 0.0f;

  int done = 0;
  while (done < att_tokens) {
    const int tile_len = min(ATTN_FLASH_TILE, att_tokens - done);
    const int tile_base_t = start_t + done;
    int slot_base = tile_base_t;
    if (slot_base >= cap) {
      slot_base %= cap;
    }

    float thread_max[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
      thread_max[qh] = -INFINITY;

    const int warp_tile_stride = nwarp * groups_per_warp;

    for (int local_t = wid * groups_per_warp + subgroup; local_t < tile_len;
         local_t += warp_tile_stride) {
      int slot = slot_base + local_t;
      while (slot >= cap)
        slot -= cap;
      const int slot_off = slot * kv_dim + kv_head_offset;
      const bf16_t *__restrict__ k_ptr = k_layer + slot_off;
      const uint2 *__restrict__ k_u2 = reinterpret_cast<const uint2 *>(k_ptr);

      float sc[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
        sc[qh] = 0.0f;

      const int D4 = D >> 2;
#pragma unroll 4
      for (int d4 = lane16; d4 < D4; d4 += 16) {
        const float4 kv = bf16quad_to_float4(k_u2[d4]);
        const int base = d4 << 2;
#pragma unroll
        for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
          const float *qv = q_heads[qh] + base;
          sc[qh] = fmaf(qv[0], kv.x, sc[qh]);
          sc[qh] = fmaf(qv[1], kv.y, sc[qh]);
          sc[qh] = fmaf(qv[2], kv.z, sc[qh]);
          sc[qh] = fmaf(qv[3], kv.w, sc[qh]);
        }
      }
      for (int d = (D4 << 2) + lane16; d < D; d += 16) {
        const float kd = static_cast<float>(k_ptr[d]);
#pragma unroll
        for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
          sc[qh] = fmaf(q_heads[qh][d], kd, sc[qh]);
        }
      }
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
        float sum = sc[qh];
        for (int offset = 8; offset > 0; offset >>= 1) {
          sum += __shfl_down(sum, offset, 16);
        }
        if (lane16 == 0) {
          const float scaled = sum * rsqrt_D;
          s_scores[qh * score_stride + local_t] = scaled;
          thread_max[qh] = fmaxf(thread_max[qh], scaled);
        }
      }
    }
    __syncthreads();

    block_reduce_max_multi(thread_max, s_warp_red, s_red_tmp, lane, wid,
                            nwarp);
    if (tid == 0) {
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
        const float mx = s_red_tmp[qh];
        if (mx > s_m[qh]) {
          const float scale = __expf(s_m[qh] - mx);
          s_l[qh] *= scale;
          s_m[qh] = mx;
          s_scale[qh] = scale;
        } else {
          s_scale[qh] = 1.0f;
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
      acc[qh] *= s_scale[qh];

    float thread_sum_multi[ATTN_FLASH_MAX_KV_MUL];
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
      thread_sum_multi[qh] = 0.0f;

    for (int local_t = tid; local_t < tile_len; local_t += blockDim.x) {
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
        float p = __expf(s_scores[qh * score_stride + local_t] - s_m[qh]);
        s_scores[qh * score_stride + local_t] = p;
        thread_sum_multi[qh] += p;
      }
    }
    __syncthreads();
    block_reduce_sum_multi(thread_sum_multi, s_warp_red, s_red_tmp, lane, wid,
                            nwarp);
    if (tid == 0) {
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh)
        s_l[qh] += s_red_tmp[qh];
    }
    __syncthreads();

    for (int local_t = worker_rank; local_t < tile_len;
         local_t += workers_this_dim) {
      int slot = slot_base + local_t;
      while (slot >= cap)
        slot -= cap;
      const int slot_off = slot * kv_dim + kv_head_offset;
      const bf16_t *__restrict__ v_ptr = v_layer + slot_off;
      const float v_d = static_cast<float>(v_ptr[d_owner]);
#pragma unroll
      for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
        const float p = s_scores[qh * score_stride + local_t];
        acc[qh] = fmaf(p, v_d, acc[qh]);
      }
    }
    __syncthreads();

    done += tile_len;
  }

  if (tid == 0) {
    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      const int q_head = kv_head * ATTN_FLASH_MAX_KV_MUL + qh;
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

  #pragma unroll
  for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
    acc[qh] *= s_scale[qh];
    s_accum[qh * acc_stride + tid] = acc[qh];
  }
  __syncthreads();

  if (worker_rank == 0) {
#pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_MAX_KV_MUL; ++qh) {
      float total = 0.0f;
      for (int w = 0; w < workers_this_dim; ++w) {
        const int peer_tid = d_owner + w * D;
        if (peer_tid >= blockDim.x)
          break;
        total += s_accum[qh * acc_stride + peer_tid];
      }
      const float inv_l = __frcp_rn(s_l[qh]);
      const float val = total * inv_l;
      const int q_head = kv_head * ATTN_FLASH_MAX_KV_MUL + qh;
      out_b[(size_t)q_head * D + d_owner] = hip_bfloat16(val);
    }
  }
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4) __global__
    void flash_decoding_even(
        bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ qkv,
        const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
        const bf16_t *__restrict__ attn_sinks, int layer_idx,
        const int *__restrict__ pos,
        const float *__restrict__ rope_inv_freq, float rope_concentration,
        const uint32_t *__restrict__ layer_offsets,
        const int *__restrict__ layer_capacity, int sliding_window,
        uint32_t kv_batch_stride, int batch_size) {
  extern __shared__ float smem[];
  flash_decoding_body<true>(
      out_tb, qkv, k_cache, v_cache, attn_sinks, layer_idx, pos,
      rope_inv_freq, rope_concentration, layer_offsets, layer_capacity,
      sliding_window, kv_batch_stride, batch_size, smem);
}

__launch_bounds__(ATTN_THREADS_PER_BLOCK, 4) __global__
    void flash_decoding_odd(
        bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ qkv,
        const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
        const bf16_t *__restrict__ attn_sinks, int layer_idx,
        const int *__restrict__ pos,
        const float *__restrict__ rope_inv_freq, float rope_concentration,
        const uint32_t *__restrict__ layer_offsets,
        const int *__restrict__ layer_capacity, uint32_t kv_batch_stride,
        int batch_size) {
  extern __shared__ float smem[];
  flash_decoding_body<false>(
      out_tb, qkv, k_cache, v_cache, attn_sinks, layer_idx, pos,
      rope_inv_freq, rope_concentration, layer_offsets, layer_capacity,
      /*sliding_window*/ 0, kv_batch_stride, batch_size, smem);
}
