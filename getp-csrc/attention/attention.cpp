#include "../common/defines.h"
#include "attention.h"
#include <math.h>
#include "../utility/utility.h"

// attention kernel: processes single sample
// Uses dynamic shared memory sized for (max_pos + 2) floats
// Optimized to read Q directly from QKV buffer
__launch_bounds__(64, 8) __global__ void attention_kernel_bf16(
    float *__restrict__ out_tb,  // [B,Hq*D]
    const float *__restrict__ qkv, // [B,(Hq+2*Hk)*D], FP32 (already rotary-applied)
  const bf16_t *__restrict__ k_cache,    // [B,L*S*KV], BF16
  const bf16_t *__restrict__ v_cache,    // [B,L*S*KV], BF16
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  const int b = blockIdx.z;

  const int pos_current = pos[b];
  if (pos_current < 0 || head >= Hq)
    return;

  const int kv_dim = Hk * D;
  const int kv_mul = Hq / Hk;
  const int kv_head = head / kv_mul;
  const float rsqrt_D = rsqrtf((float)D);

  // Base pointers for single sample and layer layer_idx
  const int q_size = Hq * D;
  const int kv_size = Hk * D;
  const int total = q_size + 2 * kv_size;
  const float *__restrict__ qkv_b = qkv + (size_t)b * total;
    const bf16_t *__restrict__ k_layer =
      k_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
    const bf16_t *__restrict__ v_layer =
      v_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
  float *__restrict__ out_b = out_tb + (size_t)b * (size_t)(Hq * D);

  extern __shared__ float s_att[];

  // Compute attention scores with enhanced vectorization
  for (int t = lane; t <= pos_current; t += WF_SIZE) {
  const bf16_t *k_ptr_bf16 = k_layer + t * kv_dim + kv_head * D;
  const float *q_ptr = qkv_b + head * D; // Read Q directly from QKV buffer

  float score = 0.0f;
  const int D4 = D >> 2;
  const uint2 *k4 = reinterpret_cast<const uint2 *>(k_ptr_bf16);
  const float4 *q4 = reinterpret_cast<const float4 *>(q_ptr);

    // Unrolled vectorized dot product for better ILP
    #pragma unroll 4
    for (int d4 = 0; d4 < D4; ++d4) {
      // load 4 bf16 -> float4
      float4 kv = bf16quad_to_float4(k4[d4]);
      float4 qv = q4[d4];
      score = fmaf(qv.x, kv.x, score);
      score = fmaf(qv.y, kv.y, score);
      score = fmaf(qv.z, kv.z, score);
      score = fmaf(qv.w, kv.w, score);
    }
    // Handle remainder (bf16 -> fp32 scalar)
    for (int d = (D4 << 2); d < D; ++d) {
      float kv = static_cast<float>(k_ptr_bf16[d]);
      score = fmaf(q_ptr[d], kv, score);
    }
    
    // Scale and apply mask
    score *= rsqrt_D;
    if (mask)
      score += mask[pos_current * S + t];
    s_att[t] = score;
  }
  __syncthreads();

  // Sink score at position pos_current + 1
  if (lane == 0) {
    float sink_score = attn_sinks[layer_idx * Hq + head];
    s_att[pos_current + 1] = sink_score;
  }
  __syncthreads();

  // Softmax over 0..pos_current+1 with optimizations from Python reference
  const int att_size = pos_current + 2;
  float maxv = -INFINITY;
  
  // Find max in warp-parallel fashion
  for (int i = lane; i < att_size; i += WF_SIZE)
    maxv = fmaxf(maxv, s_att[i]);
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    maxv = fmaxf(maxv, __shfl_down(maxv, off, WF_SIZE));
  maxv = __shfl(maxv, 0, WF_SIZE);

  // Compute exp and sum in one pass
  float sum = 0.0f;
  for (int i = lane; i < att_size; i += WF_SIZE) {
    float v = __expf(s_att[i] - maxv); // Use fast exp intrinsic
    s_att[i] = v;
    sum += v;
  }
  
  // Warp reduction for sum
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    sum += __shfl_down(sum, off, WF_SIZE);
  sum = __shfl(sum, 0, WF_SIZE);

  // Use fast reciprocal approximation (like Python's 1/sum)
  float inv_sum = __frcp_rn(sum); // Fast reciprocal
  for (int i = lane; i < att_size; i += WF_SIZE)
    s_att[i] *= inv_sum;
  __syncthreads();

  // Weighted sum of V with improved memory access pattern
  const int threads_per_dim = min(WF_SIZE, D);
  if (lane < threads_per_dim) {
    const int dims_per_thread = (D + threads_per_dim - 1) / threads_per_dim;
    const int start_dim = lane * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, D);

    // Process multiple dimensions per thread for better memory coalescing
    for (int d = start_dim; d < end_dim; ++d) {
      float result = 0.0f;
      
      // Tile the loop for better cache utilization
      const int TILE = 8;
      int t = 0;
      
      // Process in tiles
      for (; t <= pos_current - TILE + 1; t += TILE) {
        #pragma unroll
        for (int ti = 0; ti < TILE; ++ti) {
          const bf16_t *v_ptr_bf16 = v_layer + (t + ti) * kv_dim + kv_head * D;
          float vv = static_cast<float>(v_ptr_bf16[d]);
          result = fmaf(s_att[t + ti], vv, result);
        }
      }
      
      // Handle remainder
      for (; t <= pos_current; ++t) {
        const bf16_t *v_ptr_bf16 = v_layer + t * kv_dim + kv_head * D;
        float vv = static_cast<float>(v_ptr_bf16[d]);
        result = fmaf(s_att[t], vv, result);
      }
      
      out_b[head * D + d] = result;
    }
  }
}

// Batched variant: grid=(Hq,1,B), block=(WF_SIZE)
__launch_bounds__(64, 8) __global__ void attention_kernel_bf16_B(
    float *__restrict__ out_tb,  // [B, Hq*D]
    const float *__restrict__ qkv, // [B, (Hq+2*Hk)*D]
  const bf16_t *__restrict__ k_cache,    // [B, L*S*KV]
  const bf16_t *__restrict__ v_cache,    // [B, L*S*KV]
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  const int b = blockIdx.z;

  const int pos_current = pos[b];
  if (pos_current < 0 || head >= Hq)
    return;

  const int kv_dim = Hk * D;
  const int kv_mul = Hq / Hk;
  const int kv_head = head / kv_mul;
  const float rsqrt_D = rsqrtf((float)D);

  const int q_size = Hq * D;
  const int total = q_size + 2 * kv_dim;
  const float *__restrict__ qkv_b = qkv + (size_t)b * total;
  const bf16_t *__restrict__ k_base = k_cache + (size_t)b * kv_stride;
  const bf16_t *__restrict__ v_base = v_cache + (size_t)b * kv_stride;
  const bf16_t *__restrict__ k_layer = k_base + (size_t)layer_idx * S * kv_dim;
  const bf16_t *__restrict__ v_layer = v_base + (size_t)layer_idx * S * kv_dim;
  float *__restrict__ out_b = out_tb + (size_t)b * q_size;

  extern __shared__ float s_att[];

  // Compute attention scores
  for (int t = lane; t <= pos_current; t += WF_SIZE) {
    const bf16_t *k_ptr_bf16 = k_layer + t * kv_dim + kv_head * D;
    const float *q_ptr = qkv_b + head * D;
    float score = 0.0f;
    const int D4 = D >> 2;
    const uint2 *k4 = reinterpret_cast<const uint2 *>(k_ptr_bf16);
    const float4 *q4 = reinterpret_cast<const float4 *>(q_ptr);
#pragma unroll 4
    for (int d4 = 0; d4 < D4; ++d4) {
      float4 kv = bf16quad_to_float4(k4[d4]);
      float4 qv = q4[d4];
      score = fmaf(qv.x, kv.x, score);
      score = fmaf(qv.y, kv.y, score);
      score = fmaf(qv.z, kv.z, score);
      score = fmaf(qv.w, kv.w, score);
    }
    for (int d = (D4 << 2); d < D; ++d) {
      float kv = static_cast<float>(k_ptr_bf16[d]);
      score = fmaf(q_ptr[d], kv, score);
    }
    score *= rsqrt_D;
    if (mask)
      score += mask[pos_current * S + t];
    s_att[t] = score;
  }
  __syncthreads();

  if (lane == 0) {
    float sink_score = attn_sinks[layer_idx * Hq + head];
    s_att[pos_current + 1] = sink_score;
  }
  __syncthreads();

  const int att_size = pos_current + 2;
  float maxv = -INFINITY;
  for (int i = lane; i < att_size; i += WF_SIZE)
    maxv = fmaxf(maxv, s_att[i]);
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    maxv = fmaxf(maxv, __shfl_down(maxv, off, WF_SIZE));
  maxv = __shfl(maxv, 0, WF_SIZE);

  float sum = 0.0f;
  for (int i = lane; i < att_size; i += WF_SIZE) {
    float v = __expf(s_att[i] - maxv);
    s_att[i] = v;
    sum += v;
  }
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    sum += __shfl_down(sum, off, WF_SIZE);
  sum = __shfl(sum, 0, WF_SIZE);

  float inv_sum = __frcp_rn(sum);
  for (int i = lane; i < att_size; i += WF_SIZE)
    s_att[i] *= inv_sum;
  __syncthreads();

  const int threads_per_dim = min(WF_SIZE, D);
  if (lane < threads_per_dim) {
    const int dims_per_thread = (D + threads_per_dim - 1) / threads_per_dim;
    const int start_dim = lane * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, D);
    for (int d_i = start_dim; d_i < end_dim; ++d_i) {
      float result = 0.0f;
      const int TILE = 8;
      int t = 0;
      for (; t <= pos_current - TILE + 1; t += TILE) {
#pragma unroll
        for (int ti = 0; ti < TILE; ++ti) {
          const bf16_t *v_ptr_bf16 = v_layer + (t + ti) * kv_dim + kv_head * D;
          float vv = static_cast<float>(v_ptr_bf16[d_i]);
          result = fmaf(s_att[t + ti], vv, result);
        }
      }
      for (; t <= pos_current; ++t) {
        const bf16_t *v_ptr_bf16 = v_layer + t * kv_dim + kv_head * D;
        float vv = static_cast<float>(v_ptr_bf16[d_i]);
        result = fmaf(s_att[t], vv, result);
      }
      out_b[head * D + d_i] = result;
    }
  }
}
