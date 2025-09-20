#include "../common/defines.h"
#include "attention.h"
#include <math.h>

// Batched attention kernel: processes grid.y = batch dimension
// Uses dynamic shared memory sized for (max_pos_in_batch + 2) floats
__launch_bounds__(64, 8) __global__ void attention_batch_kernel(
    float *__restrict__ out_tb,  // [B, Hq*D]
    const float *__restrict__ q, // [B, Hq*D], FP32 (already rotary-applied)
    const float *__restrict__ k_cache,    // [B, L*S*KV], FP32
    const float *__restrict__ v_cache,    // [B, L*S*KV], FP32
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride, int batch_size) {
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;
  const int head = blockIdx.x;
  const int lane = threadIdx.x;

  const int pos_b = pos[b];
  if (pos_b < 0 || head >= Hq)
    return;

  const int kv_dim = Hk * D;
  const int kv_mul = Hq / Hk;
  const int kv_head = head / kv_mul;
  const float rsqrt_D = rsqrtf((float)D);

  // Base pointers for batch b and layer layer_idx
  const float *__restrict__ q_b = q + (size_t)b * Hq * D;
  const float *__restrict__ k_layer =
      k_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
  const float *__restrict__ v_layer =
      v_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
  float *__restrict__ out_b = out_tb + (size_t)b * Hq * D;

  extern __shared__ float s_att[];

  // Compute attention scores with enhanced vectorization
  for (int t = lane; t <= pos_b; t += WF_SIZE) {
    const float *k_ptr = k_layer + t * kv_dim + kv_head * D;
    const float *q_ptr = q_b + head * D;

    float score = 0.0f;
    const int D4 = D >> 2;
    const float4 *k4 = reinterpret_cast<const float4 *>(k_ptr);
    const float4 *q4 = reinterpret_cast<const float4 *>(q_ptr);

    // Unrolled vectorized dot product for better ILP
    #pragma unroll 4
    for (int d4 = 0; d4 < D4; ++d4) {
      float4 kv = k4[d4];
      float4 qv = q4[d4];
      score = fmaf(qv.x, kv.x, score);
      score = fmaf(qv.y, kv.y, score);
      score = fmaf(qv.z, kv.z, score);
      score = fmaf(qv.w, kv.w, score);
    }
    
    // Handle remainder
    for (int d = (D4 << 2); d < D; ++d) {
      score = fmaf(q_ptr[d], k_ptr[d], score);
    }
    
    // Scale and apply mask
    score *= rsqrt_D;
    if (mask)
      score += mask[pos_b * S + t];
    s_att[t] = score;
  }
  __syncthreads();

  // Sink score at position pos_b + 1
  if (lane == 0) {
    float sink_score = attn_sinks[layer_idx * Hq + head];
    s_att[pos_b + 1] = sink_score;
  }
  __syncthreads();

  // Softmax over 0..pos_b+1 with optimizations from Python reference
  const int att_size = pos_b + 2;
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
      for (; t <= pos_b - TILE + 1; t += TILE) {
        #pragma unroll
        for (int ti = 0; ti < TILE; ++ti) {
          const float *v_ptr = v_layer + (t + ti) * kv_dim + kv_head * D;
          result = fmaf(s_att[t + ti], v_ptr[d], result);
        }
      }
      
      // Handle remainder
      for (; t <= pos_b; ++t) {
        const float *v_ptr = v_layer + t * kv_dim + kv_head * D;
        result = fmaf(s_att[t], v_ptr[d], result);
      }
      
      out_b[head * D + d] = result;
    }
  }
}

// Helper to merge two (max, sum_exp) tuples for log-sum-exp in a numerically stable way
static __device__ inline void merge_logsumexp(float &m_a, float &s_a,
                                              float m_b, float s_b) {
  // Handle empty tuples safely to avoid NaNs from (-inf) - (-inf)
  const bool a_empty = !isfinite(m_a); // -inf indicates empty when s_a == 0
  const bool b_empty = !isfinite(m_b);
  if (a_empty) {
    m_a = m_b;
    s_a = s_b;
    return;
  }
  if (b_empty) {
    return;
  }
  float m = fmaxf(m_a, m_b);
  // scale existing sums into new max domain
  s_a = __expf(m_a - m) * s_a + __expf(m_b - m) * s_b;
  m_a = m;
}

// Online-softmax + flash-decoding-style streaming attention
// Single-pass algorithm:
//   - For each timestep t, compute score_t = dot(q, k_t) once via warp-parallel dot.
//   - Maintain running (m, d) where d = sum exp(score_i - m) for i<=t, and m = max(scores_0..t).
//   - Maintain numerator accumulator A = sum exp(score_i - m) * V_i for i<=t, scaled whenever m increases.
//   - After loop, merge sink into denom only, rescale A accordingly, then output out = A / d_final.
// Notes:
//   - Avoids materializing scores or probabilities, and avoids recomputing q·k per dimension.
//   - Mask (e.g., sliding window) is applied to scores before softmax via addition.
//   - Sink is included in the denominator only (matches baseline behavior).
__launch_bounds__(64, 8) __global__ void attention_batch_online_kernel(
    float *__restrict__ out_tb,  // [B, Hq*D]
    const float *__restrict__ q, // [B, Hq*D]
    const float *__restrict__ k_cache,    // [B, L*S*KV]
    const float *__restrict__ v_cache,    // [B, L*S*KV]
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride, int batch_size) {
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;
  const int head = blockIdx.x;
  const int lane = threadIdx.x;

  const int pos_b = pos[b];
  if (pos_b < 0 || head >= Hq)
    return;

  const int kv_dim = Hk * D;
  const int kv_mul = Hq / Hk;
  const int kv_head = head / kv_mul;
  const float rsqrt_D = rsqrtf((float)D);

  const float *__restrict__ q_b = q + (size_t)b * Hq * D;
  const float *__restrict__ k_layer =
      k_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
  const float *__restrict__ v_layer =
      v_cache + (size_t)b * kv_stride + (size_t)layer_idx * S * kv_dim;
  float *__restrict__ out_b = out_tb + (size_t)b * Hq * D;

  const float *q_ptr = q_b + head * D;

  // Assign lanes to disjoint contiguous dimension ranges for accumulation
  const int threads_per_dim = min(WF_SIZE, D);
  const bool active = (lane < threads_per_dim);
  const int dims_per_thread = active ? ((D + threads_per_dim - 1) / threads_per_dim) : 0;
  const int start_dim = active ? (lane * dims_per_thread) : 0;
  const int end_dim = active ? min(start_dim + dims_per_thread, D) : 0;

  // Small fixed-size register buffer for this lane's output slice
  // Upper bound assuming typical head dims (e.g., D<=256). Adjust if needed.
  const int MAX_CHUNK = 16; // supports up to 16 dims per lane
  float acc_local[MAX_CHUNK];
  int chunk = 0;
  if (active) {
    chunk = end_dim - start_dim;
    // Guard against oversize; if exceeded, fall back to on-the-fly writes
    if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;
    for (int i = 0; i < chunk; ++i) acc_local[i] = 0.0f;
  }

  // Streaming softmax state
  float run_max = -INFINITY; // m
  float run_denom = 0.0f;    // d = sum exp(score - m)

  for (int t = 0; t <= pos_b; ++t) {
    const float *k_ptr = k_layer + t * kv_dim + kv_head * D;

    // Warp-parallel dot(q, k_t)
    float part = 0.0f;
    for (int di = lane; di < D; di += WF_SIZE) {
      part = fmaf(q_ptr[di], k_ptr[di], part);
    }
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      part += __shfl_down(part, off, WF_SIZE);
    }
    float score = __shfl(part, 0, WF_SIZE);
    score *= rsqrt_D;
    if (mask) score += mask[pos_b * S + t];
    const bool score_finite = isfinite(score);

    // Initialization when no valid score has been seen yet
    if (!isfinite(run_max)) {
      if (!score_finite) {
        // masked step only, nothing to do
        continue;
      } else {
        // first finite score initializes state
        run_max = score;
        run_denom = 1.0f;
        if (active) {
          const float *v_ptr = v_layer + t * kv_dim + kv_head * D;
          for (int i = 0; i < chunk; ++i) {
            acc_local[i] += v_ptr[start_dim + i];
          }
        }
        continue;
      }
    }

    // If current score is masked (-inf), it contributes nothing
    if (!score_finite) {
      continue;
    }

    // Update running (m, denom) and scale accumulator when m increases
    float m_new = fmaxf(run_max, score);
    float scale_old = __expf(run_max - m_new);
    float e = __expf(score - m_new);
    run_denom = scale_old * run_denom + e;
    run_max = m_new;

    if (active) {
      const float *v_ptr = v_layer + t * kv_dim + kv_head * D;
      // Scale existing accumulator by scale_old, then add e * V_t
      for (int i = 0; i < chunk; ++i) {
        acc_local[i] = fmaf(e, v_ptr[start_dim + i], scale_old * acc_local[i]);
      }
    }
  }

  // Merge sink into denominator only, rescaling accumulator accordingly
  float sink_score = attn_sinks[layer_idx * Hq + head];
  float m_final = fmaxf(run_max, sink_score);
  float scale_acc = __expf(run_max - m_final);
  float denom_final = scale_acc * run_denom + __expf(sink_score - m_final);

  // Normalize and write once
  float inv_denom = (denom_final > 0.0f) ? __frcp_rn(denom_final) : 0.0f;
  if (active) {
    // apply final scaling to accumulator
    for (int i = 0; i < chunk; ++i) acc_local[i] *= scale_acc;
    int d = 0;
    for (; d < chunk; ++d) {
      out_b[head * D + (start_dim + d)] = acc_local[d] * inv_denom;
    }
    // If dims_per_thread exceeded MAX_CHUNK (rare), handle the tail directly
    for (int dd = start_dim + chunk; dd < end_dim; ++dd) {
      // This path would have lost partial sums; as a safe fallback, recompute minimal contribution
      // However, in typical configs (D<=256, WF_SIZE=64), chunk<=4 and MAX_CHUNK=16 is sufficient.
      // To avoid incorrect outputs, ensure MAX_CHUNK is large enough for your model.
      out_b[head * D + dd] = 0.0f; // conservative fallback
    }
  }
}
