#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>

typedef hip_bfloat16 bf16_t;

#define WF_SIZE 64

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

  // Compute attention scores
  for (int t = lane; t <= pos_b; t += WF_SIZE) {
    const float *k_ptr = k_layer + t * kv_dim + kv_head * D;
    const float *q_ptr = q_b + head * D;

    float score = 0.0f;
    const int D4 = D >> 2;
    const float4 *k4 = reinterpret_cast<const float4 *>(k_ptr);
    const float4 *q4 = reinterpret_cast<const float4 *>(q_ptr);

    for (int d4 = 0; d4 < D4; ++d4) {
      float4 kv = k4[d4];
      float4 qv = q4[d4];
      score = fmaf(qv.x, kv.x, score);
      score = fmaf(qv.y, kv.y, score);
      score = fmaf(qv.z, kv.z, score);
      score = fmaf(qv.w, kv.w, score);
    }
    for (int d = (D4 << 2); d < D; ++d) {
      score = fmaf(q_ptr[d], k_ptr[d], score);
    }
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

  // Softmax over 0..pos_b+1
  const int att_size = pos_b + 2;
  float maxv = -INFINITY;
  for (int i = lane; i < att_size; i += WF_SIZE)
    maxv = fmaxf(maxv, s_att[i]);
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    maxv = fmaxf(maxv, __shfl_down(maxv, off, WF_SIZE));
  maxv = __shfl(maxv, 0, WF_SIZE);

  float sum = 0.0f;
  for (int i = lane; i < att_size; i += WF_SIZE) {
    float v = __expf(s_att[i] - maxv); // Use faster exp
    s_att[i] = v;
    sum += v;
  }
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
    sum += __shfl_down(sum, off, WF_SIZE);
  sum = __shfl(sum, 0, WF_SIZE);

  float inv_sum = __frcp_rn(sum); // Use faster reciprocal
  for (int i = lane; i < att_size; i += WF_SIZE)
    s_att[i] *= inv_sum;
  __syncthreads();

  // Weighted sum of V
  const int threads_per_dim = min(WF_SIZE, D);
  if (lane < threads_per_dim) {
    const int dims_per_thread = (D + threads_per_dim - 1) / threads_per_dim;
    const int start_dim = lane * dims_per_thread;
    const int end_dim = min(start_dim + dims_per_thread, D);

    for (int d = start_dim; d < end_dim; ++d) {
      float result = 0.0f;
      for (int t = 0; t <= pos_b; ++t) {
        const float *v_ptr = v_layer + t * kv_dim + kv_head * D;
        result = fmaf(s_att[t], v_ptr[d], result);
      }
      out_b[head * D + d] = result;
    }
  }
}
