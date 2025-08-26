#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WF_SIZE 64

template <int TILE_T>
__launch_bounds__(BLOCK_SIZE, 2) __global__ void attention_kernel(
    float *__restrict__ tb,                     // [Hq, D]
    const float *__restrict__ q,                // [Hq, D]
    const float *__restrict__ key_cache,        // [S, KV] per layer
    const float *__restrict__ value_cache,      // [S, KV] per layer
    const int *__restrict__ token2row,          // [S]
    const float *__restrict__ attn_sinks_layer, // [Hq]
    const float *__restrict__ mask,             // [S,S] or nullptr
    int Hq, int Hk, int D, int KV,              // dims
    int S, int pos) {

  extern __shared__ float smem[];
  float *s_q = smem; // length D
  // reserve reduction + broadcast area at the end
  const int warp_count = (blockDim.x + 63) / 64;
  float *s_red = s_q + D; // length >= warp_count + 4

  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= Hq)
    return;

  // head mapping
  const int kv_mul = Hq / Hk;        // grouped-query attn
  const int base = (h / kv_mul) * D; // head's offset inside KV row
  const float scale = 1.0f / sqrtf((float)D);

  // shorthand row pointers for this layer's caches are passed at launch:
  const float *__restrict__ K0 = key_cache;   // [S, KV]
  const float *__restrict__ V0 = value_cache; // [S, KV]

  // stage Q into shared once
  for (int i = tid; i < D; i += blockDim.x)
    s_q[i] = q[(size_t)h * D + i];
  __syncthreads();

  // online softmax state (maintained by lane 0 and broadcast via s_red)
  float m_run = -INFINITY; // running max of logits
  float s_run = 0.0f;      // running sum of exp(logit - m_run)

  // each thread accumulates one output dimension (if tid < D)
  float acc = 0.0f;

  // convenience lambdas
  auto reduce_block_sum = [&](float v) {
// per-warp reduction (wave64)
#pragma unroll
    for (int off = 32; off > 0; off >>= 1)
      v += __shfl_down(v, off, 64);
    int lane = tid & 63;
    int wid = tid >> 6;
    if (lane == 0)
      s_red[wid] = v;
    __syncthreads();
    // warp0 reduces warps
    float out = 0.0f;
    if (wid == 0) {
      out = (lane < warp_count) ? s_red[lane] : 0.0f;
#pragma unroll
      for (int off = 32; off > 0; off >>= 1)
        out += __shfl_down(out, off, 64);
    }
    return out; // valid only on warp0 lanes; lane0 ultimately holds sum
  };

  const int T_real = pos + 1; // causal window [0..pos]
  const int lane = tid & 63;
  const int wid = tid >> 6;

  // Process tokens in tiles for better L2/LDS behavior
  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);

    // Iterate tokens in this tile, one-pass online softmax
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      int row = token2row[t];
      if (row < 0)
        row = 0;
      else if (row >= S)
        row = S - 1;

      // 1) dot = q · k(row, head)
      float part = 0.0f;
      const float *__restrict__ Krow = K0 + (size_t)row * KV + base;

      const int D4 = D >> 2;
      const float4 *__restrict__ K4 = reinterpret_cast<const float4 *>(Krow);
      const float4 *__restrict__ Q4 = reinterpret_cast<const float4 *>(s_q);
      for (int i4 = tid; i4 < D4; i4 += blockDim.x) {
        float4 k4 = K4[i4];
        float4 q4 = Q4[i4];
        part = fmaf(q4.x, k4.x, part);
        part = fmaf(q4.y, k4.y, part);
        part = fmaf(q4.z, k4.z, part);
        part = fmaf(q4.w, k4.w, part);
      }

      float block_sum = reduce_block_sum(part);
      float logit = 0.0f;
      if (wid == 0 && lane == 0) {
        logit = block_sum * scale;
        if (mask)
          logit += mask[(size_t)pos * S + t];
        if (!isfinite(logit))
          logit = -1e30f;

        // 2) online softmax update: rescale numerator/denominator to new max
        float m_new = fmaxf(m_run, logit);
        float scale_old = isfinite(m_run) ? expf(m_run - m_new) : 0.0f;
        float w = expf(logit - m_new); // exp(logit - m_new)

        // update denominator
        s_run = s_run * scale_old + w;

        // broadcast scalars needed by all threads for numerator accum
        s_red[0] = scale_old;
        s_red[1] = w;
        s_red[2] = m_new;
      }
      __syncthreads();

      const float scale_old_b = s_red[0];
      const float w_b = s_red[1];
      // m_run kept by lane0; others don't need it

      // 3) read V(row, head) and update numerator (per output dim)
      if (tid < D) {
        const float v = V0[(size_t)row * KV + base + tid];
        acc = acc * scale_old_b + w_b * v;
      }
      __syncthreads();

      if (wid == 0 && lane == 0)
        m_run = s_red[2];
      __syncthreads();
    } // tile loop
  } // t loop

  // 4) incorporate attention sink into denominator only (matches your logic)
  if (wid == 0 && lane == 0) {
    const float sinkVal = attn_sinks_layer[h];
    float m_new = fmaxf(m_run, sinkVal);
    float scale_old = isfinite(m_run) ? expf(m_run - m_new) : 0.0f;
    float w_sink = expf(sinkVal - m_new);
    float denom = s_run * scale_old + w_sink;
    float inv = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
    s_red[0] = inv;       // 1 / denominator
    s_red[1] = scale_old; // rescale for numerator
  }
  __syncthreads();

  const float inv_denom = s_red[0];
  const float final_scale = s_red[1];

  if (tid < D) {
    float num_scaled =
        acc * final_scale; // sink does not add to numerator in your impl
    tb[(size_t)h * D + tid] = num_scaled * inv_denom;
  }
}
