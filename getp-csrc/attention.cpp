#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WF_SIZE 64

template <int TILE_T, int HEADS_PER_BLOCK>
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
  const int warp_count = (blockDim.x + 63) / 64;
  
  // Each block processes HEADS_PER_BLOCK heads
  const int heads_start = blockIdx.x * HEADS_PER_BLOCK;
  const int heads_end = min(heads_start + HEADS_PER_BLOCK, Hq);
  const int actual_heads = heads_end - heads_start;
  
  if (heads_start >= Hq) return;
  
  const float scale = 1.0f / sqrtf((float)D);
  const int kv_mul = Hq / Hk;
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = tid >> 6;
  
  // Shared memory: queries for all heads in this block + reduction space
  float *s_q_all = smem; // [HEADS_PER_BLOCK * D]
  float *s_red = s_q_all + HEADS_PER_BLOCK * D; // reduction space
  
  // Load all queries for heads in this block
  for (int local_h = 0; local_h < actual_heads; ++local_h) {
    const int global_h = heads_start + local_h;
    float *s_q_h = s_q_all + local_h * D;
    for (int i = tid; i < D; i += blockDim.x) {
      s_q_h[i] = q[global_h * D + i];
    }
  }
  __syncthreads();
  
  // Each warp processes one head
  if (wid < actual_heads) {
    const int global_h = heads_start + wid;
    const int base = (global_h / kv_mul) * D;
    float *s_q_h = s_q_all + wid * D;
    
    // Online softmax state
    float m_run = -INFINITY;
    float s_run = 0.0f;
    float acc = (lane < D) ? 0.0f : 0.0f;
    
    const int T_real = pos + 1;
    
    // Process sequence tokens in tiles  
    for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
      const int tile = min(TILE_T, T_real - t0);
      
      for (int tt = 0; tt < tile; ++tt) {
        const int t = t0 + tt;
        int row = token2row[t];
        if (row < 0) row = 0;
        else if (row >= S) row = S - 1;
        
        // Compute Q·K for this token
        float part = 0.0f;
        const float *Krow = key_cache + (size_t)row * KV + base;
        
        const int D4 = D >> 2;
        const float4 *K4 = reinterpret_cast<const float4 *>(Krow);
        const float4 *Q4 = reinterpret_cast<const float4 *>(s_q_h);
        
        for (int i4 = lane; i4 < D4; i4 += 64) {
          float4 k4 = K4[i4];
          float4 q4 = Q4[i4];
          part = fmaf(q4.x, k4.x, part);
          part = fmaf(q4.y, k4.y, part);
          part = fmaf(q4.z, k4.z, part);
          part = fmaf(q4.w, k4.w, part);
        }
        
        // Warp reduction for attention score
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
          part += __shfl_down(part, off, 64);
        }
        
        // Lane 0 computes logit and updates softmax state
        if (lane == 0) {
          float logit = part * scale;
          if (mask) {
            logit += mask[(size_t)pos * S + t];
          }
          if (!isfinite(logit)) logit = -1e30f;
          
          // Online softmax update
          float m_new = fmaxf(m_run, logit);
          float scale_old = isfinite(m_run) ? expf(m_run - m_new) : 0.0f;
          float w = expf(logit - m_new);
          
          s_run = s_run * scale_old + w;
          
          // Store values for broadcasting
          s_red[wid * 4 + 0] = scale_old;
          s_red[wid * 4 + 1] = w;
          s_red[wid * 4 + 2] = m_new;
          
          m_run = m_new;
        }
        
        // Broadcast softmax values to all threads in warp
        float scale_old_b = s_red[wid * 4 + 0];
        float w_b = s_red[wid * 4 + 1];
        
        // Update output accumulator
        if (lane < D) {
          const float v = value_cache[(size_t)row * KV + base + lane];
          acc = acc * scale_old_b + w_b * v;
        }
      }
    }
    
    // Apply attention sink
    if (lane == 0) {
      const float sinkVal = attn_sinks_layer[global_h];
      float m_new = fmaxf(m_run, sinkVal);
      float scale_old = isfinite(m_run) ? expf(m_run - m_new) : 0.0f;
      float w_sink = expf(sinkVal - m_new);
      float denom = s_run * scale_old + w_sink;
      float inv = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
      
      s_red[wid * 4 + 0] = scale_old * inv;
    }
    
    // Final normalization and write output
    if (lane < D) {
      float final_scale = s_red[wid * 4 + 0];
      tb[global_h * D + lane] = acc * final_scale;
    }
  }
}
