#include "attention.hpp"
#include <math.h>

__global__ void attention_scores_kernel(
    float *att, const float *q, const float *key_cache, int h, int pos,
    int head_dim, int kv_dim, int n_attn_heads, int n_kv_heads, int seq_len,
    const float *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t > pos) return;

  int kv_mul = n_attn_heads / n_kv_heads;
  const float *k = key_cache + t * kv_dim + (h / kv_mul) * head_dim;

  float score = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    score += q[h * head_dim + i] * k[i];
  }
  score *= rsqrtf((float)head_dim);
  if (mask) score += mask[pos * seq_len + t];
  att[h * seq_len + t] = score;
}

__global__ void attention_values_kernel(
    float *tb, const float *att, const float *value_cache, int h, int pos,
    int head_dim, int kv_dim, int n_attn_heads, int n_kv_heads, int seq_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= head_dim) return;

  float res = 0.0f;
  int kv_mul = n_attn_heads / n_kv_heads;
  for (int t = 0; t <= pos; ++t) {
    const float *v = value_cache + t * kv_dim + (h / kv_mul) * head_dim;
    res += att[h * seq_len + t] * v[i];
  }
  tb[h * head_dim + i] = res;
}


// Fused attention kernel: scores + sink + softmax + values in one pass (self-attention style)
// Template TILE_T for tile size, as in getp_run.cpp
template<int TILE_T>
__global__ void attention_fused_kernel(
    float* __restrict__ tb,                 // [Hq, D] out
    const float* __restrict__ q,            // [Hq, D]
    const float* __restrict__ key_cache,    // [S, KV] for layer l
    const float* __restrict__ value_cache,  // [S, KV] for layer l
    const float* __restrict__ attn_sinks_layer, // [Hq]
    const float* __restrict__ mask,         // [S, S] or nullptr
    int Hq, int Hk, int D, int KV,          // KV = D * Hk
    int S, int pos                          // T_real = pos + 1
) {
  extern __shared__ double s_tile[]; // used for per-tile score/weight work
  const int h   = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= Hq) return;

  const int kv_mul = Hq / Hk;
  const int base   = (h / kv_mul) * D;
  const double scale = 1.0 / sqrt((double)D);

  // Pointers
  const float* qh = q + (size_t)h * D;    // [D]
  const float* K  = key_cache;            // [S, KV]
  const float* V  = value_cache;          // [S, KV]

  const int T_real = pos + 1;             // indices t ∈ [0, T_real)
  // Shared memory layout: first TILE_T for logits/weights, then blockDim.x for reductions
  double* s_logits = s_tile;
  double* s_reduce = s_tile + TILE_T;

  // 1) Find global max over all logits and sink
  double gmax = -INFINITY;
  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    // Compute logits for this tile
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      // Compute Q·K dot product with all threads participating
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)t * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      // Store logit and update local max
      if (tid == 0) {
        s_logits[tt] = logit;
        gmax = fmax(gmax, logit);
      }
      __syncthreads();
    }
  }
  // Include sink in global max
  const double sinkVal = (double)attn_sinks_layer[h];
  if (tid == 0) {
    gmax = fmax(gmax, sinkVal);
    s_reduce[0] = gmax; // Broadcast gmax
  }
  __syncthreads();
  gmax = s_reduce[0];

  // 2) Second pass: compute sumExp and numerator accumulation
  double sumExp = 0.0;
  // Each thread handles one output dimension i
  const int i0 = tid;
  double num_acc = 0.0;
  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    // Recompute logits for this tile (same as pass 1)
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      // Compute Q·K dot product
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)t * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      // Compute unnormalized weight and accumulate sumExp
      if (tid == 0) {
        double w = exp(logit - gmax);
        s_logits[tt] = w;  // Store weight as double
        sumExp += w;
      }
      __syncthreads();
    }
    // Accumulate numerator for owned output dimension
    if (i0 < D) {
      for (int tt = 0; tt < tile; ++tt) {
        const int t = t0 + tt;
        const double w = s_logits[tt];  // Double precision weight
        const float v = V[(size_t)t * KV + base + i0];
        num_acc += w * (double)v;
      }
    }
    __syncthreads();
  }
  // Add sink contribution to denominator (but not numerator)
  if (tid == 0) {
    const double sinkExp = exp(sinkVal - gmax);
    sumExp += sinkExp;
    s_reduce[0] = 1.0 / sumExp; // Store inverse for broadcast
  }
  __syncthreads();
  const double inv = s_reduce[0];
  // Write final result
  if (i0 < D) {
    tb[(size_t)h * D + i0] = (float)(num_acc * inv);
  }
}


// Paged fused attention kernel: scores + sink + softmax + values with KV indirection
// token2row[t] gives the physical row in key_cache/value_cache for logical token t.
template<int TILE_T>
__global__ void paged_attention_fused_kernel(
    float* __restrict__ tb,                 // [Hq, D] out
    const float* __restrict__ q,            // [Hq, D]
    const float* __restrict__ key_cache,    // [Rows, KV]
    const float* __restrict__ value_cache,  // [Rows, KV]
    const int* __restrict__ token2row,      // [S]
    const float* __restrict__ attn_sinks_layer, // [Hq]
    const float* __restrict__ mask,         // [S, S] or nullptr
    int Hq, int Hk, int D, int KV,          // KV = D * Hk
    int S, int pos                          // T_real = pos + 1
) {
  extern __shared__ double s_tile[]; // used for per-tile score/weight work
  const int h   = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= Hq) return;

  const int kv_mul = Hq / Hk;
  const int base   = (h / kv_mul) * D;
  const double scale = 1.0 / sqrt((double)D);

  // Pointers
  const float* qh = q + (size_t)h * D;    // [D]
  const float* K  = key_cache;            // [Rows, KV]
  const float* V  = value_cache;          // [Rows, KV]

  const int T_real = pos + 1;             // indices t ∈ [0, T_real)
  // Shared memory layout: first TILE_T for logits/weights, then blockDim.x for reductions
  double* s_logits = s_tile;
  double* s_reduce = s_tile + TILE_T;

  // 1) Find global max over all logits and sink
  double gmax = -INFINITY;
  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    // Compute logits for this tile
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      const int row = token2row[t];
      // Compute Q·K dot product with all threads participating
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)row * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      // Store logit and update local max
      if (tid == 0) {
        s_logits[tt] = logit;
        gmax = fmax(gmax, logit);
      }
      __syncthreads();
    }
  }
  // Include sink in global max
  const double sinkVal = (double)attn_sinks_layer[h];
  if (tid == 0) {
    gmax = fmax(gmax, sinkVal);
    s_reduce[0] = gmax; // Broadcast gmax
  }
  __syncthreads();
  gmax = s_reduce[0];

  // 2) Second pass: compute sumExp and numerator accumulation
  double sumExp = 0.0;
  // Each thread handles one output dimension i
  const int i0 = tid;
  double num_acc = 0.0;
  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);
    // Recompute logits for this tile (same as pass 1)
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      const int row = token2row[t];
      // Compute Q·K dot product
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)row * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      // Reduce to get full dot product
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
      // Apply scale and optional mask
      double logit = s_reduce[0] * scale;
      if (mask) {
        logit += (double)mask[(size_t)pos * S + t];
      }
      // Compute unnormalized weight and accumulate sumExp
      if (tid == 0) {
        double w = exp(logit - gmax);
        s_logits[tt] = w;  // Double precision weight
        sumExp += w;
      }
      __syncthreads();
    }
    // Accumulate numerator for owned output dimension
    if (i0 < D) {
      for (int tt = 0; tt < tile; ++tt) {
        const int t = t0 + tt;
        const int row = token2row[t];
        const double w = s_logits[tt];  // Double precision weight
        const float v = V[(size_t)row * KV + base + i0];
        num_acc += w * (double)v;
      }
    }
    __syncthreads();
  }
  // Add sink contribution to denominator (but not numerator)
  if (tid == 0) {
    const double sinkExp = exp(sinkVal - gmax);
    sumExp += sinkExp;
    s_reduce[0] = 1.0 / sumExp; // Store inverse for broadcast
  }
  __syncthreads();
  const double inv = s_reduce[0];
  // Write final result
  if (i0 < D) {
    tb[(size_t)h * D + i0] = (float)(num_acc * inv);
  }
}
