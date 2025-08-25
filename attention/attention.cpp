#include "attention.hpp"
#include <math.h>

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
  extern __shared__ double s_tile[]; // used for per-tile logits/weights
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
  // Shared memory layout: first TILE_T for logits/weights, then blockDim.x for reductions/broadcasts
  double* s_logits = s_tile;              // [TILE_T]
  double* s_reduce = s_tile + TILE_T;     // [blockDim.x]

  // Streaming softmax state
  double m_run = -INFINITY;  // running max
  double s_run = 0.0;        // running denominator sum
  const int i0 = tid;        // output dim owned by this thread
  double num_acc = 0.0;      // running numerator for dim i0

  for (int t0 = 0; t0 < T_real; t0 += TILE_T) {
    const int tile = min(TILE_T, T_real - t0);

    // 1) Compute logits for this tile and track tile_max
    double tile_max = -INFINITY;
    for (int tt = 0; tt < tile; ++tt) {
      const int t = t0 + tt;
      int row = token2row[t];
      // Clamp row to valid range to avoid OOB/NaNs if mapping is invalid
      if (row < 0) row = 0; else if (row >= S) row = S - 1;
      // Dot(Q_h, K_row)
      double part = 0.0;
      for (int i = tid; i < D; i += blockDim.x) {
        const float k = K[(size_t)row * KV + base + i];
        part += (double)qh[i] * (double)k;
      }
      // Block reduction to get full dot
      s_reduce[tid] = part;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
      }
  double logit = s_reduce[0] * scale;
      if (mask) logit += (double)mask[(size_t)pos * S + t];
  if (!isfinite(logit)) logit = -1e30; // guard
      if (tid == 0) {
        s_logits[tt] = logit;          // store logit for this tt
        tile_max = fmax(tile_max, logit);
      }
      __syncthreads();
    }

    // 2) Rescale running sums to new base m_new = max(m_run, tile_max)
    if (tid == 0) {
      double m_new = fmax(m_run, tile_max);
      double scale_old = isfinite(m_run) ? exp(m_run - m_new) : 0.0;
      s_run *= scale_old;          // rescale denominator
      s_reduce[0] = m_new;         // broadcast m_new
      s_reduce[1] = scale_old;     // broadcast scaling for numerator
      s_reduce[2] = exp(tile_max - m_new); // factor to convert tile-local weights to base m_new
    }
    __syncthreads();
    const double m_new = s_reduce[0];
    const double num_scale = s_reduce[1];
    const double factor = s_reduce[2];
    // Rescale numerator for this thread
    num_acc *= num_scale;

    // 3) Compute tile weights relative to tile_max, transform to base m_new, and accumulate
    if (i0 < D) {
      for (int tt = 0; tt < tile; ++tt) {
        // Thread 0 computes and stores weight; others read it
        if (tid == 0) {
          double w_local = exp(s_logits[tt] - (double)tile_max);
          s_logits[tt] = w_local * factor; // weight in base m_new
          s_run += s_logits[tt];
        }
        __syncthreads();
        const int t = t0 + tt;
        int row = token2row[t];
        if (row < 0) row = 0; else if (row >= S) row = S - 1;
        const float v = V[(size_t)row * KV + base + i0];
        num_acc += s_logits[tt] * (double)v;
        __syncthreads();
      }
    } else {
      // Even if this thread doesn't own an output dim, we still need s_run computed by tid 0
  for (int tt = 0; tt < tile; ++tt) {
        if (tid == 0) {
          double w_local = exp(s_logits[tt] - (double)tile_max);
          s_logits[tt] = w_local * factor; // weight in base m_new
          s_run += s_logits[tt];
        }
        __syncthreads();
      }
    }
    __syncthreads();
    // Update running max
    if (tid == 0) m_run = m_new;
    __syncthreads();
  }

  // 4) Add sink to denominator (not numerator), adjust base if needed
  const double sinkVal = (double)attn_sinks_layer[h];
  if (tid == 0) {
    double m_new = fmax(m_run, sinkVal);
    double scale_old = isfinite(m_run) ? exp(m_run - m_new) : 0.0;
    s_run = s_run * scale_old + exp(sinkVal - m_new);
    s_reduce[0] = 1.0 / s_run; // broadcast inverse denominator
    s_reduce[1] = scale_old;   // broadcast final rescale for numerator
  }
  __syncthreads();
  double inv = s_reduce[0];
  if (!isfinite(inv)) inv = 0.0; // guard
  const double final_scale = s_reduce[1];
  num_acc *= final_scale;
  if (i0 < D) tb[(size_t)h * D + i0] = (float)(num_acc * inv);
}
