#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>

template <int CB>
__device__ __forceinline__ void gemm_row_tile_fp32_multiB(
    const float *__restrict__ w_row,
    float *__restrict__ lds_x[CB],
    int k_size, int lane, float acc[CB]) {
  const int vec_k = (k_size / MFMA_K) * MFMA_K;

  for (int k = lane * MFMA_K; k < vec_k; k += WF_SIZE * MFMA_K) {
    if (k + MFMA_K <= vec_k) {
      mfma_float4 w_vec;
#pragma unroll
      for (int i = 0; i < MFMA_K; ++i) {
        w_vec[i] = w_row[k + i];
      }

#pragma unroll
      for (int b = 0; b < CB; ++b) {
        mfma_float4 x_vec;
#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          x_vec[i] = lds_x[b][k + i];
        }

        mfma_float4 acc_vec = {0};

        acc_vec = __builtin_amdgcn_mfma_f32_16x16x4f32(w_vec[0], x_vec[0],
                                                       acc_vec, 0, 0, 0);

#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          acc[b] = fmaf(w_vec[i], x_vec[i], acc[b]);
        }
      }
    }
  }

  for (int k = vec_k + lane; k < k_size; k += WF_SIZE) {
    float w_val = w_row[k];
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      acc[b] = fmaf(w_val, lds_x[b][k], acc[b]);
    }
  }
}

template<int CB>
__device__ __forceinline__ void gemm_row_tile_bf16_multiB(
    const bf16_t* __restrict__ w_row_bf16,   // [k_size] (slice of row, bf16)
    float* __restrict__ lds_x[CB],           // CB pointers -> [k_size] each (fp32)
    int k_size, int lane, float acc[CB]) {
  
  const int vec_k = (k_size / MFMA_K) * MFMA_K;

  for (int k = lane * MFMA_K; k < vec_k; k += WF_SIZE * MFMA_K) {
    if (k + MFMA_K <= vec_k) {
      mfma_float4 w_vec;
#pragma unroll
      for (int i = 0; i < MFMA_K; ++i) {
        uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row_bf16[k + i]))) << 16;
        union { uint32_t u; float f; } cvt; cvt.u = u;
        w_vec[i] = cvt.f;
      }

#pragma unroll
      for (int b = 0; b < CB; ++b) {
        mfma_float4 x_vec;
#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          x_vec[i] = lds_x[b][k + i];
        }

        mfma_float4 acc_vec = {0};

        acc_vec = __builtin_amdgcn_mfma_f32_16x16x4f32(w_vec[0], x_vec[0],
                                                       acc_vec, 0, 0, 0);

#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          acc[b] = fmaf(w_vec[i], x_vec[i], acc[b]);
        }
      }
    }
  }

  for (int k = vec_k + lane; k < k_size; k += WF_SIZE) {
    uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row_bf16[k]))) << 16;
    union { uint32_t u; float f; } cvt; cvt.u = u;
    float w_val = cvt.f;
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      acc[b] = fmaf(w_val, lds_x[b][k], acc[b]);
    }
  }
}

/**
 * Y = X @ W^T + B
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/TM), ceil(B/CB))
 */
 template <typename T>
 __launch_bounds__(BLOCK_SIZE, 1) __global__
 void matmul_bias_gemm_kernel(
     float* __restrict__ y,          // [B, d]
     const float* __restrict__ x,    // [B, n]
     const T* __restrict__ w,        // [d, n] (row-major theo n)
     const float* __restrict__ bias, // [d] (có thể null)
     const int* __restrict__ pos,    // [B] (slot inactive nếu pos[b] < 0)
     int n, int d, int batch_size)
 {
   constexpr int BATCH_TILE = 4; // denser layers can handle a larger tile
   const int batch_base = blockIdx.y * BATCH_TILE;
   const int bmax = (batch_size - batch_base > BATCH_TILE) ? BATCH_TILE : (batch_size - batch_base);
   if (bmax <= 0) return;

   __shared__ __align__(16) float lds_x[BATCH_TILE][TK + LDS_PAD];
 
   const int tid  = threadIdx.x;
   const int lane = tid & (WF_SIZE - 1);
   const int wid  = tid >> 6;                 // warp id trong block (0..TM-1)
 
   const int row  = blockIdx.x * TM + wid;    // hàng output (0..d-1)
 
   if (wid >= TM || row >= d) return;
 
   float acc[BATCH_TILE];
#pragma unroll
   for (int b = 0; b < BATCH_TILE; ++b) {
     acc[b] = 0.f;
   }
 
   // Vòng K
   for (int k_base = 0; k_base < n; k_base += TK) {
     const int k_size = min(TK, n - k_base);
 
     // 1) Load X columns for each batch in the tile
     for (int b = 0; b < BATCH_TILE; ++b) {
       if (b >= bmax) break;
       const int bb = batch_base + b;
       if (pos[bb] < 0) continue;
       const float* __restrict__ xb = x + (size_t)bb * n + k_base;
       for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[b][k] = xb[k];
     }
     __syncthreads();
 
     // 2) Compute dot products with weight row for all active batches
     const T* __restrict__ w_row = w + (size_t)row * n + k_base;
     float* __restrict__ lds_x_ptrs[BATCH_TILE];
#pragma unroll
     for (int b = 0; b < BATCH_TILE; ++b) {
       lds_x_ptrs[b] = lds_x[b];
     }
 
     if constexpr (std::is_same_v<T, bf16_t>) {
       gemm_row_tile_bf16_multiB<BATCH_TILE>(w_row, lds_x_ptrs, k_size, lane, acc);
     } else {
       gemm_row_tile_fp32_multiB<BATCH_TILE>(w_row, lds_x_ptrs, k_size, lane, acc);
     }
     __syncthreads();
   }
 
   // reduce and write outputs for each batch
   for (int b = 0; b < BATCH_TILE; ++b) {
     if (b >= bmax) break;
     const int bb = batch_base + b;
     if (pos[bb] < 0) continue;
     float v = warp_reduce_sum(acc[b]);
     if (lane == 0) {
       float* __restrict__ yb = y + (size_t)bb * d;
       yb[row] = v + (bias ? bias[row] : 0.0f);
     }
   }
 }

// ================= MLP1 (Gate & Up) : per-batch, no CB =================
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up_topk, // [K, B, IM] (K = experts_per_token)
    const float* __restrict__ x,      // [B, H]
    const bf16_t* __restrict__ w_mlp1_all, // [L, E, 2*IM, H] (row-major in last dim)
    const float* __restrict__ b_mlp1_all,  // [L, E, 2*IM]
    const int* __restrict__ topk_i,   // [B, K]
    const int* __restrict__ pos,      // [B] (inactive: pos[b] < 0)
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, int experts_per_token)
{
  constexpr int BATCH_TILE = 2; // MoE-heavy, keep small to preserve occupancy
  const int batch_base = blockIdx.y * BATCH_TILE;
  const int bmax = (batch_size - batch_base > BATCH_TILE) ? BATCH_TILE : (batch_size - batch_base);
  if (bmax <= 0) return;

  __shared__ __align__(16) float lds_x[BATCH_TILE][TK + LDS_PAD];
  __shared__ int s_expert_id[BATCH_TILE];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;          // warp id in block

  const int i    = blockIdx.x * TM + wid;   // output row in IM (0..IM-1)
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || i >= IM || kidx >= experts_per_token)
    return;

  // Load expert IDs for each batch in tile
  if (tid < BATCH_TILE) {
    if (tid < bmax) {
      const int bb = batch_base + tid;
      if (pos[bb] >= 0) {
        s_expert_id[tid] = topk_i[(size_t)bb * experts_per_token + kidx];
      } else {
        s_expert_id[tid] = -1;
      }
    } else {
      s_expert_id[tid] = -1;
    }
  }
  __syncthreads();

  float acc_gate[BATCH_TILE], acc_up[BATCH_TILE];
#pragma unroll
  for (int b = 0; b < BATCH_TILE; ++b) {
    acc_gate[b] = 0.f;
    acc_up[b] = 0.f;
  }

  // K loop over H, loading x for each batch in tile
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load X for each batch in tile
    for (int b = 0; b < BATCH_TILE; ++b) {
      if (b >= bmax) break;
      const int bb = batch_base + b;
      if (pos[bb] < 0 || s_expert_id[b] < 0) continue;
      
      const float* __restrict__ xb = x + (size_t)bb * H + k_base;
      // vec4 part
      const int vec4 = (k_size >> 2);
      float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x[b]);
      const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
      for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
      // tail
      for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
        lds_x[b][k] = xb[k];
      }
    }
    __syncthreads();

    // Compute for each batch separately (different experts may be used)
    for (int b = 0; b < BATCH_TILE; ++b) {
      if (b >= bmax) break;
      const int bb = batch_base + b;
      if (pos[bb] < 0 || s_expert_id[b] < 0) continue;

      // weights layout:
      // base = ((l * E + expert) * (2*IM)) * H + (2*i + {0,1}) * H + k_base
      const size_t base_2im = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id[b]) * (size_t)(2 * IM);
      const size_t gate_off = ((base_2im + (size_t)(2 * i + 0)) * (size_t)H) + (size_t)k_base;
      const size_t up_off   = ((base_2im + (size_t)(2 * i + 1)) * (size_t)H) + (size_t)k_base;

      const uint2* __restrict__ gate_q = reinterpret_cast<const uint2*>(w_mlp1_all + gate_off);
      const uint2* __restrict__  up_q  = reinterpret_cast<const uint2*>(w_mlp1_all + up_off);

      // vector compute (4 bf16 weights at a time)
      const int vec4k = (k_size >> 2);
      for (int v = lane; v < vec4k; v += WF_SIZE) {
        const float4 w_gate = bf16quad_to_float4(gate_q[v]);
        const float4 w_up   = bf16quad_to_float4(up_q[v]);
        const float4 xv     = *reinterpret_cast<const float4*>(&lds_x[b][v << 2]);
        acc_gate[b] = fmaf(w_gate.x, xv.x, acc_gate[b]);
        acc_gate[b] = fmaf(w_gate.y, xv.y, acc_gate[b]);
        acc_gate[b] = fmaf(w_gate.z, xv.z, acc_gate[b]);
        acc_gate[b] = fmaf(w_gate.w, xv.w, acc_gate[b]);

        acc_up[b]   = fmaf(w_up.x,   xv.x, acc_up[b]);
        acc_up[b]   = fmaf(w_up.y,   xv.y, acc_up[b]);
        acc_up[b]   = fmaf(w_up.z,   xv.z, acc_up[b]);
        acc_up[b]   = fmaf(w_up.w,   xv.w, acc_up[b]);
      }
      // tail pairs/singles
      const int tail_start = vec4k << 2;
      for (int k = tail_start + lane; k < k_size; k += WF_SIZE) {
        // bf16 -> f32 (single)
        uint32_t ug = ((uint32_t)(*reinterpret_cast<const uint16_t*>(
                          (w_mlp1_all + gate_off + k)))) << 16;
        uint32_t uu = ((uint32_t)(*reinterpret_cast<const uint16_t*>(
                          (w_mlp1_all + up_off   + k)))) << 16;
        union { uint32_t u; float f; } cg, cu; cg.u = ug; cu.u = uu;
        acc_gate[b] = fmaf(cg.f, lds_x[b][k], acc_gate[b]);
        acc_up[b]   = fmaf(cu.f, lds_x[b][k], acc_up[b]);
      }
    }
    __syncthreads();
  }

  // warp reduce and output for each batch
  for (int b = 0; b < BATCH_TILE; ++b) {
    if (b >= bmax) break;
    const int bb = batch_base + b;
    if (pos[bb] < 0 || s_expert_id[b] < 0) continue;

    float gate_sum = warp_reduce_sum(acc_gate[b]);
    float up_sum   = warp_reduce_sum(acc_up[b]);

    if (lane == 0) {
      const size_t b_gate_base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id[b]) * (size_t)(2 * IM);
      float gate = gate_sum + b_mlp1_all[b_gate_base + (size_t)(2 * i + 0)];
      float up   = up_sum   + b_mlp1_all[b_gate_base + (size_t)(2 * i + 1)];

      // clip + SwiGLU-ish (như bản bạn đang dùng)
      gate = __saturatef((gate + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
      up   = __saturatef((up   + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
      const float alpha = 1.702f;
      gate = gate * __saturatef(0.5f + 0.5f * tanhf(alpha * gate * 0.5f));
      gate = gate * (up + 1.0f);

      float* __restrict__ gu = gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)bb) * (size_t)IM);
      gu[i] = gate;
    }
  }
}


// ============ MLP2 (weighted accum) : per-batch, no CB ==============
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,              // [B, H] (accumulator)
    const float* __restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t* __restrict__ w_mlp2_all,  // [L, E, H, IM]
    const float* __restrict__ b_mlp2_all,   // [L, E, H]
    const int* __restrict__ topk_i,         // [B, K]
    const float* __restrict__ topk_v,       // [B, K]
    const int* __restrict__ pos,            // [B]
    int l_layer, int E, int IM, int H,
    int batch_size, int experts_per_token)
{
  constexpr int BATCH_TILE = 2; // MoE-heavy, keep small to preserve occupancy
  const int batch_base = blockIdx.y * BATCH_TILE;
  const int bmax = (batch_size - batch_base > BATCH_TILE) ? BATCH_TILE : (batch_size - batch_base);
  if (bmax <= 0) return;

  __shared__ __align__(16) float lds_x[BATCH_TILE][TK + LDS_PAD];
  __shared__ int   s_expert_id[BATCH_TILE];
  __shared__ float s_expert_w[BATCH_TILE];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;

  const int row  = blockIdx.x * TM + wid;   // output H row
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || row >= H || kidx >= experts_per_token)
    return;

  // Load expert IDs and weights for each batch in tile
  if (tid < BATCH_TILE) {
    if (tid < bmax) {
      const int bb = batch_base + tid;
      if (pos[bb] >= 0) {
        s_expert_id[tid] = topk_i[(size_t)bb * experts_per_token + kidx];
        s_expert_w[tid] = topk_v[(size_t)bb * experts_per_token + kidx];
      } else {
        s_expert_id[tid] = -1;
        s_expert_w[tid] = 0.f;
      }
    } else {
      s_expert_id[tid] = -1;
      s_expert_w[tid] = 0.f;
    }
  }
  __syncthreads();

  float acc[BATCH_TILE];
#pragma unroll
  for (int b = 0; b < BATCH_TILE; ++b) {
    acc[b] = 0.f;
  }

  // K loop over IM, loading gate_up_topk for each batch
  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    // Load data for each batch in tile
    for (int b = 0; b < BATCH_TILE; ++b) {
      if (b >= bmax) break;
      const int bb = batch_base + b;
      if (pos[bb] < 0 || s_expert_id[b] < 0 || s_expert_w[b] == 0.f) continue;

      const float* __restrict__ xb =
          gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)bb) * (size_t)IM + (size_t)k_base);

      // vectorized load to shared
      const int vec4 = (k_size >> 2);
      float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x[b]);
      const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
      for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
      for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
        lds_x[b][k] = xb[k];
      }
    }
    __syncthreads();

    // Compute for each batch separately (different experts may be used)
    for (int b = 0; b < BATCH_TILE; ++b) {
      if (b >= bmax) break;
      const int bb = batch_base + b;
      if (pos[bb] < 0 || s_expert_id[b] < 0 || s_expert_w[b] == 0.f) continue;

      // weight row: w[l, expert, row, k_base: ]
      const size_t base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id[b]) * (size_t)H * (size_t)IM;
      const bf16_t* __restrict__ w_row = w_mlp2_all + base + (size_t)row * (size_t)IM + (size_t)k_base;

      // vector compute
      const uint2* __restrict__ wq = reinterpret_cast<const uint2*>(w_row);
      const int vec4k = (k_size >> 2);
      for (int v = lane; v < vec4k; v += WF_SIZE) {
        const float4 wv = bf16quad_to_float4(wq[v]);
        const float4 xv = *reinterpret_cast<const float4*>(&lds_x[b][v << 2]);
        acc[b] = fmaf(wv.x, xv.x, acc[b]);
        acc[b] = fmaf(wv.y, xv.y, acc[b]);
        acc[b] = fmaf(wv.z, xv.z, acc[b]);
        acc[b] = fmaf(wv.w, xv.w, acc[b]);
      }
      // tail
      for (int k = (vec4k << 2) + lane; k < k_size; k += WF_SIZE) {
        uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
        union { uint32_t u; float f; } cvt; cvt.u = u;
        acc[b] = fmaf(cvt.f, lds_x[b][k], acc[b]);
      }
    }
    __syncthreads();
  }

  // reduce and write for each batch (1 atomic per (b,row) per expert)
  for (int b = 0; b < BATCH_TILE; ++b) {
    if (b >= bmax) break;
    const int bb = batch_base + b;
    if (pos[bb] < 0 || s_expert_id[b] < 0 || s_expert_w[b] == 0.f) continue;

    float acc_sum = warp_reduce_sum(acc[b]);
    if (lane == 0) {
      const size_t b_mlp2_base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id[b]) * (size_t)H;
      float out = acc_sum + b_mlp2_all[b_mlp2_base + (size_t)row];
      float contrib = out * s_expert_w[b];
      atomicAdd(e_agg + (size_t)bb * (size_t)H + (size_t)row, contrib);
    }
  }
}
