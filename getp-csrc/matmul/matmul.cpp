#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>

template<int CB>
__device__ __forceinline__ void gemm_row_tile_fp32_multiB(
    const float* __restrict__ w_row,         // [k_size] (slice of row)
    float* __restrict__ lds_x[CB],           // CB pointers -> [k_size] each
    int k_size, int lane, float acc[CB]) {

  const int vec4 = (k_size / 4) * 4;
  // Vector loop: 4 elements per lane-step
  for (int k = lane * 4; k < vec4; k += WF_SIZE * 4) {
    const float4 wv = *reinterpret_cast<const float4*>(&w_row[k]);
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      const float4 xv = *reinterpret_cast<const float4*>(&lds_x[b][k]);
      acc[b] = fmaf(wv.x, xv.x, acc[b]);
      acc[b] = fmaf(wv.y, xv.y, acc[b]);
      acc[b] = fmaf(wv.z, xv.z, acc[b]);
      acc[b] = fmaf(wv.w, xv.w, acc[b]);
    }
  }
  // Tail
  for (int k = vec4 + lane; k < k_size; k += WF_SIZE) {
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      acc[b] = fmaf(w_row[k], lds_x[b][k], acc[b]);
    }
  }
}

template<int CB>
__device__ __forceinline__ void gemm_row_tile_bf16_multiB(
    const bf16_t* __restrict__ w_row_bf16,   // [k_size] (slice of row, bf16)
    float* __restrict__ lds_x[CB],           // CB pointers -> [k_size] each (fp32)
    int k_size, int lane, float acc[CB]) {

  // Treat weights as bf16 quad (uint2)
  const uint2* __restrict__ wq = reinterpret_cast<const uint2*>(w_row_bf16);
  const int vec4 = (k_size / 4) * 4;

  // 4-wide vector loop
  for (int k = lane * 4; k < vec4; k += WF_SIZE * 4) {
    const float4 wv = bf16quad_to_float4(wq[k >> 2]);
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      const float4 xv = *reinterpret_cast<const float4*>(&lds_x[b][k]);
      acc[b] = fmaf(wv.x, xv.x, acc[b]);
      acc[b] = fmaf(wv.y, xv.y, acc[b]);
      acc[b] = fmaf(wv.z, xv.z, acc[b]);
      acc[b] = fmaf(wv.w, xv.w, acc[b]);
    }
  }
  // Tail (pairs + singles)
  // pairs
  const uint32_t* __restrict__ w32 = reinterpret_cast<const uint32_t*>(w_row_bf16);
  const int pairs = k_size >> 1;
  const int vec_pairs = (vec4 >> 1);
  for (int p = vec_pairs + lane; p < pairs; p += WF_SIZE) {
    const int k = p << 1;
    float w0, w1; bf16pair_to_float2(w32[p], w0, w1);
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      float a = fmaf(w0, lds_x[b][k], 0.f);
      float bacc = a;
      if (k + 1 < k_size) bacc = fmaf(w1, lds_x[b][k + 1], bacc);
      acc[b] += bacc;
    }
  }
  // odd
  if (k_size & 1) {
    const int k = k_size - 1;
    if ((k & (WF_SIZE - 1)) == lane) {
      uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row_bf16[k]))) << 16;
      union { uint32_t u; float f; } cvt; cvt.u = u;
#pragma unroll
      for (int b = 0; b < CB; ++b) acc[b] = fmaf(cvt.f, lds_x[b][k], acc[b]);
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
   __shared__ __align__(16) float lds_x[TK + LDS_PAD];
 
   const int tid  = threadIdx.x;
   const int lane = tid & (WF_SIZE - 1);
   const int wid  = tid >> 6;                 // warp id trong block (0..TM-1)
 
   const int row  = blockIdx.x * TM + wid;    // hàng output (0..d-1)
   const int b    = blockIdx.y;               // batch index (0..B-1)
 
   if (wid >= TM || row >= d || b >= batch_size) return;
   if (pos[b] < 0) return; // slot inactive
 
   float acc = 0.f;
 
   // Vòng K
   for (int k_base = 0; k_base < n; k_base += TK) {
     const int k_size = min(TK, n - k_base);
 
     // 1) Tải 1 cột X của batch b vào shared (không CB)
     const float* __restrict__ xb = x + (size_t)b * n + k_base;
     for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[k] = xb[k];
     __syncthreads();
 
     // 2) Lấy 1 lát hàng của W (row fixed), tính dot với lds_x
     const T* __restrict__ w_row = w + (size_t)row * n + k_base;
 
     if constexpr (std::is_same_v<T, bf16_t>) {
       // tải bf16 -> fp32, vector theo cặp
       const unsigned short* wbf = reinterpret_cast<const unsigned short*>(w_row);
       const int vec2 = (k_size >> 1); // 2 phần tử mỗi lần
       for (int v = lane; v < vec2; v += WF_SIZE) {
         // pack 2 bf16 thành 1 u32 rồi convert ra 2 float
         const uint32_t packed =
             ((uint32_t)wbf[(v << 1) + 0]) |
             (((uint32_t)wbf[(v << 1) + 1]) << 16);
         float f0, f1; bf16pair_to_float2(packed, f0, f1);
         const int k = v << 1;
         acc = fmaf(f0, lds_x[k + 0], acc);
         acc = fmaf(f1, lds_x[k + 1], acc);
       }
       // phần lẻ
       if (k_size & 1) {
         const int k = k_size - 1;
         if ((k & (WF_SIZE - 1)) == lane) {
           uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
           union { uint32_t u; float f; } cvt; cvt.u = u;
           acc = fmaf(cvt.f, lds_x[k], acc);
         }
       }
     } else {
       // w là float32
       const int vec4 = (k_size >> 2);
       const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w_row);
       for (int v = lane; v < vec4; v += WF_SIZE) {
         const float4 wv = w4[v];
         const float4 xv = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
         acc = fmaf(wv.x, xv.x, acc);
         acc = fmaf(wv.y, xv.y, acc);
         acc = fmaf(wv.z, xv.z, acc);
         acc = fmaf(wv.w, xv.w, acc);
       }
       for (int k = (vec4 << 2) + lane; k < k_size; k += WF_SIZE) {
         acc = fmaf(reinterpret_cast<const float*>(w_row)[k], lds_x[k], acc);
       }
     }
     __syncthreads();
   }
 
   // reduce trong warp
   float v = warp_reduce_sum(acc);
   if (lane == 0) {
     float* __restrict__ yb = y + (size_t)b * d;
     yb[row] = v + (bias ? bias[row] : 0.0f);
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
  __shared__ __align__(16) float lds_x[TK + LDS_PAD]; // one column (this b)
  __shared__ int s_expert_id;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;          // warp id in block

  const int i    = blockIdx.x * TM + wid;   // output row in IM (0..IM-1)
  const int b    = blockIdx.y;              // batch index
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || i >= IM || b >= batch_size || kidx >= experts_per_token)
    return;
  if (pos[b] < 0) return; // inactive slot

  // expert id for this (b, kidx)
  if (tid == 0) {
    s_expert_id = topk_i[(size_t)b * experts_per_token + kidx];
  }
  __syncthreads();
  if (s_expert_id < 0) return;

  float acc_gate = 0.f, acc_up = 0.f;

  // K loop over H, loading x[b, k_base : k_base+k_size) once per block
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // vectorized load x[b] -> shared
    const float* __restrict__ xb = x + (size_t)b * H + k_base;
    // vec4 part
    const int vec4 = (k_size >> 2);
    float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
    // tail
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // weights layout:
    // base = ((l * E + expert) * (2*IM)) * H + (2*i + {0,1}) * H + k_base
    const size_t base_2im = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id) * (size_t)(2 * IM);
    const size_t gate_off = ((base_2im + (size_t)(2 * i + 0)) * (size_t)H) + (size_t)k_base;
    const size_t up_off   = ((base_2im + (size_t)(2 * i + 1)) * (size_t)H) + (size_t)k_base;

    const uint2* __restrict__ gate_q = reinterpret_cast<const uint2*>(w_mlp1_all + gate_off);
    const uint2* __restrict__  up_q  = reinterpret_cast<const uint2*>(w_mlp1_all + up_off);

    // vector compute (4 bf16 weights at a time)
    const int vec4k = (k_size >> 2);
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 w_gate = bf16quad_to_float4(gate_q[v]);
      const float4 w_up   = bf16quad_to_float4(up_q[v]);
      const float4 xv     = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      acc_gate = fmaf(w_gate.x, xv.x, acc_gate);
      acc_gate = fmaf(w_gate.y, xv.y, acc_gate);
      acc_gate = fmaf(w_gate.z, xv.z, acc_gate);
      acc_gate = fmaf(w_gate.w, xv.w, acc_gate);

      acc_up   = fmaf(w_up.x,   xv.x, acc_up);
      acc_up   = fmaf(w_up.y,   xv.y, acc_up);
      acc_up   = fmaf(w_up.z,   xv.z, acc_up);
      acc_up   = fmaf(w_up.w,   xv.w, acc_up);
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
      acc_gate = fmaf(cg.f, lds_x[k], acc_gate);
      acc_up   = fmaf(cu.f, lds_x[k], acc_up);
    }
    __syncthreads();
  }

  // warp reduce per output
  acc_gate = warp_reduce_sum(acc_gate);
  acc_up   = warp_reduce_sum(acc_up);

  if (lane == 0) {
    const size_t b_gate_base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id) * (size_t)(2 * IM);
    float gate = acc_gate + b_mlp1_all[b_gate_base + (size_t)(2 * i + 0)];
    float up   = acc_up   + b_mlp1_all[b_gate_base + (size_t)(2 * i + 1)];

    // clip + SwiGLU-ish (như bản bạn đang dùng)
    gate = __saturatef((gate + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    up   = __saturatef((up   + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    const float alpha = 1.702f;
    gate = gate * __saturatef(0.5f + 0.5f * tanhf(alpha * gate * 0.5f));
    gate = gate * (up + 1.0f);

    float* __restrict__ gu = gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)b) * (size_t)IM);
    gu[i] = gate;
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
  __shared__ __align__(16) float lds_x[TK + LDS_PAD]; // one column (this b,k)
  __shared__ int   s_expert_id;
  __shared__ float s_expert_w;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;

  const int row  = blockIdx.x * TM + wid;   // output H row
  const int b    = blockIdx.y;              // batch index
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || row >= H || b >= batch_size || kidx >= experts_per_token)
    return;
  if (pos[b] < 0) return;

  // expert id & weight for this (b,kidx)
  if (tid == 0) {
    s_expert_id = topk_i[(size_t)b * experts_per_token + kidx];
    s_expert_w  = topk_v[(size_t)b * experts_per_token + kidx];
  }
  __syncthreads();
  if (s_expert_id < 0 || s_expert_w == 0.f) return;

  float acc = 0.f;

  // K loop over IM, loading gate_up_topk[kidx, b, :] into shared
  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    const float* __restrict__ xb =
        gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)b) * (size_t)IM + (size_t)k_base);

    // vectorized load to shared
    const int vec4 = (k_size >> 2);
    float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // weight row: w[l, expert, row, k_base: ]
    const size_t base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id) * (size_t)H * (size_t)IM;
    const bf16_t* __restrict__ w_row = w_mlp2_all + base + (size_t)row * (size_t)IM + (size_t)k_base;

    // vector compute
    const uint2* __restrict__ wq = reinterpret_cast<const uint2*>(w_row);
    const int vec4k = (k_size >> 2);
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 wv = bf16quad_to_float4(wq[v]);
      const float4 xv = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    // tail
    for (int k = (vec4k << 2) + lane; k < k_size; k += WF_SIZE) {
      uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
      union { uint32_t u; float f; } cvt; cvt.u = u;
      acc = fmaf(cvt.f, lds_x[k], acc);
    }
    __syncthreads();
  }

  // reduce and write once (1 atomic per (b,row) per expert)
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    const size_t b_mlp2_base = ((size_t)l_layer * (size_t)E + (size_t)s_expert_id) * (size_t)H;
    float out = acc + b_mlp2_all[b_mlp2_base + (size_t)row];
    float contrib = out * s_expert_w;
    atomicAdd(e_agg + (size_t)b * (size_t)H + (size_t)row, contrib);
  }
}
