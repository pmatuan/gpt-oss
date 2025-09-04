// matmul.cpp — GEMM-tiled, multi-batch per block (no per-batch matvec loops)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>

typedef hip_bfloat16 bf16_t;

// ============================ Configuration ============================
#define TK 512
#define TM 8
#define BLOCK_SIZE 512
#define LDS_PAD 16
#define WF_SIZE 64

#define BATCH_TILE_DEFAULT 4
#define BATCH_TILE_LIGHT   2   // dùng cho kernel có nhiều truy cập W phụ thuộc batch (MoE)

// ============================ Utility ============================

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1) {
  union { uint32_t u; float f; } a, b;
  a.u = (u & 0x0000FFFFu) << 16; // lower bf16 -> fp32
  b.u = (u & 0xFFFF0000u);       // upper bf16 -> fp32
  f0 = a.f;
  f1 = b.f;
}

__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u) {
  float4 r;
  bf16pair_to_float2(u.x, r.x, r.y);
  bf16pair_to_float2(u.y, r.z, r.w);
  return r;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    v += __shfl_down(v, off, WF_SIZE);
  }
  return v;
}

// ======== Core inner: one weight row × CB batch columns (outer-product accumulate) ========

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

// ============================ Main Kernel Functions ============================

/**
 * Y = X @ W^T + B
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/TM), ceil(B/CB))
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void matmul_bias_gemm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {

  constexpr int CB = BATCH_TILE_DEFAULT;
  __shared__ __align__(16) float lds_x[CB][TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6; // warp id in block
  const int row  = blockIdx.x * TM + wid;
  const int b0   = blockIdx.y * CB;

  if (wid >= TM || row >= d) return;

  // Valid mask per batch col
  int b_idx[CB];
  bool b_valid[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi] = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
  }

  // Accumulators per batch column
  float acc[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) acc[bi] = 0.f;

  // K loop
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // Load CB columns of X into shared once
#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      const int b = b_idx[bi];
      if (b_valid[bi]) {
        const float* xb = x + (size_t)b * n + k_base;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = xb[k];
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = 0.f;
      }
    }
    __syncthreads();

    // One weight row slice (shared across CB)
    const T* __restrict__ w_row = w + (size_t)row * n + k_base;

    // Build pointer list for inner
    float* xptrs[CB];
#pragma unroll
    for (int bi = 0; bi < CB; ++bi) xptrs[bi] = lds_x[bi];

    if constexpr (std::is_same_v<T, bf16_t>) {
      gemm_row_tile_bf16_multiB<CB>(reinterpret_cast<const bf16_t*>(w_row), xptrs, k_size, lane, acc);
    } else {
      gemm_row_tile_fp32_multiB<CB>(reinterpret_cast<const float*>(w_row), xptrs, k_size, lane, acc);
    }
    __syncthreads();
  }

  // Reduce within warp and write
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float v = warp_reduce_sum(acc[bi]);
    if (lane == 0 && b_valid[bi]) {
      float* yb = y + (size_t)b_idx[bi] * d;
      yb[row] = v + b[row];
    }
  }
}

/**
 * MLP1 (Gate & Up): per-token expert weights -> W phụ thuộc batch (ít reuse)
 * gate_up: [B, IM], x: [B, H], w_mlp1_all: [L,E,2*IM,H], b_mlp1_all: [L,E,2*IM]
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up,
    const float* __restrict__ x,
    const T* __restrict__ w_mlp1_all,
    const float* __restrict__ b_mlp1_all,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, int experts_per_token) {

  // Do weights vary per batch -> dùng tile batch nhỏ để giữ register
  constexpr int CB = BATCH_TILE_LIGHT;
  __shared__ __align__(16) float lds_x[CB][TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int i    = blockIdx.x * TM + wid; // output index in IM
  const int b0   = blockIdx.y * CB;

  if (wid >= TM || i >= IM) return;

  int b_idx[CB]; bool b_valid[CB];
  int expert_id[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi]   = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
    expert_id[bi] = -1;
    if (b_valid[bi]) {
      const int* tk = topk_i + (size_t)b_idx[bi] * experts_per_token;
      expert_id[bi] = tk[k_index];
    }
  }

  // Accumulators per batch: gate & up
  float acc_gate[CB], acc_up[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) { acc_gate[bi] = 0.f; acc_up[bi] = 0.f; }

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load X tiles once
#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (b_valid[bi]) {
        const float* xb = x + (size_t)b_idx[bi] * H + k_base;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = xb[k];
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = 0.f;
      }
    }
    __syncthreads();

    // Với mỗi batch, W_gate/W_up khác nhau (MoE). Tính outer-product trên X đã share.
#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (!b_valid[bi] || expert_id[bi] < 0) continue;

      const size_t base = ((size_t)l_layer * E + expert_id[bi]) * (2 * IM);
      const size_t gate_off = (base + (2 * i + 0)) * (size_t)H + k_base;
      const size_t up_off   = (base + (2 * i + 1)) * (size_t)H + k_base;

      float* xptrs[1] = { lds_x[bi] }; // trick: reuse inner helpers with CB=1

      // Gate
      {
        float tmp[1] = {0.f};
        if constexpr (std::is_same_v<T, bf16_t>) {
          gemm_row_tile_bf16_multiB<1>((const bf16_t*)(w_mlp1_all + gate_off), xptrs, k_size, lane, tmp);
        } else {
          gemm_row_tile_fp32_multiB<1>((const float*)(w_mlp1_all + gate_off), xptrs, k_size, lane, tmp);
        }
        acc_gate[bi] += tmp[0];
      }
      // Up
      {
        float tmp[1] = {0.f};
        if constexpr (std::is_same_v<T, bf16_t>) {
          gemm_row_tile_bf16_multiB<1>((const bf16_t*)(w_mlp1_all + up_off), xptrs, k_size, lane, tmp);
        } else {
          gemm_row_tile_fp32_multiB<1>((const float*)(w_mlp1_all + up_off), xptrs, k_size, lane, tmp);
        }
        acc_up[bi] += tmp[0];
      }
    }
    __syncthreads();
  }

  // Reduce & write + bias + SwiGLU
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float g = warp_reduce_sum(acc_gate[bi]);
    float u = warp_reduce_sum(acc_up[bi]);
    if (lane == 0 && b_valid[bi] && expert_id[bi] >= 0) {
      const size_t base = ((size_t)l_layer * E + expert_id[bi]) * (2 * IM);
      float gate = g + b_mlp1_all[base + (2 * i + 0)];
      float up   = u + b_mlp1_all[base + (2 * i + 1)];
      // clip
      gate = fminf(fmaxf(gate, -swiglu_limit), swiglu_limit);
      up   = fminf(fmaxf(up,   -swiglu_limit), swiglu_limit);
      // SwiGLU (SwiGLU = SiLU(gate) * (up + 1))
      const float alpha = 1.702f;
      gate = gate * (1.0f / (1.0f + expf(-alpha * gate)));
      gate = gate * (up + 1.0f);
      float* gu = gate_up + (size_t)b_idx[bi] * IM;
      gu[i] = gate;
    }
  }
}

/**
 * MLP2: e_agg += (gate_up @ W^T + b) * weight
 * e_agg: [B, H], gate_up: [B, IM], W: [H, IM]
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,
    const float* __restrict__ gate_up,
    const T* __restrict__ w_mlp2_all,
    const float* __restrict__ b_mlp2_all,
    const int* __restrict__ topk_i,
    const float* __restrict__ topk_v,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int IM, int H,
    int batch_size, int experts_per_token) {

  constexpr int CB = BATCH_TILE_LIGHT;
  __shared__ __align__(16) float lds_x[CB][TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid; // output dim H
  const int b0   = blockIdx.y * CB;

  if (wid >= TM || row >= H) return;

  int b_idx[CB]; bool b_valid[CB];
  int expert_id[CB]; float expert_w[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi]   = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
    expert_id[bi] = -1; expert_w[bi] = 0.f;
    if (b_valid[bi]) {
      const int* ti = topk_i + (size_t)b_idx[bi] * experts_per_token;
      const float* tv = topk_v + (size_t)b_idx[bi] * experts_per_token;
      expert_id[bi] = ti[k_index];
      expert_w[bi]  = tv[k_index];
    }
  }

  // Each bi has its own accumulator (then reduced + atomicAdd with weight)
  float acc_row[CB]; for (int bi = 0; bi < CB; ++bi) acc_row[bi] = 0.f;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (b_valid[bi]) {
        const float* xb = gate_up + (size_t)b_idx[bi] * IM + k_base;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = xb[k];
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = 0.f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (!b_valid[bi] || expert_id[bi] < 0) continue;

      const size_t base = ((size_t)l_layer * E + expert_id[bi]) * (size_t)H * (size_t)IM;
      const T* __restrict__ w_row = w_mlp2_all + base + (size_t)row * IM + k_base;

      float* xptrs[1] = { lds_x[bi] };
      float tmp[1] = {0.f};
      if constexpr (std::is_same_v<T, bf16_t>) {
        gemm_row_tile_bf16_multiB<1>((const bf16_t*)w_row, xptrs, k_size, lane, tmp);
      } else {
        gemm_row_tile_fp32_multiB<1>((const float*)w_row, xptrs, k_size, lane, tmp);
      }
      acc_row[bi] += tmp[0];
    }
    __syncthreads();
  }

#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float v = warp_reduce_sum(acc_row[bi]);
    if (lane == 0 && b_valid[bi] && expert_id[bi] >= 0) {
      float* eB = e_agg + (size_t)b_idx[bi] * H;
      const size_t bbase = ((size_t)l_layer * E + expert_id[bi]) * (size_t)H;
      float out = v + b_mlp2_all[bbase + row];
      atomicAdd(&eB[row], out * expert_w[bi]);
    }
  }
}

/**
 * Compute inv RMS per sample: inv_rms = rsqrt(mean(x^2)+eps)
 */
__global__ void compute_inv_rms_batch_kernel(
    float* __restrict__ out_inv,
    const float* __restrict__ x,
    const int* __restrict__ pos,
    int H, int batch_size) {

  const int b = blockIdx.y;
  if (b >= batch_size) return;

  if (pos[b] < 0) { if (threadIdx.x == 0) out_inv[b] = 0.0f; return; }

  const float* xb = x + (size_t)b * H;

  float sum = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = xb[i]; sum = fmaf(v, v, sum);
  }
  sum = warp_reduce_sum(sum);

  __shared__ float warp_sums[1024 / WF_SIZE];
  const int lane = threadIdx.x & (WF_SIZE - 1);
  const int wid  = threadIdx.x >> 6;

  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();

  float total = 0.f;
  if (wid == 0) {
    const int num_warps = blockDim.x / WF_SIZE;
    total = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.f;
    total = warp_reduce_sum(total);
    if (lane == 0) {
      float mean_sq = total / (float)H;
      out_inv[b] = rsqrtf(mean_sq + 1e-5f);
    }
  }
}

/**
 * y = (RMSNorm(x) * rms_w) @ W^T
 * x: [B,H], w:[V,H], y:[B,V], inv_rms:[B]
 */
template <typename T>
__launch_bounds__(1024, 1) __global__
void fused_rmsnorm_matmul_gemm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ rms_w,
    const int* __restrict__ pos,
    const float* __restrict__ inv_rms,
    int H, int V, int batch_size) {

  constexpr int CB = BATCH_TILE_DEFAULT;
  __shared__ __align__(16) float lds_x[CB][TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int warp_id = tid >> 6;
  const int warps_per_block = blockDim.x / WF_SIZE;

  const int row = blockIdx.x * warps_per_block + warp_id; // output in V
  const int b0  = blockIdx.y * CB;
  if (row >= V) return;

  int b_idx[CB]; bool b_valid[CB]; float invv[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi]   = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
    invv[bi]    = b_valid[bi] ? inv_rms[b_idx[bi]] : 0.f;
  }

  float acc[CB]; for (int bi = 0; bi < CB; ++bi) acc[bi] = 0.f;

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (b_valid[bi]) {
        const float* xb = x + (size_t)b_idx[bi] * H + k_base;
        for (int k = tid; k < k_size; k += blockDim.x) {
          lds_x[bi][k] = xb[k] * invv[bi] * rms_w[k_base + k];
        }
      } else {
        for (int k = tid; k < k_size; k += blockDim.x) lds_x[bi][k] = 0.f;
      }
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * H + k_base;
    float* xptrs[CB]; for (int bi = 0; bi < CB; ++bi) xptrs[bi] = lds_x[bi];

    if constexpr (std::is_same_v<T, bf16_t>) {
      gemm_row_tile_bf16_multiB<CB>((const bf16_t*)w_row, xptrs, k_size, lane, acc);
    } else {
      gemm_row_tile_fp32_multiB<CB>((const float*)w_row, xptrs, k_size, lane, acc);
    }
    __syncthreads();
  }

#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float v = warp_reduce_sum(acc[bi]);
    if (lane == 0 && b_valid[bi]) {
      float* yb = y + (size_t)b_idx[bi] * V; yb[row] = v;
    }
  }
}

/**
 * (RMSNorm(x)*rms_w) @ W^T + b, within one kernel (computes RMS inside)
 * x:[B,n], w:[d,n], y:[B,d]
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_rmsnorm_matmul_bias_gemm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ rms_w,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {

  constexpr int CB = BATCH_TILE_DEFAULT;
  __shared__ __align__(16) float lds_x[CB][TK + LDS_PAD];
  __shared__ float rms_sum[CB];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid;
  const int b0   = blockIdx.y * CB;

  if (wid >= TM || row >= d) return;

  int b_idx[CB]; bool b_valid[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi]   = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
    if (tid == 0) rms_sum[bi] = 0.f;
  }
  __syncthreads();

  // Compute RMS
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    if (!b_valid[bi]) continue;
    const float* xb = x + (size_t)b_idx[bi] * n;
    float s = 0.f;
    for (int i = tid; i < n; i += BLOCK_SIZE) { float v = xb[i]; s = fmaf(v, v, s); }
    s = warp_reduce_sum(s);
    if (lane == 0) atomicAdd(&rms_sum[bi], s);
  }
  __syncthreads();

  float invr[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    invr[bi] = (b_valid[bi]) ? rsqrtf(rms_sum[bi] / n + 1e-5f) : 0.f;
  }

  float acc[CB]; for (int bi = 0; bi < CB; ++bi) acc[bi] = 0.f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (b_valid[bi]) {
        const float* xb = x + (size_t)b_idx[bi] * n + k_base;
        for (int k = tid; k < k_size; k += BLOCK_SIZE)
          lds_x[bi][k] = xb[k] * invr[bi] * rms_w[k_base + k];
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[bi][k] = 0.f;
      }
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    float* xptrs[CB]; for (int bi = 0; bi < CB; ++bi) xptrs[bi] = lds_x[bi];

    if constexpr (std::is_same_v<T, bf16_t>) {
      gemm_row_tile_bf16_multiB<CB>((const bf16_t*)w_row, xptrs, k_size, lane, acc);
    } else {
      gemm_row_tile_fp32_multiB<CB>((const float*)w_row, xptrs, k_size, lane, acc);
    }
    __syncthreads();
  }

#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float v = warp_reduce_sum(acc[bi]);
    if (lane == 0 && b_valid[bi]) {
      float* yb = y + (size_t)b_idx[bi] * d; yb[row] = v + b[row];
    }
  }
}

/**
 * x = x + (y @ W^T + b)
 * y:[B,n], w:[d,n], x:[B,d]
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_matmul_bias_residual_gemm_kernel(
    float* __restrict__ x_out,   // (accumulates into x_out)
    const float* __restrict__ y_in,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {

  constexpr int CB = BATCH_TILE_DEFAULT;
  __shared__ __align__(16) float lds_y[CB][TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid;
  const int b0   = blockIdx.y * CB;

  if (wid >= TM || row >= d) return;

  int b_idx[CB]; bool b_valid[CB];
#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    b_idx[bi]   = b0 + bi;
    b_valid[bi] = (b_idx[bi] < batch_size) && (pos[b_idx[bi]] >= 0);
  }

  float acc[CB]; for (int bi = 0; bi < CB; ++bi) acc[bi] = 0.f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

#pragma unroll
    for (int bi = 0; bi < CB; ++bi) {
      if (b_valid[bi]) {
        const float* yb = y_in + (size_t)b_idx[bi] * n + k_base;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_y[bi][k] = yb[k];
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_y[bi][k] = 0.f;
      }
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    float* xptrs[CB]; for (int bi = 0; bi < CB; ++bi) xptrs[bi] = lds_y[bi];

    if constexpr (std::is_same_v<T, bf16_t>) {
      gemm_row_tile_bf16_multiB<CB>((const bf16_t*)w_row, xptrs, k_size, lane, acc);
    } else {
      gemm_row_tile_fp32_multiB<CB>((const float*)w_row, xptrs, k_size, lane, acc);
    }
    __syncthreads();
  }

#pragma unroll
  for (int bi = 0; bi < CB; ++bi) {
    float v = warp_reduce_sum(acc[bi]);
    if (lane == 0 && b_valid[bi]) {
      float* xb = x_out + (size_t)b_idx[bi] * d;
      xb[row] = xb[row] + (v + b[row]);
    }
  }
}
