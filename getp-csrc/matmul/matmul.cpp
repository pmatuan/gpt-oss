#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>


/**
 * Y = X @ W^T + B (bf16 version)
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/TM), B)
 */
__launch_bounds__(BLOCK_SIZE, 1) __global__
void matmul_bias_gemm_kernel_bf16(
    float* __restrict__ y,          // [B, d]
    const float* __restrict__ x,    // [B, n]
    const bf16_t* __restrict__ w,   // [d, n] (row-major theo n)
    const float* __restrict__ bias, // [d] (có thể null)
    int n, int d, int batch_size)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;                 // warp id trong block (0..TM-1)

  const int row  = blockIdx.x * TM + wid;    // hàng output (0..d-1)

  if (wid >= TM || row >= d) return;

  float acc = 0.f;

  // Vòng K
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // 1) Load X columns for current batch
    const float* __restrict__ xb = x + (size_t)batch_idx * n + k_base;
    for (int k = tid; k < k_size; k += BLOCK_SIZE) lds_x[k] = xb[k];
    __syncthreads();

    // 2) Compute dot products with weight row for current batch
    const bf16_t* __restrict__ w_row = w + (size_t)row * n + k_base;
    
    // Vectorized computation
    const int vec_k = (k_size / MFMA_K) * MFMA_K;

    for (int k = lane * MFMA_K; k < vec_k; k += WF_SIZE * MFMA_K) {
      if (k + MFMA_K <= vec_k) {
        mfloat4 w_vec, x_vec;
#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k + i]))) << 16;
          union { uint32_t u; float f; } cvt; cvt.u = u;
          w_vec[i] = cvt.f;
          x_vec[i] = lds_x[k + i];
        }

#pragma unroll
        for (int i = 0; i < MFMA_K; ++i) {
          acc = fmaf(w_vec[i], x_vec[i], acc);
        }
      }
    }

    for (int k = vec_k + lane; k < k_size; k += WF_SIZE) {
      uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
      union { uint32_t u; float f; } cvt; cvt.u = u;
      float w_val = cvt.f;
      acc = fmaf(w_val, lds_x[k], acc);
    }
    __syncthreads();
  }

  // reduce and write output for current batch
  float v = warp_reduce_sum(acc);
  if (lane == 0) {
    float* __restrict__ yb = y + (size_t)batch_idx * d;
    yb[row] = v + (bias ? bias[row] : 0.0f);
  }
}

/**
 * Y = X @ W^T + B (float version)
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/BN), ceil(B/BM))
 */
__launch_bounds__(BLOCK_SIZE, 1) __global__
void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [B, d]
    const float* __restrict__ x,    // [B, n]
    const float* __restrict__ w,    // [d, n] (row-major theo n)
    const float* __restrict__ bias, // [d] (có thể null)
    int n, int d, int batch_size)
{
  // Matrix dimensions
  const int M = batch_size;
  const int N = d;
  const int K = n;

  const int waveIdx = threadIdx.x / WF_SIZE;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % WF_SIZE;
  const int xInMFMA = laneIdx % MFMA_M;
  const int yInMFMA = laneIdx / MFMA_M;

  const int loadXIdx_x = threadIdx.x % BK;
  const int loadXIdx_y = threadIdx.x / BK;
  const int loadWIdx_x = threadIdx.x % BK;
  const int loadWIdx_y = threadIdx.x / BK;
  const int strideX = BLOCK_SIZE / BK;
  const int strideW = BLOCK_SIZE / BK;

  __shared__ float sx[BK][BM];
  __shared__ float sw[BK][BN];

  mfloat4 dmn = {0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load X tile
    for (int offset = 0; offset < BM; offset += strideX) {
      int index_x = bkIdx + loadXIdx_x;
      int index_y = BM * blockIdx.y + loadXIdx_y + offset;
      sx[index_x % BK][index_y % BM] =
          (index_x < K && index_y < M) ? x[index_y * K + index_x] : 0;
    }
    
    // Load W tile
    for (int offset = 0; offset < BN; offset += strideW) {
      int index_x = bkIdx + loadWIdx_x;
      int index_y = BN * blockIdx.x + loadWIdx_y + offset;
      sw[index_x % BK][index_y % BN] =
          (index_x < K && index_y < N) ? w[index_y * K + index_x] : 0;
    }
    __syncthreads();

    // MFMA computation
    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[k + yInMFMA][yInBlockTile * WM + xInMFMA];
      float bkn = sw[k + yInMFMA][xInBlockTile * WN + xInMFMA];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  // Write results
  for (int i = 0; i < 4; ++i) {
    const int xInD = laneIdx % MFMA_N;
    const int yInD = MFMA_K * (laneIdx / MFMA_N) + i;
    const int xInOutput = blockIdx.x * BN + xInBlockTile * WN + xInD;
    const int yInOutput = blockIdx.y * BM + yInBlockTile * WM + yInD;
    
    if (yInOutput < M && xInOutput < N) {
      float result = dmn[i];
      if (bias) result += bias[xInOutput];
      y[yInOutput * N + xInOutput] = result;
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
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, int experts_per_token)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;          // warp id in block

  const int i    = blockIdx.x * TM + wid;   // output row in IM (0..IM-1)
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || i >= IM || kidx >= experts_per_token)
    return;

  // Load expert ID for current batch
  expert_id = topk_i[(size_t)batch_idx * experts_per_token + kidx];
  if (expert_id < 0) return;

  float acc_gate = 0.f, acc_up = 0.f;

  // K loop over H, loading x for current batch
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load X for current batch
    const float* __restrict__ xb = x + (size_t)batch_idx * H + k_base;
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

    // Compute for current batch
    // weights layout:
    // base = ((l * E + expert) * (2*IM)) * H + (2*i + {0,1}) * H + k_base
    const size_t base_2im = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)(2 * IM);
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

      acc_up = fmaf(w_up.x, xv.x, acc_up);
      acc_up = fmaf(w_up.y, xv.y, acc_up);
      acc_up = fmaf(w_up.z, xv.z, acc_up);
      acc_up = fmaf(w_up.w, xv.w, acc_up);
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

  // warp reduce and output for current batch
  float gate_sum = warp_reduce_sum(acc_gate);
  float up_sum   = warp_reduce_sum(acc_up);

  if (lane == 0) {
    const size_t b_gate_base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)(2 * IM);
    float gate = gate_sum + b_mlp1_all[b_gate_base + (size_t)(2 * i + 0)];
    float up   = up_sum   + b_mlp1_all[b_gate_base + (size_t)(2 * i + 1)];

    // clip + SwiGLU-ish (như bản bạn đang dùng)
    gate = __saturatef((gate + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    up   = __saturatef((up   + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    const float alpha = 1.702f;
    gate = gate * __saturatef(0.5f + 0.5f * tanhf(alpha * gate * 0.5f));
    gate = gate * (up + 1.0f);

    float* __restrict__ gu = gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)batch_idx) * (size_t)IM);
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
    int l_layer, int E, int IM, int H,
    int batch_size, int experts_per_token)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;
  float expert_w;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;

  const int row  = blockIdx.x * TM + wid;   // output H row
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || row >= H || kidx >= experts_per_token)
    return;

  // Load expert ID and weight for current batch
  expert_id = topk_i[(size_t)batch_idx * experts_per_token + kidx];
  expert_w = topk_v[(size_t)batch_idx * experts_per_token + kidx];
  if (expert_id < 0 || expert_w == 0.f) return;

  float acc = 0.f;

  // K loop over IM, loading gate_up_topk for current batch
  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    // Load data for current batch
    const float* __restrict__ xb =
        gate_up_topk + (((size_t)kidx * (size_t)batch_size + (size_t)batch_idx) * (size_t)IM + (size_t)k_base);

    // vectorized load to shared
    const int vec4 = (k_size >> 2);
    float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // Compute for current batch
    // weight row: w[l, expert, row, k_base: ]
    const size_t base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)H * (size_t)IM;
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

  // reduce and write for current batch (1 atomic per (batch,row) per expert)
  float acc_sum = warp_reduce_sum(acc);
  if (lane == 0) {
    const size_t b_mlp2_base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)H;
    float out = acc_sum + b_mlp2_all[b_mlp2_base + (size_t)row];
    float contrib = out * expert_w;
    atomicAdd(e_agg + (size_t)batch_idx * (size_t)H + (size_t)row, contrib);
  }
}
