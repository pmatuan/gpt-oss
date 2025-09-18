#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>


// x:[B,n], w:[d,n], y:[B,d]
// Grid: (ceil(d/TM), ceil(B/B_TILE))
__launch_bounds__(64, 2) __global__
void matmul_bias_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const float* __restrict__ x,    // [B x n] (fp32)
    const bf16_t* __restrict__ w,   // [d x n] (bf16)
    const float* __restrict__ bias, // [d] or nullptr
    int n, int d, int B,
    const int* __restrict__ pos)    // nullable; pos[b] < 0 -> skip row
{
  // One wave64 per block: 16 x-threads, 4 y-threads
  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..3

  // 32x32 tile origin on (M=B, N=d)
  const int M0 = blockIdx.y * 32;
  const int N0 = blockIdx.x * 32;

  // Per-lane columns (two 16-wide halves)
  const int c0 = N0 + tx;
  const int c1 = N0 + tx + 16;
  const bool c0_ok = (c0 < d);
  const bool c1_ok = (c1 < d);

  // Per-lane rows (two 16-high halves)
  const int r0 = M0 + tx;
  const int r1 = M0 + tx + 16;
  const bool r0_ok = (r0 < B) && (!(pos) || pos[r0] >= 0);
  const bool r1_ok = (r1 < B) && (!(pos) || pos[r1] >= 0);

  // Accumulators for 4 MFMA fragments (32x32 tile = 2x2 of 16x16)
  f32x4 acc00 = {0.f,0.f,0.f,0.f};
  f32x4 acc01 = {0.f,0.f,0.f,0.f};
  f32x4 acc10 = {0.f,0.f,0.f,0.f};
  f32x4 acc11 = {0.f,0.f,0.f,0.f};

  // K-loop in steps of 16 (bf16 depth)
  for (int k0 = 0; k0 < n; k0 += 16) {
    // Each ty takes a 4-wide slice in this k-tile
    const int k_base = k0 + ty * 4;
    const int k_rem  = max(0, min(4, n - k_base)); // 0..4 remaining in this slice

    // A (X): row-major [B x n] -> take 4 fp32, convert to bf16 with guards
    const float* x_r0 = x + (size_t)r0 * n + k_base;
    const float* x_r1 = x + (size_t)r1 * n + k_base;
    s16x4 Avec_r0 = pack4_bf16_from_f32_guard(x_r0, 0, k_rem, r0_ok);
    s16x4 Avec_r1 = pack4_bf16_from_f32_guard(x_r1, 0, k_rem, r1_ok);

    // B (W): row-major [d x n], but we need (k, col) -> &w[col*n + k_base]
    const bf16_t* w_c0 = w + (size_t)c0 * n + k_base;
    const bf16_t* w_c1 = w + (size_t)c1 * n + k_base;
    s16x4 Bvec_c0 = pack4_bf16_from_bf16_guard(w_c0, 0, k_rem, c0_ok);
    s16x4 Bvec_c1 = pack4_bf16_from_bf16_guard(w_c1, 0, k_rem, c1_ok);

    // 4 MFMA ops (2x2 fragments)
    acc00 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c0, acc00, 0,0,0);
    acc01 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c1, acc01, 0,0,0);
    acc10 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c0, acc10, 0,0,0);
    acc11 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c1, acc11, 0,0,0);
  }

  // Runtime bias: only read when col is valid and bias != nullptr
  const float b0 = (bias && c0_ok) ? bias[c0] : 0.f;
  const float b1 = (bias && c1_ok) ? bias[c1] : 0.f;

  // Scatter back (each ty handles 4 row offsets within the 16x16 fragment)
  #pragma unroll
  for (int i=0;i<4;i++) {
    const int row_lo = M0 + (i + 4 * ty);
    const int row_hi = row_lo + 16;

    if (row_lo < B && (!(pos) || pos[row_lo] >= 0)) {
      float* y0 = y + (size_t)row_lo * d;
      if (c0_ok) y0[c0]   = acc00[i] + b0;
      if (c1_ok) y0[c1]   = acc01[i] + b1;
    }
    if (row_hi < B && (!(pos) || pos[row_hi] >= 0)) {
      float* y1 = y + (size_t)row_hi * d;
      if (c0_ok) y1[c0]   = acc10[i] + b0;
      if (c1_ok) y1[c1]   = acc11[i] + b1;
    }
  }
}

/**
 * Y = X @ W^T + B (float version)
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/TM), B)
 */
__launch_bounds__(BLOCK_SIZE, 8) __global__
void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [B, d]
    const float* __restrict__ x,    // [B, n]
    const float* __restrict__ w,    // [d, n] (row-major theo n)
    const float* __restrict__ bias, // [d] (có thể null)
    int n, int d, int batch_size, const int *pos)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;
  if (pos && pos[batch_idx] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;                 // warp id trong block (0..TM-1)

  const int row = blockIdx.x * TM + wid;    // hàng output (0..d-1)

  if (wid >= TM || row >= d) return;

  float acc = 0.f;

  // Vòng K - optimized with vectorized loads and computation
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // 1) Optimized vectorized load of X columns for current batch
    const float* __restrict__ xb = x + (size_t)batch_idx * n + k_base;
    
    // Load using vectorized operations
    const int vec4_k = (k_size >> 2);
    float4* __restrict__ lds_x4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ xb4 = reinterpret_cast<const float4*>(xb);
    
    for (int v = tid; v < vec4_k; v += BLOCK_SIZE) {
      lds_x4[v] = xb4[v];
    }
    
    // Handle remainder elements
    for (int k = (vec4_k << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // 2) Optimized computation with weight row for current batch
    const float* __restrict__ w_row = w + (size_t)row * n + k_base;
    
    // Vectorized computation - process 4 elements at a time
    const int vec_k = (k_size / MFMA_K) * MFMA_K;

    for (int k = lane * MFMA_K; k < vec_k; k += WF_SIZE * MFMA_K) {
      if (k + MFMA_K <= vec_k) {
        // Load 4 consecutive elements as vectors
        const float4 w_vec = *reinterpret_cast<const float4*>(&w_row[k]);
        const float4 x_vec = *reinterpret_cast<const float4*>(&lds_x[k]);
        
        // Perform dot product using fused multiply-add
        acc = fmaf(w_vec.x, x_vec.x, acc);
        acc = fmaf(w_vec.y, x_vec.y, acc);
        acc = fmaf(w_vec.z, x_vec.z, acc);
        acc = fmaf(w_vec.w, x_vec.w, acc);
      }
    }

    // Handle remainder elements
    for (int k = vec_k + lane; k < k_size; k += WF_SIZE) {
      acc = fmaf(w_row[k], lds_x[k], acc);
    }
    __syncthreads();
  }

  // Optimized warp reduction and write output for current batch
  float result = warp_reduce_sum(acc);
  if (lane == 0) {
    float* __restrict__ yb = y + (size_t)batch_idx * d;
    yb[row] = result + (bias ? bias[row] : 0.0f);
  }
}

// ================= MLP1 (Gate & Up) : per-batch, no CB =================
__launch_bounds__(BLOCK_SIZE, 8) __global__
void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up_topk, // [K, B, IM] (K = EXPERT_PER_TOKEN)
    const float* __restrict__ x,      // [B, H]
    const bf16_t* __restrict__ w_mlp1_all, // [L, E, 2*IM, H] (row-major in last dim)
    const float* __restrict__ b_mlp1_all,  // [L, E, 2*IM]
    const int* __restrict__ topk_i,   // [B, K]
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size,
    const int *pos)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;
  if (pos && pos[batch_idx] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;          // warp id in block

  const int i    = blockIdx.x * TM + wid;   // output row in IM (0..IM-1)
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || i >= IM || kidx >= EXPERT_PER_TOKEN)
    return;

  // Load expert ID for current batch
  expert_id = topk_i[(size_t)batch_idx * EXPERT_PER_TOKEN + kidx];
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
__launch_bounds__(BLOCK_SIZE, 8) __global__
void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,              // [B, H] (accumulator)
    const float* __restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t* __restrict__ w_mlp2_all,  // [L, E, H, IM]
    const float* __restrict__ b_mlp2_all,   // [L, E, H]
    const int* __restrict__ topk_i,         // [B, K]
    const float* __restrict__ topk_v,       // [B, K]
    int l_layer, int E, int IM, int H,
    int batch_size, const int *pos)
{
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;
  if (pos && pos[batch_idx] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;
  float expert_w;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;

  const int row  = blockIdx.x * TM + wid;   // output H row
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || row >= H || kidx >= EXPERT_PER_TOKEN)
    return;

  // Load expert ID and weight for current batch
  expert_id = topk_i[(size_t)batch_idx * EXPERT_PER_TOKEN + kidx];
  expert_w = topk_v[(size_t)batch_idx * EXPERT_PER_TOKEN + kidx];
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
