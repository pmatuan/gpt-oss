#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>


__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  union { uint32_t u; float f; } cvt;
  cvt.u = ((uint32_t)bits) << 16;
  return cvt.f;
}


// x:[B,n], w:[d,n], y:[B,d]
__global__ void matmul_bias_gemm_kernel_bf16_mfma(
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
__global__ void matmul_bias_gemm_kernel_float(
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
    const int vec_k = (k_size / K_STEP_MATMUL_FLOAT) * K_STEP_MATMUL_FLOAT;

    for (int k = lane * K_STEP_MATMUL_FLOAT; k < vec_k; k += WF_SIZE * K_STEP_MATMUL_FLOAT) {
      if (k + K_STEP_MATMUL_FLOAT <= vec_k) {
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

__global__ void count_expert_assignments_kernel(
    int* __restrict__ counts,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int batch_size,
    int experts_per_token,
    int E)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * experts_per_token;
  if (idx >= total) return;

  const int batch_idx = idx / experts_per_token;
  if (pos && pos[batch_idx] < 0) return;

  const int expert = topk_i[idx];
  if (expert < 0 || expert >= E) return;

  atomicAdd(counts + expert, 1);
}

__global__ void build_expert_assignments_kernel(
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_counters,
    int* __restrict__ assignment_batches,
    int* __restrict__ assignment_slots,
    int batch_size,
    int experts_per_token,
    int E)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * experts_per_token;
  if (idx >= total) return;

  const int batch_idx = idx / experts_per_token;
  if (pos && pos[batch_idx] < 0) return;

  const int expert = topk_i[idx];
  if (expert < 0 || expert >= E) return;

  const int slot = idx % experts_per_token;
  const int offset = atomicAdd(expert_counters + expert, 1);
  const int write_idx = expert_offsets[expert] + offset;

  assignment_batches[write_idx] = batch_idx;
  assignment_slots[write_idx] = slot;
}

// ================= MLP1 (Gate & Up) : batched per expert =================
__global__ void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up_topk,
    const float* __restrict__ x,
    const bf16_t* __restrict__ w_mlp1_all,
    const float* __restrict__ b_mlp1_all,
    const int* __restrict__ assignment_batches,
    const int* __restrict__ assignment_slots,
    const int* __restrict__ expert_offsets,
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size,
    const int *pos)
{
  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0) return;

  const int tile_start_token = blockIdx.y * MLP1_TILE_TOKENS;
  if (tile_start_token >= count) return;
  const int tile_rows = min(MLP1_TILE_TOKENS, count - tile_start_token);

  const int tile_start_im = blockIdx.x * MLP1_TILE_IM;
  if (tile_start_im >= IM) return;
  const int tile_cols = min(MLP1_TILE_IM, IM - tile_start_im);

  const int local_im = threadIdx.x;
  const int local_token = threadIdx.y;

  if (local_im >= tile_cols || local_token >= tile_rows) return;

  const int assignment_idx = start + tile_start_token + local_token;
  const int batch_idx = assignment_batches[assignment_idx];
  const int slot = assignment_slots[assignment_idx];
  if (batch_idx < 0 || slot < 0) return;
  if (pos && pos[batch_idx] < 0) return;

  const int im_idx = tile_start_im + local_im;

  const size_t base_2im = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)(2 * IM);
  const size_t gate_row_off = ((size_t)(2 * im_idx + 0)) * (size_t)H;
  const size_t up_row_off   = ((size_t)(2 * im_idx + 1)) * (size_t)H;

  const bf16_t* __restrict__ w_gate = w_mlp1_all + (base_2im * (size_t)H) + gate_row_off;
  const bf16_t* __restrict__ w_up   = w_mlp1_all + (base_2im * (size_t)H) + up_row_off;
  const float* __restrict__ x_row   = x + (size_t)batch_idx * (size_t)H;

  float acc_gate = 0.0f;
  float acc_up   = 0.0f;

  const int vec_limit = H & ~3;
  const int vec_count = vec_limit >> 2;
  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x_row);
  const uint2* __restrict__ w_gate4 = reinterpret_cast<const uint2*>(w_gate);
  const uint2* __restrict__ w_up4   = reinterpret_cast<const uint2*>(w_up);

  #pragma unroll
  for (int v = 0; v < vec_count; ++v) {
    const float4 xv = x4[v];
    const float4 gv = bf16quad_to_float4(w_gate4[v]);
    const float4 uv = bf16quad_to_float4(w_up4[v]);
    acc_gate = fmaf(gv.x, xv.x, acc_gate);
    acc_gate = fmaf(gv.y, xv.y, acc_gate);
    acc_gate = fmaf(gv.z, xv.z, acc_gate);
    acc_gate = fmaf(gv.w, xv.w, acc_gate);

    acc_up = fmaf(uv.x, xv.x, acc_up);
    acc_up = fmaf(uv.y, xv.y, acc_up);
    acc_up = fmaf(uv.z, xv.z, acc_up);
    acc_up = fmaf(uv.w, xv.w, acc_up);
  }

  const uint16_t* __restrict__ w_gate_u16 = reinterpret_cast<const uint16_t*>(w_gate);
  const uint16_t* __restrict__ w_up_u16   = reinterpret_cast<const uint16_t*>(w_up);

  for (int k = vec_limit; k < H; ++k) {
    const float xv = x_row[k];
    acc_gate = fmaf(bf16_bits_to_float(w_gate_u16[k]), xv, acc_gate);
    acc_up   = fmaf(bf16_bits_to_float(w_up_u16[k]), xv, acc_up);
  }

  const float* __restrict__ b_base = b_mlp1_all + base_2im;
  float gate_val = acc_gate + b_base[2 * im_idx + 0];
  float up_val   = acc_up   + b_base[2 * im_idx + 1];

  gate_val = __saturatef((gate_val + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
  up_val   = __saturatef((up_val   + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
  const float alpha = 1.702f;
  const float gate_act = gate_val * __saturatef(0.5f + 0.5f * tanhf(alpha * gate_val * 0.5f));
  const float final_gate = gate_act * (up_val + 1.0f);

  float* __restrict__ gu = gate_up_topk + (((size_t)slot * (size_t)batch_size + (size_t)batch_idx) * (size_t)IM);
  gu[im_idx] = final_gate;
}


// ============ MLP2 (weighted accum) : batched per expert ==============
__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,
    const float* __restrict__ gate_up_topk,
    const bf16_t* __restrict__ w_mlp2_all,
    const float* __restrict__ b_mlp2_all,
    const int* __restrict__ assignment_batches,
    const int* __restrict__ assignment_slots,
    const int* __restrict__ expert_offsets,
    const float* __restrict__ topk_v,
    int l_layer, int E, int IM, int H,
    int batch_size, const int *pos)
{
  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0) return;

  const int tile_start_token = blockIdx.y * MLP2_TILE_TOKENS;
  if (tile_start_token >= count) return;
  const int tile_rows = min(MLP2_TILE_TOKENS, count - tile_start_token);

  const int tile_start_h = blockIdx.x * MLP2_TILE_H;
  if (tile_start_h >= H) return;
  const int tile_cols = min(MLP2_TILE_H, H - tile_start_h);

  const int local_h = threadIdx.x;
  const int local_token = threadIdx.y;
  if (local_h >= tile_cols || local_token >= tile_rows) return;

  const int assignment_idx = start + tile_start_token + local_token;
  const int batch_idx = assignment_batches[assignment_idx];
  const int slot = assignment_slots[assignment_idx];
  if (batch_idx < 0 || slot < 0) return;
  if (pos && pos[batch_idx] < 0) return;

  const int h_idx = tile_start_h + local_h;
  const float expert_weight = topk_v[(size_t)batch_idx * (size_t)EXPERT_PER_TOKEN + slot];
  if (expert_weight == 0.0f) return;

  const float* __restrict__ gate_vec = gate_up_topk + (((size_t)slot * (size_t)batch_size + (size_t)batch_idx) * (size_t)IM);
  const size_t base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)H;
  const bf16_t* __restrict__ w_row = w_mlp2_all + (base + (size_t)h_idx) * (size_t)IM;

  float acc = 0.0f;
  const int vec_limit = IM & ~3;
  const int vec_count = vec_limit >> 2;
  const float4* __restrict__ gate4 = reinterpret_cast<const float4*>(gate_vec);
  const uint2* __restrict__ w_row4 = reinterpret_cast<const uint2*>(w_row);

  #pragma unroll
  for (int v = 0; v < vec_count; ++v) {
    const float4 gv = gate4[v];
    const float4 wv = bf16quad_to_float4(w_row4[v]);
    acc = fmaf(wv.x, gv.x, acc);
    acc = fmaf(wv.y, gv.y, acc);
    acc = fmaf(wv.z, gv.z, acc);
    acc = fmaf(wv.w, gv.w, acc);
  }

  const uint16_t* __restrict__ w_row_u16 = reinterpret_cast<const uint16_t*>(w_row);
  for (int k = vec_limit; k < IM; ++k) {
    acc = fmaf(bf16_bits_to_float(w_row_u16[k]), gate_vec[k], acc);
  }

  const float bias = b_mlp2_all[base + (size_t)h_idx];
  const float contrib = (acc + bias) * expert_weight;
  atomicAdd(e_agg + (size_t)batch_idx * (size_t)H + (size_t)h_idx, contrib);
}
