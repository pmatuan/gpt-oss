#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>


// Template version for unrolling the k-loop
template<int N>
__global__ void matmul_bias_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const float* __restrict__ x,    // [B x n] (fp32)
    const bf16_t* __restrict__ w,   // [d x n] (bf16)
    const float* __restrict__ bias, // [d] or nullptr
    int d, int B,
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

  // K-loop unrolled for N template parameter (steps of 1, but actual k uses *16)
  #pragma unroll
  for (int k0 = 0; k0 < N; k0++) {
    // Each ty takes a 4-wide slice in this k-tile
    const int k_base = k0 * 16 + ty * 4;
    const int k_rem  = max(0, min(4, N * 16 - k_base)); // 0..4 remaining in this slice

    // A (X): row-major [B x n] -> take 4 fp32, convert to bf16 with guards
    const float* x_r0 = x + (size_t)r0 * N * 16 + k_base;
    const float* x_r1 = x + (size_t)r1 * N * 16 + k_base;
    s16x4 Avec_r0 = pack4_bf16_from_f32_guard(x_r0, 0, k_rem, r0_ok);
    s16x4 Avec_r1 = pack4_bf16_from_f32_guard(x_r1, 0, k_rem, r1_ok);

    // B (W): row-major [d x n], but we need (k, col) -> &w[col*n + k_base]
    const bf16_t* w_c0 = w + (size_t)c0 * N * 16 + k_base;
    const bf16_t* w_c1 = w + (size_t)c1 * N * 16 + k_base;
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

template __global__ void matmul_bias_gemm_kernel_bf16_mfma<180>(
    float* __restrict__ y,
    const float* __restrict__ x,
    const bf16_t* __restrict__ w,
    const float* __restrict__ bias,
    int d, int B,
    const int* __restrict__ pos);

template __global__ void matmul_bias_gemm_kernel_bf16_mfma<256>(
    float* __restrict__ y,
    const float* __restrict__ x,
    const bf16_t* __restrict__ w,
    const float* __restrict__ bias,
    int d, int B,
    const int* __restrict__ pos);

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

__global__ void moe_count_assignments_kernel(int *counts,
                                             const int *topk_i,
                                             int total_assignments,
                                             int n_experts)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_assignments)
    return;

  const int expert = topk_i[idx];
  if (expert >= 0 && expert < n_experts) {
    atomicAdd(counts + expert, 1);
  }
}

__global__ void moe_scatter_assignments_kernel(int *assignments,
                                               int *counters,
                                               const int *offsets,
                                               const int *topk_i,
                                               int total_assignments,
                                               int n_experts)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_assignments)
    return;

  const int expert = topk_i[idx];
  if (expert < 0 || expert >= n_experts)
    return;

  const int slot = atomicAdd(counters + expert, 1);
  assignments[offsets[expert] + slot] = idx;
}

__global__ void moe_gather_tokens_kernel(float *dst, const float *src,
                                         const int *assignments,
                                         const int *assignment_active_slot,
                                         const int *active_experts,
                                         int num_active, int num_assignments,
                                         int H)
{
  const int assignment_idx = blockIdx.y;
  if (assignment_idx >= num_assignments)
    return;

  const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (col >= H)
    return;

  const int expert_slot = assignment_active_slot[assignment_idx];
  if (expert_slot < 0 || expert_slot >= num_active)
    return;

  const int expert = active_experts[expert_slot];
  const int assignment = assignments[assignment_idx];
  if (assignment < 0)
    return;

  const int batch_idx = assignment >> EXPERT_PER_TOKEN_SHIFT;
  const float *src_row = src + (size_t)batch_idx * H;
  float *dst_row = dst + (size_t)assignment_idx * H;
  dst_row[col] = src_row[col];
}

__global__ void moe_mlp1_matmul_bias_kernel(
    float *out, const float *x_grouped, const bf16_t *w_mlp1_all,
    const float *b_mlp1_all, const int *assignment_active_slot,
    const int *active_experts, int layer, int E, int H, int IM,
    int num_active, int num_assignments)
{
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int output_dim = blockIdx.x * TM + wid;

  if (wid >= TM || output_dim >= 2 * IM)
    return;

  const int assignment_idx = blockIdx.y;
  if (assignment_idx >= num_assignments)
    return;

  const int expert_slot = assignment_active_slot[assignment_idx];
  if (expert_slot < 0 || expert_slot >= num_active)
    return;

  const int expert = active_experts[expert_slot];
  const float *xb = x_grouped + (size_t)assignment_idx * H;

  float acc = 0.0f;

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    const float *xb_tile = xb + k_base;
    const int vec4 = k_size >> 2;
    float4 *s4 = reinterpret_cast<float4 *>(lds_x);
    const float4 *x4 = reinterpret_cast<const float4 *>(xb_tile);
    for (int v = tid; v < vec4; v += BLOCK_SIZE)
      s4[v] = x4[v];
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = xb_tile[k];
    __syncthreads();

    const size_t weight_row_offset =
        (((size_t)layer * (size_t)E + (size_t)expert) * (size_t)(2 * IM) +
         (size_t)output_dim) * (size_t)H + (size_t)k_base;
    const bf16_t *w_row = w_mlp1_all + weight_row_offset;

    const uint2 *wq = reinterpret_cast<const uint2 *>(w_row);
    const int vec4k = k_size >> 2;
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 wv = bf16quad_to_float4(wq[v]);
      const float4 xv = *reinterpret_cast<const float4 *>(&lds_x[v << 2]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    for (int k = (vec4k << 2) + lane; k < k_size; k += WF_SIZE) {
      uint32_t u =
          ((uint32_t)(*reinterpret_cast<const uint16_t *>(&w_row[k]))) << 16;
      union {
        uint32_t u;
        float f;
      } cvt;
      cvt.u = u;
      acc = fmaf(cvt.f, lds_x[k], acc);
    }
    __syncthreads();
  }

  float acc_sum = warp_reduce_sum(acc);
  if (lane == 0) {
    const size_t bias_base =
        ((size_t)layer * (size_t)E + (size_t)expert) * (size_t)(2 * IM) +
        (size_t)output_dim;
    float result = acc_sum + b_mlp1_all[bias_base];
    out[(size_t)assignment_idx * (size_t)(2 * IM) + (size_t)output_dim] = result;
  }
}

__global__ void moe_swiglu_activation_kernel(float *dst, const float *src,
                                             float clip, int count, int IM)
{
  const int row = blockIdx.y;
  if (row >= count)
    return;

  const int dim = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (dim >= IM)
    return;

  const size_t base = (size_t)row * (size_t)(2 * IM) + (size_t)(2 * dim);
  float gate_val = src[base + 0];
  float up_val = src[base + 1];

  if (clip > 0.0f) {
    const float inv_span = 0.5f / clip;
    float norm_gate = __saturatef(gate_val * inv_span + 0.5f);
    float norm_up = __saturatef(up_val * inv_span + 0.5f);
    gate_val = norm_gate * (2.0f * clip) - clip;
    up_val = norm_up * (2.0f * clip) - clip;
  }

  const float alpha = 1.702f;
  float activation = __saturatef(0.5f + 0.5f * tanhf(alpha * gate_val * 0.5f));
  float gate = gate_val * activation;
  dst[(size_t)row * (size_t)IM + (size_t)dim] = gate * (up_val + 1.0f);
}

__global__ void moe_mlp2_matmul_bias_kernel(
    float *out, const float *gate_up_grouped, const bf16_t *w_mlp2_all,
    const float *b_mlp2_all, const int *assignment_active_slot,
    const int *active_experts, int layer, int E, int IM, int H,
    int num_active, int num_assignments)
{
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int output_dim = blockIdx.x * TM + wid;

  if (wid >= TM || output_dim >= H)
    return;

  const int assignment_idx = blockIdx.y;
  if (assignment_idx >= num_assignments)
    return;

  const int expert_slot = assignment_active_slot[assignment_idx];
  if (expert_slot < 0 || expert_slot >= num_active)
    return;

  const int expert = active_experts[expert_slot];
  const float *xb = gate_up_grouped + (size_t)assignment_idx * IM;

  float acc = 0.0f;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    const float *xb_tile = xb + k_base;
    const int vec4 = k_size >> 2;
    float4 *s4 = reinterpret_cast<float4 *>(lds_x);
    const float4 *x4 = reinterpret_cast<const float4 *>(xb_tile);
    for (int v = tid; v < vec4; v += BLOCK_SIZE)
      s4[v] = x4[v];
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = xb_tile[k];
    __syncthreads();

    const size_t weight_row_offset =
        (((size_t)layer * (size_t)E + (size_t)expert) * (size_t)H +
         (size_t)output_dim) * (size_t)IM + (size_t)k_base;
    const bf16_t *w_row = w_mlp2_all + weight_row_offset;

    const uint2 *wq = reinterpret_cast<const uint2 *>(w_row);
    const int vec4k = k_size >> 2;
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 wv = bf16quad_to_float4(wq[v]);
      const float4 xv = *reinterpret_cast<const float4 *>(&lds_x[v << 2]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    for (int k = (vec4k << 2) + lane; k < k_size; k += WF_SIZE) {
      uint32_t u =
          ((uint32_t)(*reinterpret_cast<const uint16_t *>(&w_row[k]))) << 16;
      union {
        uint32_t u;
        float f;
      } cvt;
      cvt.u = u;
      acc = fmaf(cvt.f, lds_x[k], acc);
    }
    __syncthreads();
  }

  float acc_sum = warp_reduce_sum(acc);
  if (lane == 0) {
    const size_t bias_base =
        ((size_t)layer * (size_t)E + (size_t)expert) * (size_t)H +
        (size_t)output_dim;
    float result = acc_sum + b_mlp2_all[bias_base];
    out[(size_t)assignment_idx * (size_t)H + (size_t)output_dim] = result;
  }
}

__global__ void moe_weighted_accum_kernel(float *e_agg, const float *mlp2_out,
                                          const float *topk_v,
                                          const int *assignments,
                                          const int *assignment_active_slot,
                                          int num_active, int num_assignments,
                                          int H)
{
  const int dim = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (dim >= H)
    return;

  const int assignment_idx = blockIdx.y;
  if (assignment_idx >= num_assignments)
    return;

  const int expert_slot = assignment_active_slot[assignment_idx];
  if (expert_slot < 0 || expert_slot >= num_active)
    return;

  const int assignment = assignments[assignment_idx];
  if (assignment < 0)
    return;

  const float weight = topk_v[assignment];
  if (weight == 0.0f)
    return;

  const int batch_idx = assignment >> EXPERT_PER_TOKEN_SHIFT;
  const float *row_src = mlp2_out + (size_t)assignment_idx * H;
  float *row_dst = e_agg + (size_t)batch_idx * H;
  const float contrib = row_src[dim] * weight;
  atomicAdd(row_dst + dim, contrib);
}
