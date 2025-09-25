#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>

__device__ __forceinline__ s16x4 load_bf16x4_raw(const uint16_t* src, int valid_elems) {
  s16x4 out = {0, 0, 0, 0};
  if (!src || valid_elems <= 0) {
    return out;
  }

  if (valid_elems >= MATMUL_CHUNK_K) {
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    const uint32_t packed0 = src32[0];
    const uint32_t packed1 = src32[1];
    out[0] = static_cast<short>(packed0 & 0xFFFF);
    out[1] = static_cast<short>(packed0 >> 16);
    out[2] = static_cast<short>(packed1 & 0xFFFF);
    out[3] = static_cast<short>(packed1 >> 16);
    return out;
  }

#pragma unroll
  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    out[i] = (i < valid_elems) ? static_cast<short>(src[i]) : 0;
  }
  return out;
}


// x:[B,n], w:[d,n], y:[B,d]
__global__ __launch_bounds__(64, 2)
void matmul_bias_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const bf16_t* __restrict__ x,   // [B x n] (bf16)
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

  const uint16_t* __restrict__ w_u16 = reinterpret_cast<const uint16_t*>(w);
  const int tiles_k = (n + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;
  const size_t group_offset = (size_t)ty * group_stride;

  size_t base_tile_c0 = 0;
  const bool active_c0 = c0_ok;
  if (active_c0) {
    const int tile_col0 = c0 / MATMUL_TILE_COLS;
    const int row_in_tile0 = c0 % MATMUL_TILE_COLS;
    base_tile_c0 = ((size_t)tile_col0 * tiles_k) * tile_elems +
                   (size_t)row_in_tile0 * MATMUL_CHUNK_K;
  }

  size_t base_tile_c1 = 0;
  const bool active_c1 = c1_ok;
  if (active_c1) {
    const int tile_col1 = c1 / MATMUL_TILE_COLS;
    const int row_in_tile1 = c1 % MATMUL_TILE_COLS;
    base_tile_c1 = ((size_t)tile_col1 * tiles_k) * tile_elems +
                   (size_t)row_in_tile1 * MATMUL_CHUNK_K;
  }

  const uint16_t* w_base_c0 = active_c0 ? w_u16 + base_tile_c0 : nullptr;
  const uint16_t* w_base_c1 = active_c1 ? w_u16 + base_tile_c1 : nullptr;

  // Per-lane rows (two 16-high halves)
  const int r0 = M0 + tx;
  const int r1 = M0 + tx + 16;
  const bool has_pos = (pos != nullptr);
  const bool r0_ok = (r0 < B) && (!has_pos || pos[r0] >= 0);
  const bool r1_ok = (r1 < B) && (!has_pos || pos[r1] >= 0);

  // Accumulators for 4 MFMA fragments (32x32 tile = 2x2 of 16x16)
  f32x4 acc00 = {0.f,0.f,0.f,0.f};
  f32x4 acc01 = {0.f,0.f,0.f,0.f};
  f32x4 acc10 = {0.f,0.f,0.f,0.f};
  f32x4 acc11 = {0.f,0.f,0.f,0.f};

  // K-loop in steps of 16 (bf16 depth)
  size_t tile_offset = group_offset;
  for (int k0 = 0; k0 < n; k0 += MATMUL_TILE_K, tile_offset += tile_elems) {
    // Each ty takes a 4-wide slice in this k-tile
    const int k_base = k0 + ty * MATMUL_CHUNK_K;
    const int rem = n - k_base;
    if (rem <= 0) {
      continue;
    }

    const int chunk_elems = rem >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K : rem;
    const uint16_t* x_src0 = (r0_ok && chunk_elems > 0)
                                 ? reinterpret_cast<const uint16_t*>(
                                       x + (size_t)r0 * n + k_base)
                                 : nullptr;
    const uint16_t* x_src1 = (r1_ok && chunk_elems > 0)
                                 ? reinterpret_cast<const uint16_t*>(
                                       x + (size_t)r1 * n + k_base)
                                 : nullptr;

    const s16x4 Avec_r0 = load_bf16x4_raw(x_src0, chunk_elems);
    const s16x4 Avec_r1 = load_bf16x4_raw(x_src1, chunk_elems);

    const size_t chunk_offset = tile_offset;

    const s16x4 Bvec_c0 =
        load_bf16x4_raw((chunk_elems > 0 && active_c0) ? w_base_c0 + chunk_offset
                                                       : nullptr,
                        chunk_elems);
    const s16x4 Bvec_c1 =
        load_bf16x4_raw((chunk_elems > 0 && active_c1) ? w_base_c1 + chunk_offset
                                                       : nullptr,
                        chunk_elems);

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
  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    const int row_lo = M0 + (i + MATMUL_CHUNK_K * ty);
    const int row_hi = row_lo + 16;

    if (row_lo < B && (!has_pos || pos[row_lo] >= 0)) {
      float* y0 = y + (size_t)row_lo * d;
      if (c0_ok) y0[c0]   = acc00[i] + b0;
      if (c1_ok) y0[c1]   = acc01[i] + b1;
    }
    if (row_hi < B && (!has_pos || pos[row_hi] >= 0)) {
      float* y1 = y + (size_t)row_hi * d;
      if (c0_ok) y1[c0]   = acc10[i] + b0;
      if (c1_ok) y1[c1]   = acc11[i] + b1;
    }
  }
}

// No-bias variant: x:[B,n], w:[d,n], y:[B,d]
__global__ __launch_bounds__(64, 2)
void matmul_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const bf16_t* __restrict__ x,   // [B x n] (bf16)
    const bf16_t* __restrict__ w,   // [d x n] (bf16)
    int n, int d, int B,
    const int* __restrict__ pos)    // nullable; pos[b] < 0 -> skip row
{
  constexpr int TILE_M = MATMUL_GEMM_TILE_ROWS; // rows per block (matches packing)
  constexpr int TILE_N = MATMUL_GEMM_TILE_COLS; // expanded columns per block
  static_assert((MATMUL_GEMM_TILE_COLS % 16) == 0,
                "MATMUL_GEMM_TILE_COLS must be a multiple of 16");
  constexpr int COL_GROUPS = MATMUL_GEMM_TILE_COLS / 16;

  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..3

  const int M0 = blockIdx.y * TILE_M;
  const int N0 = blockIdx.x * TILE_N;

  int cols[COL_GROUPS];
  bool col_ok[COL_GROUPS];
  size_t base_tile[COL_GROUPS] = {0};
  const uint16_t* w_base[COL_GROUPS] = {nullptr};

  const uint16_t* __restrict__ w_u16 = reinterpret_cast<const uint16_t*>(w);
  const int tiles_k = (n + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;
  const size_t group_offset = (size_t)ty * group_stride;

  for (int idx = 0; idx < COL_GROUPS; ++idx) {
    const int c = N0 + tx + idx * 16;
    cols[idx] = c;
    const bool valid = (c < d);
    col_ok[idx] = valid;
    if (valid) {
      const int tile_col = c / MATMUL_TILE_COLS;
      const int row_in_tile = c % MATMUL_TILE_COLS;
      base_tile[idx] = ((size_t)tile_col * tiles_k) * tile_elems +
                       (size_t)row_in_tile * MATMUL_CHUNK_K;
      w_base[idx] = w_u16 + base_tile[idx];
    }
  }

  const bool has_pos = (pos != nullptr);

  const int r0 = M0 + tx;
  const int r1 = r0 + 16;
  const bool r0_ok = (r0 < B) && (!has_pos || pos[r0] >= 0);
  const bool r1_ok = (r1 < B) && (!has_pos || pos[r1] >= 0);

  f32x4 acc_lo[COL_GROUPS];
  f32x4 acc_hi[COL_GROUPS];
#pragma unroll
  for (int idx = 0; idx < COL_GROUPS; ++idx) {
    acc_lo[idx] = f32x4{0.f, 0.f, 0.f, 0.f};
    acc_hi[idx] = f32x4{0.f, 0.f, 0.f, 0.f};
  }

  const uint16_t* x_ptr0 = r0_ok
                               ? reinterpret_cast<const uint16_t*>(
                                     x + (size_t)r0 * n) +
                                     ty * MATMUL_CHUNK_K
                               : nullptr;
  const uint16_t* x_ptr1 = r1_ok
                               ? reinterpret_cast<const uint16_t*>(
                                     x + (size_t)r1 * n) +
                                     ty * MATMUL_CHUNK_K
                               : nullptr;

  const uint16_t* w_ptr[COL_GROUPS];
#pragma unroll
  for (int idx = 0; idx < COL_GROUPS; ++idx) {
    w_ptr[idx] = col_ok[idx] ? (w_base[idx] + group_offset) : nullptr;
  }

  for (int k0 = 0; k0 < n; k0 += MATMUL_TILE_K) {
    const int k_base = k0 + ty * MATMUL_CHUNK_K;
    const int rem = n - k_base;
    if (rem <= 0) {
      if (r0_ok) {
        x_ptr0 += MATMUL_TILE_K;
      }
      if (r1_ok) {
        x_ptr1 += MATMUL_TILE_K;
      }
#pragma unroll
      for (int idx = 0; idx < COL_GROUPS; ++idx) {
        if (col_ok[idx]) {
          w_ptr[idx] += tile_elems;
        }
      }
      continue;
    }

    const int chunk_elems = rem >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K : rem;
    const s16x4 Avec_r0 =
        (r0_ok && chunk_elems > 0) ? load_bf16x4_raw(x_ptr0, chunk_elems)
                                   : s16x4{0, 0, 0, 0};
    const s16x4 Avec_r1 =
        (r1_ok && chunk_elems > 0) ? load_bf16x4_raw(x_ptr1, chunk_elems)
                                   : s16x4{0, 0, 0, 0};

    s16x4 Bvec[COL_GROUPS];

#pragma unroll
    for (int idx = 0; idx < COL_GROUPS; ++idx) {
      Bvec[idx] = (col_ok[idx] && chunk_elems > 0)
                      ? load_bf16x4_raw(w_ptr[idx], chunk_elems)
                      : s16x4{0, 0, 0, 0};
    }

#pragma unroll
    for (int idx = 0; idx < COL_GROUPS; ++idx) {
      acc_lo[idx] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          Avec_r0, Bvec[idx], acc_lo[idx], 0, 0, 0);
      acc_hi[idx] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          Avec_r1, Bvec[idx], acc_hi[idx], 0, 0, 0);
    }

    if (r0_ok) {
      x_ptr0 += MATMUL_TILE_K;
    }
    if (r1_ok) {
      x_ptr1 += MATMUL_TILE_K;
    }
#pragma unroll
    for (int idx = 0; idx < COL_GROUPS; ++idx) {
      if (col_ok[idx]) {
        w_ptr[idx] += tile_elems;
      }
    }
  }

  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    const int row_lo = M0 + (i + MATMUL_CHUNK_K * ty);
    const int row_hi = row_lo + 16;

    const bool row_lo_ok = (row_lo < B) && (!has_pos || pos[row_lo] >= 0);
    const bool row_hi_ok = (row_hi < B) && (!has_pos || pos[row_hi] >= 0);

    if (row_lo_ok) {
      float* y0 = y + (size_t)row_lo * d;
#pragma unroll
      for (int idx = 0; idx < COL_GROUPS; ++idx) {
        if (col_ok[idx]) {
          y0[cols[idx]] = acc_lo[idx][i];
        }
      }
    }
    if (row_hi_ok) {
      float* y1 = y + (size_t)row_hi * d;
#pragma unroll
      for (int idx = 0; idx < COL_GROUPS; ++idx) {
        if (col_ok[idx]) {
          y1[cols[idx]] = acc_hi[idx][i];
        }
      }
    }
  }
}

/**
 * Y = X @ W^T + B (float version)
 * x: [B, n], w: [d, n], y: [B, d]
 * Grid: (ceil(d/TM), B)
 */
 __global__ __launch_bounds__(BLOCK_SIZE, 1)
void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [B, d]
    const bf16_t* __restrict__ x,   // [B, n] (bf16)
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
    const bf16_t* __restrict__ xb = x + (size_t)batch_idx * n + k_base;

    // Load using vectorized operations
    const int vec4_k = (k_size >> 2);
    float4* __restrict__ lds_x4 = reinterpret_cast<float4*>(lds_x);
    const uint2* __restrict__ xb4 = reinterpret_cast<const uint2*>(xb);

    for (int v = tid; v < vec4_k; v += BLOCK_SIZE) {
      lds_x4[v] = bf16quad_to_float4(xb4[v]);
    }
    
    // Handle remainder elements
    for (int k = (vec4_k << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = static_cast<float>(xb[k]);
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
__device__ __forceinline__ float clamp_with_limit(float v, float limit) {
  if (limit <= 0.0f)
    return v;
  const float inv_range = 1.0f / (2.0f * limit);
  const float normalized = __saturatef((v + limit) * inv_range);
  return normalized * (2.0f * limit) - limit;
}

__device__ __forceinline__ float swiglu_fused(float gate, float up,
                                              float limit) {
  const float gate_val = clamp_with_limit(gate, limit);
  const float up_val = clamp_with_limit(up, limit);
  const float alpha = 1.702f;
  const float gate_act = gate_val *
      __saturatef(0.5f + 0.5f * tanhf(alpha * gate_val * 0.5f));
  return gate_act * (up_val + 1.0f);
}

__global__ __launch_bounds__(64, 2)
void mlp1_fused_gemm_kernel(
    float *__restrict__ gate_up_topk, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w_mlp1_all, size_t stride_w_mlp1,
    const float *__restrict__ b_mlp1_all,
    const int *__restrict__ assignment_batches,
    const int *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, const int *__restrict__ pos) {
  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0)
    return;

  const int M0 = blockIdx.y * MLP_TILE_TOKENS;
  if (M0 >= count)
    return;
  const int tile_rows = min(MLP_TILE_TOKENS, count - M0);

  const int total_cols = 2 * IM;
  const int N0 = blockIdx.x * MLP_TILE_COLS;
  if (N0 >= total_cols)
    return;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int lane = ty * MLP_THREAD_X + tx;

  __shared__ int sh_batch[MLP_TILE_TOKENS];
  __shared__ int sh_slot[MLP_TILE_TOKENS];
  __shared__ int sh_valid[MLP_TILE_TOKENS];

  for (int idx = lane; idx < MLP_TILE_TOKENS;
       idx += MLP_THREAD_X * MLP_THREAD_Y) {
    int batch = -1;
    int slot = -1;
    int valid = 0;
    if (idx < tile_rows) {
      const int assignment_idx = start + M0 + idx;
      batch = assignment_batches[assignment_idx];
      slot = assignment_slots[assignment_idx];
      if (batch >= 0 && slot >= 0) {
        const bool pos_ok = (!pos) || (pos[batch] >= 0);
        if (pos_ok)
          valid = 1;
      }
    }
    sh_batch[idx] = batch;
    sh_slot[idx] = slot;
    sh_valid[idx] = valid;
  }
  __syncthreads();

  const int row_tile0 = tx;
  const int row_tile1 = tx + 16;
  const bool row0_active = (row_tile0 < tile_rows) && (sh_valid[row_tile0] != 0);
  const bool row1_active = (row_tile1 < tile_rows) && (sh_valid[row_tile1] != 0);
  const int batch0 = row0_active ? sh_batch[row_tile0] : 0;
  const int batch1 = row1_active ? sh_batch[row_tile1] : 0;

  const size_t matrix_idx = (size_t)l_layer * (size_t)E + (size_t)expert_id;
  const bf16_t *__restrict__ w_matrix =
      w_mlp1_all + matrix_idx * stride_w_mlp1;
  const uint16_t *__restrict__ w_u16 =
      reinterpret_cast<const uint16_t *>(w_matrix);
  const float *__restrict__ bias_base =
      b_mlp1_all + matrix_idx * (size_t)(2 * IM);

  const int tiles_k = (H + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;
  const size_t group_offset = (size_t)ty * group_stride;

  const int c0 = N0 + (tx << 1);
  const int c1 = c0 + 1;
  const bool c0_ok = (c0 < total_cols);
  const bool c1_ok = (c1 < total_cols);
  const int im_idx = c0 >> 1;
  const bool col_pair_ok = c0_ok && c1_ok && (im_idx < IM);

  size_t base_tile_c0 = 0;
  if (c0_ok) {
    const int tile_col0 = c0 / MATMUL_TILE_COLS;
    const int row_in_tile0 = c0 % MATMUL_TILE_COLS;
    base_tile_c0 = ((size_t)tile_col0 * tiles_k) * tile_elems +
                   (size_t)row_in_tile0 * MATMUL_CHUNK_K;
  }
  size_t base_tile_c1 = 0;
  if (c1_ok) {
    const int tile_col1 = c1 / MATMUL_TILE_COLS;
    const int row_in_tile1 = c1 % MATMUL_TILE_COLS;
    base_tile_c1 = ((size_t)tile_col1 * tiles_k) * tile_elems +
                   (size_t)row_in_tile1 * MATMUL_CHUNK_K;
  }

  const uint16_t* w_base_c0 = c0_ok ? w_u16 + base_tile_c0 : nullptr;
  const uint16_t* w_base_c1 = c1_ok ? w_u16 + base_tile_c1 : nullptr;

  f32x4 acc00 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc01 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc10 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc11 = {0.f, 0.f, 0.f, 0.f};

  size_t tile_offset = group_offset;
  for (int k0 = 0; k0 < H; k0 += MATMUL_TILE_K, tile_offset += tile_elems) {
    const int k_base = k0 + ty * MATMUL_CHUNK_K;
    const int rem = H - k_base;
    if (rem <= 0) {
      continue;
    }

    const int chunk_elems = rem >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K : rem;
    const uint16_t* x_src0 = (row0_active && chunk_elems > 0)
                                 ? reinterpret_cast<const uint16_t*>(
                                       x + (size_t)batch0 * (size_t)H +
                                       k_base)
                                 : nullptr;
    const uint16_t* x_src1 = (row1_active && chunk_elems > 0)
                                 ? reinterpret_cast<const uint16_t*>(
                                       x + (size_t)batch1 * (size_t)H +
                                       k_base)
                                 : nullptr;

    const s16x4 Avec_r0 = load_bf16x4_raw(x_src0, chunk_elems);
    const s16x4 Avec_r1 = load_bf16x4_raw(x_src1, chunk_elems);

    const size_t chunk_offset = tile_offset;

    const s16x4 Bvec_c0 =
        load_bf16x4_raw((chunk_elems > 0 && c0_ok) ? w_base_c0 + chunk_offset
                                                   : nullptr,
                        chunk_elems);
    const s16x4 Bvec_c1 =
        load_bf16x4_raw((chunk_elems > 0 && c1_ok) ? w_base_c1 + chunk_offset
                                                   : nullptr,
                        chunk_elems);

    acc00 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c0,
                                                       acc00, 0, 0, 0);
    acc01 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c1,
                                                       acc01, 0, 0, 0);
    acc10 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c0,
                                                       acc10, 0, 0, 0);
    acc11 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c1,
                                                       acc11, 0, 0, 0);
  }

  const float bias_gate = c0_ok ? bias_base[c0] : 0.0f;
  const float bias_up = c1_ok ? bias_base[c1] : 0.0f;

  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    const int row_lo = M0 + (i + MATMUL_CHUNK_K * ty);
    const int idx_lo = row_lo - M0;
    if (idx_lo >= 0 && idx_lo < tile_rows && col_pair_ok &&
        sh_valid[idx_lo]) {
      const int batch = sh_batch[idx_lo];
      const int slot = sh_slot[idx_lo];
      if (batch >= 0 && slot >= 0) {
        float gate = acc00[i] + bias_gate;
        float up = acc01[i] + bias_up;
        const float val = swiglu_fused(gate, up, swiglu_limit);
        float *dst = gate_up_topk +
            (((size_t)slot * (size_t)batch_size + (size_t)batch) *
             (size_t)IM);
        dst[im_idx] = val;
      }
    }

    const int row_hi = row_lo + 16;
    const int idx_hi = row_hi - M0;
    if (idx_hi >= 0 && idx_hi < tile_rows && col_pair_ok &&
        sh_valid[idx_hi]) {
      const int batch = sh_batch[idx_hi];
      const int slot = sh_slot[idx_hi];
      if (batch >= 0 && slot >= 0) {
        float gate = acc10[i] + bias_gate;
        float up = acc11[i] + bias_up;
        const float val = swiglu_fused(gate, up, swiglu_limit);
        float *dst = gate_up_topk +
            (((size_t)slot * (size_t)batch_size + (size_t)batch) *
             (size_t)IM);
        dst[im_idx] = val;
      }
    }
  }
}

// ============ MLP2 (weighted accum) : batched per expert ==============
__global__ __launch_bounds__(64, 2)
void mlp2_bias_weighted_accum_gemm_kernel(
    float *__restrict__ e_agg, const float *__restrict__ gate_up_topk,
    const bf16_t *__restrict__ w_mlp2_all, size_t stride_w_mlp2,
    const float *__restrict__ b_mlp2_all,
    const int *__restrict__ assignment_batches,
    const int *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, const float *__restrict__ topk_v,
    int l_layer, int E, int IM, int H, int batch_size,
    const int *__restrict__ pos) {
  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0)
    return;

  const int M0 = blockIdx.y * MLP_TILE_TOKENS;
  if (M0 >= count)
    return;
  const int tile_rows = min(MLP_TILE_TOKENS, count - M0);

  const int N0 = blockIdx.x * MLP_TILE_COLS;
  if (N0 >= H)
    return;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int lane = ty * MLP_THREAD_X + tx;

  __shared__ int sh_batch[MLP_TILE_TOKENS];
  __shared__ int sh_slot[MLP_TILE_TOKENS];
  __shared__ int sh_valid[MLP_TILE_TOKENS];
  __shared__ float sh_weight[MLP_TILE_TOKENS];
  __shared__ size_t sh_gate_offset[MLP_TILE_TOKENS];
  __shared__ short sh_gate_tile[MLP_TILE_TOKENS * MATMUL_TILE_K];

  for (int idx = lane; idx < MLP_TILE_TOKENS;
       idx += MLP_THREAD_X * MLP_THREAD_Y) {
    int batch = -1;
    int slot = -1;
    float weight = 0.0f;
    int valid = 0;
    size_t gate_offset = 0;
    if (idx < tile_rows) {
      const int assignment_idx = start + M0 + idx;
      batch = assignment_batches[assignment_idx];
      slot = assignment_slots[assignment_idx];
      if (batch >= 0 && slot >= 0) {
        const bool pos_ok = (!pos) || (pos[batch] >= 0);
        if (pos_ok) {
          weight = topk_v[(size_t)batch * (size_t)EXPERT_PER_TOKEN + slot];
          if (weight != 0.0f) {
            valid = 1;
            gate_offset = ((size_t)slot * (size_t)batch_size +
                           (size_t)batch) * (size_t)IM;
          }
        }
      }
    }
    sh_batch[idx] = batch;
    sh_slot[idx] = slot;
    sh_weight[idx] = weight;
    sh_valid[idx] = valid;
    sh_gate_offset[idx] = gate_offset;
  }
  __syncthreads();

  const int row_tile0 = tx;
  const int row_tile1 = tx + 16;
  const bool row0_active = (row_tile0 < tile_rows) && (sh_valid[row_tile0] != 0);
  const bool row1_active = (row_tile1 < tile_rows) && (sh_valid[row_tile1] != 0);


  const size_t matrix_idx = (size_t)l_layer * (size_t)E + (size_t)expert_id;
  const bf16_t *__restrict__ w_matrix =
      w_mlp2_all + matrix_idx * stride_w_mlp2;
  const uint16_t *__restrict__ w_u16 =
      reinterpret_cast<const uint16_t *>(w_matrix);
  const float *__restrict__ bias_base =
      b_mlp2_all + matrix_idx * (size_t)H;

  const int tiles_k = (IM + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;
  const size_t group_offset = (size_t)ty * group_stride;

  const int c0 = N0 + tx;
  const int c1 = N0 + tx + 16;
  const bool c0_ok = (c0 < H);
  const bool c1_ok = (c1 < H);

  size_t base_tile_c0 = 0;
  if (c0_ok) {
    const int tile_col0 = c0 / MATMUL_TILE_COLS;
    const int row_in_tile0 = c0 % MATMUL_TILE_COLS;
    base_tile_c0 = ((size_t)tile_col0 * tiles_k) * tile_elems +
                   (size_t)row_in_tile0 * MATMUL_CHUNK_K;
  }
  size_t base_tile_c1 = 0;
  if (c1_ok) {
    const int tile_col1 = c1 / MATMUL_TILE_COLS;
    const int row_in_tile1 = c1 % MATMUL_TILE_COLS;
    base_tile_c1 = ((size_t)tile_col1 * tiles_k) * tile_elems +
                   (size_t)row_in_tile1 * MATMUL_CHUNK_K;
  }

  const uint16_t* w_base_c0 = c0_ok ? w_u16 + base_tile_c0 : nullptr;
  const uint16_t* w_base_c1 = c1_ok ? w_u16 + base_tile_c1 : nullptr;

  f32x4 acc00 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc01 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc10 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc11 = {0.f, 0.f, 0.f, 0.f};

  size_t tile_offset = group_offset;
  for (int k0 = 0; k0 < IM; k0 += MATMUL_TILE_K) {
    const int rem_total = IM - k0;
    if (rem_total <= 0) {
      break;
    }

    const int chunk_total =
        rem_total >= MATMUL_TILE_K ? MATMUL_TILE_K : rem_total;
    const int stage_elems = tile_rows * MATMUL_TILE_K;
    for (int linear = lane; linear < stage_elems;
         linear += MLP_THREAD_X * MLP_THREAD_Y) {
      const int row = linear / MATMUL_TILE_K;
      const int k = linear - row * MATMUL_TILE_K;
      short val = 0;
      if (k < chunk_total && sh_valid[row]) {
        const size_t gate_offset = sh_gate_offset[row];
        const float *row_ptr = gate_up_topk + gate_offset + k0;
        const uint32_t raw = reinterpret_cast<const uint32_t *>(row_ptr)[k];
        val = static_cast<short>(raw >> 16);
      }
      sh_gate_tile[row * MATMUL_TILE_K + k] = val;
    }
    __syncthreads();

    const int k_offset = ty * MATMUL_CHUNK_K;
    const int k_remaining = rem_total - k_offset;
    const int chunk_elems =
        k_remaining > MATMUL_CHUNK_K
            ? MATMUL_CHUNK_K
            : (k_remaining > 0 ? k_remaining : 0);

    s16x4 Avec_r0 = {0, 0, 0, 0};
    if (row0_active && chunk_elems > 0) {
      const short *tile_row0 =
          sh_gate_tile + row_tile0 * MATMUL_TILE_K + k_offset;
#pragma unroll
      for (int j = 0; j < MATMUL_CHUNK_K; ++j) {
        if (j < chunk_elems) {
          Avec_r0[j] = tile_row0[j];
        }
      }
    }

    s16x4 Avec_r1 = {0, 0, 0, 0};
    if (row1_active && chunk_elems > 0) {
      const short *tile_row1 =
          sh_gate_tile + row_tile1 * MATMUL_TILE_K + k_offset;
#pragma unroll
      for (int j = 0; j < MATMUL_CHUNK_K; ++j) {
        if (j < chunk_elems) {
          Avec_r1[j] = tile_row1[j];
        }
      }
    }

    const size_t chunk_offset = tile_offset;

    const s16x4 Bvec_c0 =
        load_bf16x4_raw((chunk_elems > 0 && c0_ok) ? w_base_c0 + chunk_offset
                                                   : nullptr,
                        chunk_elems);
    const s16x4 Bvec_c1 =
        load_bf16x4_raw((chunk_elems > 0 && c1_ok) ? w_base_c1 + chunk_offset
                                                   : nullptr,
                        chunk_elems);

    acc00 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c0,
                                                       acc00, 0, 0, 0);
    acc01 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c1,
                                                       acc01, 0, 0, 0);
    acc10 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c0,
                                                       acc10, 0, 0, 0);
    acc11 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c1,
                                                       acc11, 0, 0, 0);

    __syncthreads();
    tile_offset += tile_elems;
  }

  const float bias_c0 = c0_ok ? bias_base[c0] : 0.0f;
  const float bias_c1 = c1_ok ? bias_base[c1] : 0.0f;

  for (int i = 0; i < MATMUL_CHUNK_K; ++i) {
    const int row_lo = M0 + (i + MATMUL_CHUNK_K * ty);
    const int idx_lo = row_lo - M0;
    if (idx_lo >= 0 && idx_lo < tile_rows && sh_valid[idx_lo]) {
      const int batch = sh_batch[idx_lo];
      const float weight = sh_weight[idx_lo];
      if (batch >= 0 && weight != 0.0f) {
        if (c0_ok) {
          const float contrib = (acc00[i] + bias_c0) * weight;
          atomicAdd(e_agg + (size_t)batch * (size_t)H + (size_t)c0,
                    contrib);
        }
        if (c1_ok) {
          const float contrib = (acc01[i] + bias_c1) * weight;
          atomicAdd(e_agg + (size_t)batch * (size_t)H + (size_t)c1,
                    contrib);
        }
      }
    }

    const int row_hi = row_lo + 16;
    const int idx_hi = row_hi - M0;
    if (idx_hi >= 0 && idx_hi < tile_rows && sh_valid[idx_hi]) {
      const int batch = sh_batch[idx_hi];
      const float weight = sh_weight[idx_hi];
      if (batch >= 0 && weight != 0.0f) {
        if (c0_ok) {
          const float contrib = (acc10[i] + bias_c0) * weight;
          atomicAdd(e_agg + (size_t)batch * (size_t)H + (size_t)c0,
                    contrib);
        }
        if (c1_ok) {
          const float contrib = (acc11[i] + bias_c1) * weight;
          atomicAdd(e_agg + (size_t)batch * (size_t)H + (size_t)c1,
                    contrib);
        }
      }
    }
  }
}
