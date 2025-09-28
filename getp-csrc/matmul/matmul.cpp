#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>

template <
  typename OutT, bool HasBias,
  int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH,
  int WARP_TILE_M, int WARP_TILE_N
>
__device__ __forceinline__ void matmul_bf16_mfma_body(
    OutT *__restrict__ y,               // [B x d]
    const bf16_t *__restrict__ x,       // [B x n] (bf16)
    const bf16_t *__restrict__ w,       // [d x n] (bf16, packed theo tile)
    const float *__restrict__ bias,     // [d] or nullptr (ignored if !HasBias)
    int n, int d, int B,
    const int *__restrict__ pos) {

  constexpr int SUB_TILES_M   = WARP_TILE_M / 16;
  constexpr int SUB_TILES_N   = WARP_TILE_N / 16;
  constexpr int WAVES_M       = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N       = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;
  constexpr int K_QUADS       = BLOCK_DEPTH / MATMUL_CHUNK_K;
  constexpr int LDS_STRIDE    = K_QUADS + 3;

  const int block_m = blockIdx.y * BLOCK_ROWS;
  const int block_n = blockIdx.x * BLOCK_COLS;
  if (block_m >= B || block_n >= d) return;

  if (blockDim.x != WF_SIZE || blockDim.y != WAVES_PER_BLOCK) return;

  const bool has_pos = (pos != nullptr);

  const int tiles_k_total = (n + MATMUL_TILE_K - 1) / MATMUL_TILE_K;

  const int wave = threadIdx.y;
  const int lane = threadIdx.x & (WF_SIZE - 1);
  const int tid_linear = wave * WF_SIZE + lane;
  const int threads_per_block = blockDim.y * WF_SIZE;

  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  if (wave_m >= WAVES_M || wave_n >= WAVES_N) return;

  const int lane_mod16      = lane & 15;
  const int lane_row        = lane_mod16;
  const int lane_col        = lane_mod16;
  const int lane_group      = lane >> 4;
  const int k_group         = lane_group * MATMUL_CHUNK_K;
  const int row_lane_offset = lane_group * 4;

  __shared__ __align__(16) s16x4 sh_A[BLOCK_ROWS * LDS_STRIDE];
  __shared__ __align__(16) s16x4 sh_B[BLOCK_COLS * LDS_STRIDE];

  f32x4 acc[SUB_TILES_M * SUB_TILES_N];
#pragma unroll
  for (int i = 0; i < SUB_TILES_M * SUB_TILES_N; ++i) acc[i] = f32x4{0.f,0.f,0.f,0.f};

  const uint16_t *x_u16 = reinterpret_cast<const uint16_t *>(x);
  const uint16_t *w_u16 = reinterpret_cast<const uint16_t *>(w);

  float bias_lane[SUB_TILES_N];
#pragma unroll
  for (int wn = 0; wn < SUB_TILES_N; ++wn) bias_lane[wn] = 0.f;

  if constexpr (HasBias) {
    if (bias != nullptr) {
#pragma unroll
      for (int wn = 0; wn < SUB_TILES_N; ++wn) {
        const int col = block_n + wave_n * WARP_TILE_N + wn * 16 + lane_col;
        if (col < d) bias_lane[wn] = bias[col];
      }
    }
  }

  const int total_tiles_k = (n + BLOCK_DEPTH - 1) / BLOCK_DEPTH;
  for (int tile_idx = 0; tile_idx < total_tiles_k; ++tile_idx) {
    const int k_base = tile_idx * BLOCK_DEPTH;

    const int total_a_quads = BLOCK_ROWS * K_QUADS;
    for (int linear = tid_linear; linear < total_a_quads; linear += threads_per_block) {
      const int row  = linear / K_QUADS;
      const int quad = linear - row * K_QUADS;
      const int global_row = block_m + row;
      const int k   = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K
                                                    : (remaining > 0 ? remaining : 0);
      const bool row_in_range = (global_row < B);
      const bool row_active = row_in_range && (!has_pos || pos[global_row] >= 0);
      s16x4 val = {0,0,0,0};
      if (row_active && valid > 0) {
        const uint16_t *src = x_u16 + (size_t)global_row * (size_t)n + (size_t)k;
        val = load_bf16x4(src, valid);
      }
      sh_A[row * LDS_STRIDE + quad] = val;
    }

    const int total_b_quads = BLOCK_COLS * K_QUADS;
    for (int linear = tid_linear; linear < total_b_quads; linear += threads_per_block) {
      const int col  = linear / K_QUADS;           // 0..BLOCK_COLS-1 trong block
      const int quad = linear - col * K_QUADS;     // 0..K_QUADS-1
      const int global_col = block_n + col;        // cột thật trong d
      const int k   = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K
                                                    : (remaining > 0 ? remaining : 0);

      s16x4 val = {0,0,0,0};
      if (global_col < d && valid > 0) {
        // === Inline công thức packed_weight(base, global_col, k, tiles_k_total)
        const int tile_col   = global_col / MATMUL_TILE_COLS;
        const int row_in_tile= global_col - tile_col * MATMUL_TILE_COLS;
        const int tile_k     = k / MATMUL_TILE_K;
        const int k_in_tile  = k - tile_k * MATMUL_TILE_K;
        const int group      = k_in_tile / MATMUL_CHUNK_K;           // 0..(MATMUL_TILE_K/4-1)
        const int within     = k_in_tile - group * MATMUL_CHUNK_K;   // 0..3

        const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
        const size_t tile_base  = ((size_t)tile_col * tiles_k_total + tile_k) * tile_elems;
        const size_t group_base = (size_t)group * MATMUL_TILE_COLS * MATMUL_CHUNK_K;

        const uint16_t *src_u16 =
            reinterpret_cast<const uint16_t *>(w_u16) +
            tile_base + group_base +
            (size_t)row_in_tile * MATMUL_CHUNK_K + within;

        val = load_bf16x4(src_u16, valid);
      }
      sh_B[col * LDS_STRIDE + quad] = val;
    }

    __syncthreads();

    const int wave_row_base = wave_m * WARP_TILE_M + lane_row;
    const int wave_col_base = wave_n * WARP_TILE_N + lane_col;

    for (int kk = 0; kk < BLOCK_DEPTH; kk += 16) {
      const int quad_idx = (k_group + kk) >> 2;
#pragma unroll
      for (int wm = 0; wm < SUB_TILES_M; ++wm) {
        const int a_row = wave_row_base + wm * 16;
        const s16x4 avec = sh_A[a_row * LDS_STRIDE + quad_idx];
#pragma unroll
        for (int wn = 0; wn < SUB_TILES_N; ++wn) {
          const int b_col = wave_col_base + wn * 16;
          const int acc_idx = wm * SUB_TILES_N + wn;
          const s16x4 bvec = sh_B[b_col * LDS_STRIDE + quad_idx];
          acc[acc_idx] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              avec, bvec, acc[acc_idx], 0,0,0);
        }
      }
    }

    __syncthreads();
  }

  const int lane_bias_col = lane_col;
#pragma unroll
  for (int wm = 0; wm < SUB_TILES_M; ++wm) {
    const int row_block = block_m + wave_m * WARP_TILE_M + wm * 16;
#pragma unroll
    for (int wn = 0; wn < SUB_TILES_N; ++wn) {
      const int col = block_n + wave_n * WARP_TILE_N + wn * 16 + lane_bias_col;
      if (col >= d) continue;
      const int acc_idx = wm * SUB_TILES_N + wn;
      const f32x4 v = acc[acc_idx];

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row_block + (i + row_lane_offset);
        if (row >= B) continue;
        if (has_pos && pos[row] < 0) continue;

        float out_val = v[i];
        if constexpr (HasBias) {
          out_val += bias_lane[wn];
        }

        if constexpr (std::is_same<OutT, bf16_t>::value) {
          y[(size_t)row * (size_t)d + col] = bf16_t(out_val);
        } else {
          y[(size_t)row * (size_t)d + col] = out_val;
        }
      }
    }
  }
}

__global__ void matmul_bias_gemm_kernel_bf16_mfma_qkv(
    bf16_t *__restrict__ y, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w, const float *__restrict__ bias,
    int n, int d, int B, const int *__restrict__ pos) {
  matmul_bf16_mfma_body<
      bf16_t, true,
      MATMUL_QKV_BLOCK_ROWS, MATMUL_QKV_BLOCK_COLS, MATMUL_QKV_BLOCK_DEPTH,
      MATMUL_QKV_WARP_TILE_M, MATMUL_QKV_WARP_TILE_N>(
      y, x, w, bias, n, d, B, pos);
}

__global__ void matmul_bias_gemm_kernel_bf16_mfma_att(
    bf16_t *__restrict__ y, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w, const float *__restrict__ bias,
    int n, int d, int B, const int *__restrict__ pos) {
  matmul_bf16_mfma_body<
      bf16_t, true,
      MATMUL_ATT_BLOCK_ROWS, MATMUL_ATT_BLOCK_COLS, MATMUL_ATT_BLOCK_DEPTH,
      MATMUL_ATT_WARP_TILE_M, MATMUL_ATT_WARP_TILE_N>(
      y, x, w, bias, n, d, B, pos);
}

__global__ void matmul_gemm_kernel_bf16_mfma(
    float *__restrict__ y, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w, int n, int d, int B,
    const int *__restrict__ pos) {
  matmul_bf16_mfma_body<
      float, false,
      MATMUL_LOGITS_BLOCK_ROWS, MATMUL_LOGITS_BLOCK_COLS, MATMUL_LOGITS_BLOCK_DEPTH,
      MATMUL_LOGITS_WARP_TILE_M, MATMUL_LOGITS_WARP_TILE_N>(
      y, x, w, nullptr, n, d, B, pos);
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

// ================= MLP1 (Gate & Up) : batched per expert =================

__global__ __launch_bounds__(64, 2)
void mlp1_fused_gemm_kernel(
    bf16_t *__restrict__ gate_up_topk, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w_mlp1_all, size_t stride_w_mlp1,
    const bf16_t *__restrict__ b_mlp1_all,
    const uint16_t *__restrict__ assignment_batches,
    const uint8_t *__restrict__ assignment_slots,
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
      batch = static_cast<int>(assignment_batches[assignment_idx]);
      slot = static_cast<int>(assignment_slots[assignment_idx]);
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
  const bf16_t *__restrict__ bias_base =
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
  const uint16_t* w_ptr_c0 = c0_ok ? (w_base_c0 + group_offset) : nullptr;
  const uint16_t* w_ptr_c1 = c1_ok ? (w_base_c1 + group_offset) : nullptr;

  f32x4 acc00 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc01 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc10 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc11 = {0.f, 0.f, 0.f, 0.f};

  const uint16_t* x_ptr0 = row0_active
                               ? reinterpret_cast<const uint16_t*>(
                                     x + (size_t)batch0 * (size_t)H) +
                                     ty * MATMUL_CHUNK_K
                               : nullptr;
  const uint16_t* x_ptr1 = row1_active
                               ? reinterpret_cast<const uint16_t*>(
                                     x + (size_t)batch1 * (size_t)H) +
                                     ty * MATMUL_CHUNK_K
                               : nullptr;

  for (int k0 = 0; k0 < H; k0 += MATMUL_TILE_K) {
    const int k_base = k0 + ty * MATMUL_CHUNK_K;
    const int rem = H - k_base;
    if (rem <= 0) {
      if (row0_active) {
        x_ptr0 += MATMUL_TILE_K;
      }
      if (row1_active) {
        x_ptr1 += MATMUL_TILE_K;
      }
      if (c0_ok) {
        w_ptr_c0 += tile_elems;
      }
      if (c1_ok) {
        w_ptr_c1 += tile_elems;
      }
      continue;
    }

    const int chunk_elems = rem >= MATMUL_CHUNK_K ? MATMUL_CHUNK_K : rem;
    const s16x4 Avec_r0 =
        (row0_active && chunk_elems > 0) ? load_bf16x4(x_ptr0, chunk_elems)
                                         : s16x4{0, 0, 0, 0};
    const s16x4 Avec_r1 =
        (row1_active && chunk_elems > 0) ? load_bf16x4(x_ptr1, chunk_elems)
                                         : s16x4{0, 0, 0, 0};

    const s16x4 Bvec_c0 =
        (chunk_elems > 0 && c0_ok) ? load_bf16x4(w_ptr_c0, chunk_elems)
                                   : s16x4{0, 0, 0, 0};
    const s16x4 Bvec_c1 =
        (chunk_elems > 0 && c1_ok) ? load_bf16x4(w_ptr_c1, chunk_elems)
                                   : s16x4{0, 0, 0, 0};

    acc00 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c0,
                                                       acc00, 0, 0, 0);
    acc01 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c1,
                                                       acc01, 0, 0, 0);
    acc10 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c0,
                                                       acc10, 0, 0, 0);
    acc11 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c1,
                                                       acc11, 0, 0, 0);

    if (row0_active) {
      x_ptr0 += MATMUL_TILE_K;
    }
    if (row1_active) {
      x_ptr1 += MATMUL_TILE_K;
    }
    if (c0_ok) {
      w_ptr_c0 += tile_elems;
    }
    if (c1_ok) {
      w_ptr_c1 += tile_elems;
    }
  }

  const float bias_gate =
      c0_ok ? static_cast<float>(bias_base[c0]) : 0.0f;
  const float bias_up =
      c1_ok ? static_cast<float>(bias_base[c1]) : 0.0f;

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
        bf16_t *dst = gate_up_topk +
            (((size_t)slot * (size_t)batch_size + (size_t)batch) *
             (size_t)IM);
        dst[im_idx] = bf16_t(val);
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
        bf16_t *dst = gate_up_topk +
            (((size_t)slot * (size_t)batch_size + (size_t)batch) *
             (size_t)IM);
        dst[im_idx] = bf16_t(val);
      }
    }
  }
}

// ============ MLP2 (weighted accum) : batched per expert ==============
__global__ __launch_bounds__(64, 2)
void mlp2_bias_weighted_accum_gemm_kernel(
    float *__restrict__ e_agg, const bf16_t *__restrict__ gate_up_topk,
    const bf16_t *__restrict__ w_mlp2_all, size_t stride_w_mlp2,
    const bf16_t *__restrict__ b_mlp2_all,
    const uint16_t *__restrict__ assignment_batches,
    const uint8_t *__restrict__ assignment_slots,
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
  __shared__ uint16_t sh_gate_tile[MLP_TILE_TOKENS * MATMUL_TILE_K];

  for (int idx = lane; idx < MLP_TILE_TOKENS;
       idx += MLP_THREAD_X * MLP_THREAD_Y) {
    int batch = -1;
    int slot = -1;
    float weight = 0.0f;
    int valid = 0;
    size_t gate_offset = 0;
    if (idx < tile_rows) {
      const int assignment_idx = start + M0 + idx;
      batch = static_cast<int>(assignment_batches[assignment_idx]);
      slot = static_cast<int>(assignment_slots[assignment_idx]);
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
  const bf16_t *__restrict__ bias_base =
      b_mlp2_all + matrix_idx * (size_t)H;

  const int tiles_k = (IM + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems = (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
  const size_t group_stride = (size_t)MATMUL_TILE_COLS * MATMUL_CHUNK_K;
  const size_t group_offset = (size_t)ty * group_stride;

  const uint16_t *__restrict__ gate_up_u16 =
      reinterpret_cast<const uint16_t *>(gate_up_topk);

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
  const uint16_t* w_ptr_c0 = c0_ok ? (w_base_c0 + group_offset) : nullptr;
  const uint16_t* w_ptr_c1 = c1_ok ? (w_base_c1 + group_offset) : nullptr;

  f32x4 acc00 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc01 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc10 = {0.f, 0.f, 0.f, 0.f};
  f32x4 acc11 = {0.f, 0.f, 0.f, 0.f};

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
      uint16_t val = 0;
      if (k < chunk_total && sh_valid[row]) {
        const size_t gate_offset = sh_gate_offset[row];
        const size_t elem_idx = gate_offset + k0 + k;
        val = gate_up_u16[elem_idx];
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

    const uint16_t *tile_row0 = row0_active
                                ? sh_gate_tile +
                                      row_tile0 * MATMUL_TILE_K + k_offset
                                : nullptr;
    const uint16_t *tile_row1 = row1_active
                                ? sh_gate_tile +
                                      row_tile1 * MATMUL_TILE_K + k_offset
                                : nullptr;

    const s16x4 Avec_r0 =
        (row0_active && chunk_elems > 0)
            ? load_bf16x4(tile_row0, chunk_elems)
            : s16x4{0, 0, 0, 0};

    const s16x4 Avec_r1 =
        (row1_active && chunk_elems > 0)
            ? load_bf16x4(tile_row1, chunk_elems)
            : s16x4{0, 0, 0, 0};

    const s16x4 Bvec_c0 =
        (chunk_elems > 0 && c0_ok) ? load_bf16x4(w_ptr_c0, chunk_elems)
                                   : s16x4{0, 0, 0, 0};
    const s16x4 Bvec_c1 =
        (chunk_elems > 0 && c1_ok) ? load_bf16x4(w_ptr_c1, chunk_elems)
                                   : s16x4{0, 0, 0, 0};

    acc00 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c0,
                                                       acc00, 0, 0, 0);
    acc01 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r0, Bvec_c1,
                                                       acc01, 0, 0, 0);
    acc10 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c0,
                                                       acc10, 0, 0, 0);
    acc11 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(Avec_r1, Bvec_c1,
                                                       acc11, 0, 0, 0);

    __syncthreads();
    if (c0_ok) {
      w_ptr_c0 += tile_elems;
    }
    if (c1_ok) {
      w_ptr_c1 += tile_elems;
    }
  }

  const float bias_c0 =
      c0_ok ? static_cast<float>(bias_base[c0]) : 0.0f;
  const float bias_c1 =
      c1_ok ? static_cast<float>(bias_base[c1]) : 0.0f;

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
