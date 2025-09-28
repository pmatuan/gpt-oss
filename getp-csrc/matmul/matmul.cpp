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

__global__ __launch_bounds__(WF_SIZE * MATMUL_MLP1_WAVES_PER_BLOCK, 1)
void mlp1_fused_gemm_kernel(
    bf16_t *__restrict__ gate_up_topk, const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w_mlp1_all, size_t stride_w_mlp1,
    const bf16_t *__restrict__ b_mlp1_all,
    const uint16_t *__restrict__ assignment_batches,
    const uint8_t *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, const int *__restrict__ pos) {
  constexpr int BLOCK_ROWS = MATMUL_MLP1_BLOCK_ROWS;
  constexpr int BLOCK_COLS = MATMUL_MLP1_BLOCK_COLS;
  constexpr int BLOCK_DEPTH = MATMUL_MLP1_BLOCK_DEPTH;
  constexpr int WARP_TILE_M = MATMUL_MLP1_WARP_TILE_M;
  constexpr int WARP_TILE_N = MATMUL_MLP1_WARP_TILE_N;
  constexpr int SUB_TILES_M = WARP_TILE_M / 16;
  constexpr int SUB_TILES_N = WARP_TILE_N / 16;
  constexpr int WAVES_M = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = MATMUL_MLP1_WAVES_PER_BLOCK;
  constexpr int K_QUADS = BLOCK_DEPTH / MATMUL_CHUNK_K;
  constexpr int LDS_STRIDE = K_QUADS + 3;

  static_assert(WAVES_M * WAVES_N == WAVES_PER_BLOCK,
                "Invalid MLP1 wave configuration");

  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0)
    return;

  const int block_m = blockIdx.y * BLOCK_ROWS;
  if (block_m >= count)
    return;
  const int tile_rows = min(BLOCK_ROWS, count - block_m);

  const int total_cols = 2 * IM;
  const int block_n = blockIdx.x * BLOCK_COLS;
  if (block_n >= total_cols)
    return;

  if (blockDim.x != WF_SIZE || blockDim.y != WAVES_PER_BLOCK)
    return;

  const int lane = threadIdx.x & (WF_SIZE - 1);
  const int wave = threadIdx.y;
  const int tid_linear = wave * WF_SIZE + lane;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  if (wave_m >= WAVES_M || wave_n >= WAVES_N)
    return;

  const int lane_mod16 = lane & 15;
  const int lane_row = lane_mod16;
  const int lane_col = lane_mod16;
  const int lane_group = lane >> 4;
  const int k_group = lane_group * MATMUL_CHUNK_K;
  const int row_lane_offset = lane_group * 4;

  __shared__ int sh_batch[BLOCK_ROWS];
  __shared__ int sh_slot[BLOCK_ROWS];
  __shared__ uint8_t sh_valid[BLOCK_ROWS];
  __shared__ __align__(16) s16x4 sh_A[BLOCK_ROWS * LDS_STRIDE];
  __shared__ __align__(16) s16x4 sh_B[BLOCK_COLS * LDS_STRIDE];

  const uint16_t *__restrict__ x_u16 =
      reinterpret_cast<const uint16_t *>(x);

  for (int idx = tid_linear; idx < BLOCK_ROWS; idx += threads_per_block) {
    int batch = -1;
    int slot = -1;
    uint8_t valid = 0;
    if (idx < tile_rows) {
      const int assignment_idx = start + block_m + idx;
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

  const size_t matrix_idx = (size_t)l_layer * (size_t)E + (size_t)expert_id;
  const bf16_t *__restrict__ w_matrix =
      w_mlp1_all + matrix_idx * stride_w_mlp1;
  const uint16_t *__restrict__ w_u16 =
      reinterpret_cast<const uint16_t *>(w_matrix);
  const bf16_t *__restrict__ bias_base =
      b_mlp1_all + matrix_idx * (size_t)(2 * IM);

  const int n = H;
  const int tiles_k_total = (n + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const int total_tiles = (n + BLOCK_DEPTH - 1) / BLOCK_DEPTH;

  f32x4 acc[SUB_TILES_M * SUB_TILES_N];
#pragma unroll
  for (int i = 0; i < SUB_TILES_M * SUB_TILES_N; ++i)
    acc[i] = f32x4{0.f, 0.f, 0.f, 0.f};

  float bias_lane[SUB_TILES_N];
#pragma unroll
  for (int wn = 0; wn < SUB_TILES_N; ++wn) {
    const int col = block_n + wave_n * WARP_TILE_N + wn * 16 + lane_col;
    bias_lane[wn] = (col < total_cols)
                        ? static_cast<float>(bias_base[col])
                        : 0.0f;
  }

  for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
    const int k_base = tile_idx * BLOCK_DEPTH;

    const int total_a_quads = BLOCK_ROWS * K_QUADS;
    for (int linear = tid_linear; linear < total_a_quads;
         linear += threads_per_block) {
      const int row = linear / K_QUADS;
      const int quad = linear - row * K_QUADS;
      const int k = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K
                             ? MATMUL_CHUNK_K
                             : (remaining > 0 ? remaining : 0);
      s16x4 val = {0, 0, 0, 0};
      if (row < tile_rows && sh_valid[row] && valid > 0) {
        const int batch = sh_batch[row];
        const size_t offset = (size_t)batch * (size_t)n + (size_t)k;
        val = load_bf16x4(x_u16 + offset, valid);
      }
      sh_A[row * LDS_STRIDE + quad] = val;
    }

    const int total_b_quads = BLOCK_COLS * K_QUADS;
    for (int linear = tid_linear; linear < total_b_quads;
         linear += threads_per_block) {
      const int col = linear / K_QUADS;
      const int quad = linear - col * K_QUADS;
      const int global_col = block_n + col;
      const int k = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K
                             ? MATMUL_CHUNK_K
                             : (remaining > 0 ? remaining : 0);
      s16x4 val = {0, 0, 0, 0};
      if (global_col < total_cols && valid > 0) {
        const int tile_col = global_col / MATMUL_TILE_COLS;
        const int row_in_tile = global_col - tile_col * MATMUL_TILE_COLS;
        const int tile_k = k / MATMUL_TILE_K;
        const int k_in_tile = k - tile_k * MATMUL_TILE_K;
        const int group = k_in_tile / MATMUL_CHUNK_K;
        const int within = k_in_tile - group * MATMUL_CHUNK_K;
        const size_t tile_elems =
            (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
        const size_t tile_base =
            ((size_t)tile_col * tiles_k_total + tile_k) * tile_elems;
        const size_t group_base =
            (size_t)group * MATMUL_TILE_COLS * MATMUL_CHUNK_K;
        const uint16_t *src =
            w_u16 + tile_base + group_base +
            (size_t)row_in_tile * MATMUL_CHUNK_K + within;
        val = load_bf16x4(src, valid);
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
              avec, bvec, acc[acc_idx], 0, 0, 0);
        }
      }
    }

    __syncthreads();
  }

  const float limit = swiglu_limit;

#pragma unroll
  for (int wm = 0; wm < SUB_TILES_M; ++wm) {
    const int row_block = wave_m * WARP_TILE_M + wm * 16;
#pragma unroll
    for (int wn = 0; wn < SUB_TILES_N; ++wn) {
      const int col_base = wave_n * WARP_TILE_N + wn * 16;
      const int col = block_n + col_base + lane_col;
      const bool col_valid = (col < total_cols);
      const int im_idx = col >> 1;
      const bool im_valid = (im_idx < IM);
      const bool is_gate = ((col & 1) == 0);
      const float bias_val = bias_lane[wn];
      const int acc_idx = wm * SUB_TILES_N + wn;
      const f32x4 vec = acc[acc_idx];
      const int row_base = row_block + row_lane_offset;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row_base + i;
        if (row >= tile_rows)
          continue;
        if (!sh_valid[row])
          continue;

        float value = vec[i] + bias_val;
        int col_ok = (col_valid && im_valid) ? 1 : 0;
        if (!col_ok)
          value = 0.0f;

        const float partner_value = __shfl_xor(value, 1, WF_SIZE);
        const int partner_ok = __shfl_xor(col_ok, 1, WF_SIZE);

        if (!is_gate || !col_ok || !partner_ok)
          continue;

        const int batch = sh_batch[row];
        const int slot = sh_slot[row];
        if (batch < 0 || slot < 0)
          continue;

        const float gate = value;
        const float up = partner_value;
        const float fused = swiglu_fused(gate, up, limit);
        const size_t dst_offset =
            (((size_t)slot * (size_t)batch_size + (size_t)batch) *
             (size_t)IM) + (size_t)im_idx;
        gate_up_topk[dst_offset] = bf16_t(fused);
      }
    }
  }
}

// ============ MLP2 (weighted accum) : batched per expert ==============
__global__ __launch_bounds__(WF_SIZE * MATMUL_MLP2_WAVES_PER_BLOCK, 1)
void mlp2_bias_weighted_accum_gemm_kernel(
    float *__restrict__ e_agg, const bf16_t *__restrict__ gate_up_topk,
    const bf16_t *__restrict__ w_mlp2_all, size_t stride_w_mlp2,
    const bf16_t *__restrict__ b_mlp2_all,
    const uint16_t *__restrict__ assignment_batches,
    const uint8_t *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, const float *__restrict__ topk_v,
    int l_layer, int E, int IM, int H, int batch_size,
    const int *__restrict__ pos) {
  constexpr int BLOCK_ROWS = MATMUL_MLP2_BLOCK_ROWS;
  constexpr int BLOCK_COLS = MATMUL_MLP2_BLOCK_COLS;
  constexpr int BLOCK_DEPTH = MATMUL_MLP2_BLOCK_DEPTH;
  constexpr int WARP_TILE_M = MATMUL_MLP2_WARP_TILE_M;
  constexpr int WARP_TILE_N = MATMUL_MLP2_WARP_TILE_N;
  constexpr int SUB_TILES_M = WARP_TILE_M / 16;
  constexpr int SUB_TILES_N = WARP_TILE_N / 16;
  constexpr int WAVES_M = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = MATMUL_MLP2_WAVES_PER_BLOCK;
  constexpr int K_QUADS = BLOCK_DEPTH / MATMUL_CHUNK_K;
  constexpr int LDS_STRIDE = K_QUADS + 3;

  static_assert(WAVES_M * WAVES_N == WAVES_PER_BLOCK,
                "Invalid MLP2 wave configuration");

  const int expert_id = blockIdx.z;
  const int start = expert_offsets[expert_id];
  const int end = expert_offsets[expert_id + 1];
  const int count = end - start;
  if (count <= 0)
    return;

  const int block_m = blockIdx.y * BLOCK_ROWS;
  if (block_m >= count)
    return;
  const int tile_rows = min(BLOCK_ROWS, count - block_m);

  const int total_cols = H;
  const int block_n = blockIdx.x * BLOCK_COLS;
  if (block_n >= total_cols)
    return;

  if (blockDim.x != WF_SIZE || blockDim.y != WAVES_PER_BLOCK)
    return;

  const int lane = threadIdx.x & (WF_SIZE - 1);
  const int wave = threadIdx.y;
  const int tid_linear = wave * WF_SIZE + lane;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  if (wave_m >= WAVES_M || wave_n >= WAVES_N)
    return;

  const int lane_mod16 = lane & 15;
  const int lane_row = lane_mod16;
  const int lane_col = lane_mod16;
  const int lane_group = lane >> 4;
  const int k_group = lane_group * MATMUL_CHUNK_K;
  const int row_lane_offset = lane_group * 4;

  __shared__ int sh_batch[BLOCK_ROWS];
  __shared__ float sh_weight[BLOCK_ROWS];
  __shared__ uint8_t sh_valid[BLOCK_ROWS];
  __shared__ size_t sh_gate_offset[BLOCK_ROWS];
  __shared__ __align__(16) s16x4 sh_A[BLOCK_ROWS * LDS_STRIDE];
  __shared__ __align__(16) s16x4 sh_B[BLOCK_COLS * LDS_STRIDE];

  const uint16_t *__restrict__ gate_up_u16 =
      reinterpret_cast<const uint16_t *>(gate_up_topk);

  for (int idx = tid_linear; idx < BLOCK_ROWS; idx += threads_per_block) {
    int batch = -1;
    float weight = 0.0f;
    uint8_t valid = 0;
    size_t gate_offset = 0;
    if (idx < tile_rows) {
      const int assignment_idx = start + block_m + idx;
      batch = static_cast<int>(assignment_batches[assignment_idx]);
      const int slot = static_cast<int>(assignment_slots[assignment_idx]);
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
    sh_weight[idx] = weight;
    sh_valid[idx] = valid;
    sh_gate_offset[idx] = gate_offset;
  }
  __syncthreads();

  const size_t matrix_idx = (size_t)l_layer * (size_t)E + (size_t)expert_id;
  const bf16_t *__restrict__ w_matrix =
      w_mlp2_all + matrix_idx * stride_w_mlp2;
  const uint16_t *__restrict__ w_u16 =
      reinterpret_cast<const uint16_t *>(w_matrix);
  const bf16_t *__restrict__ bias_base =
      b_mlp2_all + matrix_idx * (size_t)H;

  const int n = IM;
  const int tiles_k_total = (n + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const int total_tiles = (n + BLOCK_DEPTH - 1) / BLOCK_DEPTH;

  f32x4 acc[SUB_TILES_M * SUB_TILES_N];
#pragma unroll
  for (int i = 0; i < SUB_TILES_M * SUB_TILES_N; ++i)
    acc[i] = f32x4{0.f, 0.f, 0.f, 0.f};

  float bias_lane[SUB_TILES_N];
#pragma unroll
  for (int wn = 0; wn < SUB_TILES_N; ++wn) {
    const int col = block_n + wave_n * WARP_TILE_N + wn * 16 + lane_col;
    bias_lane[wn] = (col < total_cols)
                        ? static_cast<float>(bias_base[col])
                        : 0.0f;
  }

  for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
    const int k_base = tile_idx * BLOCK_DEPTH;

    const int total_a_quads = BLOCK_ROWS * K_QUADS;
    for (int linear = tid_linear; linear < total_a_quads;
         linear += threads_per_block) {
      const int row = linear / K_QUADS;
      const int quad = linear - row * K_QUADS;
      const int k = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K
                             ? MATMUL_CHUNK_K
                             : (remaining > 0 ? remaining : 0);
      s16x4 val = {0, 0, 0, 0};
      if (row < tile_rows && sh_valid[row] && valid > 0) {
        const size_t offset = sh_gate_offset[row] + (size_t)k;
        val = load_bf16x4(gate_up_u16 + offset, valid);
      }
      sh_A[row * LDS_STRIDE + quad] = val;
    }

    const int total_b_quads = BLOCK_COLS * K_QUADS;
    for (int linear = tid_linear; linear < total_b_quads;
         linear += threads_per_block) {
      const int col = linear / K_QUADS;
      const int quad = linear - col * K_QUADS;
      const int global_col = block_n + col;
      const int k = k_base + quad * MATMUL_CHUNK_K;
      const int remaining = n - k;
      const int valid = remaining >= MATMUL_CHUNK_K
                             ? MATMUL_CHUNK_K
                             : (remaining > 0 ? remaining : 0);
      s16x4 val = {0, 0, 0, 0};
      if (global_col < total_cols && valid > 0) {
        const int tile_col = global_col / MATMUL_TILE_COLS;
        const int row_in_tile = global_col - tile_col * MATMUL_TILE_COLS;
        const int tile_k = k / MATMUL_TILE_K;
        const int k_in_tile = k - tile_k * MATMUL_TILE_K;
        const int group = k_in_tile / MATMUL_CHUNK_K;
        const int within = k_in_tile - group * MATMUL_CHUNK_K;
        const size_t tile_elems =
            (size_t)MATMUL_TILE_COLS * MATMUL_TILE_K;
        const size_t tile_base =
            ((size_t)tile_col * tiles_k_total + tile_k) * tile_elems;
        const size_t group_base =
            (size_t)group * MATMUL_TILE_COLS * MATMUL_CHUNK_K;
        const uint16_t *src =
            w_u16 + tile_base + group_base +
            (size_t)row_in_tile * MATMUL_CHUNK_K + within;
        val = load_bf16x4(src, valid);
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
              avec, bvec, acc[acc_idx], 0, 0, 0);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int wm = 0; wm < SUB_TILES_M; ++wm) {
    const int row_block = wave_m * WARP_TILE_M + wm * 16;
#pragma unroll
    for (int wn = 0; wn < SUB_TILES_N; ++wn) {
      const int col_base = wave_n * WARP_TILE_N + wn * 16;
      const int col = block_n + col_base + lane_col;
      const bool col_valid = (col < total_cols);
      const int acc_idx = wm * SUB_TILES_N + wn;
      const f32x4 vec = acc[acc_idx];
      const float bias_val = bias_lane[wn];
      const int row_base = row_block + row_lane_offset;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row_base + i;
        if (row >= tile_rows)
          continue;
        if (!sh_valid[row])
          continue;
        if (!col_valid)
          continue;

        const int batch = sh_batch[row];
        const float weight = sh_weight[row];
        if (batch < 0 || weight == 0.0f)
          continue;

        const float value = (vec[i] + bias_val) * weight;
        atomicAdd(e_agg + (size_t)batch * (size_t)H + (size_t)col, value);
      }
    }
  }
}

