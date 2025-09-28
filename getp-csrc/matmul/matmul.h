#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"
#include <type_traits>

template <int BlockM, int BlockN, int BlockK, int WaveTileM, int WaveTileN>
struct MatmulMfmaConfig {
  static_assert(BlockM > 0 && BlockN > 0 && BlockK > 0,
                "Tile sizes must be positive");
  static_assert(WaveTileM > 0 && WaveTileN > 0,
                "Wave tiles must be positive");
  static_assert(BlockM % WaveTileM == 0,
                "BlockM must be divisible by the per-wave M tile");
  static_assert(BlockN % WaveTileN == 0,
                "BlockN must be divisible by the per-wave N tile");
  static_assert(WaveTileM % 16 == 0 && WaveTileN % 16 == 0,
                "Wave tiles must align with MFMA dimensions");
  static_assert(BlockK % 16 == 0,
                "BlockK must be a multiple of 16 for MFMA");
  static_assert(BlockK % MATMUL_CHUNK_K == 0,
                "BlockK must be divisible by MATMUL_CHUNK_K");

  static constexpr int BlockRows = BlockM;
  static constexpr int BlockCols = BlockN;
  static constexpr int BlockDepth = BlockK;
  static constexpr int WarpTileM = WaveTileM;
  static constexpr int WarpTileN = WaveTileN;
  static constexpr int SubTilesM = WarpTileM / 16;
  static constexpr int SubTilesN = WarpTileN / 16;
  static constexpr int WavesM = BlockRows / WarpTileM;
  static constexpr int WavesN = BlockCols / WarpTileN;
  static constexpr int WavesPerBlock = WavesM * WavesN;
  static_assert(WavesPerBlock > 0 && WavesPerBlock <= 16,
                "Unsupported number of waves per workgroup");
  static constexpr int ThreadsPerBlock = WF_SIZE * WavesPerBlock;
  static_assert(ThreadsPerBlock <= 1024,
                "Workgroup size exceeds hardware limit");
  static constexpr int KQuads = BlockDepth / MATMUL_CHUNK_K;
  static constexpr int LdsStride = KQuads + 3; // small padding cuts bank conflicts
};

using MatmulQkvConfig = MatmulMfmaConfig<256, 256, 32, 128, 32>;
using MatmulAttConfig = MatmulMfmaConfig<128, 128, 32, 32, 64>;
using MatmulLogitsConfig = MatmulMfmaConfig<256, 256, 32, 128, 32>;

// Bias variants tuned for QKV and attention output projections
__global__ void matmul_bias_gemm_kernel_bf16_mfma_qkv(
    bf16_t* __restrict__ y,         // [B x d]
    const bf16_t* __restrict__ x,   // [B x n] (bf16)
    const bf16_t* __restrict__ w,   // [d x n] (bf16 packed)
    const float* __restrict__ bias, // [d] or nullptr
    int n, int d, int B,
    const int* __restrict__ pos);

__global__ void matmul_bias_gemm_kernel_bf16_mfma_att(
    bf16_t* __restrict__ y,         // [B x d]
    const bf16_t* __restrict__ x,   // [B x n] (bf16)
    const bf16_t* __restrict__ w,   // [d x n] (bf16 packed)
    const float* __restrict__ bias, // [d] or nullptr
    int n, int d, int B,
    const int* __restrict__ pos);

// No-bias variant: Y = X @ W^T
__global__ void matmul_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const bf16_t* __restrict__ x,   // [B x n] (bf16)
    const bf16_t* __restrict__ w,   // [d x n] (bf16 packed)
    int n, int d, int B,
    const int* __restrict__ pos);

__global__ void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [B, d]
    const bf16_t* __restrict__ x,   // [B, n] (bf16)
    const float* __restrict__ w,    // [d, n] (row-major theo n)
    const float* __restrict__ bias, // [d] (có thể null)
    int n, int d, int batch_size, const int *pos);

__global__ void
mlp1_fused_gemm_kernel(bf16_t *__restrict__ gate_up_topk,     // [K, B, IM]
                       const bf16_t *__restrict__ x,          // [B, H] (bf16)
                       const bf16_t *__restrict__ w_mlp1_all, // [L, E, 2*IM, H]
                       size_t stride_w_mlp1,
                       const bf16_t *__restrict__ b_mlp1_all, // [L, E, 2*IM]
                       const uint16_t *__restrict__ assignment_batches,
                       const uint8_t *__restrict__ assignment_slots,
                       const int *__restrict__ expert_offsets, int l_layer,
                       int E, int H, int IM, float swiglu_limit, int batch_size,
                       const int *pos);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float *__restrict__ e_agg,             // [B, H]
    const bf16_t *__restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t *__restrict__ w_mlp2_all,  // [L, E, H, IM]
    size_t stride_w_mlp2,
    const bf16_t *__restrict__ b_mlp2_all, // [L, E, H]
    const uint16_t *__restrict__ assignment_batches,
    const uint8_t *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, const float *__restrict__ topk_v,
    int l_layer, int E, int IM, int H, int batch_size, const int *pos);
#endif // GETP_MATMUL_H
