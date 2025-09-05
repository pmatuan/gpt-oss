#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"

// Matrix multiplication utility functions and kernels
// Designed for modularity - can be extended with different implementations

// Core GEMM operations
template<int CB>
__device__ __forceinline__ void gemm_row_tile_fp32_multiB(
    const float* __restrict__ w_row,         // [k_size] (slice of row)
    float* __restrict__ lds_x[CB],           // CB pointers -> [k_size] each
    int k_size, int lane, float acc[CB]);

template<int CB>
__device__ __forceinline__ void gemm_row_tile_bf16_multiB(
    const bf16_t* __restrict__ w_row,        // [k_size] (slice of row)
    float* __restrict__ lds_x[CB],           // CB pointers -> [k_size] each  
    int k_size, int lane, float acc[CB]);

// Utility functions
__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Grid dimension helpers
inline dim3 get_gemv_grid_dim(int d);
inline dim3 get_gemm_grid_dim(int d, int batch_size, int batch_tile = BATCH_TILE_DEFAULT);

// Kernel declarations
template<typename T>
__global__ void fused_rmsnorm_matmul_bias_gemm_kernel(
    float *out, const float *x, const T *w, const float *b,
    const float *rms_w, const int *pos, int H, int D, int batch_size);

template<typename T>
__global__ void fused_matmul_bias_residual_gemm_kernel(
    float *x, const float *tb, const T *w, const float *b,
    const int *pos, int O_N, int H, int batch_size);

template<typename T>
__global__ void matmul_bias_gemm_kernel(
    float *out, const float *x, const T *w, const float *b,
    const int *pos, int H, int D, int batch_size);

template<typename T>
__global__ void fused_rmsnorm_matmul_gemm_kernel(
    float *out, const float *x, const T *w, const float *rms_w,
    const int *pos, const float *inv_rms, int H, int V, int batch_size);

// MLP kernels
__global__ void mlp1_fused_gemm_kernel(
    float *gate_up_topk, const float *x, const bf16_t *w, const float *b,
    const int *topk_i, const int *pos, int layer, int E, int H, int IM,
    float clip, int B, int K);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float *e_agg, const float *gate_up, const bf16_t *w, const float *b,
    const int *topk_i, const float *topk_v, const int *pos, int layer,
    int E, int IM, int H, int B, int K);

// Batch processing kernels
__global__ void copy_embedding_bf16_batch_kernel(float *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 const int *pos, int batch_size,
                                                 int hidden_dim);

__global__ void rmsnorm_batch_kernel(float *out, const float *x,
                                     const float *weight, const int *pos,
                                     int dim, int batch_size);

__global__ void residual_add_batch_kernel(float *x, const float *residual,
                                          const int *pos, int dim, int batch_size);

__global__ void compute_inv_rms_batch_kernel(float *inv_rms, const float *x,
                                             const int *pos, int dim, int batch_size);

// Utility function for FP32 to BF16 conversion
void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                               int n_streams, size_t chunk_bytes);

#endif // GETP_MATMUL_H