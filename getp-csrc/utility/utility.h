#ifndef GETP_UTILITY_H
#define GETP_UTILITY_H

#include "../common/defines.h"

// GPU Memory Debugging
static inline void debug_print_gpu_memory(const char *tag, int device_id = 0);

// Grid Dimension Utilities
inline dim3 get_gemv_grid_dim(int d);
inline dim3 get_gemm_grid_dim(int d, int batch_size);
inline dim3 get_gemm_grid_dim(int d, int batch_size, int batch_tile);

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Embedding and Data Movement Kernels
__global__ void copy_embedding_bf16_batch_kernel(float *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 const int *pos, int batch_size,
                                                 int hidden_dim);

// Normalization Kernels
__global__ void rmsnorm_batch_kernel(float *out, const float *x,
                                     const float *weight, const int *pos,
                                     int dim, int batch_size);

__global__ void compute_inv_rms_batch_kernel(float *inv_rms, const float *x,
                                             const int *pos, int dim, int batch_size);

// Utility Operations
__global__ void residual_add_batch_kernel(float *x, const float *residual,
                                          const int *pos, int dim, int batch_size);

// QKV Processing Kernels
__global__ void split_qkv_scatter_to_cache_batch_kernel(
    float *q, float *k_cache, float *v_cache, const float *qkv, int Hq, int Hk,
    int D, int loff, const int *pos, int batch_size, int kv_total_size);

// RoPE (Rotary Position Embedding) Kernels
__global__ void fused_inline_rope_qkv_batch_kernel(
    float *q, float *k_cache, const int *pos, float theta, int Hq, int Hk,
    int D, float rope_scaling_factor, int initial_context_length, int loff,
    int kv_total_size, int batch_size);

// Expert/MoE Utility Kernels
__global__ void fused_topk_softmax_batch_kernel(
    float *topk_v, int *topk_i, const float *scores, const int *pos, int E,
    int K, int batch_size);

// Data Type Conversion Utilities
void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                               int n_streams, size_t chunk_bytes);

#endif // GETP_UTILITY_H