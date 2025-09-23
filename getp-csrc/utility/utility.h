#ifndef GETP_UTILITY_H
#define GETP_UTILITY_H

#include "../common/defines.h"

// GPU Memory Debugging
void debug_print_gpu_memory(const char *tag, int device_id = 0);

// Grid Dimension Utilities (removed - using direct dim3 initialization)
__device__ __forceinline__ short f32_to_bf16_bits_short(float f);
__device__ __forceinline__ s16x4 pack4_bf16_from_f32_guard(
    const float* base_f32, int k_off, int k_rem, bool row_valid);
__device__ __forceinline__ s16x4 pack4_bf16_from_bf16_guard(
    const bf16_t* base_bf16, int k_off, int k_rem, bool col_valid);
__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Embedding and Data Movement Kernels
__global__ void copy_embedding_bf16_batch_kernel(float *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 int batch_size,
                                                 int hidden_dim);

// Normalization Kernels
__global__ void rmsnorm_batch_kernel(float *out, const float *x,
                                     const float *weight, int dim,
                                     const int *pos);

// Utility Operations
__global__ void residual_add_batch_kernel(float *x, const float *residual,
                                          int dim, int batch_size,
                                          const int *pos);

__global__ void fused_split_rope_scatter_qkv_batch_kernel(
    float* __restrict__ q_out,
    bf16_t* __restrict__ key_cache,
    bf16_t* __restrict__ value_cache,
    const float* __restrict__ qkv,     // [B, Hq*D + 2*Hk*D]
    const int* __restrict__ pos,       // [B]
    // model params
    int Hq, int Hk, int D,
    // RoPE params
    float theta, float rope_scaling_factor, int initial_context_length,
    // cache params
    int layer_offset,   // = l * S * (Hk*D)
    int kv_total_size,  // = L * S * (Hk*D)
    int batch_size);

// Expert/MoE Utility Kernels
__global__ void fused_topk_softmax_batch_kernel(
    float *topk_v, int *topk_i, const float *scores, int E, int K,
    int batch_size, const int *pos);

__global__ void argmax_batch_kernel(const float *logits, int *out_indices,
                                    int vocab_size, int batch_size,
                                    const int *pos);

// Data Type Conversion Utilities
void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                              int n_streams, size_t chunk_bytes);

size_t matmul_packed_elems(int rows, int cols);
void pack_fp32_to_bf16_matmul(const float *src, int rows, int cols,
                              bf16_t *dst);

#endif // GETP_UTILITY_H