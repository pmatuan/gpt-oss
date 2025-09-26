#ifndef GETP_UTILITY_H
#define GETP_UTILITY_H

#include "../common/defines.h"

// GPU Memory Debugging
void debug_print_gpu_memory(const char *tag, int device_id = 0);

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Embedding and Data Movement Kernels
__global__ void copy_embedding_bf16_batch_kernel(bf16_t *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 int batch_size,
                                                 int hidden_dim);

// Normalization Kernels
__global__ void rmsnorm_batch_kernel(bf16_t *out, const bf16_t *x,
                                     const float *weight, int dim,
                                     const int *pos);

// Utility Operations
__global__ void residual_add_batch_kernel(bf16_t *x, const float *residual,
                                          int dim, int batch_size,
                                          const int *pos);

__global__ void residual_add_batch_kernel_bf16(bf16_t *x,
                                               const bf16_t *residual,
                                               int dim, int batch_size,
                                               const int *pos);

__global__ void fused_split_rope_scatter_qkv_batch_kernel(
    bf16_t* __restrict__ q_out,
    bf16_t* __restrict__ key_cache,
    bf16_t* __restrict__ value_cache,
    const bf16_t* __restrict__ qkv,    // [B, Hq*D + 2*Hk*D]
    const int* __restrict__ pos,       // [B]
    // model params
    int Hq, int Hk, int D,
    // RoPE params
    float theta, float rope_scaling_factor, int initial_context_length,
    // cache params
    int layer_idx,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity,
    uint32_t kv_batch_stride,
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

// Matmul Helper Functions
__device__ __forceinline__ s16x4 load_bf16x4(const uint16_t* src, int valid_elems);

// Expert Assignment Kernels
__global__ void count_expert_assignments_kernel(
    int* __restrict__ counts,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int batch_size,
    int experts_per_token,
    int E);

__global__ void build_expert_assignments_kernel(
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_counters,
    uint16_t* __restrict__ assignment_batches,
    uint8_t* __restrict__ assignment_slots,
    int batch_size,
    int experts_per_token,
    int E);

// Activation Functions
__device__ __forceinline__ float clamp_with_limit(float v, float limit);
__device__ __forceinline__ float swiglu_fused(float gate, float up, float limit);

#endif // GETP_UTILITY_H
