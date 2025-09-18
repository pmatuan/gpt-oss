#ifndef GETP_UTILITY_H
#define GETP_UTILITY_H

#include "../common/defines.h"

// GPU Memory Debugging
static inline void debug_print_gpu_memory(const char *tag, int device_id = 0);

// Grid Dimension Utilities (removed - using direct dim3 initialization)

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Embedding and Data Movement Kernels
__global__ void copy_embedding_bf16_kernel(float *dst, const bf16_t *src,
                                           const int *tokens,
                                           int hidden_dim);

// Normalization Kernels
__global__ void rmsnorm_kernel(float *out, const float *x,
                               const float *weight, int dim,
                               const int *pos);

__global__ void compute_inv_rms_kernel(float *inv_rms, const float *x,
                                       int dim);

// Utility Operations
__global__ void residual_add_kernel(float *x, const float *residual,
                                    int dim, const int *pos);

// QKV Processing Kernels
__global__ void split_qkv_scatter_to_cache_kernel(
    float *q, bf16_t *k_cache, bf16_t *v_cache, const float *qkv, int Hq, int Hk,
    int D, int loff, const int *pos, int kv_total_size);

// RoPE (Rotary Position Embedding) Kernels
__global__ void fused_inline_rope_qkv_kernel(
    float *q, bf16_t *k_cache, const int *pos, float theta, int Hq, int Hk,
    int D, float rope_scaling_factor, int initial_context_length, int loff,
    int kv_total_size);

// Expert/MoE Utility Kernels
__global__ void fused_topk_softmax_kernel(
    float *topk_v, int *topk_i, const float *scores, int E, int K,
    const int *pos);

// Simple on-GPU sampling: greedy argmax per batch sample
// logits: [B, V], pos: [B] (inactive if pos[b] < 0), outputs next_tokens[B]
__global__ void argmax_logits_kernel(const float *logits, int V,
                                     const int *pos, int *next_tokens);

// Data Type Conversion Utilities
void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                               int n_streams, size_t chunk_bytes);

// MoE bucketing (group by expert)
__global__ void moe_count_assignments_kernel(const int *topk_i, const int *pos,
                                             int B, int K, int E,
                                             int *counts);

__global__ void moe_fill_indices_kernel(const int *topk_i, const int *pos,
                                        int B, int K, int E, int *offsets,
                                        const int *indptr, int *assign_b,
                                        int *assign_k);

__global__ void moe_gather_gate_up_kernel(const float *gate_up, int B, int K,
                                          int IM, const int *assign_b,
                                          const int *assign_k, int start,
                                          int N, float *GU_out);

__global__ void moe_scatter_mlp2_accum_kernel(const float *Z, int N, int H,
                                              const int *assign_b,
                                              const int *assign_k, int start,
                                              const float *topk_v, int K,
                                              float *x, const int *pos);

#endif // GETP_UTILITY_H