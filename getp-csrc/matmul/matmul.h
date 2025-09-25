#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"


__global__ void matmul_bias_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
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
    int* __restrict__ assignment_batches,
    int* __restrict__ assignment_slots,
    int batch_size,
    int experts_per_token,
    int E);

__global__ void
mlp1_fused_gemm_kernel(float *__restrict__ gate_up_topk,      // [K, B, IM]
                       const bf16_t *__restrict__ x,          // [B, H] (bf16)
                       const bf16_t *__restrict__ w_mlp1_all, // [L, E, 2*IM, H]
                       size_t stride_w_mlp1,
                       const float *__restrict__ b_mlp1_all, // [L, E, 2*IM]
                       const int *__restrict__ assignment_batches,
                       const int *__restrict__ assignment_slots,
                       const int *__restrict__ expert_offsets, int l_layer,
                       int E, int H, int IM, float swiglu_limit, int batch_size,
                       const int *pos);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float *__restrict__ e_agg,              // [B, H]
    const float *__restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t *__restrict__ w_mlp2_all,  // [L, E, H, IM]
    size_t stride_w_mlp2,
    const float *__restrict__ b_mlp2_all, // [L, E, H]
    const int *__restrict__ assignment_batches,
    const int *__restrict__ assignment_slots,
    const int *__restrict__ expert_offsets, const float *__restrict__ topk_v,
    int l_layer, int E, int IM, int H, int batch_size, const int *pos);

// Accumulate compact partials [cnt, H] into dest [B, H] using batch_ids[cnt]
__global__ void accumulate_partials_kernel(
    float* __restrict__ dest,           // [B, H]
    const float* __restrict__ src,      // [cnt, H]
    const int* __restrict__ batch_ids,  // [cnt]
    int H,
    int cnt);
#endif // GETP_MATMUL_H
