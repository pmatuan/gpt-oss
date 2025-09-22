#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"


__global__ void matmul_bias_gemm_kernel_bf16_mfma(
    float* __restrict__ y,          // [B x d]
    const float* __restrict__ x,    // [B x n] (fp32)
    const bf16_t* __restrict__ w,   // [d x n] (bf16)
    const float* __restrict__ bias, // [d] or nullptr
    int n, int d, int B,
    const int* __restrict__ pos);

__global__ void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [B, d]
    const float* __restrict__ x,    // [B, n]
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

__global__ void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up_topk, // [K, B, IM]
    const float* __restrict__ x,      // [B, H]
    const bf16_t* __restrict__ w_mlp1_all, // [L, E, 2*IM, H]
    const float* __restrict__ b_mlp1_all,  // [L, E, 2*IM]
    const int* __restrict__ assignment_batches,
    const int* __restrict__ assignment_slots,
    const int* __restrict__ expert_offsets,
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size,
    const int *pos);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,              // [B, H]
    const float* __restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t* __restrict__ w_mlp2_all,  // [L, E, H, IM]
    const float* __restrict__ b_mlp2_all,   // [L, E, H]
    const int* __restrict__ assignment_batches,
    const int* __restrict__ assignment_slots,
    const int* __restrict__ expert_offsets,
    const float* __restrict__ topk_v,
    int l_layer, int E, int IM, int H,
    int batch_size, const int *pos);

__global__ void mlp1_fused_gemm_kernel_old(
    float* __restrict__ gate_up_topk, // [K, B, IM] (K = EXPERT_PER_TOKEN)
    const float* __restrict__ x,      // [B, H]
    const bf16_t* __restrict__ w_mlp1_all, // [L, E, 2*IM, H] (row-major in last dim)
    const float* __restrict__ b_mlp1_all,  // [L, E, 2*IM]
    const int* __restrict__ topk_i,   // [B, K]
    int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size,
    const int *pos);

__global__ void mlp2_bias_weighted_accum_gemm_kernel_old(
    float* __restrict__ e_agg,              // [B, H] (accumulator)
    const float* __restrict__ gate_up_topk, // [K, B, IM]
    const bf16_t* __restrict__ w_mlp2_all,  // [L, E, H, IM]
    const float* __restrict__ b_mlp2_all,   // [L, E, H]
    const int* __restrict__ topk_i,         // [B, K]
    const float* __restrict__ topk_v,       // [B, K]
    int l_layer, int E, int IM, int H,
    int batch_size, const int *pos);
#endif // GETP_MATMUL_H
