#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"


__global__ void matmul_bias_gemm_kernel_bf16(
    float *out, const float *x, const bf16_t *w, const float *b,
    int H, int D, const int *pos);

__global__ void matmul_bias_gemm_kernel_float(
    float *out, const float *x, const float *w, const float *b,
    int H, int D, const int *pos);

// Batched (grid.z = B) variants. Layout assumptions:
// - out has leading dimension ld_out == D (output size)
// - x   has leading dimension ld_x   == H (input size)
// Each batch b uses slice offsets b*ld_*.
__global__ void matmul_bias_gemm_kernel_bf16_B(
    float *out, const float *x, const bf16_t *w, const float *b,
    int H, int D, const int *pos);

__global__ void matmul_bias_gemm_kernel_float_B(
    float *out, const float *x, const float *w, const float *b,
    int H, int D, const int *pos);

__global__ void mlp1_fused_gemm_kernel(
    float *gate_up_topk, const float *x, const bf16_t *w, const float *b,
    const int *topk_i, int layer, int E, int H, int IM,
    float clip, const int *pos);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float *x, const float *gate_up, const bf16_t *w, const float *b,
    const int *topk_i, const float *topk_v, int layer,
    int E, int IM, int H, const int *pos);

// Fused output GEMM + argmax over vocab; writes only next_tokens per batch sample
__global__ void out_gemm_argmax_kernel(
    const float *t, const bf16_t *w_out, int V, int H,
    const int *pos, int *next_tokens);

#endif // GETP_MATMUL_H