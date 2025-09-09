#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"


__global__ void matmul_bias_gemm_kernel_bf16(
    float *out, const float *x, const bf16_t *w, const float *b,
    const int *pos, int H, int D, int batch_size);

__global__ void matmul_bias_gemm_kernel_float(
    float *out, const float *x, const float *w, const float *b,
    const int *pos, int H, int D, int batch_size);

__global__ void mlp1_fused_gemm_kernel(
    float *gate_up_topk, const float *x, const bf16_t *w, const float *b,
    const int *topk_i, const int *pos, int layer, int E, int H, int IM,
    float clip, int B, int K);

__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float *e_agg, const float *gate_up, const bf16_t *w, const float *b,
    const int *topk_i, const float *topk_v, const int *pos, int layer,
    int E, int IM, int H, int B, int K);

#endif // GETP_MATMUL_H