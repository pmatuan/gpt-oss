#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"


__global__ void matmul_bias_gemm_kernel_bf16(
    float *out, const float *x, const bf16_t *w, const float *b,
    int H, int D, int batch_size, const int *pos);

__global__ void matmul_bias_gemm_kernel_float(
    float *out, const float *x, const float *w, const float *b,
    int H, int D, int batch_size, const int *pos);

__global__ void mlp1_matmul_bias_kernel(
    float *out, const float *x, const bf16_t *w, const float *b,
    const int *topk_i, int layer, int E, int H, int IM,
    int B, const int *pos);

__global__ void mlp1_split_swiglu_kernel(
    float *gate_up_out, const float *gate_up_raw, float clip,
    int IM, int B, int K, const int *pos);

__global__ void mlp2_matmul_bias_kernel(
    float *out, const float *gate_up, const bf16_t *w, const float *b,
    const int *topk_i, int layer, int E, int IM, int H, int B,
    const int *pos);

__global__ void mlp2_weighted_accum_kernel(
    float *e_agg, const float *mlp2_out, const float *topk_v,
    int H, int B, int K, const int *pos);

__global__ void moe_count_assignments_kernel(int *counts,
                                             const int *topk_i,
                                             int total_assignments,
                                             int n_experts);

__global__ void moe_scatter_assignments_kernel(int *assignments,
                                               int *counters,
                                               const int *offsets,
                                               const int *topk_i,
                                               int total_assignments,
                                               int n_experts);

__global__ void moe_gather_tokens_kernel(float *dst, const float *src,
                                         const int *assignments, int start,
                                         int count, int H);

__global__ void moe_swiglu_activation_kernel(float *dst, const float *src,
                                             float clip, int count, int IM);

__global__ void moe_weighted_accum_kernel(float *e_agg, const float *mlp2_out,
                                          const float *topk_v,
                                          const int *assignments, int start,
                                          int count, int H);

#endif // GETP_MATMUL_H
