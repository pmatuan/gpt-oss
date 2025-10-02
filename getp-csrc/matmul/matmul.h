#ifndef GETP_MATMUL_H
#define GETP_MATMUL_H

#include "../common/defines.h"

// Bias variants tuned for QKV and attention output projections
__global__ void
matmul_bias_qkv_kernel(bf16_t *__restrict__ y,         // [B x d]
                       const bf16_t *__restrict__ x,   // [B x n] (bf16)
                       const bf16_t *__restrict__ w,   // [d x n] (bf16 packed)
                       const bf16_t *__restrict__ bias, // [d] or nullptr
                       int n, int d, int B, const int *__restrict__ pos);

__global__ void
matmul_bias_att_kernel(bf16_t *__restrict__ y,         // [B x d]
                       const bf16_t *__restrict__ x,   // [B x n] (bf16)
                       const bf16_t *__restrict__ w,   // [d x n] (bf16 packed)
                       const bf16_t *__restrict__ bias, // [d] or nullptr
                       int n, int d, int B, const int *__restrict__ pos);

// No-bias variant: Y = X @ W^T
__global__ void
matmul_logits_kernel(float *__restrict__ y,        // [B x d]
                     const bf16_t *__restrict__ x, // [B x n] (bf16)
                     const bf16_t *__restrict__ w, // [d x n] (bf16 packed)
                     int n, int d, int B, const int *__restrict__ pos);

__global__ void
matmul_router_kernel(float *__restrict__ y,        // [B, d]
                     const bf16_t *__restrict__ x, // [B, n] (bf16)
                     const bf16_t *__restrict__ w,  // [d, n] (row-major theo n)
                     const bf16_t *__restrict__ bias, // [d] (có thể null)
                     int n, int d, int batch_size, const int *pos);

__global__ void
mlp1_kernel(bf16_t *__restrict__ gate_up_topk,     // [K, B, IM]
            const bf16_t *__restrict__ x,          // [B, H] (bf16)
            const bf16_t *__restrict__ w_mlp1_all, // [L, E, 2*IM, H]
            size_t stride_w_mlp1,
            const bf16_t *__restrict__ b_mlp1_all, // [L, E, 2*IM]
            const uint16_t *__restrict__ assignment_batches,
            const uint8_t *__restrict__ assignment_slots,
            const int *__restrict__ expert_offsets, int l_layer, int E, int H,
            int IM, float swiglu_limit, int batch_size, const int *pos);

__global__ void
mlp1_120b_kernel(bf16_t *__restrict__ gate_up_topk,     // [K, B, IM]
                 const bf16_t *__restrict__ x,          // [B, H] (bf16)
                 const bf16_t *__restrict__ w_mlp1_all, // [L, E, 2*IM, H]
                 size_t stride_w_mlp1,
                 const bf16_t *__restrict__ b_mlp1_all, // [L, E, 2*IM]
                 const uint16_t *__restrict__ assignment_batches,
                 const uint8_t *__restrict__ assignment_slots,
                 const int *__restrict__ expert_offsets, int l_layer, int E,
                 int H, int IM, float swiglu_limit, int batch_size,
                 const int *pos);

__global__ void
mlp2_kernel(float *__restrict__ e_agg,               // [B, H]
            const bf16_t *__restrict__ gate_up_topk, // [K, B, IM]
            const bf16_t *__restrict__ w_mlp2_all,   // [L, E, H, IM]
            size_t stride_w_mlp2,
            const bf16_t *__restrict__ b_mlp2_all, // [L, E, H]
            const uint16_t *__restrict__ assignment_batches,
            const uint8_t *__restrict__ assignment_slots,
            const int *__restrict__ expert_offsets,
            const float *__restrict__ topk_v, int l_layer, int E, int IM, int H,
            int batch_size, const int *pos);

__global__ void
mlp2_120b_kernel(float *__restrict__ e_agg,               // [B, H]
                 const bf16_t *__restrict__ gate_up_topk, // [K, B, IM]
                 const bf16_t *__restrict__ w_mlp2_all,   // [L, E, H, IM]
                 size_t stride_w_mlp2,
                 const bf16_t *__restrict__ b_mlp2_all, // [L, E, H]
                 const uint16_t *__restrict__ assignment_batches,
                 const uint8_t *__restrict__ assignment_slots,
                 const int *__restrict__ expert_offsets,
                 const float *__restrict__ topk_v, int l_layer, int E, int IM,
                 int H, int batch_size, const int *pos);
#endif // GETP_MATMUL_H
