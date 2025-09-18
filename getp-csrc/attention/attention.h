#ifndef GETP_ATTENTION_H
#define GETP_ATTENTION_H

#include "../common/defines.h"

__global__ void attention_kernel_bf16(
    float *__restrict__ out_tb,  // [Hq*D]
    const float *__restrict__ q, // [Hq*D], FP32 (already rotary-applied)
    const bf16_t *__restrict__ k_cache,    // [L*S*KV], BF16
    const bf16_t *__restrict__ v_cache,    // [L*S*KV], BF16
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride);

// Batched variant: grid=(Hq,1,B). Slices per b with strides:
// out_tb: b*(Hq*D), qkv: b*(Hq+2*Hk)*D, caches: b*(L*S*KV). pos indexed by b.
__global__ void attention_kernel_bf16_B(
    float *__restrict__ out_tb,
    const float *__restrict__ qkv,
    const bf16_t *__restrict__ k_cache,
    const bf16_t *__restrict__ v_cache,
    const float *__restrict__ attn_sinks,
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride);

#endif // GETP_ATTENTION_H