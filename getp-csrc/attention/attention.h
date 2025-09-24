#ifndef GETP_ATTENTION_H
#define GETP_ATTENTION_H

#include "../common/defines.h"

__global__ void attention_batch_kernel(
    bf16_t *__restrict__ out_tb,  // [B, Hq*D] (bf16 storage)
    const bf16_t *__restrict__ q, // [B, Hq*D], BF16 (already rotary-applied)
    const bf16_t *__restrict__ k_cache,    // [B, L*S*KV], BF16
    const bf16_t *__restrict__ v_cache,    // [B, L*S*KV], BF16
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size);

#endif // GETP_ATTENTION_H
