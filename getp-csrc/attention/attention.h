#ifndef GETP_ATTENTION_H
#define GETP_ATTENTION_H

#include "../common/defines.h"

__global__ void attention_flashdecode_mqa_even(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const bf16_t *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, int sliding_window,
    uint32_t kv_batch_stride, int batch_size);

__global__ void attention_flashdecode_mqa_odd(
    bf16_t *__restrict__ out_tb, const bf16_t *__restrict__ q,
    const bf16_t *__restrict__ k_cache, const bf16_t *__restrict__ v_cache,
    const bf16_t *__restrict__ attn_sinks, int layer_idx,
    const int *__restrict__ pos, int D, int Hq, int Hk,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity, uint32_t kv_batch_stride,
    int batch_size);

#endif // GETP_ATTENTION_H
