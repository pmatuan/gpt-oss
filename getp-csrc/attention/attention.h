#ifndef GETP_ATTENTION_H
#define GETP_ATTENTION_H

#include "../common/defines.h"

// Attention mechanism interface - designed for modularity
// This allows switching between different attention implementations

// Base attention kernel interface
__global__ void attention_batch_kernel(
    float *__restrict__ out_tb,  // [B, Hq*D]
    const float *__restrict__ q, // [B, Hq*D], FP32 (already rotary-applied)
    const float *__restrict__ k_cache,    // [B, L*S*KV], FP32
    const float *__restrict__ v_cache,    // [B, L*S*KV], FP32
    const float *__restrict__ attn_sinks, // [L*Hq]
    int layer_idx, const int *__restrict__ pos, int D, int Hq, int Hk, int S,
    const float *__restrict__ mask, int kv_stride, int batch_size);

// Future implementations can include:
// - Flash Attention: flash_attention_kernel(...)
// - Paged Attention: paged_attention_kernel(...)
// - Sparse Attention: sparse_attention_kernel(...)

#endif // GETP_ATTENTION_H