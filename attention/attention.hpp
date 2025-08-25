#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include <hip/hip_runtime.h>

// Fused attention kernel: scores + sink + softmax + values in one pass (self-attention style)
// Template TILE_T for tile size, as in getp_run.cpp
template<int TILE_T>
__global__ void attention_fused_kernel(
    float* __restrict__ tb,                 // [Hq, D] out
    const float* __restrict__ q,            // [Hq, D]
    const float* __restrict__ key_cache,    // [S, KV] for layer l
    const float* __restrict__ value_cache,  // [S, KV] for layer l
    const float* __restrict__ attn_sinks_layer, // [Hq]
    const float* __restrict__ mask,         // [S, S] or nullptr
    int Hq, int Hk, int D, int KV,          // KV = D * Hk
    int S, int pos                          // T_real = pos + 1
);

// Paged fused attention kernel: same math as attention_fused_kernel,
// but K/V rows are gathered via token->row indirection (paged KV cache).
// token2row maps logical token t in [0, pos] to a physical row index in caches.
template<int TILE_T>
__global__ void paged_attention_fused_kernel(
    float* __restrict__ tb,                      // [Hq, D] out
    const float* __restrict__ q,                 // [Hq, D]
    const float* __restrict__ key_cache,         // [Rows, KV]
    const float* __restrict__ value_cache,       // [Rows, KV]
    const int* __restrict__ token2row,           // [S] or at least pos+1
    const float* __restrict__ attn_sinks_layer,  // [Hq]
    const float* __restrict__ mask,              // [S, S] or nullptr
    int Hq, int Hk, int D, int KV,               // KV = D * Hk
    int S, int pos                               // T_real = pos + 1
);

#endif // ATTENTION_HPP
