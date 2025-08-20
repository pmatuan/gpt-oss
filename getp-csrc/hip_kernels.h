#pragma once

#include <hip/hip_runtime.h>

// Simple HIP kernels used for verification and experimentation. These
// implementations are intentionally naive and prioritise correctness over
// performance. Each kernel operates on a single thread block and is suitable
// for small tensor sizes used in unit tests or as reference implementations.

void hip_rmsnorm(float *o, const float *x, const float *w, int n);
void hip_linear(float *out, const float *x, const float *w, const float *b,
                int out_dim, int in_dim);
void hip_attention_dense(float *out, const float *q, const float *k,
                         const float *v, int seq_len, int head_dim);
void hip_attention_banded(float *out, const float *q, const float *k,
                          const float *v, int seq_len, int head_dim, int band);
void hip_moe_expert(float *out, const float *x, const float *w1,
                    const float *b1, const float *w2, const float *b2,
                    int hidden, int intermediate);

struct Transformer;
float *hip_forward(Transformer *transformer, int token, int pos,
                   int vocab_size);

#endif // HIP_KERNELS_H
