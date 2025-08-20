#include "hip_kernels.h"

// Each kernel below launches a single block with a single thread. The
// intention is to provide a reference implementation of the corresponding
// operation on the GPU. They are not optimised and primarily serve to
// demonstrate how HIP kernels can be wired into the existing inference
// pipeline.

__global__ void rmsnorm_kernel(float *o, const float *x, const float *w,
                               int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  ss = rsqrtf(ss / n + 1e-5f);
  for (int i = 0; i < n; ++i) {
    o[i] = x[i] * ss * w[i];
  }
}

void hip_rmsnorm(float *o, const float *x, const float *w, int n) {
  hipLaunchKernelGGL(rmsnorm_kernel, dim3(1), dim3(1), 0, 0, o, x, w, n);
}

__global__ void linear_kernel(float *out, const float *x, const float *w,
                              const float *b, int out_dim, int in_dim) {
  for (int i = 0; i < out_dim; ++i) {
    float val = b ? b[i] : 0.0f;
    for (int j = 0; j < in_dim; ++j) {
      val += w[i * in_dim + j] * x[j];
    }
    out[i] = val;
  }
}

void hip_linear(float *out, const float *x, const float *w, const float *b,
                int out_dim, int in_dim) {
  hipLaunchKernelGGL(linear_kernel, dim3(1), dim3(1), 0, 0, out, x, w, b,
                     out_dim, in_dim);
}

__global__ void attention_dense_kernel(float *out, const float *q,
                                       const float *k, const float *v,
                                       int seq_len, int head_dim) {
  for (int t = 0; t < seq_len; ++t) {
    // Compute attention weights for token t
    float scores[512]; // assumes head_dim <= 512 for simplicity
    for (int s = 0; s <= t; ++s) {
      float sc = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        sc += q[t * head_dim + i] * k[s * head_dim + i];
      }
      scores[s] = sc / sqrtf((float)head_dim);
    }
    // softmax over 0..t
    float max_val = scores[0];
    for (int i = 1; i <= t; ++i)
      max_val = max(max_val, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i <= t; ++i) {
      scores[i] = expf(scores[i] - max_val);
      sum += scores[i];
    }
    for (int i = 0; i <= t; ++i)
      scores[i] /= sum;
    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int s = 0; s <= t; ++s) {
        val += scores[s] * v[s * head_dim + d];
      }
      out[t * head_dim + d] = val;
    }
  }
}

void hip_attention_dense(float *out, const float *q, const float *k,
                         const float *v, int seq_len, int head_dim) {
  hipLaunchKernelGGL(attention_dense_kernel, dim3(1), dim3(1), 0, 0, out, q, k,
                     v, seq_len, head_dim);
}

__global__ void attention_banded_kernel(float *out, const float *q,
                                        const float *k, const float *v,
                                        int seq_len, int head_dim, int band) {
  for (int t = 0; t < seq_len; ++t) {
    int start = max(0, t - band);
    float scores[512];
    for (int s = start; s <= t; ++s) {
      float sc = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        sc += q[t * head_dim + i] * k[s * head_dim + i];
      }
      scores[s - start] = sc / sqrtf((float)head_dim);
    }
    int count = t - start + 1;
    float max_val = scores[0];
    for (int i = 1; i < count; ++i)
      max_val = max(max_val, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
      scores[i] = expf(scores[i] - max_val);
      sum += scores[i];
    }
    for (int i = 0; i < count; ++i)
      scores[i] /= sum;
    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int s = 0; s < count; ++s) {
        val += scores[s] * v[(start + s) * head_dim + d];
      }
      out[t * head_dim + d] = val;
    }
  }
}

void hip_attention_banded(float *out, const float *q, const float *k,
                          const float *v, int seq_len, int head_dim, int band) {
  hipLaunchKernelGGL(attention_banded_kernel, dim3(1), dim3(1), 0, 0, out, q, k,
                     v, seq_len, head_dim, band);
}

__global__ void moe_expert_kernel(float *out, const float *x, const float *w1,
                                  const float *b1, const float *w2,
                                  const float *b2, int hidden,
                                  int intermediate) {
  extern __shared__ float buf[];
  float *tmp = buf; // size intermediate
  for (int i = 0; i < intermediate; ++i) {
    float val = b1 ? b1[i] : 0.0f;
    for (int j = 0; j < hidden; ++j) {
      val += w1[i * hidden + j] * x[j];
    }
    // simple GELU approximation
    float c = val * val * val * 0.044715f + val;
    tmp[i] = 0.5f * val * (1.0f + tanhf(c));
  }
  for (int o = 0; o < hidden; ++o) {
    float val = b2 ? b2[o] : 0.0f;
    for (int i = 0; i < intermediate; ++i) {
      val += w2[o * intermediate + i] * tmp[i];
    }
    out[o] = val;
  }
}

void hip_moe_expert(float *out, const float *x, const float *w1,
                    const float *b1, const float *w2, const float *b2,
                    int hidden, int intermediate) {
  size_t shared = intermediate * sizeof(float);
  hipLaunchKernelGGL(moe_expert_kernel, dim3(1), dim3(1), shared, 0, out, x, w1,
                     b1, w2, b2, hidden, intermediate);
}

// -----------------------------------------------------------------------------
// Minimal GPU wrapper around the existing forward() implementation. The logits
// are passed through a simple identity kernel executed on the GPU to verify
// HIP integration while preserving baseline correctness.

struct Transformer;
extern float *forward(Transformer *transformer, int token, int pos);

__global__ void identity_kernel(float *dst, const float *src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

float *hip_forward(Transformer *transformer, int token, int pos,
                   int vocab_size) {
  float *logits = forward(transformer, token, pos);
  float *d_in = nullptr, *d_out = nullptr;
  size_t bytes = sizeof(float) * vocab_size;
  if (hipMalloc(&d_in, bytes) != hipSuccess ||
      hipMalloc(&d_out, bytes) != hipSuccess) {
    return logits;
  }
  hipMemcpy(d_in, logits, bytes, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(identity_kernel, dim3(1), dim3(1), 0, 0, d_out, d_in,
                     vocab_size);
  hipMemcpy(logits, d_out, bytes, hipMemcpyDeviceToHost);
  hipFree(d_in);
  hipFree(d_out);
  return logits;
}
