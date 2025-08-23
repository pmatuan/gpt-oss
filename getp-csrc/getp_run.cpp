#include "../profiler.h"
#include "getp_eval.cpp"
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef GETP_RUN
#define GETP_RUN

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = call;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static inline void debug_print_gpu_memory(const char *tag) {
  size_t free_b = 0, total_b = 0;
  hipError_t err = hipMemGetInfo(&free_b, &total_b);
  if (err != hipSuccess) {
    fprintf(stderr, "HIP hipMemGetInfo failed: %s\n", hipGetErrorString(err));
    return;
  }
  double free_gib = (double)free_b / (1024.0 * 1024.0 * 1024.0);
  double total_gib = (double)total_b / (1024.0 * 1024.0 * 1024.0);
  double used_gib = total_gib - free_gib;
  printf("[HIP] %s: HBM free %.2f GiB / total %.2f GiB (used %.2f GiB)\n", tag,
         free_gib, total_gib, used_gib);
  fflush(stdout);
}

typedef uint16_t bf16_t;

static inline bf16_t f32_to_bf16(float f) {
  union {
    float f;
    uint32_t u;
  } v = {f};
  uint32_t x = v.u;
  uint32_t lsb = (x >> 16) & 1;
  uint32_t rounding_bias = 0x00007FFF + lsb;
  x += rounding_bias;
  bf16_t h = (bf16_t)(x >> 16);
  return h;
}

__device__ __forceinline__ float bf16_to_f32(bf16_t h) {
  uint32_t x = ((uint32_t)h) << 16;
  float f = __uint_as_float(x);
  return f;
}

// Global GPU Buffers
// Activations / temporaries (FP32)
static float *d_x, *d_t, *d_tb, *d_tb2;
static float *d_router_score, *d_topk_v, *d_mlp1_out;
static int *d_topk_i;
static float *d_gate, *d_up, *d_gate_up, *d_e_agg;
static float *d_qkv, *d_q, *d_k, *d_v;
static float *d_key_cache, *d_value_cache;
static float *d_att, *d_logits, *d_mask;
static float *d_cos_vals, *d_sin_vals;

// Small weights (keep FP32)
static float *d_rms_attn_w, *d_rms_ffn_w;
static float *d_b_qkv, *d_b_o, *d_attn_sinks;
static float *d_w_router, *d_b_router;
static float *d_rms_out_w;

// Expert biases on device (FP32)
static float *g_b_mlp1 = nullptr;
static float *g_b_mlp2 = nullptr;

// Large weights (store BF16)
static bf16_t *d_token_embedding_table_bf16;
static bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
static bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
static bf16_t *d_out_bf16;

// Config pointer
static Config *h_config = nullptr;

// Kernels (FP32 activations)
__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight,
                               int size) {
  // Block-wide reduction for sum of squares
  extern __shared__ double s_rms[];
  double sum = 0.0;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = x[i];
    sum += (double)v * (double)v;
  }
  s_rms[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_rms[threadIdx.x] += s_rms[threadIdx.x + s];
    __syncthreads();
  }
  double inv = rsqrt(s_rms[0] / (double)size + 1e-5);

  // Apply scale
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    o[i] = weight[i] * (float)(inv * (double)x[i]);
  }
}

__global__ void softmax_kernel(float *x, int size) {
  extern __shared__ float s_soft[]; // reuse: reduction buffer
  float maxv = -INFINITY;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    maxv = fmaxf(maxv, x[i]);
  }
  s_soft[threadIdx.x] = maxv;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_soft[threadIdx.x] = fmaxf(s_soft[threadIdx.x], s_soft[threadIdx.x + s]);
    __syncthreads();
  }
  maxv = s_soft[0];

  double sum = 0.0;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float v = expf(x[i] - maxv);
    x[i] = v;
    sum += v;
  }
  // reduce sum
  s_soft[threadIdx.x] = (float)sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_soft[threadIdx.x] += s_soft[threadIdx.x + s];
    __syncthreads();
  }
  float inv = 1.0f / s_soft[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x)
    x[i] *= inv;
}

__global__ void add_bias_kernel(float *y, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    y[i] += b[i];
}

__global__ void copy_embedding_bf16_row_kernel(float *dst, const bf16_t *src,
                                               int token, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    dst[i] = bf16_to_f32(src[(size_t)token * hidden_dim + i]);
  }
}

__global__ void split_qkv_kernel(float *q, float *k, float *v, const float *qkv,
                                 int n_attn_heads, int n_kv_heads,
                                 int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int q_size = n_attn_heads * head_dim;
  int kv_size = n_kv_heads * head_dim;
  int total = q_size + 2 * kv_size;
  if (idx >= total)
    return;
  if (idx < q_size)
    q[idx] = qkv[idx];
  else if (idx < q_size + kv_size)
    k[idx - q_size] = qkv[idx];
  else
    v[idx - q_size - kv_size] = qkv[idx];
}

__global__ void apply_rotary_emb_kernel(float *x, const float *cosv,
                                        const float *sinv, int n_heads,
                                        int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim >> 1;
  if (h < n_heads && i < half) {
    float x1 = x[h * head_dim + i];
    float x2 = x[h * head_dim + half + i];
    float c = cosv[i], s = sinv[i];
    x[h * head_dim + i] = x1 * c - x2 * s;
    x[h * head_dim + half + i] = x2 * c + x1 * s;
  }
}

__global__ void compute_cos_sin_kernel(float *cosv, float *sinv, int pos,
                                       float rope_theta, int head_dim,
                                       float scaling_factor,
                                       float initial_context_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int d_half = head_dim >> 1;
  if (i >= d_half)
    return;
  float freq = powf(rope_theta, (float)(2 * i) / (float)head_dim);
  float inv_freq;
  float concentration = 1.0f;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = d_half *
                logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                logf(rope_theta);
    float high = d_half *
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(rope_theta);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    ramp = fmaxf(0.0f, fminf(1.0f, ramp));
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }
  float val = pos * inv_freq;
  cosv[i] = cosf(val) * concentration;
  sinv[i] = sinf(val) * concentration;
}

// Attention score for head h
__global__ void attention_scores_kernel(float *att, const float *q,
                                        const float *key_cache, int h, int pos,
                                        int head_dim, int kv_dim,
                                        int n_attn_heads, int n_kv_heads,
                                        int seq_len, const float *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t > pos)
    return;
  int kv_mul = n_attn_heads / n_kv_heads;
  const float *k = key_cache + t * kv_dim + (h / kv_mul) * head_dim;
  float score = 0.0f;
  for (int i = 0; i < head_dim; ++i)
    score += q[h * head_dim + i] * k[i];
  score *= rsqrtf((float)head_dim);
  if (mask)
    score += mask[pos * seq_len + t];
  att[h * seq_len + t] = score;
}

__global__ void attention_values_kernel(float *tb, const float *att,
                                        const float *value_cache, int h,
                                        int pos, int head_dim, int kv_dim,
                                        int n_attn_heads, int n_kv_heads,
                                        int seq_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= head_dim)
    return;
  float res = 0.0f;
  int kv_mul = n_attn_heads / n_kv_heads;
  for (int t = 0; t <= pos; ++t) {
    const float *v = value_cache + t * kv_dim + (h / kv_mul) * head_dim;
    res += att[h * seq_len + t] * v[i];
  }
  tb[h * head_dim + i] = res;
}

__global__ void residual_add_kernel(float *x, const float *residual, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    x[i] += residual[i];
}

__global__ void topk_kernel_1token(float *topk_values, int *topk_indices,
                                   const float *router_score, int num_experts,
                                   int experts_per_token) {
  for (int k = 0; k < experts_per_token; ++k) {
    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < num_experts; ++i) {
      bool used = false;
      for (int j = 0; j < k; ++j)
        if (topk_indices[j] == i) {
          used = true;
          break;
        }
      if (!used && router_score[i] > max_val) {
        max_val = router_score[i];
        max_idx = i;
      }
    }
    topk_values[k] = max_val;
    topk_indices[k] = max_idx;
  }
}

__global__ void split_gate_up_kernel(float *gate, float *up,
                                     const float *mlp1_out,
                                     int intermediate_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < intermediate_dim) {
    gate[i] = mlp1_out[2 * i];
    up[i] = mlp1_out[2 * i + 1];
  }
}

__global__ void swiglu_kernel(float *gate_up, const float *gate,
                              const float *up, int intermediate_dim,
                              float swiglu_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < intermediate_dim) {
    float g = gate[i];
    float u = up[i];
    g = fminf(fmaxf(g, -swiglu_limit), swiglu_limit);
    u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
    const float alpha = 1.702f; // SiLU approx
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);
    gate_up[i] = g;
  }
}

__global__ void weighted_sum_kernel(float *e_agg, const float *expert_out,
                                    float weight, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim)
    e_agg[i] += expert_out[i] * weight;
}

__global__ void memset_zero_kernel(float *ptr, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    ptr[i] = 0.0f;
}

// Matmul: W(d,n)[bf16] @ x(n)[f32] -> y(d)[f32]; FP32 accumulate
__global__ void matmul_bf16_wxf32_yf32(float *y, const float *x,
                                       const bf16_t *w, int n, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x; // 0..d-1
  if (row >= d)
    return;
  const bf16_t *wrow = w + (size_t)row * n;
  float acc = 0.0f;
  for (int j = 0; j < n; ++j) {
    acc += bf16_to_f32(wrow[j]) * x[j];
  }
  y[row] = acc;
}

// Minimal FP32 matmul: W(d,n)[f32] @ x(n)[f32] -> y(d)[f32]
__global__ void matmul_f32_wxf32_yf32(float *y, const float *x, const float *w,
                                      int n, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= d)
    return;
  const float *wr = w + (size_t)row * n;
  float acc = 0.0f;
  for (int j = 0; j < n; ++j)
    acc += wr[j] * x[j];
  y[row] = acc;
}

static void copy_fp32_to_bf16_device(const float *h_src, size_t count,
                                     bf16_t *d_dst) {
  const size_t CHUNK_ELEMS =
      (size_t)8 * 1024 * 1024; // 8M elems (~16MB) per chunk
  bf16_t *h_chunk = nullptr;
  HIP_CHECK(hipHostMalloc((void **)&h_chunk, CHUNK_ELEMS * sizeof(bf16_t)));
  size_t done = 0;
  while (done < count) {
    size_t todo = (count - done > CHUNK_ELEMS) ? CHUNK_ELEMS : (count - done);
    // convert
    for (size_t i = 0; i < todo; ++i)
      h_chunk[i] = f32_to_bf16(h_src[done + i]);
    // copy
    HIP_CHECK(hipMemcpy(d_dst + done, h_chunk, todo * sizeof(bf16_t),
                        hipMemcpyHostToDevice));
    done += todo;
  }
  HIP_CHECK(hipHostFree(h_chunk));
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  PROFILE_FUNCTION();
  h_config = &transformer->config;
  HIP_CHECK(hipSetDevice(0));

  const int H = h_config->hidden_dim;
  const int V = h_config->vocab_size;
  const int L = h_config->n_layers;
  const int E = h_config->n_experts;
  const int D = h_config->head_dim;
  const int Hq = h_config->n_attn_heads;
  const int Hk = h_config->n_kv_heads;
  const int KV = D * Hk;
  const int S = h_config->seq_len;
  const int IM = h_config->intermediate_dim;

  debug_print_gpu_memory("before allocations");

  // ---------------- Activations ----------------
  HIP_CHECK(hipMalloc(&d_x, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_t, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb, D * Hq * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_tb2, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_router_score, E * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_v, h_config->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_i, h_config->experts_per_token * sizeof(int)));

  HIP_CHECK(hipMalloc(&d_mlp1_out, 2 * IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_up, IM * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_e_agg, H * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv, (D * (Hq + 2 * Hk)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, Hq * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, Hk * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, Hk * D * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_key_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_value_cache, L * S * KV * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_att, Hq * S * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, V * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_cos_vals, (D / 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sin_vals, (D / 2) * sizeof(float)));

  if (h_config->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&d_mask, S * S * sizeof(float)));
    float *h_mask = (float *)malloc(S * S * sizeof(float));
    for (int i = 0; i < S; ++i) {
      for (int j = 0; j < S; ++j) {
        h_mask[i * S + j] =
            (i - j >= h_config->sliding_window) ? -INFINITY : 0.0f;
      }
    }
    HIP_CHECK(hipMemcpy(d_mask, h_mask, S * S * sizeof(float),
                        hipMemcpyHostToDevice));
    free(h_mask);
  } else {
    d_mask = nullptr;
  }

  debug_print_gpu_memory("after activations");

  // ---------------- Weights ----------------
  TransformerWeights *w = &transformer->weights;

  // Small FP32 weights
  HIP_CHECK(hipMalloc(&d_rms_attn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_attn_w, w->rms_attn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_ffn_w, L * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_ffn_w, w->rms_ffn_w, L * H * sizeof(float),
                      hipMemcpyHostToDevice));

  const int QKV_D = D * (Hq + 2 * Hk);
  HIP_CHECK(hipMalloc(&d_b_qkv, L * QKV_D * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_qkv, w->b_qkv, L * QKV_D * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_o, L * H * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_b_o, w->b_o, L * H * sizeof(float), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_attn_sinks, L * Hq * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_attn_sinks, w->attn_sinks, L * Hq * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_w_router, L * H * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w_router, w->w_router, L * H * E * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_b_router, L * E * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_b_router, w->b_router, L * E * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&d_rms_out_w, H * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_rms_out_w, w->rms_out_w, H * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after small FP32 weights");

  // Expert biases FP32
  HIP_CHECK(hipMalloc(&g_b_mlp1, (size_t)L * E * (2 * IM) * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp1, w->b_mlp1,
                      (size_t)L * E * (2 * IM) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&g_b_mlp2, (size_t)L * E * H * sizeof(float)));
  HIP_CHECK(hipMemcpy(g_b_mlp2, w->b_mlp2, (size_t)L * E * H * sizeof(float),
                      hipMemcpyHostToDevice));

  debug_print_gpu_memory("after expert biases");

  // Large BF16 weights
  HIP_CHECK(
      hipMalloc(&d_token_embedding_table_bf16, (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->token_embedding_table, (size_t)V * H,
                           d_token_embedding_table_bf16);

  HIP_CHECK(hipMalloc(&d_w_qkv_bf16, (size_t)L * QKV_D * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_qkv, (size_t)L * QKV_D * H, d_w_qkv_bf16);

  const int O_N = D * Hq;
  HIP_CHECK(hipMalloc(&d_w_o_bf16, (size_t)L * H * O_N * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_o, (size_t)L * H * O_N, d_w_o_bf16);

  HIP_CHECK(
      hipMalloc(&d_w_mlp1_bf16, (size_t)L * E * (2 * IM) * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp1, (size_t)L * E * (2 * IM) * H,
                           d_w_mlp1_bf16);

  HIP_CHECK(hipMalloc(&d_w_mlp2_bf16, (size_t)L * E * H * IM * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->w_mlp2, (size_t)L * E * H * IM, d_w_mlp2_bf16);

  HIP_CHECK(hipMalloc(&d_out_bf16, (size_t)V * H * sizeof(bf16_t)));
  copy_fp32_to_bf16_device(w->out, (size_t)V * H, d_out_bf16);

  debug_print_gpu_memory("after large BF16 weights (model loaded)");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_t));
  HIP_CHECK(hipFree(d_tb));
  HIP_CHECK(hipFree(d_tb2));
  HIP_CHECK(hipFree(d_router_score));
  HIP_CHECK(hipFree(d_topk_v));
  HIP_CHECK(hipFree(d_topk_i));
  HIP_CHECK(hipFree(d_mlp1_out));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_gate_up));
  HIP_CHECK(hipFree(d_e_agg));
  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_key_cache));
  HIP_CHECK(hipFree(d_value_cache));
  HIP_CHECK(hipFree(d_att));
  HIP_CHECK(hipFree(d_logits));
  HIP_CHECK(hipFree(d_cos_vals));
  HIP_CHECK(hipFree(d_sin_vals));
  if (d_mask)
    HIP_CHECK(hipFree(d_mask));

  HIP_CHECK(hipFree(d_rms_attn_w));
  HIP_CHECK(hipFree(d_rms_ffn_w));
  HIP_CHECK(hipFree(d_b_qkv));
  HIP_CHECK(hipFree(d_b_o));
  HIP_CHECK(hipFree(d_attn_sinks));
  HIP_CHECK(hipFree(d_w_router));
  HIP_CHECK(hipFree(d_b_router));
  HIP_CHECK(hipFree(d_rms_out_w));

  if (g_b_mlp1)
    HIP_CHECK(hipFree(g_b_mlp1));
  if (g_b_mlp2)
    HIP_CHECK(hipFree(g_b_mlp2));

  HIP_CHECK(hipFree(d_token_embedding_table_bf16));
  HIP_CHECK(hipFree(d_w_qkv_bf16));
  HIP_CHECK(hipFree(d_w_o_bf16));
  HIP_CHECK(hipFree(d_w_mlp1_bf16));
  HIP_CHECK(hipFree(d_w_mlp2_bf16));
  HIP_CHECK(hipFree(d_out_bf16));
}

float *gpu_forward(Transformer *transformer, int token, int pos) {
  PROFILE_FUNCTION();
  const Config *p = h_config;
  const int H = p->hidden_dim;
  const int D = p->head_dim;
  const int Hq = p->n_attn_heads;
  const int Hk = p->n_kv_heads;
  const int KV = D * Hk;
  const int IM = p->intermediate_dim;
  const int E = p->n_experts;
  const int S = p->seq_len;

  dim3 block(256);
  dim3 gridH((H + block.x - 1) / block.x);

  // x <- embedding[token] (BF16 -> FP32)
  PROFILE_KERNEL_LAUNCH("copy_embedding_bf16_row_kernel",
                        copy_embedding_bf16_row_kernel<<<gridH, block>>>(
                            d_x, d_token_embedding_table_bf16, token, H));

  for (int l = 0; l < p->n_layers; ++l) {
    // RMSNorm (attn)
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel(attn)",
                          rmsnorm_kernel<<<1, 256, 256 * sizeof(double)>>>(
                              d_t, d_x, d_rms_attn_w + l * H, H));

    // QKV
    const int QKV_D = D * (Hq + 2 * Hk);
    dim3 gridQKV((QKV_D + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bf16_wxf32_yf32(QKV)",
        matmul_bf16_wxf32_yf32<<<gridQKV, block>>>(
            d_qkv, d_t, d_w_qkv_bf16 + (size_t)l * QKV_D * H, H, QKV_D));
    PROFILE_KERNEL_LAUNCH(
        "add_bias_kernel(b_qkv)",
        add_bias_kernel<<<gridQKV, block>>>(d_qkv, d_b_qkv + l * QKV_D, QKV_D));

    PROFILE_KERNEL_LAUNCH(
        "split_qkv_kernel",
        split_qkv_kernel<<<gridQKV, block>>>(d_q, d_k, d_v, d_qkv, Hq, Hk, D));

    int loff = l * S * KV;
    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float),
                        hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(d_value_cache + loff + pos * KV, d_v,
                        KV * sizeof(float), hipMemcpyDeviceToDevice));

    dim3 gridR(((D / 2) + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH("compute_cos_sin_kernel",
                          compute_cos_sin_kernel<<<gridR, block>>>(
                              d_cos_vals, d_sin_vals, pos, p->rope_theta, D,
                              p->rope_scaling_factor,
                              p->initial_context_length));

    dim3 gridApplyQ(Hq), gridApplyK(Hk);
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel(Q)",
                          apply_rotary_emb_kernel<<<gridApplyQ, D / 2>>>(
                              d_q, d_cos_vals, d_sin_vals, Hq, D));
    PROFILE_KERNEL_LAUNCH("apply_rotary_emb_kernel(K)",
                          apply_rotary_emb_kernel<<<gridApplyK, D / 2>>>(
                              d_k, d_cos_vals, d_sin_vals, Hk, D));

    HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * KV, d_k, KV * sizeof(float),
                        hipMemcpyDeviceToDevice));

    for (int h = 0; h < Hq; ++h) {
      dim3 gridAtt((pos + 1 + block.x - 1) / block.x);
      PROFILE_KERNEL_LAUNCH(
          "attention_scores_kernel",
          attention_scores_kernel<<<gridAtt, block>>>(
              d_att, d_q, d_key_cache + loff, h, pos, D, KV, Hq, Hk, S,
              (p->sliding_window > 0 && (l % 2 == 0)) ? d_mask : nullptr));
      float sink_val;
      HIP_CHECK(hipMemcpy(&sink_val, d_attn_sinks + l * Hq + h, sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(d_att + h * S + pos + 1, &sink_val, sizeof(float),
                          hipMemcpyHostToDevice));

      PROFILE_KERNEL_LAUNCH("softmax_kernel(att)",
                            softmax_kernel<<<1, 256, 256 * sizeof(float)>>>(
                                d_att + h * S, pos + 2));

      dim3 gridVal((D + block.x - 1) / block.x);
      PROFILE_KERNEL_LAUNCH(
          "attention_values_kernel",
          attention_values_kernel<<<gridVal, block>>>(
              d_tb, d_att, d_value_cache + loff, h, pos, D, KV, Hq, Hk, S));
    }

    const int O_N = D * Hq;
    dim3 gridO((H + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH(
        "matmul_bf16_wxf32_yf32(W_o)",
        matmul_bf16_wxf32_yf32<<<gridO, block>>>(
            d_tb2, d_tb, d_w_o_bf16 + (size_t)l * H * O_N, O_N, H));
    PROFILE_KERNEL_LAUNCH(
        "add_bias_kernel(b_o)",
        add_bias_kernel<<<gridO, block>>>(d_tb2, d_b_o + l * H, H));
    PROFILE_KERNEL_LAUNCH("residual_add_kernel(attn)",
                          residual_add_kernel<<<gridH, block>>>(d_x, d_tb2, H));

    // FFN RMSNorm
    PROFILE_KERNEL_LAUNCH("rmsnorm_kernel(ffn)",
                          rmsnorm_kernel<<<1, 256, 256 * sizeof(double)>>>(
                              d_t, d_x, d_rms_ffn_w + l * H, H));

    // Router FP32: d_router_score(E) = W_router(E,H) @ d_t(H)
    dim3 gridE((E + block.x - 1) / block.x);
    PROFILE_KERNEL_LAUNCH(
        "matmul_f32_wxf32_yf32(router)",
        matmul_f32_wxf32_yf32<<<gridE, block>>>(
            d_router_score, d_t, d_w_router + (size_t)l * H * E, H, E));
    PROFILE_KERNEL_LAUNCH("add_bias_kernel(router_b)",
                          add_bias_kernel<<<gridE, block>>>(
                              d_router_score, d_b_router + l * E, E));

    // Top-k experts
    PROFILE_KERNEL_LAUNCH("topk_kernel_1token",
                          topk_kernel_1token<<<1, 1>>>(d_topk_v, d_topk_i,
                                                       d_router_score, E,
                                                       p->experts_per_token));
    PROFILE_KERNEL_LAUNCH("softmax_kernel(topk)",
                          softmax_kernel<<<1, 256, 256 * sizeof(float)>>>(
                              d_topk_v, p->experts_per_token));

    // Aggregate experts
    PROFILE_KERNEL_LAUNCH("memset_zero_kernel(e_agg)",
                          memset_zero_kernel<<<gridH, block>>>(d_e_agg, H));

    // Read topk to host (small)
    float *h_topk_v = (float *)malloc(p->experts_per_token * sizeof(float));
    int *h_topk_i = (int *)malloc(p->experts_per_token * sizeof(int));
    HIP_CHECK(hipMemcpy(h_topk_v, d_topk_v,
                        p->experts_per_token * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_topk_i, d_topk_i, p->experts_per_token * sizeof(int),
                        hipMemcpyDeviceToHost));

    for (int kk = 0; kk < p->experts_per_token; ++kk) {
      int e = h_topk_i[kk];
      float ew = h_topk_v[kk];
      // MLP1: (2IM x H) @ d_t
      size_t off = ((size_t)l * E + e) * (size_t)(2 * IM) * (size_t)H;
      dim3 gridM1((2 * IM + block.x - 1) / block.x);
      PROFILE_KERNEL_LAUNCH(
          "matmul_bf16_wxf32_yf32(MLP1)",
          matmul_bf16_wxf32_yf32<<<gridM1, block>>>(
              d_mlp1_out, d_t, d_w_mlp1_bf16 + off, H, 2 * IM));

      // Add bias b_mlp1
      size_t b1off = ((size_t)l * E + e) * (size_t)(2 * IM);
      PROFILE_KERNEL_LAUNCH("add_bias_kernel(b_mlp1)",
                            add_bias_kernel<<<gridM1, block>>>(
                                d_mlp1_out, g_b_mlp1 + b1off, 2 * IM));

      // Split & SwiGLU
      dim3 gridIM((IM + block.x - 1) / block.x);
      PROFILE_KERNEL_LAUNCH("split_gate_up_kernel",
                            split_gate_up_kernel<<<gridIM, block>>>(
                                d_gate, d_up, d_mlp1_out, IM));
      PROFILE_KERNEL_LAUNCH("swiglu_kernel",
                            swiglu_kernel<<<gridIM, block>>>(
                                d_gate_up, d_gate, d_up, IM, p->swiglu_limit));

      // MLP2: (H x IM) @ gate_up
      off = ((size_t)l * E + e) * (size_t)H * (size_t)IM;
      PROFILE_KERNEL_LAUNCH("matmul_bf16_wxf32_yf32(MLP2)",
                            matmul_bf16_wxf32_yf32<<<gridH, block>>>(
                                d_tb2, d_gate_up, d_w_mlp2_bf16 + off, IM, H));

      // Add b_mlp2, then weighted sum into e_agg
      size_t b2off = ((size_t)l * E + e) * (size_t)H;
      PROFILE_KERNEL_LAUNCH(
          "add_bias_kernel(b_mlp2)",
          add_bias_kernel<<<gridH, block>>>(d_tb2, g_b_mlp2 + b2off, H));
      PROFILE_KERNEL_LAUNCH(
          "weighted_sum_kernel",
          weighted_sum_kernel<<<gridH, block>>>(d_e_agg, d_tb2, ew, H));
    }
    free(h_topk_v);
    free(h_topk_i);

    // Residual
    PROFILE_KERNEL_LAUNCH(
        "residual_add_kernel(ffn)",
        residual_add_kernel<<<gridH, block>>>(d_x, d_e_agg, H));
  }

  // Final RMSNorm
  PROFILE_KERNEL_LAUNCH("rmsnorm_kernel(final)",
                        rmsnorm_kernel<<<1, 256, 256 * sizeof(double)>>>(
                            d_t, d_x, d_rms_out_w, H));
  HIP_CHECK(hipMemcpy(d_x, d_t, H * sizeof(float), hipMemcpyDeviceToDevice));

  // LM head: (V x H) @ x -> logits(V)
  const int V = p->vocab_size;
  dim3 gridV((V + block.x - 1) / block.x);
  PROFILE_KERNEL_LAUNCH("matmul_bf16_wxf32_yf32(OUT)",
                        matmul_bf16_wxf32_yf32<<<gridV, block>>>(
                            d_logits, d_x, d_out_bf16, H, V));

  return d_logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  PROFILE_FUNCTION();
  const char *empty_prompt = "";
  if (!input_seq)
    input_seq = empty_prompt;

  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "Expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  int next;
  int token = prompt_tokens[0];
  int pos = 0;

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);
  
  while (pos < steps) {
    float *d_log = gpu_forward(transformer, token, pos);

    float *h_logits =
        (float *)malloc(transformer->config.vocab_size * sizeof(float));
    HIP_CHECK(hipMemcpy(h_logits, d_log,
                        transformer->config.vocab_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    pos++;
    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      next = sample(sampler, h_logits);
      output_tokens[pos - num_prompt_tokens] = next;
    }

    if (next == 199999 || next == 200002)
      break;

    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);

    token = next;
    free(h_logits);
  }
  printf("\n");
  output_tokens[pos - num_prompt_tokens + 1] = -1;
  free(prompt_tokens);
  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
