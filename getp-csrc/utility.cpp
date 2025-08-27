#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef hip_bfloat16 bf16_t;

#define TM 8
#define BLOCK_SIZE 512
#define WF_SIZE 64
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

inline dim3 get_gemv_grid_dim(int d) { return dim3((d + TM - 1) / TM, 1, 1); }

__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight, int size) {
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    float v = x[i];
    sum += v * v;
  }
#pragma unroll
  for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
    sum += __shfl_down(sum, off, WF_SIZE);
  }
  __shared__ float warp_sums[BLOCK_SIZE / WF_SIZE];
  if (lane == 0)
    warp_sums[wid] = sum;
  __syncthreads();

  float total = 0.f;
  if (tid < BLOCK_SIZE / WF_SIZE)
    total = warp_sums[tid];

  if (wid == 0) {
    float t = (tid < BLOCK_SIZE / WF_SIZE) ? total : 0.f;
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      t += __shfl_down(t, off, WF_SIZE);
    }
    if (lane == 0)
      warp_sums[0] = t;
  }
  __syncthreads();

  const float inv = rsqrtf(warp_sums[0] / (float)size + 1e-5f);
  for (int i = tid; i < size; i += blockDim.x) {
    o[i] = weight[i] * (x[i] * inv);
  }
}

__global__ void softmax_kernel(float *x, int size) {
  if (threadIdx.x == 0) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
      max_val = fmax(max_val, x[i]);
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
      float ev = expf(x[i] - max_val);
      x[i] = ev;
      sum += ev;
    }
    float inv_sum = 1.0 / sum;
    for (int i = 0; i < size; i++)
      x[i] = x[i] * inv_sum;
  }
}

__global__ void copy_embedding_bf16_row_kernel(float *dst, const bf16_t *src, int token, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    dst[i] = static_cast<float>(src[(size_t)token * hidden_dim + i]);
  }
}

__global__ void split_qkv_kernel(float *q, float *k, float *v, const float *qkv,
                                 int n_attn_heads, int n_kv_heads, int head_dim) {
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

__global__ void split_qkv_scatter_to_cache_kernel(float *q, float *key_cache, float *value_cache, 
                                                  const float *qkv, int n_attn_heads, int n_kv_heads, 
                                                  int head_dim, int layer_offset, int pos_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int q_size = n_attn_heads * head_dim;
  int kv_size = n_kv_heads * head_dim;
  int total = q_size + 2 * kv_size;
  
  if (idx >= total)
    return;
    
  if (idx < q_size) {
    // Copy Q to output buffer
    q[idx] = qkv[idx];
  } else if (idx < q_size + kv_size) {
    // Scatter K directly to key cache
    int k_idx = idx - q_size;
    key_cache[layer_offset + pos_offset + k_idx] = qkv[idx];
  } else {
    // Scatter V directly to value cache  
    int v_idx = idx - q_size - kv_size;
    value_cache[layer_offset + pos_offset + v_idx] = qkv[idx];
  }
}

__global__ void apply_rotary_emb_kernel(float *x, const float *cosv, const float *sinv,
                                       int n_heads, int head_dim) {
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

__global__ void inline_rope_kernel(float *x, int pos, float rope_theta, int n_heads, int head_dim,
                                  float scaling_factor, float initial_context_length) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim >> 1;
  
  if (h >= n_heads || i >= half)
    return;
    
  // Inline computation of cos/sin values
  float freq = powf(rope_theta, (float)(2 * i) / (float)head_dim);
  float inv_freq;
  float concentration = 1.0f;
  
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(rope_theta);
    float high = half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(rope_theta);
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
  float c = cosf(val) * concentration;
  float s = sinf(val) * concentration;
  
  // Inline application of rotary embedding
  float x1 = x[h * head_dim + i];
  float x2 = x[h * head_dim + half + i];
  x[h * head_dim + i] = x1 * c - x2 * s;
  x[h * head_dim + half + i] = x2 * c + x1 * s;
}

__global__ void inline_rope_to_cache_kernel(float *key_cache, int pos, float rope_theta, int n_heads, int head_dim,
                                           float scaling_factor, float initial_context_length, 
                                           int layer_offset, int pos_offset) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim >> 1;
  
  if (h >= n_heads || i >= half)
    return;
    
  // Inline computation of cos/sin values
  float freq = powf(rope_theta, (float)(2 * i) / (float)head_dim);
  float inv_freq;
  float concentration = 1.0f;
  
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float ntk_beta = 32.0f, ntk_alpha = 1.0f;
    float low = half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(rope_theta);
    float high = half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(rope_theta);
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
  float c = cosf(val) * concentration;
  float s = sinf(val) * concentration;
  
  // Apply rotary embedding directly to key cache
  int cache_idx = layer_offset + pos_offset + h * head_dim;
  float x1 = key_cache[cache_idx + i];
  float x2 = key_cache[cache_idx + half + i];
  key_cache[cache_idx + i] = x1 * c - x2 * s;
  key_cache[cache_idx + half + i] = x2 * c + x1 * s;
}

__global__ void add_bias_kernel(float *y, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    y[i] += b[i];
}

__global__ void residual_add_kernel(float *x, const float *residual, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    x[i] += residual[i];
}

__global__ void add_bias_residual_inplace_kernel(float *x, const float *y, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    x[i] = x[i] + (y[i] + b[i]);
}

__global__ void topk_kernel_1token(float *topk_values, int *topk_indices,
                                   float *router_score, int num_experts, int experts_per_token) {
  extern __shared__ float smem[];
  float *scores = smem;
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;

  for (int i = tid; i < num_experts; i += blockDim.x) {
    scores[i] = router_score[i];
  }
  __syncthreads();

  __shared__ float warp_vals[BLOCK_SIZE / WF_SIZE];
  __shared__ int warp_idxs[BLOCK_SIZE / WF_SIZE];

  for (int k = 0; k < experts_per_token; k++) {
    float local_best = -INFINITY;
    int local_idx = -1;
    for (int i = tid; i < num_experts; i += blockDim.x) {
      float v = scores[i];
      if (v > local_best) {
        local_best = v;
        local_idx = i;
      }
    }
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      float oVal = __shfl_down(local_best, off, WF_SIZE);
      int oIdx = __shfl_down(local_idx, off, WF_SIZE);
      if (oVal > local_best) {
        local_best = oVal;
        local_idx = oIdx;
      }
    }
    if (lane == 0) {
      warp_vals[wid] = local_best;
      warp_idxs[wid] = local_idx;
    }
    __syncthreads();

    if (wid == 0) {
      float bVal = (lane < BLOCK_SIZE / WF_SIZE) ? warp_vals[lane] : -INFINITY;
      int bIdx = (lane < BLOCK_SIZE / WF_SIZE) ? warp_idxs[lane] : -1;
#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
        float oVal = __shfl_down(bVal, off, WF_SIZE);
        int oIdx = __shfl_down(bIdx, off, WF_SIZE);
        if (oVal > bVal) {
          bVal = oVal;
          bIdx = oIdx;
        }
      }
      if (lane == 0) {
        topk_values[k] = bVal;
        topk_indices[k] = bIdx;
        if (bIdx >= 0)
          scores[bIdx] = -INFINITY;
      }
    }
    __syncthreads();
  }
}

static void copy_fp32_to_bf16_device(const float *h_src, size_t count, bf16_t *d_dst,
                                     int n_streams, size_t chunk_bytes) {
  if (count == 0)
    return;

  const size_t chunk_elems = chunk_bytes / sizeof(bf16_t);
  const size_t actual_chunk_elems = (chunk_elems > count) ? count : chunk_elems;

  hipStream_t *streams = nullptr;
  hipEvent_t *events = nullptr;
  bf16_t **pinned_chunks = nullptr;
  bool async_success = true;

  streams = (hipStream_t *)malloc(n_streams * sizeof(hipStream_t));
  if (!streams)
    async_success = false;
  if (async_success) {
    events = (hipEvent_t *)malloc(n_streams * sizeof(hipEvent_t));
    if (!events)
      async_success = false;
  }
  if (async_success) {
    pinned_chunks = (bf16_t **)malloc(n_streams * sizeof(bf16_t *));
    if (!pinned_chunks)
      async_success = false;
    else {
      for (int i = 0; i < n_streams; i++)
        pinned_chunks[i] = nullptr;
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipEventCreateWithFlags(&events[i], hipEventDisableTiming);
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }
  if (async_success) {
    for (int i = 0; i < n_streams; i++) {
      hipError_t err = hipHostMalloc((void **)&pinned_chunks[i], actual_chunk_elems * sizeof(bf16_t));
      if (err != hipSuccess) {
        async_success = false;
        break;
      }
    }
  }

  if (!async_success) {
    if (pinned_chunks) {
      for (int i = 0; i < n_streams; i++) {
        if (pinned_chunks[i]) {
          (void)hipHostFree(pinned_chunks[i]);
        }
      }
      free(pinned_chunks);
    }
    if (events) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipEventDestroy(events[i]);
      }
      free(events);
    }
    if (streams) {
      for (int i = 0; i < n_streams; i++) {
        (void)hipStreamDestroy(streams[i]);
      }
      free(streams);
    }

    const size_t SYNC_CHUNK_ELEMS = 8 * 1024 * 1024;
    bf16_t *h_chunk = nullptr;
    HIP_CHECK(hipHostMalloc((void **)&h_chunk, SYNC_CHUNK_ELEMS * sizeof(bf16_t)));
    size_t done = 0;
    while (done < count) {
      size_t todo = (count - done > SYNC_CHUNK_ELEMS) ? SYNC_CHUNK_ELEMS : (count - done);
      for (size_t i = 0; i < todo; ++i)
        h_chunk[i] = hip_bfloat16(h_src[done + i]);
      HIP_CHECK(hipMemcpy(d_dst + done, h_chunk, todo * sizeof(bf16_t), hipMemcpyHostToDevice));
      done += todo;
    }
    HIP_CHECK(hipHostFree(h_chunk));
    return;
  }

  size_t done = 0;
  int stream_idx = 0;
  bool *buffer_ready = (bool *)calloc(n_streams, sizeof(bool));
  for (int i = 0; i < n_streams; i++)
    buffer_ready[i] = true;

  while (done < count) {
    size_t todo = (count - done > actual_chunk_elems) ? actual_chunk_elems : (count - done);
    if (!buffer_ready[stream_idx]) {
      HIP_CHECK(hipEventSynchronize(events[stream_idx]));
      buffer_ready[stream_idx] = true;
    }
    bf16_t *chunk = pinned_chunks[stream_idx];
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < todo; ++i)
      chunk[i] = hip_bfloat16(h_src[done + i]);

    HIP_CHECK(hipMemcpyAsync(d_dst + done, chunk, todo * sizeof(bf16_t),
                             hipMemcpyHostToDevice, streams[stream_idx]));
    HIP_CHECK(hipEventRecord(events[stream_idx], streams[stream_idx]));
    buffer_ready[stream_idx] = false;

    done += todo;
    stream_idx = (stream_idx + 1) % n_streams;
  }
  for (int i = 0; i < n_streams; i++)
    HIP_CHECK(hipEventSynchronize(events[i]));

  free(buffer_ready);
  for (int i = 0; i < n_streams; i++) {
    HIP_CHECK(hipHostFree(pinned_chunks[i]));
    HIP_CHECK(hipEventDestroy(events[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  free(pinned_chunks);
  free(events);
  free(streams);
}
