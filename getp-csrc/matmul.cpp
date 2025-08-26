#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <math.h>

#define TK 128
#define TM 4
#define BLOCK_SIZE 256
#define LDS_PAD 8
#define WF_SIZE 64

typedef hip_bfloat16 bf16_t;

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2) __global__
    void matmul_kernel(float *__restrict__ y, const float *__restrict__ x,
                       const T *__restrict__ w, int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d)
    return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE)
      lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w + (size_t)row * n + k_base;

    // Use optimized BF16 operations when possible
    if constexpr (std::is_same_v<T, bf16_t>) {
      float acc = 0.0f;

      // Process 4 elements at a time for better memory access
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          // Vectorized BF16 operations
          float x_val0 = lds_x[k];
          float x_val1 = lds_x[k + 1];
          float x_val2 = lds_x[k + 2];
          float x_val3 = lds_x[k + 3];

          bf16_t w_val0 = w_row[k];
          bf16_t w_val1 = w_row[k + 1];
          bf16_t w_val2 = w_row[k + 2];
          bf16_t w_val3 = w_row[k + 3];

          // Accumulate with FMA for better precision
          acc = fmaf(static_cast<float>(w_val0), x_val0, acc);
          acc = fmaf(static_cast<float>(w_val1), x_val1, acc);
          acc = fmaf(static_cast<float>(w_val2), x_val2, acc);
          acc = fmaf(static_cast<float>(w_val3), x_val3, acc);
        }
      }

      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        bf16_t w_val = w_row[k];
        acc = fmaf(static_cast<float>(w_val), x_val, acc);
      }

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    } else {
      // Fall back to standard computation for FP32
      float acc = 0.0f;
#pragma unroll 4
      for (int k = lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        float w_val = w_row[k];
        acc = fmaf(w_val, x_val, acc);
      }
#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    }
    __syncthreads();
  }
  if (lane == 0)
    y[row] = acc_all;
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2) __global__
    void matmul_bias_kernel(float *__restrict__ y, const float *__restrict__ x,
                            const T *__restrict__ w,
                            const float *__restrict__ b, int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d)
    return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    for (int k = lane; k < k_size; k += WF_SIZE)
      lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w + (size_t)row * n + k_base;

    if constexpr (std::is_same_v<T, bf16_t>) {
      float acc = 0.0f;

      // Process 4 elements at a time for better memory access
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          // Vectorized BF16 operations
          float x_val0 = lds_x[k];
          float x_val1 = lds_x[k + 1];
          float x_val2 = lds_x[k + 2];
          float x_val3 = lds_x[k + 3];

          bf16_t w_val0 = w_row[k];
          bf16_t w_val1 = w_row[k + 1];
          bf16_t w_val2 = w_row[k + 2];
          bf16_t w_val3 = w_row[k + 3];

          // Accumulate with FMA for better precision
          acc = fmaf(static_cast<float>(w_val0), x_val0, acc);
          acc = fmaf(static_cast<float>(w_val1), x_val1, acc);
          acc = fmaf(static_cast<float>(w_val2), x_val2, acc);
          acc = fmaf(static_cast<float>(w_val3), x_val3, acc);
        }
      }

      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        bf16_t w_val = w_row[k];
        acc = fmaf(static_cast<float>(w_val), x_val, acc);
      }

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    } else {
      float acc = 0.0f;
#pragma unroll 4
      for (int k = lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        float w_val = w_row[k];
        acc = fmaf(w_val, x_val, acc);
      }
#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    }
    __syncthreads();
  }
  if (lane == 0)
    y[row] = acc_all + b[row];
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2) __global__
    void mlp1_fused_kernel(float *__restrict__ gate_up,
                           const float *__restrict__ x,
                           const T *__restrict__ w_mlp1_all,
                           const float *__restrict__ b_mlp1_all,
                           const int *__restrict__ topk_i, int k_index,
                           int l_layer, int E, int H, int IM,
                           float swiglu_limit) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int i = blockIdx.x * TM + wid; // output index in [0..IM)
  if (wid >= TM || i >= IM)
    return;

  // read expert id from device
  const int e = topk_i[k_index];

  float acc_g_all = 0.0f;
  float acc_u_all = 0.0f;

  // Offsets
  const size_t base_rows =
      ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
  const size_t off_gate = (base_rows + (size_t)(2 * i + 0)) * (size_t)H;
  const size_t off_up = (base_rows + (size_t)(2 * i + 1)) * (size_t)H;

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);
    for (int k = lane; k < k_size; k += WF_SIZE)
      lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ wg = w_mlp1_all + off_gate + k_base;
    const T *__restrict__ wu = w_mlp1_all + off_up + k_base;

    float acc_g = 0.0f, acc_u = 0.0f;
#pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float xv = lds_x[k];
      float wgk = std::is_same_v<T, bf16_t> ? (float)wg[k]
                                            : (float)((const float *)wg)[k];
      float wuk = std::is_same_v<T, bf16_t> ? (float)wu[k]
                                            : (float)((const float *)wu)[k];
      acc_g = fmaf(wgk, xv, acc_g);
      acc_u = fmaf(wuk, xv, acc_u);
    }
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1) {
      acc_g += __shfl_down(acc_g, off, WF_SIZE);
      acc_u += __shfl_down(acc_u, off, WF_SIZE);
    }
    if (lane == 0) {
      acc_g_all += acc_g;
      acc_u_all += acc_u;
    }
    __syncthreads();
  }

  if (lane == 0) {
    // Add bias
    const size_t bbase =
        ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
    float g = acc_g_all + b_mlp1_all[bbase + (size_t)(2 * i + 0)];
    float u = acc_u_all + b_mlp1_all[bbase + (size_t)(2 * i + 1)];

    // Clamp (match previous kernel)
    if (g > swiglu_limit)
      g = swiglu_limit;
    if (u > swiglu_limit)
      u = swiglu_limit;
    if (u < -swiglu_limit)
      u = -swiglu_limit;

    // SiLU approx then * (u + 1)
    const float alpha = 1.702f;
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);

    gate_up[i] = g;
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 2) __global__
    void mlp2_bias_weighted_accum_kernel(
        float *__restrict__ e_agg, const float *__restrict__ gate_up,
        const T *__restrict__ w_mlp2_all, const float *__restrict__ b_mlp2_all,
        const int *__restrict__ topk_i, const float *__restrict__ topk_v,
        int k_index, int l_layer, int E, int IM, int H) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid; // output index in [0..H)
  if (wid >= TM || row >= H)
    return;

  const int e = topk_i[k_index];
  const float weight = topk_v[k_index];

  float acc_all = 0.0f;

  const size_t base =
      ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)H * (size_t)IM;
  const T *__restrict__ w_row_base =
      w_mlp2_all + (size_t)row * (size_t)IM + base;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);
    for (int k = lane; k < k_size; k += WF_SIZE)
      lds_x[k] = gate_up[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w_row_base + k_base;
    float acc = 0.0f;
#pragma unroll 4
    for (int k = lane; k < k_size; k += WF_SIZE) {
      float x_val = lds_x[k];
      if constexpr (std::is_same_v<T, bf16_t>) {
        bf16_t w_val = w_row[k];
        acc = fmaf((float)w_val, x_val, acc);
      } else {
        float w_val = ((const float *)w_row)[k];
        acc = fmaf(w_val, x_val, acc);
      }
    }
#pragma unroll
    for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
      acc += __shfl_down(acc, off, WF_SIZE);
    if (lane == 0)
      acc_all += acc;
    __syncthreads();
  }

  if (lane == 0) {
    const size_t bbase = ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)H;
    float y = acc_all + b_mlp2_all[bbase + (size_t)row];
    e_agg[row] += y * weight;
  }
}
