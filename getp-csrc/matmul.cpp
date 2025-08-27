#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <math.h>

#define TK 256
#define TM 8
#define BLOCK_SIZE 512
#define LDS_PAD 16
#define WF_SIZE 64

typedef hip_bfloat16 bf16_t;

template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void matmul_kernel(float *__restrict__ y, const float *__restrict__ x,
                       const T *__restrict__ w, int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];
  __shared__ float lds_w[TM * TK];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  
  if (wid >= TM || row >= d)
    return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // Cooperative loading of x vector into shared memory
    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = x[k_base + k];
    
    // Load weight matrix tiles cooperatively
    const T *__restrict__ w_base = w + (size_t)blockIdx.x * TM * n + k_base;
    for (int idx = tid; idx < TM * k_size; idx += BLOCK_SIZE) {
      int row_offset = idx / k_size;
      int col_offset = idx % k_size;
      if (row_offset < TM && blockIdx.x * TM + row_offset < d) {
        if constexpr (std::is_same_v<T, bf16_t>) {
          lds_w[idx] = static_cast<float>(w_base[row_offset * n + col_offset]);
        } else {
          lds_w[idx] = ((const float*)w_base)[row_offset * n + col_offset];
        }
      }
    }
    __syncthreads();

    const float *__restrict__ w_row = lds_w + wid * k_size;

    if constexpr (std::is_same_v<T, bf16_t>) {
      float acc = 0.0f;

      // Process 8 elements at a time for better vectorization
      int k_vec8 = (k_size / 8) * 8;
      for (int k = lane * 8; k < k_vec8; k += WF_SIZE * 8) {
        if (k + 7 < k_vec8) {
          // Vectorized operations with 8-wide loads
          float4 x_vals01 = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 x_vals23 = *reinterpret_cast<const float4*>(&lds_x[k + 4]);
          float4 w_vals01 = *reinterpret_cast<const float4*>(&w_row[k]);
          float4 w_vals23 = *reinterpret_cast<const float4*>(&w_row[k + 4]);

          // Dual FMA operations for better throughput
          acc = fmaf(w_vals01.x, x_vals01.x, acc);
          acc = fmaf(w_vals01.y, x_vals01.y, acc);
          acc = fmaf(w_vals01.z, x_vals01.z, acc);
          acc = fmaf(w_vals01.w, x_vals01.w, acc);
          acc = fmaf(w_vals23.x, x_vals23.x, acc);
          acc = fmaf(w_vals23.y, x_vals23.y, acc);
          acc = fmaf(w_vals23.z, x_vals23.z, acc);
          acc = fmaf(w_vals23.w, x_vals23.w, acc);
        }
      }

      // Process 4 elements at a time for remaining
      int k_vec4 = (k_size / 4) * 4;
      for (int k = max(k_vec8, lane * 4); k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 w_vals = *reinterpret_cast<const float4*>(&w_row[k]);
          
          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }

      // Handle remaining elements
      for (int k = max(k_vec4, lane); k < k_size; k += WF_SIZE) {
        acc = fmaf(w_row[k], lds_x[k], acc);
      }

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    } else {
      // Optimized FP32 path
      float acc = 0.0f;
      
      // Process 4 elements at a time with vectorized loads
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 w_vals = *reinterpret_cast<const float4*>(&w_row[k]);
          
          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }
      
      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        acc = fmaf(w_row[k], lds_x[k], acc);
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
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void matmul_bias_kernel(float *__restrict__ y, const float *__restrict__ x,
                            const T *__restrict__ w,
                            const float *__restrict__ b, int n, int d) {
  __shared__ float lds_x[TK + LDS_PAD];
  __shared__ float lds_w[TM * TK];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  if (wid >= TM || row >= d)
    return;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // Cooperative loading of x vector into shared memory
    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = x[k_base + k];
    
    // Load weight matrix tiles cooperatively
    const T *__restrict__ w_base = w + (size_t)blockIdx.x * TM * n + k_base;
    for (int idx = tid; idx < TM * k_size; idx += BLOCK_SIZE) {
      int row_offset = idx / k_size;
      int col_offset = idx % k_size;
      if (row_offset < TM && blockIdx.x * TM + row_offset < d) {
        if constexpr (std::is_same_v<T, bf16_t>) {
          lds_w[idx] = static_cast<float>(w_base[row_offset * n + col_offset]);
        } else {
          lds_w[idx] = ((const float*)w_base)[row_offset * n + col_offset];
        }
      }
    }
    __syncthreads();

    const float *__restrict__ w_row = lds_w + wid * k_size;

    if constexpr (std::is_same_v<T, bf16_t>) {
      float acc = 0.0f;

      // Process 8 elements at a time for better vectorization
      int k_vec8 = (k_size / 8) * 8;
      for (int k = lane * 8; k < k_vec8; k += WF_SIZE * 8) {
        if (k + 7 < k_vec8) {
          float4 x_vals01 = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 x_vals23 = *reinterpret_cast<const float4*>(&lds_x[k + 4]);
          float4 w_vals01 = *reinterpret_cast<const float4*>(&w_row[k]);
          float4 w_vals23 = *reinterpret_cast<const float4*>(&w_row[k + 4]);

          acc = fmaf(w_vals01.x, x_vals01.x, acc);
          acc = fmaf(w_vals01.y, x_vals01.y, acc);
          acc = fmaf(w_vals01.z, x_vals01.z, acc);
          acc = fmaf(w_vals01.w, x_vals01.w, acc);
          acc = fmaf(w_vals23.x, x_vals23.x, acc);
          acc = fmaf(w_vals23.y, x_vals23.y, acc);
          acc = fmaf(w_vals23.z, x_vals23.z, acc);
          acc = fmaf(w_vals23.w, x_vals23.w, acc);
        }
      }

      // Process 4 elements at a time for remaining
      int k_vec4 = (k_size / 4) * 4;
      for (int k = max(k_vec8, lane * 4); k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 w_vals = *reinterpret_cast<const float4*>(&w_row[k]);
          
          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }

      // Handle remaining elements
      for (int k = max(k_vec4, lane); k < k_size; k += WF_SIZE) {
        acc = fmaf(w_row[k], lds_x[k], acc);
      }

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);
      if (lane == 0)
        acc_all += acc;
    } else {
      float acc = 0.0f;
      
      // Process 4 elements at a time with vectorized loads
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 w_vals = *reinterpret_cast<const float4*>(&w_row[k]);
          
          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }
      
      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        acc = fmaf(w_row[k], lds_x[k], acc);
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
__launch_bounds__(BLOCK_SIZE, 1) __global__
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
    
    // Cooperative loading of x vector
    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = x[k_base + k];
    __syncthreads();

    const T *__restrict__ wg = w_mlp1_all + off_gate + k_base;
    const T *__restrict__ wu = w_mlp1_all + off_up + k_base;

    float acc_g = 0.0f, acc_u = 0.0f;
    
    if constexpr (std::is_same_v<T, bf16_t>) {
      // Vectorized processing for BF16
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          
          bf16_t wg0 = wg[k], wg1 = wg[k+1], wg2 = wg[k+2], wg3 = wg[k+3];
          bf16_t wu0 = wu[k], wu1 = wu[k+1], wu2 = wu[k+2], wu3 = wu[k+3];
          
          acc_g = fmaf((float)wg0, x_vals.x, acc_g);
          acc_g = fmaf((float)wg1, x_vals.y, acc_g);
          acc_g = fmaf((float)wg2, x_vals.z, acc_g);
          acc_g = fmaf((float)wg3, x_vals.w, acc_g);
          
          acc_u = fmaf((float)wu0, x_vals.x, acc_u);
          acc_u = fmaf((float)wu1, x_vals.y, acc_u);
          acc_u = fmaf((float)wu2, x_vals.z, acc_u);
          acc_u = fmaf((float)wu3, x_vals.w, acc_u);
        }
      }
      
      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = (float)wg[k];
        float wuk = (float)wu[k];
        acc_g = fmaf(wgk, xv, acc_g);
        acc_u = fmaf(wuk, xv, acc_u);
      }
    } else {
      // Vectorized processing for FP32
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 wg_vals = *reinterpret_cast<const float4*>(&((const float*)wg)[k]);
          float4 wu_vals = *reinterpret_cast<const float4*>(&((const float*)wu)[k]);
          
          acc_g = fmaf(wg_vals.x, x_vals.x, acc_g);
          acc_g = fmaf(wg_vals.y, x_vals.y, acc_g);
          acc_g = fmaf(wg_vals.z, x_vals.z, acc_g);
          acc_g = fmaf(wg_vals.w, x_vals.w, acc_g);
          
          acc_u = fmaf(wu_vals.x, x_vals.x, acc_u);
          acc_u = fmaf(wu_vals.y, x_vals.y, acc_u);
          acc_u = fmaf(wu_vals.z, x_vals.z, acc_u);
          acc_u = fmaf(wu_vals.w, x_vals.w, acc_u);
        }
      }
      
      // Handle remaining elements  
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = ((const float*)wg)[k];
        float wuk = ((const float*)wu)[k];
        acc_g = fmaf(wgk, xv, acc_g);
        acc_u = fmaf(wuk, xv, acc_u);
      }
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
    g = fminf(fmaxf(g, -swiglu_limit), swiglu_limit);
    u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);

    // SiLU approx then * (u + 1)  
    const float alpha = 1.702f;
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);

    gate_up[i] = g;
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
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
    
    // Cooperative loading of gate_up vector
    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = gate_up[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w_row_base + k_base;
    float acc = 0.0f;
    
    if constexpr (std::is_same_v<T, bf16_t>) {
      // Vectorized processing for BF16
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          
          bf16_t w0 = w_row[k], w1 = w_row[k+1], w2 = w_row[k+2], w3 = w_row[k+3];
          
          acc = fmaf((float)w0, x_vals.x, acc);
          acc = fmaf((float)w1, x_vals.y, acc);
          acc = fmaf((float)w2, x_vals.z, acc);
          acc = fmaf((float)w3, x_vals.w, acc);
        }
      }
      
      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        bf16_t w_val = w_row[k];
        acc = fmaf((float)w_val, x_val, acc);
      }
    } else {
      // Vectorized processing for FP32
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
          float4 w_vals = *reinterpret_cast<const float4*>(&((const float*)w_row)[k]);
          
          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }
      
      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
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
    atomicAdd(&e_agg[row], y * weight);
  }
}
