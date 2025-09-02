#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>

#define TK 256
#define TM 8
#define BLOCK_SIZE 512
#define LDS_PAD 16
#define WF_SIZE 64

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0,
                                                   float &f1) {
  union {
    uint32_t u;
    float f;
  } a, b;
  a.u = (u & 0x0000FFFFu) << 16; // lower bf16 -> fp32
  b.u = (u & 0xFFFF0000u);       // upper bf16 -> fp32
  f0 = a.f;
  f1 = b.f;
}

template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void matmul_bias_batch_kernel(float *__restrict__ y,
                                  const float *__restrict__ x,
                                  const T *__restrict__ w,
                                  const float *__restrict__ b,
                                  const int *__restrict__ pos, int n, int d,
                                  int batch_size) {
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int bidx = blockIdx.y;
  if (bidx >= batch_size)
    return;
  if (wid >= TM || row >= d)
    return;
  if (pos[bidx] < 0)
    return;

  const float *x_b = x + (size_t)bidx * n;
  float *y_b = y + (size_t)bidx * d;

  float acc_all = 0.0f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = (k_base + TK < n) ? TK : (n - k_base);

    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = x_b[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row_g = w + (size_t)row * (size_t)n + k_base;

    if constexpr (std::is_same_v<T, bf16_t>) {
      const uint32_t *__restrict__ w32 =
          reinterpret_cast<const uint32_t *>(w_row_g);
      const int pairs = k_size >> 1;
      const int step_pairs = 4;
      const int vec_pairs = (pairs / step_pairs) * step_pairs;

      float acc0 = 0.f, acc1 = 0.f;

      for (int p = lane * step_pairs; p < vec_pairs;
           p += WF_SIZE * step_pairs) {
        const int k_elem = (p << 1);
        float4 x01 = *reinterpret_cast<const float4 *>(&lds_x[k_elem + 0]);
        float4 x23 = *reinterpret_cast<const float4 *>(&lds_x[k_elem + 4]);

        uint32_t u0 = w32[p + 0], u1 = w32[p + 1], u2 = w32[p + 2],
                 u3 = w32[p + 3];
        float w0, w1, w2, w3, w4, w5, w6, w7;
        bf16pair_to_float2(u0, w0, w1);
        bf16pair_to_float2(u1, w2, w3);
        bf16pair_to_float2(u2, w4, w5);
        bf16pair_to_float2(u3, w6, w7);

        acc0 = fmaf(w0, x01.x, acc0);
        acc1 = fmaf(w1, x01.y, acc1);
        acc0 = fmaf(w2, x01.z, acc0);
        acc1 = fmaf(w3, x01.w, acc1);
        acc0 = fmaf(w4, x23.x, acc0);
        acc1 = fmaf(w5, x23.y, acc1);
        acc0 = fmaf(w6, x23.z, acc0);
        acc1 = fmaf(w7, x23.w, acc1);
      }

      float acc = acc0 + acc1;

      for (int p = vec_pairs + lane; p < pairs; p += WF_SIZE) {
        const int k_elem = (p << 1);
        uint32_t up = reinterpret_cast<const uint32_t *>(w_row_g)[p];
        float w0, w1;
        bf16pair_to_float2(up, w0, w1);
        acc = fmaf(w0, lds_x[k_elem + 0], acc);
        if (k_elem + 1 < k_size)
          acc = fmaf(w1, lds_x[k_elem + 1], acc);
      }
      if (k_size & 1) {
        const int k = k_base + k_size - 1;
        if ((k - k_base) % WF_SIZE == lane) {
          uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t *>(
                           &w_row_g[k_size - 1])))
                       << 16;
          union {
            uint32_t u;
            float f;
          } cvt;
          cvt.u = u;
          acc = fmaf(cvt.f, lds_x[k_size - 1], acc);
        }
      }

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);

      if (lane == 0)
        acc_all += acc;

    } else {
      const float *__restrict__ wf = reinterpret_cast<const float *>(w_row_g);
      int vec = (k_size / 8) * 8;
      float acc0 = 0.f, acc1 = 0.f;

      for (int k = lane * 8; k < vec; k += WF_SIZE * 8) {
        float4 x01 = *reinterpret_cast<const float4 *>(&lds_x[k + 0]);
        float4 x23 = *reinterpret_cast<const float4 *>(&lds_x[k + 4]);
        float4 w01 = *reinterpret_cast<const float4 *>(&wf[k + 0]);
        float4 w23 = *reinterpret_cast<const float4 *>(&wf[k + 4]);

        acc0 = fmaf(w01.x, x01.x, acc0);
        acc1 = fmaf(w01.y, x01.y, acc1);
        acc0 = fmaf(w01.z, x01.z, acc0);
        acc1 = fmaf(w01.w, x01.w, acc1);
        acc0 = fmaf(w23.x, x23.x, acc0);
        acc1 = fmaf(w23.y, x23.y, acc1);
        acc0 = fmaf(w23.z, x23.z, acc0);
        acc1 = fmaf(w23.w, x23.w, acc1);
      }

      float acc = acc0 + acc1;

      int vec4 = ((k_size - vec) / 4) * 4 + vec;
      for (int k = max(vec, lane * 4); k < vec4; k += WF_SIZE * 4) {
        float4 xv = *reinterpret_cast<const float4 *>(&lds_x[k]);
        float4 wv = *reinterpret_cast<const float4 *>(&wf[k]);
        acc = fmaf(wv.x, xv.x, acc);
        acc = fmaf(wv.y, xv.y, acc);
        acc = fmaf(wv.z, xv.z, acc);
        acc = fmaf(wv.w, xv.w, acc);
      }
      for (int k = vec4 + lane; k < k_size; k += WF_SIZE)
        acc = fmaf(wf[k], lds_x[k], acc);

#pragma unroll
      for (int off = WF_SIZE >> 1; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, WF_SIZE);

      if (lane == 0)
        acc_all += acc;
    }

    __syncthreads();
  }

  if (lane == 0)
    y_b[row] = acc_all + b[row];
}

// Batched MLP1 kernel
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__ void mlp1_fused_batch_kernel(
    float *__restrict__ gate_up, const float *__restrict__ x,
    const T *__restrict__ w_mlp1_all, const float *__restrict__ b_mlp1_all,
    const int *__restrict__ topk_i, const int *__restrict__ pos, int k_index,
    int l_layer, int E, int H, int IM, float swiglu_limit, int batch_size,
    int experts_per_token) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int i = blockIdx.x * TM + wid;
  const int b = blockIdx.y;
  if (b >= batch_size || pos[b] < 0)
    return;
  if (wid >= TM || i >= IM)
    return;

  const float *x_b = x + (size_t)b * H;
  float *gate_up_b = gate_up + (size_t)b * IM;
  const int *topk_i_b = topk_i + (size_t)b * experts_per_token;
  const int e = topk_i_b[k_index];

  float acc_g_all = 0.0f;
  float acc_u_all = 0.0f;

  const size_t base_rows =
      ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
  const size_t off_gate = (base_rows + (size_t)(2 * i + 0)) * (size_t)H;
  const size_t off_up = (base_rows + (size_t)(2 * i + 1)) * (size_t)H;

  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = x_b[k_base + k];
    __syncthreads();

    const T *__restrict__ wg = w_mlp1_all + off_gate + k_base;
    const T *__restrict__ wu = w_mlp1_all + off_up + k_base;

    float acc_g = 0.0f, acc_u = 0.0f;

    if constexpr (std::is_same_v<T, bf16_t>) {
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4 *>(&lds_x[k]);

          bf16_t wg0 = wg[k], wg1 = wg[k + 1], wg2 = wg[k + 2], wg3 = wg[k + 3];
          bf16_t wu0 = wu[k], wu1 = wu[k + 1], wu2 = wu[k + 2], wu3 = wu[k + 3];

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

      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = (float)wg[k];
        float wuk = (float)wu[k];
        acc_g = fmaf(wgk, xv, acc_g);
        acc_u = fmaf(wuk, xv, acc_u);
      }
    } else {
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4 *>(&lds_x[k]);
          float4 wg_vals =
              *reinterpret_cast<const float4 *>(&((const float *)wg)[k]);
          float4 wu_vals =
              *reinterpret_cast<const float4 *>(&((const float *)wu)[k]);

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

      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = ((const float *)wg)[k];
        float wuk = ((const float *)wu)[k];
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
    const size_t bbase =
        ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)(2 * IM);
    float g = acc_g_all + b_mlp1_all[bbase + (size_t)(2 * i + 0)];
    float u = acc_u_all + b_mlp1_all[bbase + (size_t)(2 * i + 1)];

    g = fminf(fmaxf(g, -swiglu_limit), swiglu_limit);
    u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);

    const float alpha = 1.702f;
    g *= (1.0f / (1.0f + expf(-alpha * g)));
    g *= (u + 1.0f);

    gate_up_b[i] = g;
  }
}

// Batched MLP2 kernel
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void mlp2_bias_weighted_accum_batch_kernel(
        float *__restrict__ e_agg, const float *__restrict__ gate_up,
        const T *__restrict__ w_mlp2_all, const float *__restrict__ b_mlp2_all,
        const int *__restrict__ topk_i, const float *__restrict__ topk_v,
        const int *__restrict__ pos, int k_index, int l_layer, int E, int IM,
        int H, int batch_size, int experts_per_token) {
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int b = blockIdx.y;
  if (b >= batch_size || pos[b] < 0)
    return;
  if (wid >= TM || row >= H)
    return;

  const float *gate_up_b = gate_up + (size_t)b * IM;
  float *e_agg_b = e_agg + (size_t)b * H;
  const int *topk_i_b = topk_i + (size_t)b * experts_per_token;
  const float *topk_v_b = topk_v + (size_t)b * experts_per_token;

  const int e = topk_i_b[k_index];
  const float weight = topk_v_b[k_index];

  float acc_all = 0.0f;

  const size_t base =
      ((size_t)l_layer * (size_t)E + (size_t)e) * (size_t)H * (size_t)IM;
  const T *__restrict__ w_row_base =
      w_mlp2_all + (size_t)row * (size_t)IM + base;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    for (int k = tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = gate_up_b[k_base + k];
    __syncthreads();

    const T *__restrict__ w_row = w_row_base + k_base;
    float acc = 0.0f;

    if constexpr (std::is_same_v<T, bf16_t>) {
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4 *>(&lds_x[k]);

          bf16_t w0 = w_row[k], w1 = w_row[k + 1], w2 = w_row[k + 2],
                 w3 = w_row[k + 3];

          acc = fmaf((float)w0, x_vals.x, acc);
          acc = fmaf((float)w1, x_vals.y, acc);
          acc = fmaf((float)w2, x_vals.z, acc);
          acc = fmaf((float)w3, x_vals.w, acc);
        }
      }

      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float x_val = lds_x[k];
        bf16_t w_val = w_row[k];
        acc = fmaf((float)w_val, x_val, acc);
      }
    } else {
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        if (k + 3 < k_vec4) {
          float4 x_vals = *reinterpret_cast<const float4 *>(&lds_x[k]);
          float4 w_vals =
              *reinterpret_cast<const float4 *>(&((const float *)w_row)[k]);

          acc = fmaf(w_vals.x, x_vals.x, acc);
          acc = fmaf(w_vals.y, x_vals.y, acc);
          acc = fmaf(w_vals.z, x_vals.z, acc);
          acc = fmaf(w_vals.w, x_vals.w, acc);
        }
      }

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
    atomicAdd(&e_agg_b[row], y * weight);
  }
}
