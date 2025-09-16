#include "../common/defines.h"
#include "../utility/utility.h"
#include "matmul.h"
#include <math.h>


// x:[n], w:[d,n], y:[d]
// Grid: (ceil(d/TM), 1)
__global__ void matmul_bias_gemm_kernel_bf16(
    float* __restrict__ y,
    const float* __restrict__ x,
    const bf16_t* __restrict__ w,
    const float* __restrict__ bias,
    int n, int d, const int *pos)
{
  if (pos && pos[0] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;
  const int row  = blockIdx.x * TM + wid;

  if (wid >= TM || row >= d) return;

  float acc = 0.f;

  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // Load X for single sample
    const float* __restrict__ xb = x + k_base;
    // vec4 load
    const int vec4 = (k_size >> 2);
    float4* s4 = reinterpret_cast<float4*>(lds_x);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) s4[v] = x4[v];
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE)
      lds_x[k] = xb[k];
    __syncthreads();

    // Weight computation for single sample
    const bf16_t* __restrict__ w_row = w + (size_t)row * n + k_base;

    const int vec4k = (k_size >> 2);
    const uint2* wq = reinterpret_cast<const uint2*>(w_row);
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 wv = bf16quad_to_float4(wq[v]);
      const float4 xv = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    for (int k = vec4k * 4 + lane; k < k_size; k += WF_SIZE) {
      const float wv = (float)w_row[k];
      acc = fmaf(wv, lds_x[k], acc);
    }
    __syncthreads();
  }

  // Reduce and write for single sample
  float v = warp_reduce_sum(acc);
  if (lane == 0) {
    y[row] = v + (bias ? bias[row] : 0.0f);
  }
}

/**
 * Y = X @ W^T + B (float version)
 * x: [n], w: [d, n], y: [d]
 * Grid: (ceil(d/TM), 1)
 */
__global__ void matmul_bias_gemm_kernel_float(
    float* __restrict__ y,          // [d]
    const float* __restrict__ x,    // [n]
    const float* __restrict__ w,    // [d, n] (row-major theo n)
    const float* __restrict__ bias, // [d] (có thể null)
    int n, int d, const int *pos)
{
  if (pos && pos[0] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;                 // warp id trong block (0..TM-1)

  const int row = blockIdx.x * TM + wid;    // hàng output (0..d-1)

  if (wid >= TM || row >= d) return;

  float acc = 0.f;

  // Vòng K - optimized with vectorized loads and computation
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // 1) Optimized vectorized load of X columns for single sample
    const float* __restrict__ xb = x + k_base;
    
    // Load using vectorized operations
    const int vec4_k = (k_size >> 2);
    float4* __restrict__ lds_x4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ xb4 = reinterpret_cast<const float4*>(xb);
    
    for (int v = tid; v < vec4_k; v += BLOCK_SIZE) {
      lds_x4[v] = xb4[v];
    }
    
    // Handle remainder elements
    for (int k = (vec4_k << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // 2) Optimized computation with weight row for single sample
    const float* __restrict__ w_row = w + (size_t)row * n + k_base;
    
    // Vectorized computation - process 4 elements at a time
    const int vec4_loop = (k_size >> 2);

    for (int v = lane; v < vec4_loop; v += WF_SIZE) {
      // Load 4 consecutive elements as vectors
      const float4 w_vec = *reinterpret_cast<const float4*>(&w_row[v << 2]);
      const float4 x_vec = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      
      // Perform dot product using fused multiply-add
      acc = fmaf(w_vec.x, x_vec.x, acc);
      acc = fmaf(w_vec.y, x_vec.y, acc);
      acc = fmaf(w_vec.z, x_vec.z, acc);
      acc = fmaf(w_vec.w, x_vec.w, acc);
    }

    // Handle remainder elements
    for (int k = (vec4_loop << 2) + lane; k < k_size; k += WF_SIZE) {
      acc = fmaf(w_row[k], lds_x[k], acc);
    }
    __syncthreads();
  }

  // Optimized warp reduction and write output for single sample
  float result = warp_reduce_sum(acc);
  if (lane == 0) {
    y[row] = result + (bias ? bias[row] : 0.0f);
  }
}

// ================= MLP1 (Gate & Up) : single sample, no CB =================
__global__ void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up_topk, // [K, IM] (K = EXPERT_PER_TOKEN)
    const float* __restrict__ x,      // [H]
    const bf16_t* __restrict__ w_mlp1_all, // [L, E, 2*IM, H] (row-major in last dim)
    const float* __restrict__ b_mlp1_all,  // [L, E, 2*IM]
    const int* __restrict__ topk_i,   // [K]
    int l_layer, int E, int H, int IM,
    float swiglu_limit,
    const int *pos)
{
  if (pos && pos[0] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;          // warp id in block

  const int i    = blockIdx.x * TM + wid;   // output row in IM (0..IM-1)
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || i >= IM || kidx >= EXPERT_PER_TOKEN)
    return;

  // Load expert ID for single sample
  expert_id = topk_i[kidx];
  if (expert_id < 0) return;

  float acc_gate = 0.f, acc_up = 0.f;

  // K loop over H, loading x for single sample
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load X for single sample
    const float* __restrict__ xb = x + k_base;
    // vec4 part
    const int vec4 = (k_size >> 2);
    float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
    // tail
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // Compute for single sample
    // weights layout:
    // base = ((l * E + expert) * (2*IM)) * H + (2*i + {0,1}) * H + k_base
    const size_t base_2im = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)(2 * IM);
    const size_t gate_off = ((base_2im + (size_t)(2 * i + 0)) * (size_t)H) + (size_t)k_base;
    const size_t up_off   = ((base_2im + (size_t)(2 * i + 1)) * (size_t)H) + (size_t)k_base;

    const uint2* __restrict__ gate_q = reinterpret_cast<const uint2*>(w_mlp1_all + gate_off);
    const uint2* __restrict__  up_q  = reinterpret_cast<const uint2*>(w_mlp1_all + up_off);

    // vector compute (4 bf16 weights at a time)
    const int vec4k = (k_size >> 2);
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 w_gate = bf16quad_to_float4(gate_q[v]);
      const float4 w_up   = bf16quad_to_float4(up_q[v]);
      const float4 xv     = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      acc_gate = fmaf(w_gate.x, xv.x, acc_gate);
      acc_gate = fmaf(w_gate.y, xv.y, acc_gate);
      acc_gate = fmaf(w_gate.z, xv.z, acc_gate);
      acc_gate = fmaf(w_gate.w, xv.w, acc_gate);

      acc_up = fmaf(w_up.x, xv.x, acc_up);
      acc_up = fmaf(w_up.y, xv.y, acc_up);
      acc_up = fmaf(w_up.z, xv.z, acc_up);
      acc_up = fmaf(w_up.w, xv.w, acc_up);
    }
    // tail pairs/singles
    const int tail_start = vec4k << 2;
    for (int k = tail_start + lane; k < k_size; k += WF_SIZE) {
      // bf16 -> f32 (single)
      uint32_t ug = ((uint32_t)(*reinterpret_cast<const uint16_t*>(
                        (w_mlp1_all + gate_off + k)))) << 16;
      uint32_t uu = ((uint32_t)(*reinterpret_cast<const uint16_t*>(
                        (w_mlp1_all + up_off   + k)))) << 16;
      union { uint32_t u; float f; } cg, cu; cg.u = ug; cu.u = uu;
      acc_gate = fmaf(cg.f, lds_x[k], acc_gate);
      acc_up   = fmaf(cu.f, lds_x[k], acc_up);
    }
    __syncthreads();
  }

  // warp reduce and output for single sample
  float gate_sum = warp_reduce_sum(acc_gate);
  float up_sum   = warp_reduce_sum(acc_up);

  if (lane == 0) {
    const size_t b_gate_base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)(2 * IM);
    float gate = gate_sum + b_mlp1_all[b_gate_base + (size_t)(2 * i + 0)];
    float up   = up_sum   + b_mlp1_all[b_gate_base + (size_t)(2 * i + 1)];

    // clip + SwiGLU-ish (như bản bạn đang dùng)
    gate = __saturatef((gate + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    up   = __saturatef((up   + swiglu_limit) / (2.0f * swiglu_limit)) * (2.0f * swiglu_limit) - swiglu_limit;
    const float alpha = 1.702f;
    gate = gate * __saturatef(0.5f + 0.5f * tanhf(alpha * gate * 0.5f));
    gate = gate * (up + 1.0f);

    float* __restrict__ gu = gate_up_topk + ((size_t)kidx * (size_t)IM);
    gu[i] = gate;
  }
}


// ============ MLP2 (weighted accum) : single sample, no CB ==============
__global__ void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,              // [H] (accumulator)
    const float* __restrict__ gate_up_topk, // [K, IM]
    const bf16_t* __restrict__ w_mlp2_all,  // [L, E, H, IM]
    const float* __restrict__ b_mlp2_all,   // [L, E, H]
    const int* __restrict__ topk_i,         // [K]
    const float* __restrict__ topk_v,       // [K]
    int l_layer, int E, int IM, int H,
    const int *pos)
{
  if (pos && pos[0] < 0) return;

  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  int expert_id;
  float expert_w;

  const int tid  = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid  = tid >> 6;

  const int row  = blockIdx.x * TM + wid;   // output H row
  const int kidx = blockIdx.z;              // expert index per token

  if (wid >= TM || row >= H || kidx >= EXPERT_PER_TOKEN)
    return;

  // Load expert ID and weight for single sample
  expert_id = topk_i[kidx];
  expert_w = topk_v[kidx];
  if (expert_id < 0 || expert_w == 0.f) return;

  float acc = 0.f;

  // K loop over IM, loading gate_up_topk for single sample
  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    // Load data for single sample
    const float* __restrict__ xb =
        gate_up_topk + ((size_t)kidx * (size_t)IM + (size_t)k_base);

    // vectorized load to shared
    const int vec4 = (k_size >> 2);
    float4* __restrict__ s4 = reinterpret_cast<float4*>(lds_x);
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
    for (int v = tid; v < vec4; v += BLOCK_SIZE) { s4[v] = x4[v]; }
    for (int k = (vec4 << 2) + tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = xb[k];
    }
    __syncthreads();

    // Compute for single sample
    // weight row: w[l, expert, row, k_base: ]
    const size_t base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)H * (size_t)IM;
    const bf16_t* __restrict__ w_row = w_mlp2_all + base + (size_t)row * (size_t)IM + (size_t)k_base;

    // vector compute
    const uint2* __restrict__ wq = reinterpret_cast<const uint2*>(w_row);
    const int vec4k = (k_size >> 2);
    for (int v = lane; v < vec4k; v += WF_SIZE) {
      const float4 wv = bf16quad_to_float4(wq[v]);
      const float4 xv = *reinterpret_cast<const float4*>(&lds_x[v << 2]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    // tail
    for (int k = (vec4k << 2) + lane; k < k_size; k += WF_SIZE) {
      uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
      union { uint32_t u; float f; } cvt; cvt.u = u;
      acc = fmaf(cvt.f, lds_x[k], acc);
    }
    __syncthreads();
  }

  // reduce and write for single sample (1 atomic per row per expert)
  float acc_sum = warp_reduce_sum(acc);
  if (lane == 0) {
    const size_t b_mlp2_base = ((size_t)l_layer * (size_t)E + (size_t)expert_id) * (size_t)H;
    float out = acc_sum + b_mlp2_all[b_mlp2_base + (size_t)row];
    float contrib = out * expert_w;
    atomicAdd(e_agg + (size_t)row, contrib);
  }
}
