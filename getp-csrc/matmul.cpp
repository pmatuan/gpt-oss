#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>

// ============================ Configuration ============================
#define TK 512           // Tile size for K dimension - optimized for cache
#define TM 8             // Threads per warp handling multiple output rows
#define BLOCK_SIZE 512   // Threads per block
#define LDS_PAD 16       // Padding to avoid bank conflicts
#define WF_SIZE 64       // AMD wavefront size
#define VEC_SIZE 8       // Vectorization width

// ============================ Utility Functions ============================

/**
 * Convert packed bf16 pair to two float32 values
 * Optimized for minimal register usage
 */
__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1) {
  union { uint32_t u; float f; } a, b;
  a.u = (u & 0x0000FFFFu) << 16; // Lower bf16 -> fp32
  b.u = (u & 0xFFFF0000u);       // Upper bf16 -> fp32
  f0 = a.f;
  f1 = b.f;
}

/**
 * Convert packed bf16 quad (uint2) to four float32 values
 * High-throughput conversion for vectorized operations
 */
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u) {
  float4 result;
  bf16pair_to_float2(u.x, result.x, result.y);
  bf16pair_to_float2(u.y, result.z, result.w);
  return result;
}

/**
 * Warp-level reduction using shuffle operations
 * Reduces a single float value across all lanes in a warp
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WF_SIZE >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, WF_SIZE);
  }
  return val;
}

// ============================ Vectorized Compute Kernels ============================

/**
 * Optimized bf16 matrix-vector multiplication for a tile
 * Uses quad-based vectorization for maximum throughput
 */
__device__ __forceinline__ float compute_matvec_bf16_tile(
    const bf16_t* __restrict__ w_row,
    const float* __restrict__ lds_x,
    int k_size,
    int lane) {
  
  const uint2* __restrict__ w64 = reinterpret_cast<const uint2*>(w_row);
  const int quads = k_size >> 2;
  const int vec_quads = (quads / 2) * 2; // Process 2 quads per iteration
  
  float acc = 0.f;
  
  // Main vectorized loop - process 8 elements per iteration
  for (int q = lane * 2; q < vec_quads; q += WF_SIZE * 2) {
    const int k_elem = q << 2;
    
    // Load data
    float4 x0 = *reinterpret_cast<const float4*>(&lds_x[k_elem + 0]);
    float4 x1 = *reinterpret_cast<const float4*>(&lds_x[k_elem + 4]);
    float4 w0 = bf16quad_to_float4(w64[q + 0]);
    float4 w1 = bf16quad_to_float4(w64[q + 1]);
    
    // Compute
    acc = fmaf(w0.x, x0.x, acc);
    acc = fmaf(w0.y, x0.y, acc);
    acc = fmaf(w0.z, x0.z, acc);
    acc = fmaf(w0.w, x0.w, acc);
    acc = fmaf(w1.x, x1.x, acc);
    acc = fmaf(w1.y, x1.y, acc);
    acc = fmaf(w1.z, x1.z, acc);
    acc = fmaf(w1.w, x1.w, acc);
  }
  
  // Handle remaining quads
  for (int q = vec_quads + lane; q < quads; q += WF_SIZE) {
    const int k_elem = q << 2;
    float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k_elem]);
    float4 w_vals = bf16quad_to_float4(w64[q]);
    acc = fmaf(w_vals.x, x_vals.x, acc);
    acc = fmaf(w_vals.y, x_vals.y, acc);
    acc = fmaf(w_vals.z, x_vals.z, acc);
    acc = fmaf(w_vals.w, x_vals.w, acc);
  }
  
  // Handle remaining pairs and single elements
  const uint32_t* __restrict__ w32 = reinterpret_cast<const uint32_t*>(w_row);
  const int pairs = k_size >> 1;
  
  for (int p = (quads << 1) + lane; p < pairs; p += WF_SIZE) {
    const int k_elem = p << 1;
    uint32_t up = w32[p];
    float w0, w1;
    bf16pair_to_float2(up, w0, w1);
    acc = fmaf(w0, lds_x[k_elem + 0], acc);
    if (k_elem + 1 < k_size)
      acc = fmaf(w1, lds_x[k_elem + 1], acc);
  }
  
  // Handle odd element
  if (k_size & 1) {
    const int k = k_size - 1;
    if (k % WF_SIZE == lane) {
      uint32_t u = ((uint32_t)(*reinterpret_cast<const uint16_t*>(&w_row[k]))) << 16;
      union { uint32_t u; float f; } cvt;
      cvt.u = u;
      acc = fmaf(cvt.f, lds_x[k], acc);
    }
  }
  
  return acc;
}

/**
 * Optimized fp32 matrix-vector multiplication for a tile
 * Uses float4 vectorization for maximum bandwidth
 */
__device__ __forceinline__ float compute_matvec_fp32_tile(
    const float* __restrict__ w_row,
    const float* __restrict__ lds_x,
    int k_size,
    int lane) {
  
  const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w_row);
  int vec4 = (k_size / 4) * 4;
  float acc = 0.f;
  
  // Main vectorized loop
  for (int k = lane * 4; k < vec4; k += WF_SIZE * 4) {
    float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
    float4 w_vals = w4[k / 4];
    acc = fmaf(w_vals.x, x_vals.x, acc);
    acc = fmaf(w_vals.y, x_vals.y, acc);
    acc = fmaf(w_vals.z, x_vals.z, acc);
    acc = fmaf(w_vals.w, x_vals.w, acc);
  }
  
  // Handle remaining elements
  for (int k = vec4 + lane; k < k_size; k += WF_SIZE) {
    acc = fmaf(w_row[k], lds_x[k], acc);
  }
  
  return acc;
}

// ============================ Main Kernel Functions ============================

/**
 * Batched matrix multiplication with bias: Y = X @ W^T + B
 * Supports both bf16 and fp32 weights with template specialization
 * 
 * @param y: Output tensor [batch_size, d]
 * @param x: Input tensor [batch_size, n] 
 * @param w: Weight matrix [d, n]
 * @param b: Bias vector [d]
 * @param pos: Position mask [batch_size] - negative values skip computation
 * @param n: Input dimension
 * @param d: Output dimension  
 * @param batch_size: Batch size
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void matmul_bias_batch_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  // Shared memory for input tile caching
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  // Thread and warp identifiers
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_idx = blockIdx.y;
  
  // Early exit conditions
  if (batch_idx >= batch_size || wid >= TM || row >= d) return;
  if (pos[batch_idx] < 0) return;

  // Batch-specific pointers
  const float* x_batch = x + (size_t)batch_idx * n;
  float* y_batch = y + (size_t)batch_idx * d;

  float acc_total = 0.0f;

  // Process input in tiles
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);

    // Cooperative loading of input tile into shared memory
    for (int k = tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = x_batch[k_base + k];
    }
    __syncthreads();

    // Get weight row pointer for current tile
    const T* __restrict__ w_row = w + (size_t)row * n + k_base;

    // Compute matrix-vector product for this tile
    float tile_acc;
    if constexpr (std::is_same_v<T, bf16_t>) {
      tile_acc = compute_matvec_bf16_tile(
          reinterpret_cast<const bf16_t*>(w_row), lds_x, k_size, lane);
    } else {
      tile_acc = compute_matvec_fp32_tile(
          reinterpret_cast<const float*>(w_row), lds_x, k_size, lane);
    }

    // Warp-level reduction
    tile_acc = warp_reduce_sum(tile_acc);
    
    if (lane == 0) {
      acc_total += tile_acc;
    }
    
    __syncthreads();
  }

  // Write final result with bias
  if (lane == 0) {
    y_batch[row] = acc_total + b[row];
  }
}

/**
 * Batched MLP1 (Gate-Up) kernel with SwiGLU activation
 * Computes gate and up projections simultaneously for efficiency
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__ 
void mlp1_fused_batch_kernel(
    float* __restrict__ gate_up,
    const float* __restrict__ x,
    const T* __restrict__ w_mlp1_all,
    const float* __restrict__ b_mlp1_all,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, int experts_per_token) {
  
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int i = blockIdx.x * TM + wid;
  const int batch_idx = blockIdx.y;
  
  if (batch_idx >= batch_size || pos[batch_idx] < 0) return;
  if (wid >= TM || i >= IM) return;

  // Batch-specific pointers
  const float* x_batch = x + (size_t)batch_idx * H;
  float* gate_up_batch = gate_up + (size_t)batch_idx * IM;
  const int* topk_i_batch = topk_i + (size_t)batch_idx * experts_per_token;
  const int expert_id = topk_i_batch[k_index];

  float acc_gate_total = 0.0f;
  float acc_up_total = 0.0f;

  // Calculate weight offsets for current expert
  const size_t base_offset = ((size_t)l_layer * E + expert_id) * (2 * IM);
  const size_t gate_offset = (base_offset + (2 * i + 0)) * H;
  const size_t up_offset = (base_offset + (2 * i + 1)) * H;

  // Process in tiles
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load input tile
    for (int k = tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = x_batch[k_base + k];
    }
    __syncthreads();

    const T* __restrict__ w_gate = w_mlp1_all + gate_offset + k_base;
    const T* __restrict__ w_up = w_mlp1_all + up_offset + k_base;

    float acc_gate = 0.0f, acc_up = 0.0f;

    if constexpr (std::is_same_v<T, bf16_t>) {
      // Vectorized bf16 processing for gate and up projections
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
        
        // Vectorized bf16 loads and conversions
        float4 wg_vals = {(float)w_gate[k], (float)w_gate[k+1], 
                         (float)w_gate[k+2], (float)w_gate[k+3]};
        float4 wu_vals = {(float)w_up[k], (float)w_up[k+1],
                         (float)w_up[k+2], (float)w_up[k+3]};

        acc_gate = fmaf(wg_vals.x, x_vals.x, acc_gate);
        acc_gate = fmaf(wg_vals.y, x_vals.y, acc_gate);
        acc_gate = fmaf(wg_vals.z, x_vals.z, acc_gate);
        acc_gate = fmaf(wg_vals.w, x_vals.w, acc_gate);

        acc_up = fmaf(wu_vals.x, x_vals.x, acc_up);
        acc_up = fmaf(wu_vals.y, x_vals.y, acc_up);
        acc_up = fmaf(wu_vals.z, x_vals.z, acc_up);
        acc_up = fmaf(wu_vals.w, x_vals.w, acc_up);
      }

      // Handle remaining elements
      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = (float)w_gate[k];
        float wuk = (float)w_up[k];
        acc_gate = fmaf(wgk, xv, acc_gate);
        acc_up = fmaf(wuk, xv, acc_up);
      }
    } else {
      // Optimized fp32 path
      const float4* wg4 = reinterpret_cast<const float4*>(w_gate);
      const float4* wu4 = reinterpret_cast<const float4*>(w_up);
      const float* wgf = reinterpret_cast<const float*>(w_gate);
      const float* wuf = reinterpret_cast<const float*>(w_up);
      
      int k_vec4 = (k_size / 4) * 4;
      for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
        float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[k]);
        float4 wg_vals = wg4[k / 4];
        float4 wu_vals = wu4[k / 4];

        acc_gate = fmaf(wg_vals.x, x_vals.x, acc_gate);
        acc_gate = fmaf(wg_vals.y, x_vals.y, acc_gate);
        acc_gate = fmaf(wg_vals.z, x_vals.z, acc_gate);
        acc_gate = fmaf(wg_vals.w, x_vals.w, acc_gate);

        acc_up = fmaf(wu_vals.x, x_vals.x, acc_up);
        acc_up = fmaf(wu_vals.y, x_vals.y, acc_up);
        acc_up = fmaf(wu_vals.z, x_vals.z, acc_up);
        acc_up = fmaf(wu_vals.w, x_vals.w, acc_up);
      }

      for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
        float xv = lds_x[k];
        float wgk = wgf[k];
        float wuk = wuf[k];
        acc_gate = fmaf(wgk, xv, acc_gate);
        acc_up = fmaf(wuk, xv, acc_up);
      }
    }

    // Warp reduction for both accumulators
    acc_gate = warp_reduce_sum(acc_gate);
    acc_up = warp_reduce_sum(acc_up);
    
    if (lane == 0) {
      acc_gate_total += acc_gate;
      acc_up_total += acc_up;
    }
    __syncthreads();
  }

  // Apply bias, clipping, and SwiGLU activation
  if (lane == 0) {
    const size_t bias_base = ((size_t)l_layer * E + expert_id) * (2 * IM);
    float gate = acc_gate_total + b_mlp1_all[bias_base + (2 * i + 0)];
    float up = acc_up_total + b_mlp1_all[bias_base + (2 * i + 1)];

    // Apply clipping
    gate = fminf(fmaxf(gate, -swiglu_limit), swiglu_limit);
    up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);

    // SwiGLU activation: gate * sigmoid(gate) * (up + 1)
    const float alpha = 1.702f;
    gate *= (1.0f / (1.0f + expf(-alpha * gate)));
    gate *= (up + 1.0f);

    gate_up_batch[i] = gate;
  }
}

/**
 * Batched MLP2 (Down projection) kernel with weighted accumulation
 * Performs the final MLP projection with expert mixing weights
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp2_bias_weighted_accum_batch_kernel(
    float* __restrict__ e_agg,
    const float* __restrict__ gate_up,
    const T* __restrict__ w_mlp2_all,
    const float* __restrict__ b_mlp2_all,
    const int* __restrict__ topk_i,
    const float* __restrict__ topk_v,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int IM, int H,
    int batch_size, int experts_per_token) {
  
  __shared__ float lds_x[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_idx = blockIdx.y;
  
  if (batch_idx >= batch_size || pos[batch_idx] < 0) return;
  if (wid >= TM || row >= H) return;

  const float* gate_up_batch = gate_up + (size_t)batch_idx * IM;
  float* e_agg_batch = e_agg + (size_t)batch_idx * H;
  const int* topk_i_batch = topk_i + (size_t)batch_idx * experts_per_token;
  const float* topk_v_batch = topk_v + (size_t)batch_idx * experts_per_token;

  const int expert_id = topk_i_batch[k_index];
  const float expert_weight = topk_v_batch[k_index];

  float acc_total = 0.0f;

  // Calculate weight offset for current expert and output row
  const size_t base_offset = ((size_t)l_layer * E + expert_id) * H * IM;
  const T* __restrict__ w_row_base = w_mlp2_all + (size_t)row * IM + base_offset;

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    // Load gate_up tile
    for (int k = tid; k < k_size; k += BLOCK_SIZE) {
      lds_x[k] = gate_up_batch[k_base + k];
    }
    __syncthreads();

    const T* __restrict__ w_row = w_row_base + k_base;
    float acc;

    if constexpr (std::is_same_v<T, bf16_t>) {
      acc = compute_matvec_bf16_tile(
          reinterpret_cast<const bf16_t*>(w_row), lds_x, k_size, lane);
    } else {
      acc = compute_matvec_fp32_tile(
          reinterpret_cast<const float*>(w_row), lds_x, k_size, lane);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      acc_total += acc;
    }
    __syncthreads();
  }

  // Apply bias and weighted accumulation
  if (lane == 0) {
    const size_t bias_base = ((size_t)l_layer * E + expert_id) * H;
    float result = acc_total + b_mlp2_all[bias_base + row];
    atomicAdd(&e_agg_batch[row], result * expert_weight);
  }
}

/**
 * Compute per-sample RMS normalization factor
 * Computes inv_rms = rsqrt(mean(x^2) + eps) for each batch sample
 */
__global__ void compute_inv_rms_batch_kernel(
    float* __restrict__ out_inv,
    const float* __restrict__ x,
    const int* __restrict__ pos,
    int H, int batch_size) {
  
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) return;
  
  if (pos[batch_idx] < 0) {
    if (threadIdx.x == 0) out_inv[batch_idx] = 0.0f;
    return;
  }

  const float* __restrict__ x_batch = x + (size_t)batch_idx * H;
  
  // Compute sum of squares
  float sum = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = x_batch[i];
    sum = fmaf(v, v, sum);
  }
  
  // Warp-level reduction
  sum = warp_reduce_sum(sum);
  
  // Block-level reduction using shared memory
  __shared__ float warp_sums[1024 / WF_SIZE];
  const int lane = threadIdx.x & (WF_SIZE - 1);
  const int warp_id = threadIdx.x >> 6;
  
  if (lane == 0) warp_sums[warp_id] = sum;
  __syncthreads();
  
  // Final reduction
  float total = 0.f;
  if (warp_id == 0) {
    const int num_warps = blockDim.x / WF_SIZE;
    total = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.f;
    total = warp_reduce_sum(total);
    
    if (lane == 0) {
      float mean_sq = total / (float)H;
      out_inv[batch_idx] = rsqrtf(mean_sq + 1e-5f);
    }
  }
}

/**
 * Fused RMSNorm + MatMul kernel for maximum efficiency
 * Combines normalization and matrix multiplication in a single pass
 * Uses double buffering for optimal memory bandwidth utilization
 */
template <typename T>
__launch_bounds__(1024, 1) __global__ 
void fused_rmsnorm_matmul_batch_kernel(
    float* __restrict__ y,              // [B, V]
    const float* __restrict__ x,        // [B, H]
    const T* __restrict__ w,            // [V, H]
    const float* __restrict__ rms_w,    // [H]
    const int* __restrict__ pos,        // [B]
    const float* __restrict__ inv_rms,  // [B]
    int H, int V, int batch_size) {
  
  // Thread identifiers
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int warp_id = tid >> 6;
  const int warps_per_block = blockDim.x / WF_SIZE;
  
  const int row_block = blockIdx.x;
  const int row = row_block * warps_per_block + warp_id;
  const int batch_idx = blockIdx.y;

  if (batch_idx >= batch_size || pos[batch_idx] < 0) return;
  if (warp_id >= warps_per_block || row >= V) return;

  // Batch-specific pointers
  const float* __restrict__ x_batch = x + (size_t)batch_idx * H;
  float* __restrict__ y_batch = y + (size_t)batch_idx * V;
  const float inv = inv_rms[batch_idx];

  // Double-buffered shared memory for optimal throughput
  constexpr int TK_LOCAL = 512;
  __shared__ __align__(16) float lds_x0[TK_LOCAL + LDS_PAD];
  __shared__ __align__(16) float lds_x1[TK_LOCAL + LDS_PAD];

  float acc_total = 0.f;

  // Preload first tile
  int k_base = 0;
  int current_buffer = 0;
  
  {
    const int k_size = min(TK_LOCAL, H - k_base);
    for (int k = tid; k < k_size; k += blockDim.x) {
      float normalized_val = x_batch[k_base + k] * inv * rms_w[k_base + k];
      lds_x0[k] = normalized_val;
    }
    __syncthreads();
  }

  // Main processing loop with double buffering
  while (k_base < H) {
    const int k_size = min(TK_LOCAL, H - k_base);
    const float* __restrict__ lds_current = (current_buffer == 0) ? lds_x0 : lds_x1;

    // Get weight row pointer
    const T* __restrict__ w_row = w + (size_t)row * H + k_base;

    // Compute matrix-vector product for current tile
    float tile_acc;
    if constexpr (std::is_same_v<T, bf16_t>) {
      tile_acc = compute_matvec_bf16_tile(
          reinterpret_cast<const bf16_t*>(w_row), lds_current, k_size, lane);
    } else {
      tile_acc = compute_matvec_fp32_tile(
          reinterpret_cast<const float*>(w_row), lds_current, k_size, lane);
    }

    // Warp reduction
    tile_acc = warp_reduce_sum(tile_acc);
    if (lane == 0) acc_total += tile_acc;
    __syncthreads();

    // Prefetch next tile while computing
    k_base += k_size;
    current_buffer ^= 1;
    
    if (k_base < H) {
      const int next_k_size = min(TK_LOCAL, H - k_base);
      float* __restrict__ lds_next = (current_buffer == 0) ? lds_x0 : lds_x1;
      
      for (int k = tid; k < next_k_size; k += blockDim.x) {
        float normalized_val = x_batch[k_base + k] * inv * rms_w[k_base + k];
        lds_next[k] = normalized_val;
      }
      __syncthreads();
    }
  }

  // Write final result
  if (lane == 0) {
    y_batch[row] = acc_total;
  }
}

/**
 * Fused RMSNorm + MatMul + Bias kernel for QKV projections
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_rmsnorm_matmul_bias_batch_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ rms_w,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];
  __shared__ float shared_rms_sum;

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_idx = blockIdx.y;
  
  if (batch_idx >= batch_size || wid >= TM || row >= d) return;
  if (pos[batch_idx] < 0) return;

  const float* x_batch = x + (size_t)batch_idx * n;
  float* y_batch = y + (size_t)batch_idx * d;

  // Compute RMS normalization factor
  if (tid == 0) shared_rms_sum = 0.0f;
  __syncthreads();

  float sum = 0.0f;
  for (int i = tid; i < n; i += BLOCK_SIZE) {
    float v = x_batch[i];
    sum = fmaf(v, v, sum);
  }
  
  sum = warp_reduce_sum(sum);
  if (lane == 0) atomicAdd(&shared_rms_sum, sum);
  __syncthreads();

  const float rms_factor = rsqrtf(shared_rms_sum / n + 1e-5f);
  float acc_total = 0.0f;

  // Process in tiles
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);
    
    // Load and normalize input tile
    for (int k = tid; k < k_size; k += BLOCK_SIZE) {
      float v = x_batch[k_base + k];
      lds_x[k] = v * rms_factor * rms_w[k_base + k];
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    
    float tile_acc;
    if constexpr (std::is_same_v<T, bf16_t>) {
      tile_acc = compute_matvec_bf16_tile(
          reinterpret_cast<const bf16_t*>(w_row), lds_x, k_size, lane);
    } else {
      tile_acc = compute_matvec_fp32_tile(
          reinterpret_cast<const float*>(w_row), lds_x, k_size, lane);
    }

    tile_acc = warp_reduce_sum(tile_acc);
    if (lane == 0) acc_total += tile_acc;
    __syncthreads();
  }
  
  if (lane == 0) {
    y_batch[row] = acc_total + b[row];
  }
}

/**
 * Fused MatMul + Bias + Residual kernel for output projections
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_matmul_bias_residual_batch_kernel(
    float* __restrict__ x,
    const float* __restrict__ y,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  __shared__ __align__(16) float lds_y[TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_idx = blockIdx.y;
  
  if (batch_idx >= batch_size || wid >= TM || row >= d) return;
  if (pos[batch_idx] < 0) return;

  float* x_batch = x + (size_t)batch_idx * d;
  const float* y_batch = y + (size_t)batch_idx * n;

  float acc_total = 0.0f;
  
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);
    
    for (int k = tid; k < k_size; k += BLOCK_SIZE) {
      lds_y[k] = y_batch[k_base + k];
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    
    float tile_acc;
    if constexpr (std::is_same_v<T, bf16_t>) {
      tile_acc = compute_matvec_bf16_tile(
          reinterpret_cast<const bf16_t*>(w_row), lds_y, k_size, lane);
    } else {
      tile_acc = compute_matvec_fp32_tile(
          reinterpret_cast<const float*>(w_row), lds_y, k_size, lane);
    }

    tile_acc = warp_reduce_sum(tile_acc);
    if (lane == 0) acc_total += tile_acc;
    __syncthreads();
  }
  
  if (lane == 0) {
    x_batch[row] = x_batch[row] + (acc_total + b[row]);
  }
}
