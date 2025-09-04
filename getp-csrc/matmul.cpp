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
 * GEMM version: Matrix-matrix multiplication with bias: Y = X @ W^T + B
 * Processes multiple batch items simultaneously for better parallelization
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
void matmul_bias_gemm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  // Shared memory for input tiles from multiple batch items
  __shared__ __align__(16) float lds_x[TK + LDS_PAD];

  // Thread and warp identifiers
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  
  // Early exit conditions
  if (wid >= TM || row >= d) return;

  // Process multiple batch items per block
  const int batch_start = blockIdx.y * TM;
  const int batch_end = min(batch_start + TM, batch_size);
  
  for (int batch_idx = batch_start; batch_idx < batch_end; batch_idx++) {
    if (pos[batch_idx] < 0) continue;
    
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
}

/**
 * GEMM version of MLP1 (Gate-Up) kernel with SwiGLU activation
 * Processes multiple batch items simultaneously for better GPU utilization
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__ 
void mlp1_fused_gemm_kernel(
    float* __restrict__ gate_up,
    const float* __restrict__ x,
    const T* __restrict__ w_mlp1_all,
    const float* __restrict__ b_mlp1_all,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, int experts_per_token) {
  
  constexpr int BATCH_TILE = 2; // Process 2 batch items per block
  __shared__ float lds_x[BATCH_TILE][TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int i = blockIdx.x * TM + wid;
  const int batch_block_start = blockIdx.y * BATCH_TILE;
  
  if (wid >= TM || i >= IM) return;

  // Accumulation for multiple batch items
  float acc_gate[BATCH_TILE], acc_up[BATCH_TILE];
  int expert_ids[BATCH_TILE];
  
  #pragma unroll
  for (int b = 0; b < BATCH_TILE; b++) {
    acc_gate[b] = 0.0f;
    acc_up[b] = 0.0f;
    expert_ids[b] = -1;
    
    const int batch_idx = batch_block_start + b;
    if (batch_idx < batch_size && pos[batch_idx] >= 0) {
      const int* topk_i_batch = topk_i + (size_t)batch_idx * experts_per_token;
      expert_ids[b] = topk_i_batch[k_index];
    }
  }

  // Process in tiles
  for (int k_base = 0; k_base < H; k_base += TK) {
    const int k_size = min(TK, H - k_base);

    // Load input tiles for multiple batch items
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        const float* x_batch = x + (size_t)batch_idx * H;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_x[b][k] = x_batch[k_base + k];
        }
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_x[b][k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Process each batch item
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      if (expert_ids[b] < 0) continue;
      
      const size_t base_offset = ((size_t)l_layer * E + expert_ids[b]) * (2 * IM);
      const size_t gate_offset = (base_offset + (2 * i + 0)) * H + k_base;
      const size_t up_offset = (base_offset + (2 * i + 1)) * H + k_base;

      const T* __restrict__ w_gate = w_mlp1_all + gate_offset;
      const T* __restrict__ w_up = w_mlp1_all + up_offset;

      float acc_gate_tile = 0.0f, acc_up_tile = 0.0f;

      if constexpr (std::is_same_v<T, bf16_t>) {
        int k_vec4 = (k_size / 4) * 4;
        for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[b][k]);
          
          float4 wg_vals = {(float)w_gate[k], (float)w_gate[k+1], 
                           (float)w_gate[k+2], (float)w_gate[k+3]};
          float4 wu_vals = {(float)w_up[k], (float)w_up[k+1],
                           (float)w_up[k+2], (float)w_up[k+3]};

          acc_gate_tile = fmaf(wg_vals.x, x_vals.x, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.y, x_vals.y, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.z, x_vals.z, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.w, x_vals.w, acc_gate_tile);

          acc_up_tile = fmaf(wu_vals.x, x_vals.x, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.y, x_vals.y, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.z, x_vals.z, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.w, x_vals.w, acc_up_tile);
        }

        for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
          float xv = lds_x[b][k];
          acc_gate_tile = fmaf((float)w_gate[k], xv, acc_gate_tile);
          acc_up_tile = fmaf((float)w_up[k], xv, acc_up_tile);
        }
      } else {
        const float4* wg4 = reinterpret_cast<const float4*>(w_gate);
        const float4* wu4 = reinterpret_cast<const float4*>(w_up);
        
        int k_vec4 = (k_size / 4) * 4;
        for (int k = lane * 4; k < k_vec4; k += WF_SIZE * 4) {
          float4 x_vals = *reinterpret_cast<const float4*>(&lds_x[b][k]);
          float4 wg_vals = wg4[k / 4];
          float4 wu_vals = wu4[k / 4];

          acc_gate_tile = fmaf(wg_vals.x, x_vals.x, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.y, x_vals.y, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.z, x_vals.z, acc_gate_tile);
          acc_gate_tile = fmaf(wg_vals.w, x_vals.w, acc_gate_tile);

          acc_up_tile = fmaf(wu_vals.x, x_vals.x, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.y, x_vals.y, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.z, x_vals.z, acc_up_tile);
          acc_up_tile = fmaf(wu_vals.w, x_vals.w, acc_up_tile);
        }

        const float* wgf = reinterpret_cast<const float*>(w_gate);
        const float* wuf = reinterpret_cast<const float*>(w_up);
        for (int k = k_vec4 + lane; k < k_size; k += WF_SIZE) {
          float xv = lds_x[b][k];
          acc_gate_tile = fmaf(wgf[k], xv, acc_gate_tile);
          acc_up_tile = fmaf(wuf[k], xv, acc_up_tile);
        }
      }

      // Warp reduction
      acc_gate_tile = warp_reduce_sum(acc_gate_tile);
      acc_up_tile = warp_reduce_sum(acc_up_tile);
      
      if (lane == 0) {
        acc_gate[b] += acc_gate_tile;
        acc_up[b] += acc_up_tile;
      }
    }
    __syncthreads();
  }

  // Apply bias, clipping, and SwiGLU activation for all batch items
  if (lane == 0) {
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0 && expert_ids[b] >= 0) {
        float* gate_up_batch = gate_up + (size_t)batch_idx * IM;
        
        const size_t bias_base = ((size_t)l_layer * E + expert_ids[b]) * (2 * IM);
        float gate = acc_gate[b] + b_mlp1_all[bias_base + (2 * i + 0)];
        float up = acc_up[b] + b_mlp1_all[bias_base + (2 * i + 1)];

        // Apply clipping
        gate = fminf(fmaxf(gate, -swiglu_limit), swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);

        // SwiGLU activation
        const float alpha = 1.702f;
        gate *= (1.0f / (1.0f + expf(-alpha * gate)));
        gate *= (up + 1.0f);

        gate_up_batch[i] = gate;
      }
    }
  }
}

/**
 * GEMM version of MLP2 (Down projection) kernel with weighted accumulation
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void mlp2_bias_weighted_accum_gemm_kernel(
    float* __restrict__ e_agg,
    const float* __restrict__ gate_up,
    const T* __restrict__ w_mlp2_all,
    const float* __restrict__ b_mlp2_all,
    const int* __restrict__ topk_i,
    const float* __restrict__ topk_v,
    const int* __restrict__ pos,
    int k_index, int l_layer, int E, int IM, int H,
    int batch_size, int experts_per_token) {
  
  constexpr int BATCH_TILE = 2;
  __shared__ float lds_x[BATCH_TILE][TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_block_start = blockIdx.y * BATCH_TILE;
  
  if (wid >= TM || row >= H) return;

  float acc_total[BATCH_TILE];
  int expert_ids[BATCH_TILE];
  float expert_weights[BATCH_TILE];
  
  // Initialize accumulators and get expert info
  for (int b = 0; b < BATCH_TILE; b++) {
    const int batch_idx = batch_block_start + b;
    acc_total[b] = 0.0f;
    expert_ids[b] = -1;
    expert_weights[b] = 0.0f;
    
    if (batch_idx < batch_size && pos[batch_idx] >= 0) {
      const int* topk_i_batch = topk_i + (size_t)batch_idx * experts_per_token;
      const float* topk_v_batch = topk_v + (size_t)batch_idx * experts_per_token;
      expert_ids[b] = topk_i_batch[k_index];
      expert_weights[b] = topk_v_batch[k_index];
    }
  }

  for (int k_base = 0; k_base < IM; k_base += TK) {
    const int k_size = min(TK, IM - k_base);

    // Load gate_up tiles for multiple batch items
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        const float* gate_up_batch = gate_up + (size_t)batch_idx * IM;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_x[b][k] = gate_up_batch[k_base + k];
        }
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_x[b][k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Process each batch item
    for (int b = 0; b < BATCH_TILE; b++) {
      if (expert_ids[b] < 0) continue;
      
      const size_t base_offset = ((size_t)l_layer * E + expert_ids[b]) * H * IM;
      const T* __restrict__ w_row = w_mlp2_all + (size_t)row * IM + base_offset + k_base;
      
      float acc;
      if constexpr (std::is_same_v<T, bf16_t>) {
        acc = compute_matvec_bf16_tile(
            reinterpret_cast<const bf16_t*>(w_row), lds_x[b], k_size, lane);
      } else {
        acc = compute_matvec_fp32_tile(
            reinterpret_cast<const float*>(w_row), lds_x[b], k_size, lane);
      }

      acc = warp_reduce_sum(acc);
      if (lane == 0) {
        acc_total[b] += acc;
      }
    }
    __syncthreads();
  }

  // Apply bias and weighted accumulation
  if (lane == 0) {
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0 && expert_ids[b] >= 0) {
        float* e_agg_batch = e_agg + (size_t)batch_idx * H;
        const size_t bias_base = ((size_t)l_layer * E + expert_ids[b]) * H;
        float result = acc_total[b] + b_mlp2_all[bias_base + row];
        atomicAdd(&e_agg_batch[row], result * expert_weights[b]);
      }
    }
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
 * Optimized GEMM version of fused RMSNorm + MatMul kernel
 * Uses true matrix-matrix multiplication for multiple batch items simultaneously
 * Each block processes multiple batch items and output rows in parallel
 */
template <typename T>
__launch_bounds__(1024, 1) __global__ 
void fused_rmsnorm_matmul_gemm_kernel(
    float* __restrict__ y,              // [B, V]
    const float* __restrict__ x,        // [B, H]
    const T* __restrict__ w,            // [V, H]
    const float* __restrict__ rms_w,    // [H]
    const int* __restrict__ pos,        // [B]
    const float* __restrict__ inv_rms,  // [B]
    int H, int V, int batch_size) {
  
  // Shared memory for multiple input vectors and weight tiles
  constexpr int TK_LOCAL = 512;
  constexpr int BATCH_TILE = 4; // Process 4 batch items per block
  
  __shared__ __align__(16) float lds_x[BATCH_TILE][TK_LOCAL + LDS_PAD];
  
  // Thread identifiers
  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int warp_id = tid >> 6;
  const int warps_per_block = blockDim.x / WF_SIZE;
  
  const int row_block = blockIdx.x;
  const int row = row_block * warps_per_block + warp_id;
  const int batch_block_start = blockIdx.y * BATCH_TILE;

  if (warp_id >= warps_per_block || row >= V) return;

  // Accumulation for multiple batch items
  float acc_batch[BATCH_TILE];
  #pragma unroll
  for (int b = 0; b < BATCH_TILE; b++) {
    acc_batch[b] = 0.0f;
  }

  // Main processing loop
  for (int k_base = 0; k_base < H; k_base += TK_LOCAL) {
    const int k_size = min(TK_LOCAL, H - k_base);
    
    // Cooperatively load multiple batch inputs
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        const float* __restrict__ x_batch = x + (size_t)batch_idx * H;
        const float inv = inv_rms[batch_idx];
        
        for (int k = tid; k < k_size; k += blockDim.x) {
          float normalized_val = x_batch[k_base + k] * inv * rms_w[k_base + k];
          lds_x[b][k] = normalized_val;
        }
      } else {
        // Fill with zeros for invalid batch items
        for (int k = tid; k < k_size; k += blockDim.x) {
          lds_x[b][k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Get weight row pointer
    const T* __restrict__ w_row = w + (size_t)row * H + k_base;

    // Compute matrix-matrix product for current tile
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      float tile_acc;
      if constexpr (std::is_same_v<T, bf16_t>) {
        tile_acc = compute_matvec_bf16_tile(
            reinterpret_cast<const bf16_t*>(w_row), lds_x[b], k_size, lane);
      } else {
        tile_acc = compute_matvec_fp32_tile(
            reinterpret_cast<const float*>(w_row), lds_x[b], k_size, lane);
      }
      
      // Warp reduction
      tile_acc = warp_reduce_sum(tile_acc);
      if (lane == 0) acc_batch[b] += tile_acc;
    }
    __syncthreads();
  }

  // Write final results for all valid batch items
  if (lane == 0) {
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
      const int batch_idx = batch_block_start + b;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        float* __restrict__ y_batch = y + (size_t)batch_idx * V;
        y_batch[row] = acc_batch[b];
      }
    }
  }
}

/**
 * GEMM version of fused RMSNorm + MatMul + Bias kernel for QKV projections
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_rmsnorm_matmul_bias_gemm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ rms_w,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  constexpr int BATCH_TILE = 2;
  __shared__ __align__(16) float lds_x[BATCH_TILE][TK + LDS_PAD];
  __shared__ float shared_rms_sum[BATCH_TILE];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_block_start = blockIdx.y * BATCH_TILE;
  
  if (wid >= TM || row >= d) return;

  // Compute RMS for multiple batch items
  for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
    const int batch_idx = batch_block_start + batch_i;
    if (tid == 0) shared_rms_sum[batch_i] = 0.0f;
  }
  __syncthreads();

  for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
    const int batch_idx = batch_block_start + batch_i;
    if (batch_idx < batch_size && pos[batch_idx] >= 0) {
      const float* x_batch = x + (size_t)batch_idx * n;
      
      float sum = 0.0f;
      for (int i = tid; i < n; i += BLOCK_SIZE) {
        float v = x_batch[i];
        sum = fmaf(v, v, sum);
      }
      
      sum = warp_reduce_sum(sum);
      if (lane == 0) atomicAdd(&shared_rms_sum[batch_i], sum);
    }
  }
  __syncthreads();

  float rms_factors[BATCH_TILE];
  for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
    const int batch_idx = batch_block_start + batch_i;
    rms_factors[batch_i] = (batch_idx < batch_size && pos[batch_idx] >= 0) ? 
                     rsqrtf(shared_rms_sum[batch_i] / n + 1e-5f) : 0.0f;
  }

  float acc_total[BATCH_TILE];
  for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
    acc_total[batch_i] = 0.0f;
  }

  // Process in tiles
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);
    
    // Load and normalize input tiles for multiple batches
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      const int batch_idx = batch_block_start + batch_i;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        const float* x_batch = x + (size_t)batch_idx * n;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          float v = x_batch[k_base + k];
          lds_x[batch_i][k] = v * rms_factors[batch_i] * rms_w[k_base + k];
        }
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_x[batch_i][k] = 0.0f;
        }
      }
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      float tile_acc;
      if constexpr (std::is_same_v<T, bf16_t>) {
        tile_acc = compute_matvec_bf16_tile(
            reinterpret_cast<const bf16_t*>(w_row), lds_x[batch_i], k_size, lane);
      } else {
        tile_acc = compute_matvec_fp32_tile(
            reinterpret_cast<const float*>(w_row), lds_x[batch_i], k_size, lane);
      }

      tile_acc = warp_reduce_sum(tile_acc);
      if (lane == 0) acc_total[batch_i] += tile_acc;
    }
    __syncthreads();
  }
  
  if (lane == 0) {
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      const int batch_idx = batch_block_start + batch_i;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        float* y_batch = y + (size_t)batch_idx * d;
        y_batch[row] = acc_total[batch_i] + b[row];
      }
    }
  }
}

/**
 * GEMM version of Fused MatMul + Bias + Residual kernel for output projections
 */
template <typename T>
__launch_bounds__(BLOCK_SIZE, 1) __global__
void fused_matmul_bias_residual_gemm_kernel(
    float* __restrict__ x,
    const float* __restrict__ y,
    const T* __restrict__ w,
    const float* __restrict__ b,
    const int* __restrict__ pos,
    int n, int d, int batch_size) {
  
  constexpr int BATCH_TILE = 2;
  __shared__ __align__(16) float lds_y[BATCH_TILE][TK + LDS_PAD];

  const int tid = threadIdx.x;
  const int lane = tid & (WF_SIZE - 1);
  const int wid = tid >> 6;
  const int row = blockIdx.x * TM + wid;
  const int batch_block_start = blockIdx.y * BATCH_TILE;
  
  if (wid >= TM || row >= d) return;

  float acc_total[BATCH_TILE];
  for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
    acc_total[batch_i] = 0.0f;
  }
  
  for (int k_base = 0; k_base < n; k_base += TK) {
    const int k_size = min(TK, n - k_base);
    
    // Load input tiles for multiple batch items
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      const int batch_idx = batch_block_start + batch_i;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        const float* y_batch = y + (size_t)batch_idx * n;
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_y[batch_i][k] = y_batch[k_base + k];
        }
      } else {
        for (int k = tid; k < k_size; k += BLOCK_SIZE) {
          lds_y[batch_i][k] = 0.0f;
        }
      }
    }
    __syncthreads();

    const T* w_row = w + (size_t)row * n + k_base;
    
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      float tile_acc;
      if constexpr (std::is_same_v<T, bf16_t>) {
        tile_acc = compute_matvec_bf16_tile(
            reinterpret_cast<const bf16_t*>(w_row), lds_y[batch_i], k_size, lane);
      } else {
        tile_acc = compute_matvec_fp32_tile(
            reinterpret_cast<const float*>(w_row), lds_y[batch_i], k_size, lane);
      }

      tile_acc = warp_reduce_sum(tile_acc);
      if (lane == 0) acc_total[batch_i] += tile_acc;
    }
    __syncthreads();
  }
  
  if (lane == 0) {
    for (int batch_i = 0; batch_i < BATCH_TILE; batch_i++) {
      const int batch_idx = batch_block_start + batch_i;
      if (batch_idx < batch_size && pos[batch_idx] >= 0) {
        float* x_batch = x + (size_t)batch_idx * d;
        x_batch[row] = x_batch[row] + (acc_total[batch_i] + b[row]);
      }
    }
  }
}
