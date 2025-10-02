#ifndef GETP_UTILITY_H
#define GETP_UTILITY_H

#include "../common/defines.h"

// GPU Memory Debugging
void debug_print_gpu_memory(const char *tag, int device_id = 0);

__device__ __forceinline__ void bf16pair_to_float2(uint32_t u, float &f0, float &f1);
__device__ __forceinline__ float4 bf16quad_to_float4(uint2 u);
__device__ __forceinline__ float warp_reduce_sum(float v);

// Embedding and Data Movement Kernels
__global__ void copy_embedding_bf16_batch_kernel(bf16_t *dst, const bf16_t *src,
                                                 const int *tokens,
                                                 int batch_size,
                                                 int hidden_dim);

// Normalization Kernels
__global__ void rmsnorm_batch_kernel(bf16_t *out, const bf16_t *x,
                                     const bf16_t *weight, int dim,
                                     const int *pos);

// Utility Operations
__global__ void residual_add_batch_kernel(bf16_t *x, const float *residual,
                                          int dim, int batch_size,
                                          const int *pos);

__global__ void residual_add_batch_kernel_bf16(bf16_t *x,
                                               const bf16_t *residual,
                                               int dim, int batch_size,
                                               const int *pos);

__global__ void fused_split_rope_scatter_qkv_batch_kernel(
    bf16_t* __restrict__ q_out,
    bf16_t* __restrict__ key_cache,
    bf16_t* __restrict__ value_cache,
    const bf16_t* __restrict__ qkv,    // [B, Hq*D + 2*Hk*D]
    const int* __restrict__ pos,       // [B]
    // model params
    int Hq, int Hk, int D,
    // RoPE params
    const float* __restrict__ rope_inv_freq,
    float rope_concentration,
    // cache params
    int layer_idx,
    const uint32_t *__restrict__ layer_offsets,
    const int *__restrict__ layer_capacity,
    uint32_t kv_batch_stride,
    int batch_size);

// Expert/MoE Utility Kernels
__global__ void fused_topk_softmax_batch_kernel(
    float *topk_v, int *topk_i, const float *scores, int E, int K,
    int batch_size, const int *pos);

__global__ void argmax_batch_kernel(const float *logits, int *out_indices,
                                    int vocab_size, int batch_size,
                                    const int *pos);

// Data Type Conversion Utilities
void copy_fp32_to_bf16_device(const float *src, size_t n, bf16_t *dst,
                              int n_streams, size_t chunk_bytes);

size_t matmul_packed_elems(int rows, int cols);
void pack_fp32_to_bf16_matmul(const float *src, int rows, int cols,
                              bf16_t *dst);

// Accumulate compact partials [cnt, H] into dest [B, H] using batch_ids[cnt]
__global__ void accumulate_partials_kernel(
    float* __restrict__ dest,           // [B, H]
    const float* __restrict__ src,      // [cnt, H]
    const int* __restrict__ batch_ids,  // [cnt]
    int H,
    int cnt);

// ================= Device-side bucketization & packing (MoE routing) ================
// Count per-local-expert assignments for a specific owner at layer l.
// e2lid_owner_l: [E] mapping global expert id -> local id on this owner for layer l, or -1.
__global__ void route_count_owner_kernel(
    int* __restrict__ expert_counts,   // [E_local]
    const int* __restrict__ topk_i,    // [B*K]
    const int* __restrict__ pos,       // [B]
    const int* __restrict__ e2lid_owner_l, // [E]
    int B,
    int K,
    int E);

// Simple exclusive scan for small arrays (one block). counts -> offsets (size n+1).
__global__ void exclusive_scan_small_kernel(
    const int* __restrict__ counts, // [n]
    int* __restrict__ offsets,      // [n+1]
    int n);

// Build assignments and compact batch maps per owner.
// Produces:
//  - b2local[B] initialized to -1; local2b filled for 0..B_local-1
//  - owner_B (single int) final B_local
//  - assignment_batches/slots filled using expert_offsets and per-expert write counters
__global__ void route_pack_owner_kernel(
    int* __restrict__ b2local,          // [B], init -1
    int* __restrict__ local2b,          // [B]
    int* __restrict__ owner_B,          // [1]
    const int* __restrict__ expert_offsets, // [E_local+1]
    int* __restrict__ expert_writes,    // [E_local], init 0
    int* __restrict__ assignment_batches, // [total_assignments]
    int* __restrict__ assignment_slots,   // [total_assignments]
    const int* __restrict__ topk_i,     // [B*K]
    const int* __restrict__ pos,        // [B]
    const int* __restrict__ e2lid_owner_l, // [E]
    int B,
    int K,
    int E);

// Pack rows x[b,:] for lb=b2local[b] >=0 into dst[lb,:].
__global__ void pack_rows_owner_kernel(
    bf16_t* __restrict__ dst,          // [B_local, H]
    const bf16_t* __restrict__ src,    // [B, H]
    const int* __restrict__ b2local,   // [B]
    int B,
    int H);

// Pack pos[B_local] and topk_v[B_local*K] using b2local map.
__global__ void pack_meta_owner_kernel(
    int* __restrict__ pos_owner,         // [B_local]
    float* __restrict__ topk_v_owner,    // [B_local*K]
    const int* __restrict__ b2local,     // [B]
    const int* __restrict__ pos,         // [B]
    const float* __restrict__ topk_v,    // [B*K]
    int B,
    int K);

// Matmul Helper Functions
__device__ __forceinline__ s16x4 load_bf16x4(const uint16_t* src, int valid_elems);

// Expert Assignment Kernels
__global__ void count_expert_assignments_kernel(
    int* __restrict__ counts,
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    int batch_size,
    int experts_per_token,
    int E);

__global__ void build_expert_assignments_kernel(
    const int* __restrict__ topk_i,
    const int* __restrict__ pos,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_counters,
    uint16_t* __restrict__ assignment_batches,
    uint8_t* __restrict__ assignment_slots,
    int batch_size,
    int experts_per_token,
    int E);

// Activation Functions
__device__ __forceinline__ float clamp_with_limit(float v, float limit);
__device__ __forceinline__ float swiglu_fused(float gate, float up, float limit);

#endif // GETP_UTILITY_H
