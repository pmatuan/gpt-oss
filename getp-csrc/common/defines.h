#ifndef GETP_COMMON_DEFINES_H
#define GETP_COMMON_DEFINES_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string>
#include <stdlib.h>
#include <vector>

typedef hip_bfloat16 bf16_t;

// GPU Compute Constants
#define WF_SIZE 64
#define TM 8  
#define BLOCK_SIZE (WF_SIZE * TM)
#define TK 512
#define LDS_PAD 16
#define K_STEP_MATMUL_FLOAT 4
#define EXPERT_PER_TOKEN 4
#define EXPERT_PER_TOKEN_SHIFT 2
#define EXPERT_PER_TOKEN_MASK (EXPERT_PER_TOKEN - 1)
#define MAX_BATCH_SIZE 1024
#define TM_MM 32
#define TN_MM 32
#define MLP_TILE_TOKENS 32
#define MLP_TILE_COLS 32
#define MLP_THREAD_X 16
#define MLP_THREAD_Y 4
#define MATMUL_TILE_COLS 32
#define MATMUL_TILE_K 16
#define MATMUL_CHUNK_K 4
// Specialized tiling for logits GEMM (no-bias) where output dimension is huge
#define MATMUL_GEMM_TILE_COLS 64
#define MATMUL_GEMM_TILE_ROWS 32

using f32x4 = float __attribute__((ext_vector_type(4)));
using s16x4 = short __attribute__((ext_vector_type(4)));

// HIP Error Checking Macro
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = call;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__,        \
              hipGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// GPU Buffer Structures
struct GPUActivationBuffers {
  bf16_t *d_x;
  bf16_t *d_t;
  bf16_t *d_tb;
  float *d_t_fp32;
  float *d_router_score, *d_topk_v;
  int *d_topk_i;
  float *d_e_agg;
  float *d_gate_up_workspace; // Pre-allocated workspace for MLP
  size_t gate_up_workspace_bytes;
  float *d_qkv;
  bf16_t *d_q;
  bf16_t *d_key_cache, *d_value_cache;
  int kv_seq_capacity;
  int kv_window_capacity;
  int kv_seq_limit;
  uint32_t kv_batch_stride;
  float *d_logits;
  int *d_next_tokens;
  int *d_tokens;
  int *d_pos;
  float *d_inv_rms;
};

struct GPUWeightBuffersFP32 {
  float *d_rms_attn_w, *d_rms_ffn_w;
  float *d_b_qkv, *d_b_o, *d_attn_sinks;
  float *d_w_router, *d_b_router;
  float *d_rms_out_w;
};

struct GPUExpertBiasBuffers {
  float *g_b_mlp1;
  float *g_b_mlp2;
};

struct GPUWeightBuffersBF16 {
  bf16_t *d_token_embedding_table_bf16;
  bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
  bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
  bf16_t *d_out_bf16;
};

struct DeviceContext {
  int device_id;
  int E_local = 0; // number of local experts owned by this device (per layer)

  GPUActivationBuffers gpu_activations;
  GPUWeightBuffersFP32 gpu_weights_fp32;
  GPUExpertBiasBuffers gpu_expert_bias;
  GPUWeightBuffersBF16 gpu_weights_bf16;
  int capacity_B = 1;
  hipStream_t *streams = nullptr;
  int n_streams = 0;
  // Dedicated streams for MoE routing/p2p
  hipStream_t compute_stream = nullptr;   // QKV/Attn/Res/FFN-router
  hipStream_t pack_stream = nullptr;      // route_count + scan + route_pack
  hipStream_t mlp_stream = nullptr;       // run MLP on owner
  hipStream_t *comm_streams = nullptr;    // [ndev]
  size_t stride_w_qkv_bf16 = 0;
  size_t stride_w_o_bf16 = 0;
  size_t stride_w_out_bf16 = 0;
  size_t stride_w_mlp1_bf16 = 0;
  size_t stride_w_mlp2_bf16 = 0;
  std::vector<uint32_t> h_kv_layer_offsets;
  std::vector<int> h_kv_layer_capacity;
  uint32_t *d_kv_layer_offsets = nullptr;
  int *d_kv_layer_capacity = nullptr;

  // Routing meta on device (home needs all owners' local-id maps)
  int *d_e2lid_allowners = nullptr; // [ndev * L * E]
  int *d_E_local_per_owner_layer = nullptr; // [ndev * L]

  // Home-side ring buffers (per peer)
  bf16_t **send_x_peer = nullptr;        // [ndev] -> device pointers allocated on HOME
  int **send_pos_peer = nullptr;         // [ndev] -> [B_local]
  float **send_topk_v_peer = nullptr;    // [ndev] -> [B_local*K]
  int **send_assignment_batches_peer = nullptr; // [ndev] -> [total_assign]
  int **send_assignment_slots_peer = nullptr;   // [ndev] -> [total_assign]
  int **send_expert_offsets_peer = nullptr;     // [ndev] -> [E_local+1]
  int **d_owner_B_peer = nullptr;        // [ndev] -> [1]
  int **d_b2local_peer = nullptr;        // [ndev]
  int **d_local2b_peer = nullptr;        // [ndev]
  int **d_expert_counts_peer = nullptr;  // [ndev]
  int **d_expert_offsets_peer = nullptr; // [ndev]
  int **d_expert_writes_peer = nullptr;  // [ndev]
  int **d_pos_peer = nullptr;            // [ndev]
  float **d_topk_v_peer = nullptr;       // [ndev]

  // Owner-side receive buffers (per home) on this device
  bf16_t **recv_x_from_home = nullptr;     // [ndev] -> [B_local, H]
  int **recv_pos_from_home = nullptr;      // [ndev]
  float **recv_topk_v_from_home = nullptr; // [ndev]
  int **recv_assignment_batches = nullptr; // [ndev]
  int **recv_assignment_slots = nullptr;   // [ndev]
  int **recv_expert_offsets = nullptr;     // [ndev]

  // Owner-side partials per home and home-side recv partials
  float **partial_owner_per_home = nullptr; // [ndev] -> [B_local,H]
  float **recv_partial_home = nullptr;      // [ndev] -> [B_local,H] (allocated on home device)
  float **gate_up_owner_per_home = nullptr; // [ndev] -> [K, B_local, IM]
};

// Prompt Context Structure
struct PromptCtx {
  int idx;                // original order in requests
  std::string input_seq;  // input text (raw)
  int *prompt_tokens;     // tokenized prompt buffer
  int num_prompt_tokens;  // number of prompt tokens
  int *output_tokens;     // output token buffer (caller provided)
  std::string output_str; // output text (raw)
  int pos;                // current decode step (position in sequence)
  int token;              // current token being processed
  bool finished;          // EOS reached or step limit
  int max_steps;          // maximum steps allowed for this prompt

  // Runtime state useful for batching / scheduling
  float *h_logits;  // host logits buffer (per step)
  int logits_size;  // vocab size (for allocation)
  Sampler *sampler; // pointer to sampler (per prompt, if needed)

  // Metadata for extensions
  long long num_generated; // number of tokens generated so far
  double start_time;       // timestamp when generation started
  double end_time;         // timestamp when finished
  bool is_context_phase;   // true if still consuming prompt tokens
  void *user_data;         // reserved for external features

  PromptCtx()
      : idx(0), input_seq(""), prompt_tokens(nullptr), num_prompt_tokens(0),
        output_tokens(nullptr), output_str(""), pos(0), token(0), finished(false),
        max_steps(0), h_logits(nullptr), logits_size(0), sampler(nullptr),
        num_generated(0), start_time(0.0), end_time(0.0),
        is_context_phase(true), user_data(nullptr) {}
};

struct EPAssignHost {
  int b;      // batch row index
  int e;      // global expert id
  float w;    // gate weight
};

// Utility function for PromptCtx cleanup
static inline void free_prompt_ctx_heap_buffers(PromptCtx &ctx) {
  if (ctx.h_logits) {
    free(ctx.h_logits);
    ctx.h_logits = nullptr;
  }
  if (ctx.prompt_tokens) {
    free(ctx.prompt_tokens);
    ctx.prompt_tokens = nullptr;
  }
  ctx.output_str.clear();
}

#endif // GETP_COMMON_DEFINES_H
