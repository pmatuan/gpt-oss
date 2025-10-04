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
#define ATTN_WARPS_PER_BLOCK 4
#define ATTN_THREADS_PER_BLOCK (WF_SIZE * ATTN_WARPS_PER_BLOCK)
#define ATTN_FLASH_TILE 128
#define ATTN_FLASH_MAX_KV_MUL 8
#define TK 512
#define LDS_PAD 16
#define K_STEP_MATMUL_FLOAT 4
#define EXPERT_PER_TOKEN 4
#define EXPERT_PER_TOKEN_SHIFT 2
#define EXPERT_PER_TOKEN_MASK (EXPERT_PER_TOKEN - 1)
#define MAX_BATCH_SIZE 896
#define MAX_BATCH_SIZE_120B 768
#define MLP_TILE_TOKENS 32
#define MLP_TILE_COLS 32
#define MLP_THREAD_X 16
#define MLP_THREAD_Y 4
#define MATMUL_TILE_COLS 32
#define MATMUL_TILE_K 16
#define MATMUL_CHUNK_K 4

// Parameterize MFMA launch configuration
#define MATMUL_QKV_BLOCK_ROWS 192
#define MATMUL_QKV_BLOCK_COLS 128
#define MATMUL_QKV_BLOCK_DEPTH 32
#define MATMUL_QKV_WARP_TILE_M 32
#define MATMUL_QKV_WARP_TILE_N 64
#define MATMUL_QKV_WAVES_PER_BLOCK (MATMUL_QKV_BLOCK_ROWS / MATMUL_QKV_WARP_TILE_M) * (MATMUL_QKV_BLOCK_COLS / MATMUL_QKV_WARP_TILE_N)

#define MATMUL_ATT_BLOCK_ROWS 96
#define MATMUL_ATT_BLOCK_COLS 160
#define MATMUL_ATT_BLOCK_DEPTH 32
#define MATMUL_ATT_WARP_TILE_M 32
#define MATMUL_ATT_WARP_TILE_N 32
#define MATMUL_ATT_WAVES_PER_BLOCK (MATMUL_ATT_BLOCK_ROWS / MATMUL_ATT_WARP_TILE_M) * (MATMUL_ATT_BLOCK_COLS / MATMUL_ATT_WARP_TILE_N)

#define MATMUL_LOGITS_BLOCK_ROWS 192
#define MATMUL_LOGITS_BLOCK_COLS 128
#define MATMUL_LOGITS_BLOCK_DEPTH 32
#define MATMUL_LOGITS_WARP_TILE_M 32
#define MATMUL_LOGITS_WARP_TILE_N 64
#define MATMUL_LOGITS_WAVES_PER_BLOCK (MATMUL_LOGITS_BLOCK_ROWS / MATMUL_LOGITS_WARP_TILE_M) * (MATMUL_LOGITS_BLOCK_COLS / MATMUL_LOGITS_WARP_TILE_N)

#define MATMUL_MLP1_BLOCK_ROWS 192
#define MATMUL_MLP1_BLOCK_COLS 96
#define MATMUL_MLP1_BLOCK_DEPTH 32
#define MATMUL_MLP1_WARP_TILE_M 32
#define MATMUL_MLP1_WARP_TILE_N 48
#define MATMUL_MLP1_WAVES_PER_BLOCK (MATMUL_MLP1_BLOCK_ROWS / MATMUL_MLP1_WARP_TILE_M) * (MATMUL_MLP1_BLOCK_COLS / MATMUL_MLP1_WARP_TILE_N)

#define MATMUL_MLP1_BLOCK_ROWS_120B 192
#define MATMUL_MLP1_BLOCK_COLS_120B 96
#define MATMUL_MLP1_BLOCK_DEPTH_120B 32
#define MATMUL_MLP1_WARP_TILE_M_120B 32
#define MATMUL_MLP1_WARP_TILE_N_120B 48
#define MATMUL_MLP1_WAVES_PER_BLOCK_120B (MATMUL_MLP1_BLOCK_ROWS_120B / MATMUL_MLP1_WARP_TILE_M_120B) * (MATMUL_MLP1_BLOCK_COLS_120B / MATMUL_MLP1_WARP_TILE_N_120B)

#define MATMUL_MLP2_BLOCK_ROWS 192
#define MATMUL_MLP2_BLOCK_COLS 96
#define MATMUL_MLP2_BLOCK_DEPTH 32
#define MATMUL_MLP2_WARP_TILE_M 32
#define MATMUL_MLP2_WARP_TILE_N 48
#define MATMUL_MLP2_WAVES_PER_BLOCK (MATMUL_MLP2_BLOCK_ROWS / MATMUL_MLP2_WARP_TILE_M) * (MATMUL_MLP2_BLOCK_COLS / MATMUL_MLP2_WARP_TILE_N)

#define MATMUL_MLP2_BLOCK_ROWS_120B 192
#define MATMUL_MLP2_BLOCK_COLS_120B 96
#define MATMUL_MLP2_BLOCK_DEPTH_120B 32
#define MATMUL_MLP2_WARP_TILE_M_120B 32
#define MATMUL_MLP2_WARP_TILE_N_120B 48
#define MATMUL_MLP2_WAVES_PER_BLOCK_120B (MATMUL_MLP2_BLOCK_ROWS_120B / MATMUL_MLP2_WARP_TILE_M_120B) * (MATMUL_MLP2_BLOCK_COLS_120B / MATMUL_MLP2_WARP_TILE_N_120B)

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
  float *d_router_score, *d_topk_v;
  int *d_topk_i;
  float *d_e_agg;
  bf16_t *d_gate_up_workspace; // Pre-allocated workspace for MLP
  size_t gate_up_workspace_bytes;
  bf16_t *d_mlp2_partial_bf16; // [K, B, H] intermediate buffer for MLP2 (no atomic)
  size_t mlp2_partial_bytes;
  bf16_t *d_qkv;
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
  float *d_rope_inv_freq;
  float rope_concentration;
};

struct DeviceExpertWorkspace {
  int *d_expert_counts = nullptr;
  int *d_expert_offsets = nullptr;
  uint16_t *d_assignment_batches = nullptr;
  uint8_t *d_assignment_slots = nullptr;
  size_t assignment_capacity = 0;
  size_t expert_capacity = 0;
};

struct HostPinnedBatchBuffers {
  int *tokens = nullptr;
  int *pos = nullptr;
  int *next_tokens = nullptr;
  int *expert_counts = nullptr;
  int *expert_offsets = nullptr;
  size_t batch_capacity = 0;
  size_t expert_capacity = 0;
};

struct GPUWeightBuffersFP32 {
  bf16_t *d_rms_attn_w, *d_rms_ffn_w;
  bf16_t *d_b_qkv, *d_b_o, *d_attn_sinks;
  bf16_t *d_w_router, *d_b_router;
  bf16_t *d_rms_out_w;
};

struct GPUExpertBiasBuffers {
  bf16_t *g_b_mlp1;
  bf16_t *g_b_mlp2;
};

struct GPUWeightBuffersBF16 {
  bf16_t *d_token_embedding_table_bf16;
  bf16_t *d_w_qkv_bf16, *d_w_o_bf16;
  bf16_t *d_w_mlp1_bf16, *d_w_mlp2_bf16;
  bf16_t *d_out_bf16;
};

struct RoutingMeta {
  int *d_e2lid_allowners = nullptr;        // [ndev * L * E]
  int *d_E_local_per_owner_layer = nullptr; // [ndev * L]
};

struct HomePeerBuffers {
  bf16_t **send_x_peer = nullptr;        // [ndev] -> device pointers allocated on HOME
  int **send_pos_peer = nullptr;         // [ndev] -> [B_local]
  float **send_topk_v_peer = nullptr;    // [ndev] -> [B_local*K]
  uint16_t **send_assignment_batches_peer = nullptr; // [ndev] -> [total_assign]
  uint8_t **send_assignment_slots_peer = nullptr;   // [ndev] -> [total_assign]
  int **send_expert_offsets_peer = nullptr;         // [ndev] -> [E_local+1]
  int **d_owner_B_peer = nullptr;        // [ndev] -> [1]
  int **d_b2local_peer = nullptr;        // [ndev]
  int **d_local2b_peer = nullptr;        // [ndev]
  int **d_expert_counts_peer = nullptr;  // [ndev]
  int **d_expert_offsets_peer = nullptr; // [ndev]
  int **d_pos_peer = nullptr;            // [ndev]
  float **d_topk_v_peer = nullptr;       // [ndev]
  int *h_owner_B = nullptr;              // [ndev] host pinned
  int *h_total_assign = nullptr;         // [ndev] host pinned
  int *h_prev_owner_B = nullptr;         // [ndev] host-persistent for selective resets
};

struct OwnerReceiveBuffers {
  bf16_t **recv_x_from_home = nullptr;     // [ndev] -> [B_local, H]
  int **recv_pos_from_home = nullptr;      // [ndev]
  float **recv_topk_v_from_home = nullptr; // [ndev]
  uint16_t **recv_assignment_batches = nullptr; // [ndev]
  uint8_t **recv_assignment_slots = nullptr;   // [ndev]
  int **recv_expert_offsets = nullptr;     // [ndev]
};

struct OwnerPartialBuffers {
  float **partial_owner_per_home = nullptr; // [ndev] -> [B_local,H]
  bf16_t **partial_owner_per_home_bf16 = nullptr; // [ndev] -> [B_local,H]
  bf16_t **recv_partial_home_bf16 = nullptr;      // [ndev] -> [B_local,H]
  bf16_t **gate_up_owner_per_home = nullptr; // [ndev] -> [K, B_local, IM]
};

struct DeviceContext {
  int device_id;
  int E_local = 0; // number of local experts owned by this device (per layer)

  GPUActivationBuffers gpu_activations;
  GPUWeightBuffersFP32 gpu_weights_fp32;
  GPUExpertBiasBuffers gpu_expert_bias;
  GPUWeightBuffersBF16 gpu_weights_bf16;
  int capacity_B = 1;

  // Dedicated streams for MoE routing/p2p
  hipStream_t compute_stream = nullptr;   // QKV/Attn/Res/FFN-router
  hipStream_t pack_stream = nullptr;      // route_count + scan + route_pack
  hipStream_t mlp_stream = nullptr;       // run MLP on owner
  hipStream_t *comm_streams = nullptr;    // [ndev]
  hipEvent_t router_ready_event = nullptr;
  hipEvent_t *pack_done_events = nullptr;   // [ndev]
  hipEvent_t *copy_ready_events = nullptr;  // [ndev]
  hipEvent_t *mlp_done_events = nullptr;    // [ndev]

  size_t stride_w_qkv_bf16 = 0;
  size_t stride_w_o_bf16 = 0;
  size_t stride_w_out_bf16 = 0;
  size_t stride_w_mlp1_bf16 = 0;
  size_t stride_w_mlp2_bf16 = 0;
  std::vector<uint32_t> h_kv_layer_offsets;
  std::vector<int> h_kv_layer_capacity;
  uint32_t *d_kv_layer_offsets = nullptr;
  int *d_kv_layer_capacity = nullptr;
  DeviceExpertWorkspace expert_workspace;
  HostPinnedBatchBuffers host_pinned_batch;

  // Routing meta on device (home needs all owners' local-id maps)
  RoutingMeta routing_meta;
  // Home-side ring buffers (per peer)
  HomePeerBuffers home_peer_buffers;
  // Owner-side receive buffers (per home) on this device
  OwnerReceiveBuffers owner_receive_buffers;
  // Owner-side partials per home and home-side recv partials
  OwnerPartialBuffers partial_buffers;
};

// Prompt Context Structure
struct PromptCtx {
  int idx;                // original order in requests
  std::string input_seq;  // input text (raw)
  int *prompt_tokens;     // tokenized prompt buffer
  int num_prompt_tokens;  // number of prompt tokens
  int *output_tokens;     // output token buffer (caller provided)
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
        output_tokens(nullptr), pos(0), token(0), finished(false),
        max_steps(0), h_logits(nullptr), logits_size(0), sampler(nullptr),
        num_generated(0), start_time(0.0), end_time(0.0),
        is_context_phase(true), user_data(nullptr) {}
};

struct EPAssignHost {
  int b;      // batch row index
  int e;      // global expert id
  float w;    // gate weight
};

static Config *model_config;

static std::vector<DeviceContext> g_devices;
static int g_num_devices = 0;

static bool use_expert_parallelism = false;

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
}

#endif // GETP_COMMON_DEFINES_H
