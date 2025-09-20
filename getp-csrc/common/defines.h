#ifndef GETP_COMMON_DEFINES_H
#define GETP_COMMON_DEFINES_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <string>
#include <stdlib.h>

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
#define MAX_BATCH_SIZE 64
#define TM_MM 32
#define TN_MM 32

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
  float *d_x, *d_t, *d_tb, *d_tb2;
  float *d_router_score, *d_topk_v;
  int *d_topk_i;
  float *d_gate_up, *d_e_agg;
  float *d_gate_up_workspace; // Pre-allocated workspace for MLP
  int *d_expert_counts;
  int *d_expert_offsets;
  int *d_expert_assignments;
  size_t expert_assign_capacity;
  int *d_assignment_active_slot;
  size_t assignment_active_capacity;
  int *d_active_experts;
  int *d_active_counts;
  int active_expert_capacity;
  float *d_moe_x_workspace;
  float *d_mlp1_workspace;
  float *d_mlp2_workspace;
  size_t gate_up_workspace_bytes;
  size_t moe_x_workspace_bytes;
  size_t mlp1_workspace_bytes;
  size_t mlp2_workspace_bytes;
  float *d_qkv, *d_q;
  bf16_t *d_key_cache, *d_value_cache;
  float *d_att, *d_logits, *d_mask;
  float *d_cos_vals, *d_sin_vals;
  int *d_token2row;
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

  GPUActivationBuffers gpu_activations;
  GPUWeightBuffersFP32 gpu_weights_fp32;
  GPUExpertBiasBuffers gpu_expert_bias;
  GPUWeightBuffersBF16 gpu_weights_bf16;
  int capacity_B = 1;
  hipStream_t *streams = nullptr;
  int n_streams = 0;
};

// Prompt Context Structure
struct PromptCtx {
  int idx;                // original order in requests
  std::string input_seq;  // input text (raw)
  int *prompt_tokens;     // tokenized prompt buffer
  int num_prompt_tokens;  // number of prompt tokens
  int *output_tokens;     // output token buffer (caller provided)
  std::string output_str; // C++ string for efficient concatenation
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
        output_tokens(nullptr), output_str(), pos(0), token(0), finished(false),
        max_steps(0), h_logits(nullptr), logits_size(0), sampler(nullptr),
        num_generated(0), start_time(0.0), end_time(0.0),
        is_context_phase(true), user_data(nullptr) {}
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
  ctx.output_str.clear(); // Clear the string
}

#endif // GETP_COMMON_DEFINES_H
