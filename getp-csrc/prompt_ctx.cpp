#include <stdio.h>

struct PromptCtx {
  int idx;               // original order in requests
  const char *input_seq; // input text (raw)
  int *prompt_tokens;    // tokenized prompt buffer
  int num_prompt_tokens; // number of prompt tokens
  int *output_tokens;    // output token buffer (caller provided)
  char *output_buffer;   // accumulate decoded pieces for printing
  size_t buffer_size;    // size allocated for output_buffer
  size_t buffer_pos;     // current filled position in output_buffer
  int pos;               // current decode step (position in sequence)
  int token;             // current token being processed
  bool finished;         // EOS reached or step limit
  int max_steps;         // maximum steps allowed for this prompt

  // --- Runtime state useful for batching / scheduling ---
  float *h_logits;  // host logits buffer (per step)
  int logits_size;  // vocab size (for allocation)
  Sampler *sampler; // pointer to sampler (per prompt, if needed)

  // --- Metadata for extensions ---
  long long num_generated; // number of tokens generated so far
  double start_time;       // timestamp when generation started
  double end_time;         // timestamp when finished
  bool is_context_phase;   // true if still consuming prompt tokens
  void *user_data;         // reserved for external features

  PromptCtx()
      : idx(0), input_seq(nullptr), prompt_tokens(nullptr),
        num_prompt_tokens(0), output_tokens(nullptr), output_buffer(nullptr),
        buffer_size(0), buffer_pos(0), pos(0), token(0), finished(false),
        max_steps(0), h_logits(nullptr), logits_size(0), sampler(nullptr),
        num_generated(0), start_time(0.0), end_time(0.0),
        is_context_phase(true), user_data(nullptr) {}
};

static inline void ensure_buffer_capacity(PromptCtx &ctx, size_t extra_needed) {
  if (ctx.buffer_pos + extra_needed + 1 <= ctx.buffer_size)
    return; // +1 for '\0'
  size_t new_size = ctx.buffer_size ? ctx.buffer_size : 4096;
  while (ctx.buffer_pos + extra_needed + 1 > new_size)
    new_size *= 2;
  char *new_buf = (char *)realloc(ctx.output_buffer, new_size);
  if (!new_buf) {
    fprintf(stderr, "OOM: realloc output_buffer\n");
    exit(EXIT_FAILURE);
  }
  ctx.output_buffer = new_buf;
  ctx.buffer_size = new_size;
}

static inline void free_prompt_ctx_heap_buffers(PromptCtx &ctx) {
  if (ctx.h_logits) {
    free(ctx.h_logits);
    ctx.h_logits = nullptr;
  }
  if (ctx.prompt_tokens) {
    free(ctx.prompt_tokens);
    ctx.prompt_tokens = nullptr;
  }
  if (ctx.output_buffer) {
    free(ctx.output_buffer);
    ctx.output_buffer = nullptr;
  }
  ctx.buffer_size = ctx.buffer_pos = 0;
}
