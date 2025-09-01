#include <string>

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
      : idx(0), input_seq(""), prompt_tokens(nullptr), num_prompt_tokens(0),
        output_tokens(nullptr), output_str(), pos(0), token(0), finished(false),
        max_steps(0), h_logits(nullptr), logits_size(0), sampler(nullptr),
        num_generated(0), start_time(0.0), end_time(0.0),
        is_context_phase(true), user_data(nullptr) {}
};

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
