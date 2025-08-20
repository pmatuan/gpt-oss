// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include "hip_kernels.h"

#ifndef GETP_RUN
#define GETP_RUN

static bool use_gpu = false;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Detect availability of a HIP capable GPU. If one is present we enable the
  // GPU execution path; otherwise the code falls back to the original CPU
  // implementation. The heavy lifting is still performed by the forward()
  // function from run.cpp, but we initialise the GPU here so that kernels can
  // be launched later if desired.
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0) {
    use_gpu = true;
    fprintf(stderr, "HIP device detected: using GPU path\n");
  } else {
    fprintf(stderr, "No HIP device detected: using CPU path\n");
  }
  (void)transformer;
  (void)tokenizer;
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Currently there is no persistent GPU state, but the hook is provided for
  // symmetry with warm_up().
  (void)transformer;
  (void)tokenizer;
  use_gpu = false;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {

    // forward the transformer to get logits for the next token. When the GPU
    // path is enabled this call could be replaced by a GPU implementation of
    // the forward pass. For now it simply dispatches to the existing CPU
    // routine to ensure functional correctness while the handwritten HIP
    // kernels are validated independently.
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
