# Repository Guidelines

## Project Structure & Module Organization
Core inference lives in `run.cpp`, which wires the tokenizer, profiler hooks, and model execution. Auxiliary CUDA/HIP-aware kernels and math utilities sit under `getp-csrc/` (`attention/`, `matmul/`, `utility/`, `profiler/`). Tokenization logic is in `tokenizer.cpp` and the shared header `tokenizer.hpp`. Use `data/` for sample prompts and outputs, and `evaluation/` for benchmarking helpers. `decode.cpp` converts raw token IDs back to text, while `export_tokenizer_bin.py` and `run_transformers.py` bridge Python tooling. The root `Makefile` orchestrates every build target, and `requirements.txt` pins the Python sidecar tools.

## Build, Test, and Development Commands
- `make run` — unoptimized build for quick iteration; produces `./run`.
- `make runfast` / `make runomp` / `make runprof` — optimized, OpenMP, and profiled builds respectively; set `OMP_NUM_THREADS` when using OpenMP.
- `./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat` — launch inference with an exported model.
- `make decode && ./decode -1 -i data/output.txt` — decode token IDs emitted by `getp` runs.
- `make tokenizer-bin` — regenerate `tokenizer.bin` before any tokenizer testing.
- `make tokenizer-test && ./test_tokenizer -t tokenizer.bin -i "Hello world"` — validate the C++ tokenizer.
- `python3 test_tokenizer.py --bin ./test_tokenizer --tok ./tokenizer.bin --prompt data/input.txt` — cross-check against Tiktoken.
- `make runprof` is default run with it to test the compile

## Coding Style & Naming Conventions
Target C++17 with either `hipcc` or `g++`; keep headers self-contained and prefer internal `#include` paths relative to the repo root. Match the existing two-space indentation, brace-on-same-line layout, and snake_case function naming (`find_token_id`, `run_eval_step`). Favor descriptive struct and enum names and guard new utilities with `#pragma once` headers. For Python helpers, follow PEP 8 (four-space indents, snake_case) and run `python -m compileall` locally when in doubt.

## Testing Guidelines
Run make runprof to test the code

## Commit & Pull Request Guidelines
Use short, imperative commit subjects mirroring the existing history (`optimize`, `remove batching`). Group unrelated edits into separate commits, and add a focused body when explaining performance or numerical changes. Pull requests should describe intent, mention affected targets (e.g., `make runomp`), link related issues or benchmarks, and include screenshots or logs for profiling updates. Tag reviewers when touching `run.cpp`, the profiler, or tokenizer assets.
