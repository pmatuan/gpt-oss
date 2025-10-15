<div align="center">

# GPT-OSS from Scratch on GPU AMD MI250

[Overview](#overview) • [Highlights](#highlights) • [Repository-layout](#repository-layout) • [Getting-started](#getting-started) • [Evaluation](#evaluation) • [Performance](#performance) • [Acknowledgements](#acknowledgements)

</div>

## Overview

This project delivers a pure C++17/HIP implementation of OpenAI's GPT-OSS 20B and 120B models for AMD MI250 GPUs.

## Highlights

- Pure C++/HIP implementation of GPT-OSS 20B and 120B inference without library dependencies.
- Optimized and validated on AMD MI250 GPUs.

## Repository layout

```text
gpt-oss/
├── run.cpp                  # Entry point with chat/generate/getp modes
├── tokenizer.{hpp,cpp}      # C++ tokenizer matching o200k_harmony
├── getp-csrc/               # GPU kernels (attention, matmul, routing, profiler)
├── export_model_bin/        # Safetensor → binary conversion scripts
├── data/                    # Sample prompts and helper assets
├── evaluation/              # METEOR & BERTScore evaluation pipeline
├── decode.cpp               # Utility to detokenize model outputs
├── Makefile                 # hipcc/g++ build targets
└── requirements.txt         # Python dependencies for tooling
```

## Getting started

### Prerequisites

- ROCm-enabled AMD Instinct MI250 GPUs and a working HIP toolchain (`hipcc` preferred).
- GCC/Clang with OpenMP support if you plan to use CPU-assisted runs.
- Python 3.10+ with virtualenv tooling.
- Access to the GPT-OSS checkpoints (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, optional `tiny-random/gpt-oss`).

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the tokenizer binary once:

```bash
python3 export_tokenizer_bin.py -o tokenizer.bin
```

### Convert model weights

1. Download the safetensor checkpoints from Hugging Face into `export_model_bin/{gpt-oss-20b,gpt-oss-120b}`.
2. Run the exporter matching your target model, for example:

   ```bash
   python3 export_model_bin/gpt-oss-20b/export_model_bin.py \
       --model openai/gpt-oss-20b \
       --output /path/to/model_bins/gpt-oss-20b.bin
   ```

The resulting `.bin` is a memory-mappable blob consumed by `./run`.

### Build targets

```bash
make run        # Development build (no optimizations)
make runfast    # Optimized build (-O3) using hipcc or g++
make runomp     # OpenMP-enabled build for multi-threaded runs
make runprof    # Build with profiling instrumentation enabled
make decode     # Detokenizer CLI (./decode)
make clean      # Remove build outputs
```

If `hipcc` is available it is used automatically; otherwise the Makefile falls back to `g++`.

### Run inference

```bash
# Batch GETP job
./run /path/to/gpt-oss-120b.bin -m getp -i data/input.txt -o data/output.txt

# Convert token IDs back to text
./decode -i data/output.txt -o data/output.txt.decoded
```

## Evaluation

The `evaluation/` folder provides a metrics pipeline built around METEOR and BERTScore F1. Typical usage:

```bash
python3 evaluation/eval.py -m 20b \
    -s submission/output_20b_token_ids.txt \
    -r references/output_20b_token_ids.txt
```

Key features:

- Token IDs are decoded with `tiktoken` to align with the tokenizer used during inference.
- Optional `threshold.json` asserts minimum acceptable scores during CI.
- Supports overriding submission/reference paths and encodings for custom datasets.
- Designed to run efficiently on multi-GPU ROCm systems; CPU fallback is available but slower.

## Performance

End-to-end results collected on a single node with 8× AMD Instinct MI250 GPUs:

| Model        | Mode | Batch Size | Throughput (TPS) | METEOR  | BERTScore F1 |
| ------------ | ---- | ---------- | ---------------- | ------- | ------------ |
| gpt-oss-20b  | GETP | 7,168      | 35,454.61        | 0.49883 | 0.976014     |
| gpt-oss-120b | GETP | 6,144      | 10,296.66        | 0.42150 | 0.971202     |

## Acknowledgements

Developed as part of the GPU Engineer Training Program jointly organized by [Moreh](https://www.linkedin.com/company/moreh-vietnam/) and the [THUNDER Research Group](http://snuvm.snu.ac.kr/) at Seoul National University. GPT-OSS weights are provided by OpenAI.

## Contributing

Issues and pull requests are welcome. If you are adding new kernels, include profiler captures or benchmarks, and keep changes scoped to existing files when possible. For larger features, please open an issue first so we can coordinate API changes.
