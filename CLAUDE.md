# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance C++ implementation of GPT-OSS models with optimized GPU inference using HIP/ROCm. The project focuses on efficient transformer model inference with support for models ranging from 7M to 120B parameters.

## Core Architecture

### Key Components
- **Main inference engine**: `run.cpp` - Entry point for model inference with chat, generate, and getp modes
- **GETP implementation**: `getp-csrc/` - GPU-optimized inference kernels for batch processing
  - `getp_run.cpp`: Main GETP runner with GPU kernels
  - `getp_eval.cpp`: Evaluation logic (DO NOT MODIFY)
  - `attention/`: Flash attention implementation
  - `matmul/`: Optimized matrix multiplication kernels
  - `profiler/`: Performance profiling utilities
- **Tokenizer**: BPE tokenizer based on tiktoken with C++ implementation
- **Model export**: Python scripts to convert Hugging Face models to binary format

### GPU Optimization Strategy
- Uses HIP/ROCm for AMD GPUs (gfx90a architecture)
- Optimized kernels with configurable block sizes (BLOCK_SIZE=512)
- Warp-level optimizations (WF_SIZE=64)
- Batch processing with tiling (BATCH_TILE_DEFAULT=4)
- Mixed precision with bfloat16 support

## Essential Commands

### Build Commands
```bash
make run          # Basic build (slow, debug-friendly)
make runfast      # Optimized build with -O3
make runomp       # OpenMP multi-threaded build
make runprof      # Build with profiling enabled
make decode       # Build output decoder utility
make tokenizer-bin # Generate tokenizer binary
make tokenizer-test # Build tokenizer test
make clean        # Clean all build artifacts
```

### Running the Model
```bash
# Generate mode (text completion)
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "1+1="

# Chat mode (interactive)
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat

# GETP mode (batch processing)
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/input.txt -o data/output.txt
./decode -i data/output.txt  # Convert output tokens to text
```

### Testing
```bash
# Test tokenizer compatibility with tiktoken
python3 test_tokenizer.py --bin ./test_tokenizer --tok ./tokenizer.bin --verbose --prompt data/input.txt

# Run evaluation
python3 evaluation/eval.py -p ../data/input.txt -s ../data/output.txt -r refs_openai_gpt5.jsonl
```

### Development Tools
```bash
# Format code (pre-commit hooks configured)
pre-commit run --all-files

# Profile performance (requires runprof build)
OMP_NUM_THREADS=4 ./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "test"
```

## Important Constraints

### EDITING RESTRICTIONS
- **ONLY edit files in `getp-csrc/` folder** (this is the only area for modifications)
- **DO NOT MODIFY**:
  - `run.cpp` - Main program file
  - `getp-csrc/getp_eval.cpp` - Evaluation logic  
  - `Makefile` - Build configuration
  - Any files outside `getp-csrc/` folder

### EXECUTION ENVIRONMENT
- **This is a development host only** - DO NOT run any build or execution commands here
- All `make` commands and program execution happens on the server
- Only perform code editing and file modifications on this host

### Code Style
- C++ standard: C++17
- Python: Google style (isort profile)
- Formatting: clang-format for C++, yapf for Python, prettier for Markdown/YAML
- Pre-commit hooks enforce consistent formatting

## Environment Setup
```bash
# Required environment variables
export GPT_OSS_REPO_ROOT="/path/to/gpt-oss"
export MODELS_ROOT="/path/to/models"
export MODELBIN_ROOT="/path/to/modelbin"

# Python environment (requires Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Support
- **gpt-oss-7m**: Tiny model for testing
- **gpt-oss-20b**: Medium-scale model
- **gpt-oss-120b**: Large-scale model

Models must be converted from Hugging Face safetensors format to binary using the export scripts in `export_model_bin/`.

## Performance Optimization Tips
- Use `make runomp` for CPU parallelism
- Set `OMP_NUM_THREADS` for OpenMP builds
- Enable profiling with `make runprof` to identify bottlenecks
- GPU kernels are tuned for AMD MI250X (gfx90a)