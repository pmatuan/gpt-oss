struct Config;

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct Sampler;

#include "matmul.cpp"
#include "../utility/utility.cpp"

template <int BM_, int BN_, int BK_, int WM_, int WN_, int TM_, int TN_>
struct MatmulConfig {
  static_assert(BM_ > 0 && BN_ > 0 && BK_ > 0 && WM_ > 0 && WN_ > 0,
                "Tile sizes must be positive");
  static_assert(BM_ % WM_ == 0, "BM must be divisible by WM");
  static_assert(BN_ % WN_ == 0, "BN must be divisible by WN");
  static_assert(WM_ % TM_ == 0, "WM must be divisible by MFMA tile");
  static_assert(WN_ % TN_ == 0, "WN must be divisible by MFMA tile");
  static_assert(BK_ % TM_ == 0, "BK must be divisible by MFMA tile");
  static_assert(BK_ % 8 == 0, "BK must be divisible by 8 for vector loads");
  static_assert(TM_ % 16 == 0 && TN_ % 16 == 0,
                "MFMA tile dimensions must be multiples of 16");

  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int BK = BK_;
  static constexpr int WM = WM_;
  static constexpr int WN = WN_;
  static constexpr int WavesM = BM / WM;
  static constexpr int WavesN = BN / WN;
  static constexpr int WavesPerWg = WavesM * WavesN;

  static_assert(WavesM > 0 && WavesN > 0, "Invalid wave tiling");
  static_assert(WavesPerWg <= 16, "Too many waves per workgroup (max 16)");
};

template <typename Config>
std::string format_config_label(const char *prefix) {
  std::string label(prefix);
  label += "_BM" + std::to_string(Config::BM);
  label += "_BN" + std::to_string(Config::BN);
  label += "_BK" + std::to_string(Config::BK);
  label += "_WM" + std::to_string(Config::WM);
  label += "_WN" + std::to_string(Config::WN);
  return label;
}

#define MATMUL_CONFIG_LIST(MACRO)                                                \
  MACRO(64, 64, 32, 32, 32)                                                     \
  MACRO(64, 128, 32, 32, 32)                                                    \
  MACRO(64, 256, 32, 32, 64)                                                    \
  MACRO(96, 96, 32, 32, 32)                                                     \
  MACRO(96, 160, 32, 32, 32)                                                    \
  MACRO(128, 64, 32, 32, 32)                                                    \
  MACRO(128, 256, 32, 32, 64)                                                   \
  MACRO(160, 96, 32, 32, 32)                                                    \
  MACRO(192, 96, 32, 32, 48)                                                    \
  MACRO(192, 128, 32, 32, 64)                                                   \
  MACRO(192, 192, 32, 64, 64)                                                   \
  MACRO(224, 96, 32, 32, 48)                                                    \
  MACRO(256, 64, 32, 32, 32)                                                    \
  MACRO(256, 128, 32, 32, 64)                                                   \
  MACRO(256, 256, 32, 64, 64)                                                   \
  MACRO(128, 128, 64, 32, 32)                                                   \
  MACRO(192, 128, 64, 32, 64)                                                   \
  MACRO(192, 192, 64, 64, 64)                                                   \
  MACRO(256, 64, 64, 32, 32)                                                    \
  MACRO(256, 128, 64, 32, 64)                                                   \
  MACRO(64, 256, 64, 32, 64)                                                    \
  MACRO(64, 320, 64, 32, 64)   /* Waves=2*5=10  */                              \
  MACRO(64, 288, 64, 32, 96)   /* Waves=2*3=6   */                              \
  MACRO(96, 288, 64, 32, 96)   /* Waves=3*3=9   */                              \
  MACRO(96, 192, 96, 32, 64)   /* Waves=3*3=9   */                              

using bf16_t = hip_bfloat16;

struct ErrStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double max_rel = 0.0;
  double l2_rel = 0.0;
};

static inline bf16_t f32_to_bf16(float x) {
  uint32_t bits = *reinterpret_cast<uint32_t *>(&x);
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  uint16_t hi = static_cast<uint16_t>(bits >> 16);
  bf16_t out{};
  *reinterpret_cast<uint16_t *>(&out) = hi;
  return out;
}

static inline float bf16_to_f32(bf16_t h) {
  uint16_t lo = *reinterpret_cast<uint16_t *>(&h);
  uint32_t hi = static_cast<uint32_t>(lo) << 16;
  float f = *reinterpret_cast<float *>(&hi);
  return f;
}

static ErrStats compare_vectors(const std::vector<float> &a,
                                const std::vector<float> &b) {
  assert(a.size() == b.size());
  ErrStats s;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double sum_ref_sq = 0.0;
  constexpr double eps = 1e-9;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    double ad = std::abs(diff);
    sum_abs += ad;
    sum_sq += diff * diff;
    if (ad > s.max_abs)
      s.max_abs = ad;
    double ref = static_cast<double>(b[i]);
    double denom = std::max(std::abs(ref), eps);
    double rel = ad / denom;
    if (rel > s.max_rel)
      s.max_rel = rel;
    sum_ref_sq += ref * ref;
  }
  s.mean_abs = sum_abs / std::max<size_t>(1, a.size());
  s.rmse = std::sqrt(sum_sq / std::max<size_t>(1, a.size()));
  s.l2_rel = std::sqrt(sum_sq / std::max(sum_ref_sq, eps));
  return s;
}

static void print_device_info() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  std::cout << "Device " << dev << ": " << prop.name
            << " | MultiProcessorCount=" << prop.multiProcessorCount
            << " | MaxThreadsPerBlock=" << prop.maxThreadsPerBlock << "\n";
}

struct TimingOptions {
  int warmup = 10;
  int iters = 100;
};

struct ModelSpec {
  std::string name;
  int hidden_size;
  int num_layers;
  int num_experts;
  int experts_per_token;
  int intermediate_size;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  int vocab_size;
  float swiglu_limit;
};

struct BenchmarkSetting {
  ModelSpec spec;
  int batch_size;
};

static const ModelSpec MODEL_20B{
    "20B", 2880, 24, 32, 4, 2880, 64, 8, 64, 201088, 7.0f};
static const ModelSpec MODEL_120B{
    "120B", 2880, 36, 128, 4, 2880, 64, 8, 64, 201088, 7.0f};

static const std::array<BenchmarkSetting, 2> g_settings = {
    BenchmarkSetting{MODEL_20B, 1536},
    BenchmarkSetting{MODEL_120B, 1024},
};

static inline std::vector<float> copy_device_float(float *d_ptr, size_t elems) {
  std::vector<float> out(elems, 0.0f);
  HIP_CHECK(hipMemcpy(out.data(), d_ptr, elems * sizeof(float),
                      hipMemcpyDeviceToHost));
  return out;
}

static inline std::vector<float> copy_device_bf16(bf16_t *d_ptr, size_t elems) {
  std::vector<bf16_t> tmp(elems);
  HIP_CHECK(hipMemcpy(tmp.data(), d_ptr, elems * sizeof(bf16_t),
                      hipMemcpyDeviceToHost));
  std::vector<float> out(elems);
  for (size_t i = 0; i < elems; ++i)
    out[i] = bf16_to_f32(tmp[i]);
  return out;
}

static void fill_random_bf16(std::vector<bf16_t> &dst, std::mt19937 &rng,
                             float min_val = -1.0f, float max_val = 1.0f) {
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (auto &v : dst)
    v = f32_to_bf16(dist(rng));
}

static void fill_random_float(std::vector<float> &dst, std::mt19937 &rng,
                              float min_val = -1.0f, float max_val = 1.0f) {
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (auto &v : dst)
    v = dist(rng);
}

static void fill_random_packed_matmul(bf16_t *dst, int rows, int cols,
                                      std::mt19937 &rng,
                                      float min_val = -1.0f,
                                      float max_val = 1.0f) {
  const size_t tiles_cols =
      ((size_t)rows + MATMUL_TILE_COLS - 1) / MATMUL_TILE_COLS;
  const size_t tiles_k =
      ((size_t)cols + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  const size_t tile_elems =
      (size_t)MATMUL_TILE_COLS * (size_t)MATMUL_TILE_K;
  const size_t group_stride =
      (size_t)MATMUL_TILE_COLS * (size_t)MATMUL_CHUNK_K;

  std::uniform_real_distribution<float> dist(min_val, max_val);

  for (size_t tile_col = 0; tile_col < tiles_cols; ++tile_col) {
    const int col_base = static_cast<int>(tile_col * MATMUL_TILE_COLS);
    const int col_block =
        std::max(0, std::min(rows - col_base, MATMUL_TILE_COLS));
    for (size_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
      const int k_base = static_cast<int>(tile_k * MATMUL_TILE_K);
      const int k_block =
          std::max(0, std::min(cols - k_base, MATMUL_TILE_K));
      bf16_t *tile_ptr =
          dst + (tile_col * tiles_k + tile_k) * tile_elems;
      std::fill(tile_ptr, tile_ptr + tile_elems, bf16_t(0.0f));
      if (col_block <= 0 || k_block <= 0)
        continue;

      for (int group = 0; group < MATMUL_TILE_K; group += MATMUL_CHUNK_K) {
        const int k_off = k_base + group;
        const int remaining = k_block - group;
        if (remaining <= 0)
          break;
        const int actual_chunk =
            remaining > MATMUL_CHUNK_K ? MATMUL_CHUNK_K : remaining;
        const size_t group_base =
            (size_t)(group / MATMUL_CHUNK_K) * group_stride;

        for (int col = 0; col < col_block; ++col) {
          const size_t dst_base = group_base + (size_t)col * MATMUL_CHUNK_K;
          for (int i = 0; i < actual_chunk; ++i) {
            tile_ptr[dst_base + i] = f32_to_bf16(dist(rng));
          }
        }
      }
    }
  }
}

static inline double matmul_flops(int m, int n, int k) {
  return 2.0 * static_cast<double>(m) * static_cast<double>(n) *
         static_cast<double>(k);
}

static bool accuracy_ok(const ErrStats &s, double max_abs = 5e-2,
                        double max_rel = 5e-3, double l2_rel = 5e-3) {
  return s.max_abs <= max_abs && s.max_rel <= max_rel && s.l2_rel <= l2_rel;
}

static float benchmark_kernel(const std::function<void()> &launch, int warmup,
                              int iters, hipStream_t stream) {
  for (int i = 0; i < warmup; ++i)
    launch();
  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  HIP_CHECK(hipEventRecord(ev_start, stream));
  for (int i = 0; i < iters; ++i)
    launch();
  HIP_CHECK(hipEventRecord(ev_stop, stream));
  HIP_CHECK(hipEventSynchronize(ev_stop));

  float ms_total = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms_total, ev_start, ev_stop));
  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));
  return ms_total / std::max(1, iters);
}

// -------------------------- MatMul: QKV --------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
__global__ void matmul_bias_qkv_kernel_variant(
    bf16_t *y, const bf16_t *x, const bf16_t *w, const float *bias, int n,
    int d, int B, const int *pos) {
  matmul_bias_bf16_mfma_body<BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                             WARP_TILE_N>(y, x, w, bias, n, d, B, pos);
}

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
static float launch_qkv_kernel(bf16_t *y, const bf16_t *x, const bf16_t *w,
                               const float *bias, int n, int d, int B,
                               const int *pos, hipStream_t stream) {
  static_assert(BLOCK_ROWS % WARP_TILE_M == 0, "invalid BLOCK_ROWS/WARP_TILE_M");
  static_assert(BLOCK_COLS % WARP_TILE_N == 0, "invalid BLOCK_COLS/WARP_TILE_N");
  constexpr int WAVES_M = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;
  dim3 grid((d + BLOCK_COLS - 1) / BLOCK_COLS,
            (B + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
  dim3 block(WF_SIZE, WAVES_PER_BLOCK, 1);
  hipLaunchKernelGGL((matmul_bias_qkv_kernel_variant<
                          BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                          WARP_TILE_N>),
                     grid, block, 0, stream, y, x, w, bias, n, d, B, pos);
  return 0.0f;
}

struct MatmulQKVProblem {
  int B;
  int n;
  int d;
  size_t output_elems;
  double flops;
};

struct MatmulQKVCandidate {
  std::string label;
  float (*launcher)(bf16_t *, const bf16_t *, const bf16_t *, const float *,
                    int, int, int, const int *, hipStream_t);
};

static MatmulQKVCandidate make_qkv_candidate_default() {
  MatmulQKVCandidate c{};
  c.label = "defines";
  c.launcher = [](bf16_t *y, const bf16_t *x, const bf16_t *w,
                  const float *bias, int n, int d, int B, const int *pos,
                  hipStream_t stream) -> float {
    dim3 grid((d + MATMUL_QKV_BLOCK_COLS - 1) / MATMUL_QKV_BLOCK_COLS,
              (B + MATMUL_QKV_BLOCK_ROWS - 1) / MATMUL_QKV_BLOCK_ROWS, 1);
    dim3 block(WF_SIZE, MATMUL_QKV_WAVES_PER_BLOCK, 1);
    hipLaunchKernelGGL(matmul_bias_qkv_kernel, grid, block, 0,
                       stream, y, x, w, bias, n, d, B, pos);
    return 0.0f;
  };
  return c;
}

template <typename Config>
MatmulQKVCandidate make_qkv_candidate(const char *prefix) {
  MatmulQKVCandidate cand{};
  cand.label = format_config_label<Config>(prefix);
  cand.launcher = [](bf16_t *y, const bf16_t *x, const bf16_t *w,
                     const float *bias, int n, int d, int B, const int *pos,
                     hipStream_t stream) -> float {
    return launch_qkv_kernel<Config::BM, Config::BN, Config::BK, Config::WM,
                             Config::WN>(y, x, w, bias, n, d, B, pos, stream);
  };
  return cand;
}

// -------------------------- MatMul: ATT --------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
__global__ void matmul_bias_att_kernel_variant(
    bf16_t *y, const bf16_t *x, const bf16_t *w, const float *bias, int n,
    int d, int B, const int *pos) {
  matmul_bias_bf16_mfma_body<BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                             WARP_TILE_N>(y, x, w, bias, n, d, B, pos);
}

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
static float launch_att_kernel(bf16_t *y, const bf16_t *x, const bf16_t *w,
                               const float *bias, int n, int d, int B,
                               const int *pos, hipStream_t stream) {
  static_assert(BLOCK_ROWS % WARP_TILE_M == 0, "invalid BLOCK_ROWS/WARP_TILE_M");
  static_assert(BLOCK_COLS % WARP_TILE_N == 0, "invalid BLOCK_COLS/WARP_TILE_N");
  constexpr int WAVES_M = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;
  dim3 grid((d + BLOCK_COLS - 1) / BLOCK_COLS,
            (B + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
  dim3 block(WF_SIZE, WAVES_PER_BLOCK, 1);
  hipLaunchKernelGGL((matmul_bias_att_kernel_variant<
                          BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                          WARP_TILE_N>),
                     grid, block, 0, stream, y, x, w, bias, n, d, B, pos);
  return 0.0f;
}

struct MatmulATTCandidate {
  std::string label;
  float (*launcher)(bf16_t *, const bf16_t *, const bf16_t *, const float *,
                    int, int, int, const int *, hipStream_t);
};

static MatmulATTCandidate make_att_candidate_default() {
  MatmulATTCandidate c{};
  c.label = "defines";
  c.launcher = [](bf16_t *y, const bf16_t *x, const bf16_t *w,
                  const float *bias, int n, int d, int B, const int *pos,
                  hipStream_t stream) -> float {
    dim3 grid((d + MATMUL_ATT_BLOCK_COLS - 1) / MATMUL_ATT_BLOCK_COLS,
              (B + MATMUL_ATT_BLOCK_ROWS - 1) / MATMUL_ATT_BLOCK_ROWS, 1);
    dim3 block(WF_SIZE, MATMUL_ATT_WAVES_PER_BLOCK, 1);
    hipLaunchKernelGGL(matmul_bias_att_kernel, grid, block, 0,
                       stream, y, x, w, bias, n, d, B, pos);
    return 0.0f;
  };
  return c;
}

template <typename Config>
MatmulATTCandidate make_att_candidate(const char *prefix) {
  MatmulATTCandidate cand{};
  cand.label = format_config_label<Config>(prefix);
  cand.launcher = [](bf16_t *y, const bf16_t *x, const bf16_t *w,
                     const float *bias, int n, int d, int B, const int *pos,
                     hipStream_t stream) -> float {
    return launch_att_kernel<Config::BM, Config::BN, Config::BK, Config::WM,
                             Config::WN>(y, x, w, bias, n, d, B, pos, stream);
  };
  return cand;
}

// -------------------------- MatMul: Logits --------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
__global__ void matmul_logits_kernel_variant(float *y,
                                                     const bf16_t *x,
                                                     const bf16_t *w, int n,
                                                     int d, int B,
                                                     const int *pos) {
  matmul_bf16_mfma_body<BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                        WARP_TILE_N>(y, x, w, n, d, B, pos);
}

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N>
static float launch_logits_kernel(float *y, const bf16_t *x, const bf16_t *w,
                                  int n, int d, int B, const int *pos,
                                  hipStream_t stream) {
  static_assert(BLOCK_ROWS % WARP_TILE_M == 0,
                "invalid BLOCK_ROWS/WARP_TILE_M");
  static_assert(BLOCK_COLS % WARP_TILE_N == 0,
                "invalid BLOCK_COLS/WARP_TILE_N");
  constexpr int WAVES_M = BLOCK_ROWS / WARP_TILE_M;
  constexpr int WAVES_N = BLOCK_COLS / WARP_TILE_N;
  constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;
  dim3 grid((d + BLOCK_COLS - 1) / BLOCK_COLS,
            (B + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
  dim3 block(WF_SIZE, WAVES_PER_BLOCK, 1);
  hipLaunchKernelGGL((matmul_logits_kernel_variant<
                          BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                          WARP_TILE_N>),
                     grid, block, 0, stream, y, x, w, n, d, B, pos);
  return 0.0f;
}

struct MatmulLogitsCandidate {
  std::string label;
  float (*launcher)(float *, const bf16_t *, const bf16_t *, int, int, int,
                    const int *, hipStream_t);
};

static MatmulLogitsCandidate make_logits_candidate_default() {
  MatmulLogitsCandidate c{};
  c.label = "defines";
  c.launcher = [](float *y, const bf16_t *x, const bf16_t *w, int n, int d,
                  int B, const int *pos, hipStream_t stream) -> float {
    dim3 grid((d + MATMUL_LOGITS_BLOCK_COLS - 1) / MATMUL_LOGITS_BLOCK_COLS,
              (B + MATMUL_LOGITS_BLOCK_ROWS - 1) / MATMUL_LOGITS_BLOCK_ROWS,
              1);
    dim3 block(WF_SIZE, MATMUL_LOGITS_WAVES_PER_BLOCK, 1);
    hipLaunchKernelGGL(matmul_logits_kernel, grid, block, 0, stream, y,
                       x, w, n, d, B, pos);
    return 0.0f;
  };
  return c;
}

template <typename Config>
MatmulLogitsCandidate make_logits_candidate(const char *prefix) {
  MatmulLogitsCandidate cand{};
  cand.label = format_config_label<Config>(prefix);
  cand.launcher = [](float *y, const bf16_t *x, const bf16_t *w, int n, int d,
                     int B, const int *pos, hipStream_t stream) -> float {
    return launch_logits_kernel<Config::BM, Config::BN, Config::BK, Config::WM,
                                Config::WN>(y, x, w, n, d, B, pos, stream);
  };
  return cand;
}

// -------------------------- MLP assignments --------------------------

struct MoeAssignments {
  std::vector<uint16_t> batches;
  std::vector<uint8_t> slots;
  std::vector<int> offsets;
  std::vector<int> counts;
  int max_count = 0;
};

static MoeAssignments build_moe_assignments(int B, int experts,
                                            int experts_per_token) {
  MoeAssignments out;
  out.offsets.resize(experts + 1, 0);
  out.counts.resize(experts, 0);

  const int total = B * experts_per_token;
  std::vector<std::vector<std::pair<int, int>>> per_expert(experts);
  int global_idx = 0;
  for (int b = 0; b < B; ++b) {
    for (int slot = 0; slot < experts_per_token; ++slot) {
      int expert = global_idx % experts;
      per_expert[expert].emplace_back(b, slot);
      ++global_idx;
    }
  }

  out.batches.reserve(total);
  out.slots.reserve(total);

  for (int e = 0; e < experts; ++e) {
    out.offsets[e + 1] = out.offsets[e] + static_cast<int>(per_expert[e].size());
    out.counts[e] = static_cast<int>(per_expert[e].size());
    out.max_count = std::max(out.max_count, out.counts[e]);
    for (auto [batch, slot] : per_expert[e]) {
      out.batches.push_back(static_cast<uint16_t>(batch));
      out.slots.push_back(static_cast<uint8_t>(slot));
    }
  }

  return out;
}

// -------------------------- MLP1 --------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N, int WAVES_PER_BLOCK>
__global__ void mlp1_kernel_variant(
    bf16_t *gate_up_topk, const bf16_t *x, const bf16_t *w_mlp1_all,
    size_t stride_w_mlp1, const bf16_t *b_mlp1_all,
    const uint16_t *assignment_batches, const uint8_t *assignment_slots,
    const int *expert_offsets, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, const int *pos) {
  mlp1_kernel_body<BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH,
                              WARP_TILE_M, WARP_TILE_N, WAVES_PER_BLOCK>(
      gate_up_topk, x, w_mlp1_all, stride_w_mlp1, b_mlp1_all,
      assignment_batches, assignment_slots, expert_offsets, l_layer, E, H, IM,
      swiglu_limit, batch_size, pos);
}

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N, int WAVES_PER_BLOCK>
static float launch_mlp1_kernel(
    bf16_t *gate_up_topk, const bf16_t *x, const bf16_t *w_mlp1_all,
    size_t stride_w_mlp1, const bf16_t *b_mlp1_all,
    const uint16_t *assignment_batches, const uint8_t *assignment_slots,
    const int *expert_offsets, int l_layer, int E, int H, int IM,
    float swiglu_limit, int batch_size, const int *pos, int max_assign_per_expert,
    hipStream_t stream) {
  static_assert(BLOCK_ROWS % WARP_TILE_M == 0,
                "invalid BLOCK_ROWS/WARP_TILE_M");
  static_assert(BLOCK_COLS % WARP_TILE_N == 0,
                "invalid BLOCK_COLS/WARP_TILE_N");
  constexpr int COMPUTED_WAVES =
      (BLOCK_ROWS / WARP_TILE_M) * (BLOCK_COLS / WARP_TILE_N);
  static_assert(COMPUTED_WAVES == WAVES_PER_BLOCK,
                "waves per block mismatch");
  const int max_tiles =
      (max_assign_per_expert + BLOCK_ROWS - 1) / BLOCK_ROWS;
  dim3 grid((2 * IM + BLOCK_COLS - 1) / BLOCK_COLS, max_tiles, E);
  dim3 block(WF_SIZE, WAVES_PER_BLOCK, 1);
  hipLaunchKernelGGL((mlp1_kernel_variant<
                          BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                          WARP_TILE_N, WAVES_PER_BLOCK>),
                     grid, block, 0, stream, gate_up_topk, x, w_mlp1_all,
                     stride_w_mlp1, b_mlp1_all, assignment_batches,
                     assignment_slots, expert_offsets, l_layer, E, H, IM,
                     swiglu_limit, batch_size, pos);
  return 0.0f;
}

struct MLP1Candidate {
  std::string label;
  float (*launcher)(bf16_t *, const bf16_t *, const bf16_t *, size_t,
                    const bf16_t *, const uint16_t *, const uint8_t *,
                    const int *, int, int, int, int, float, int, const int *,
                    int, hipStream_t);
};

static MLP1Candidate make_mlp1_candidate_default() {
  MLP1Candidate c{};
  c.label = "defines";
  c.launcher = [](bf16_t *gate_up_topk, const bf16_t *x,
                  const bf16_t *w_mlp1_all, size_t stride_w_mlp1,
                  const bf16_t *b_mlp1_all, const uint16_t *assignment_batches,
                  const uint8_t *assignment_slots, const int *expert_offsets,
                  int l_layer, int E, int H, int IM, float swiglu_limit,
                  int batch_size, const int *pos, int max_assign_per_expert,
                  hipStream_t stream) -> float {
    const int max_tiles =
        (max_assign_per_expert + MATMUL_MLP1_BLOCK_ROWS - 1) /
        MATMUL_MLP1_BLOCK_ROWS;
    dim3 grid((2 * IM + MATMUL_MLP1_BLOCK_COLS - 1) /
                  MATMUL_MLP1_BLOCK_COLS,
              max_tiles, E);
    dim3 block(WF_SIZE, MATMUL_MLP1_WAVES_PER_BLOCK, 1);
    hipLaunchKernelGGL(mlp1_kernel, grid, block, 0, stream,
                       gate_up_topk, x, w_mlp1_all, stride_w_mlp1, b_mlp1_all,
                       assignment_batches, assignment_slots, expert_offsets,
                       l_layer, E, H, IM, swiglu_limit, batch_size, pos);
    return 0.0f;
  };
  return c;
}

template <typename Config>
MLP1Candidate make_mlp1_candidate(const char *prefix) {
  MLP1Candidate cand{};
  cand.label = format_config_label<Config>(prefix);
  cand.launcher = [](bf16_t *gate_up_topk, const bf16_t *x,
                     const bf16_t *w_mlp1_all, size_t stride_w_mlp1,
                     const bf16_t *b_mlp1_all,
                     const uint16_t *assignment_batches,
                     const uint8_t *assignment_slots,
                     const int *expert_offsets, int l_layer, int E, int H,
                     int IM, float swiglu_limit, int batch_size,
                     const int *pos, int max_assign_per_expert,
                     hipStream_t stream) -> float {
    return launch_mlp1_kernel<Config::BM, Config::BN, Config::BK, Config::WM,
                              Config::WN, Config::WavesPerWg>(
        gate_up_topk, x, w_mlp1_all, stride_w_mlp1, b_mlp1_all,
        assignment_batches, assignment_slots, expert_offsets, l_layer, E, H,
        IM, swiglu_limit, batch_size, pos, max_assign_per_expert, stream);
  };
  return cand;
}

// -------------------------- MLP2 --------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N, int WAVES_PER_BLOCK>
__global__ void mlp2_kernel_variant(
    float *e_agg, const bf16_t *gate_up_topk, const bf16_t *w_mlp2_all,
    size_t stride_w_mlp2, const bf16_t *b_mlp2_all,
    const uint16_t *assignment_batches, const uint8_t *assignment_slots,
    const int *expert_offsets, const float *topk_v, int l_layer, int E, int IM,
    int H, int batch_size, const int *pos) {
  mlp2_kernel_body<
      BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M, WARP_TILE_N,
      WAVES_PER_BLOCK>(e_agg, gate_up_topk, w_mlp2_all, stride_w_mlp2,
                       b_mlp2_all, assignment_batches, assignment_slots,
                       expert_offsets, topk_v, l_layer, E, IM, H, batch_size,
                       pos);
}

template <int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int WARP_TILE_M,
          int WARP_TILE_N, int WAVES_PER_BLOCK>
static float launch_mlp2_kernel(
    float *e_agg, const bf16_t *gate_up_topk, const bf16_t *w_mlp2_all,
    size_t stride_w_mlp2, const bf16_t *b_mlp2_all,
    const uint16_t *assignment_batches, const uint8_t *assignment_slots,
    const int *expert_offsets, const float *topk_v, int l_layer, int E, int IM,
    int H, int batch_size, const int *pos, int max_assign_per_expert,
    hipStream_t stream) {
  static_assert(BLOCK_ROWS % WARP_TILE_M == 0,
                "invalid BLOCK_ROWS/WARP_TILE_M");
  static_assert(BLOCK_COLS % WARP_TILE_N == 0,
                "invalid BLOCK_COLS/WARP_TILE_N");
  constexpr int COMPUTED_WAVES =
      (BLOCK_ROWS / WARP_TILE_M) * (BLOCK_COLS / WARP_TILE_N);
  static_assert(COMPUTED_WAVES == WAVES_PER_BLOCK,
                "waves per block mismatch");
  const int max_tiles =
      (max_assign_per_expert + BLOCK_ROWS - 1) / BLOCK_ROWS;
  dim3 grid((H + BLOCK_COLS - 1) / BLOCK_COLS, max_tiles, E);
  dim3 block(WF_SIZE, WAVES_PER_BLOCK, 1);
  hipLaunchKernelGGL((mlp2_kernel_variant<
                          BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, WARP_TILE_M,
                          WARP_TILE_N, WAVES_PER_BLOCK>),
                     grid, block, 0, stream, e_agg, gate_up_topk, w_mlp2_all,
                     stride_w_mlp2, b_mlp2_all, assignment_batches,
                     assignment_slots, expert_offsets, topk_v, l_layer, E, IM,
                     H, batch_size, pos);
  return 0.0f;
}

struct MLP2Candidate {
  std::string label;
  float (*launcher)(float *, const bf16_t *, const bf16_t *, size_t,
                    const bf16_t *, const uint16_t *, const uint8_t *,
                    const int *, const float *, int, int, int, int, int,
                    const int *, int, hipStream_t);
};

static MLP2Candidate make_mlp2_candidate_default() {
  MLP2Candidate c{};
  c.label = "defines";
  c.launcher = [](float *e_agg, const bf16_t *gate_up_topk,
                  const bf16_t *w_mlp2_all, size_t stride_w_mlp2,
                  const bf16_t *b_mlp2_all,
                  const uint16_t *assignment_batches,
                  const uint8_t *assignment_slots, const int *expert_offsets,
                  const float *topk_v, int l_layer, int E, int IM, int H,
                  int batch_size, const int *pos, int max_assign_per_expert,
                  hipStream_t stream) -> float {
    const int max_tiles =
        (max_assign_per_expert + MATMUL_MLP2_BLOCK_ROWS - 1) /
        MATMUL_MLP2_BLOCK_ROWS;
    dim3 grid((H + MATMUL_MLP2_BLOCK_COLS - 1) / MATMUL_MLP2_BLOCK_COLS,
              max_tiles, E);
    dim3 block(WF_SIZE, MATMUL_MLP2_WAVES_PER_BLOCK, 1);
    hipLaunchKernelGGL(mlp2_kernel, grid, block, 0,
                       stream, e_agg, gate_up_topk, w_mlp2_all, stride_w_mlp2,
                       b_mlp2_all, assignment_batches, assignment_slots,
                       expert_offsets, topk_v, l_layer, E, IM, H, batch_size,
                       pos);
    return 0.0f;
  };
  return c;
}

template <typename Config>
MLP2Candidate make_mlp2_candidate(const char *prefix) {
  MLP2Candidate cand{};
  cand.label = format_config_label<Config>(prefix);
  cand.launcher = [](float *e_agg, const bf16_t *gate_up_topk,
                     const bf16_t *w_mlp2_all, size_t stride_w_mlp2,
                     const bf16_t *b_mlp2_all,
                     const uint16_t *assignment_batches,
                     const uint8_t *assignment_slots,
                     const int *expert_offsets, const float *topk_v,
                     int l_layer, int E, int IM, int H, int batch_size,
                     const int *pos, int max_assign_per_expert,
                     hipStream_t stream) -> float {
    return launch_mlp2_kernel<Config::BM, Config::BN, Config::BK, Config::WM,
                              Config::WN, Config::WavesPerWg>(
        e_agg, gate_up_topk, w_mlp2_all, stride_w_mlp2, b_mlp2_all,
        assignment_batches, assignment_slots, expert_offsets, topk_v, l_layer,
        E, IM, H, batch_size, pos, max_assign_per_expert, stream);
  };
  return cand;
}

// -------------------------- Utilities --------------------------

struct RunResult {
  float ms = 0.0f;
  double tflops = 0.0;
  std::vector<float> output;
};

static RunResult run_qkv_variant(const MatmulQKVProblem &prob,
                                 bf16_t *d_y, const bf16_t *d_x,
                                 const bf16_t *d_w, const float *d_bias,
                                 const int *d_pos, hipStream_t stream,
                                 const TimingOptions &opts,
                                 const MatmulQKVCandidate &cand) {
  HIP_CHECK(hipMemsetAsync(d_y, 0, prob.output_elems * sizeof(bf16_t), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto launch = [&]() {
    cand.launcher(d_y, d_x, d_w, d_bias, prob.n, prob.d, prob.B, d_pos, stream);
  };

  float ms = benchmark_kernel(launch, opts.warmup, opts.iters, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  RunResult out;
  out.ms = ms;
  out.tflops = (prob.flops / (ms * 1e-3)) / 1e12;
  out.output = copy_device_bf16(d_y, prob.output_elems);
  return out;
}

struct MatmulATTProblem {
  int B;
  int n;
  int d;
  size_t output_elems;
  double flops;
};

static RunResult run_att_variant(const MatmulATTProblem &prob, bf16_t *d_y,
                                 const bf16_t *d_x, const bf16_t *d_w,
                                 const float *d_bias, const int *d_pos,
                                 hipStream_t stream, const TimingOptions &opts,
                                 const MatmulATTCandidate &cand) {
  HIP_CHECK(hipMemsetAsync(d_y, 0, prob.output_elems * sizeof(bf16_t), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto launch = [&]() {
    cand.launcher(d_y, d_x, d_w, d_bias, prob.n, prob.d, prob.B, d_pos, stream);
  };

  float ms = benchmark_kernel(launch, opts.warmup, opts.iters, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  RunResult out;
  out.ms = ms;
  out.tflops = (prob.flops / (ms * 1e-3)) / 1e12;
  out.output = copy_device_bf16(d_y, prob.output_elems);
  return out;
}

struct MatmulLogitsProblem {
  int B;
  int n;
  int d;
  size_t output_elems;
  double flops;
};

static RunResult run_logits_variant(const MatmulLogitsProblem &prob, float *d_y,
                                    const bf16_t *d_x, const bf16_t *d_w,
                                    const int *d_pos, hipStream_t stream,
                                    const TimingOptions &opts,
                                    const MatmulLogitsCandidate &cand) {
  HIP_CHECK(hipMemsetAsync(d_y, 0, prob.output_elems * sizeof(float), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto launch = [&]() {
    cand.launcher(d_y, d_x, d_w, prob.n, prob.d, prob.B, d_pos, stream);
  };

  float ms = benchmark_kernel(launch, opts.warmup, opts.iters, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  RunResult out;
  out.ms = ms;
  out.tflops = (prob.flops / (ms * 1e-3)) / 1e12;
  out.output = copy_device_float(d_y, prob.output_elems);
  return out;
}

struct MLP1Problem {
  int B;
  int E;
  int H;
  int IM;
  int max_assign;
  int total_assign;
  size_t output_elems;
  double flops;
};

static RunResult run_mlp1_variant(
    const MLP1Problem &prob, bf16_t *d_gate_up, const bf16_t *d_x,
    const bf16_t *d_w, size_t stride_w, const bf16_t *d_bias,
    const uint16_t *d_batches, const uint8_t *d_slots, const int *d_offsets,
    int l_layer, float swiglu_limit, const int *d_pos, hipStream_t stream,
    const TimingOptions &opts, const MLP1Candidate &cand) {
  HIP_CHECK(hipMemsetAsync(d_gate_up, 0, prob.output_elems * sizeof(bf16_t),
                           stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto launch = [&]() {
    cand.launcher(d_gate_up, d_x, d_w, stride_w, d_bias, d_batches, d_slots,
                  d_offsets, l_layer, prob.E, prob.H, prob.IM, swiglu_limit,
                  prob.B, d_pos, prob.max_assign, stream);
  };

  float ms = benchmark_kernel(launch, opts.warmup, opts.iters, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  RunResult out;
  out.ms = ms;
  out.tflops = (prob.flops / (ms * 1e-3)) / 1e12;
  out.output = copy_device_bf16(d_gate_up, prob.output_elems);
  return out;
}

struct MLP2Problem {
  int B;
  int E;
  int H;
  int IM;
  int max_assign;
  int total_assign;
  size_t output_elems;
  double flops;
};

static RunResult run_mlp2_variant(
    const MLP2Problem &prob, float *d_out, const bf16_t *d_gate_up,
    const bf16_t *d_w, size_t stride_w, const bf16_t *d_bias,
    const uint16_t *d_batches, const uint8_t *d_slots, const int *d_offsets,
    const float *d_topk_v, int l_layer, const int *d_pos, hipStream_t stream,
    const TimingOptions &opts, const MLP2Candidate &cand) {
  HIP_CHECK(hipMemsetAsync(d_out, 0, prob.output_elems * sizeof(float),
                           stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto launch = [&]() {
    cand.launcher(d_out, d_gate_up, d_w, stride_w, d_bias, d_batches, d_slots,
                  d_offsets, d_topk_v, l_layer, prob.E, prob.IM, prob.H,
                  prob.B, d_pos, prob.max_assign, stream);
  };

  float ms = benchmark_kernel(launch, opts.warmup, opts.iters, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  RunResult out;
  out.ms = ms;
  out.tflops = (prob.flops / (ms * 1e-3)) / 1e12;
  out.output = copy_device_float(d_out, prob.output_elems);
  return out;
}

static std::vector<MatmulQKVCandidate> build_qkv_candidates() {
  std::vector<MatmulQKVCandidate> out;
  out.reserve(24);
#define ADD_QKV_CONFIG(BM, BN, BK, WM, WN)                                      \
  out.emplace_back(                                                            \
      make_qkv_candidate<MatmulConfig<BM, BN, BK, WM, WN, 16, 16>>("QKV"));
  MATMUL_CONFIG_LIST(ADD_QKV_CONFIG)
#undef ADD_QKV_CONFIG
  return out;
}

static std::vector<MatmulATTCandidate> build_att_candidates() {
  std::vector<MatmulATTCandidate> out;
  out.reserve(24);
#define ADD_ATT_CONFIG(BM, BN, BK, WM, WN)                                      \
  out.emplace_back(                                                            \
      make_att_candidate<MatmulConfig<BM, BN, BK, WM, WN, 16, 16>>("ATT"));
  MATMUL_CONFIG_LIST(ADD_ATT_CONFIG)
#undef ADD_ATT_CONFIG
  return out;
}

static std::vector<MatmulLogitsCandidate> build_logits_candidates() {
  std::vector<MatmulLogitsCandidate> out;
  out.reserve(24);
#define ADD_LOGITS_CONFIG(BM, BN, BK, WM, WN)                                   \
  out.emplace_back(make_logits_candidate<MatmulConfig<BM, BN, BK, WM, WN, 16,   \
                                           16>>("LOGITS"));
  MATMUL_CONFIG_LIST(ADD_LOGITS_CONFIG)
#undef ADD_LOGITS_CONFIG
  return out;
}

static std::vector<MLP1Candidate> build_mlp1_candidates() {
  std::vector<MLP1Candidate> out;
  out.reserve(24);
#define ADD_MLP1_CONFIG(BM, BN, BK, WM, WN)                                     \
  out.emplace_back(                                                            \
      make_mlp1_candidate<MatmulConfig<BM, BN, BK, WM, WN, 16, 16>>("MLP1"));
  MATMUL_CONFIG_LIST(ADD_MLP1_CONFIG)
#undef ADD_MLP1_CONFIG
  return out;
}

static std::vector<MLP2Candidate> build_mlp2_candidates() {
  std::vector<MLP2Candidate> out;
  out.reserve(24);
#define ADD_MLP2_CONFIG(BM, BN, BK, WM, WN)                                     \
  out.emplace_back(                                                            \
      make_mlp2_candidate<MatmulConfig<BM, BN, BK, WM, WN, 16, 16>>("MLP2"));
  MATMUL_CONFIG_LIST(ADD_MLP2_CONFIG)
#undef ADD_MLP2_CONFIG
  return out;
}

#undef MATMUL_CONFIG_LIST

// -------------------------- Tuners --------------------------

static void tune_matmul_qkv(const BenchmarkSetting &setting,
                            const TimingOptions &opts, hipStream_t stream) {
  const ModelSpec &spec = setting.spec;
  const int B = setting.batch_size;
  const int H = spec.hidden_size;
  const int D = spec.head_dim;
  const int Hq = spec.num_attention_heads;
  const int Hk = spec.num_key_value_heads;
  const int QKV_D = D * (Hq + 2 * Hk);

  MatmulQKVProblem prob{B, H, QKV_D, (size_t)B * QKV_D,
                        matmul_flops(B, QKV_D, H)};

  std::cout << "\n== MatMul QKV | Model " << spec.name << " | B=" << B
            << " ==\n";

  std::mt19937 rng(1337);
  std::vector<bf16_t> h_x((size_t)B * H);
  std::vector<float> h_bias(QKV_D);

  fill_random_bf16(h_x, rng);
  fill_random_float(h_bias, rng);

  const size_t stride = matmul_packed_elems(QKV_D, H);
  std::vector<bf16_t> h_w(stride);
  fill_random_packed_matmul(h_w.data(), QKV_D, H, rng);

  std::vector<int> h_pos(B, 0);

  bf16_t *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
  float *d_bias = nullptr;
  int *d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_y, prob.output_elems * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_bias, h_bias.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  MatmulQKVCandidate baseline = make_qkv_candidate_default();
  RunResult base_run = run_qkv_variant(prob, d_y, d_x, d_w, d_bias, d_pos,
                                       stream, opts, baseline);

  std::cout << "baseline (" << baseline.label << "): " << base_run.ms
            << " ms | " << base_run.tflops << " TFLOPS\n";

  static const auto candidates = build_qkv_candidates();

  for (const auto &cand : candidates) {
    RunResult trial = run_qkv_variant(prob, d_y, d_x, d_w, d_bias, d_pos,
                                      stream, opts, cand);
    ErrStats err = compare_vectors(trial.output, base_run.output);
    const bool ok = accuracy_ok(err);
    const double speedup = trial.ms > 0.0 ? base_run.ms / trial.ms : 0.0;
    const bool has_speedup = speedup >= 1.0;
    if (!ok) {
      if (has_speedup) {
        std::cout << "  [discard] " << cand.label << " | " << trial.ms
                  << " ms | " << trial.tflops << " TFLOPS | speedup x"
                  << speedup << " | accuracy fail (max_abs=" << err.max_abs
                  << ", max_rel=" << err.max_rel << ", l2_rel="
                  << err.l2_rel << ")\n";
      }
      continue;
    }
    if (has_speedup) {
      std::cout << "  [faster] " << cand.label << " | " << trial.ms << " ms | "
                << trial.tflops << " TFLOPS | speedup x" << speedup << "\n";
    }
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_bias));
  HIP_CHECK(hipFree(d_pos));
}

static void tune_matmul_att(const BenchmarkSetting &setting,
                            const TimingOptions &opts, hipStream_t stream) {
  const ModelSpec &spec = setting.spec;
  const int B = setting.batch_size;
  const int H = spec.hidden_size;
  const int D = spec.head_dim;
  const int Hq = spec.num_attention_heads;
  const int O_N = D * Hq;

  MatmulATTProblem prob{B, O_N, H, (size_t)B * H, matmul_flops(B, H, O_N)};

  std::cout << "\n== MatMul AttnOut | Model " << spec.name << " | B=" << B
            << " ==\n";

  std::mt19937 rng(2024);
  std::vector<bf16_t> h_x((size_t)B * O_N);
  std::vector<float> h_bias(H);
  fill_random_bf16(h_x, rng);
  fill_random_float(h_bias, rng);

  const size_t stride = matmul_packed_elems(H, O_N);
  std::vector<bf16_t> h_w(stride);
  fill_random_packed_matmul(h_w.data(), H, O_N, rng);

  std::vector<int> h_pos(B, 0);

  bf16_t *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
  float *d_bias = nullptr;
  int *d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_y, prob.output_elems * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_bias, h_bias.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  MatmulATTCandidate baseline = make_att_candidate_default();
  RunResult base_run = run_att_variant(prob, d_y, d_x, d_w, d_bias, d_pos,
                                       stream, opts, baseline);

  std::cout << "baseline (" << baseline.label << "): " << base_run.ms
            << " ms | " << base_run.tflops << " TFLOPS\n";

  static const auto candidates = build_att_candidates();

  for (const auto &cand : candidates) {
    RunResult trial = run_att_variant(prob, d_y, d_x, d_w, d_bias, d_pos,
                                      stream, opts, cand);
    ErrStats err = compare_vectors(trial.output, base_run.output);
    const bool ok = accuracy_ok(err);
    const double speedup = trial.ms > 0.0 ? base_run.ms / trial.ms : 0.0;
    const bool has_speedup = speedup >= 1.0;
    if (!ok) {
      if (has_speedup) {
        std::cout << "  [discard] " << cand.label << " | " << trial.ms
                  << " ms | " << trial.tflops << " TFLOPS | speedup x"
                  << speedup << " | accuracy fail (max_abs=" << err.max_abs
                  << ", max_rel=" << err.max_rel << ", l2_rel="
                  << err.l2_rel << ")\n";
      }
      continue;
    }
    if (has_speedup) {
      std::cout << "  [faster] " << cand.label << " | " << trial.ms << " ms | "
                << trial.tflops << " TFLOPS | speedup x" << speedup << "\n";
    }
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_bias));
  HIP_CHECK(hipFree(d_pos));
}

static void tune_matmul_logits(const BenchmarkSetting &setting,
                               const TimingOptions &opts,
                               hipStream_t stream) {
  const ModelSpec &spec = setting.spec;
  const int B = setting.batch_size;
  const int H = spec.hidden_size;
  const int V = spec.vocab_size;

  MatmulLogitsProblem prob{B, H, V, (size_t)B * V, matmul_flops(B, V, H)};

  std::cout << "\n== MatMul Logits | Model " << spec.name << " | B=" << B
            << " ==\n";

  std::mt19937 rng(4096);
  std::vector<bf16_t> h_x((size_t)B * H);
  fill_random_bf16(h_x, rng);

  const size_t stride = matmul_packed_elems(V, H);
  std::vector<bf16_t> h_w(stride);
  fill_random_packed_matmul(h_w.data(), V, H, rng);

  std::vector<int> h_pos(B, 0);

  bf16_t *d_x = nullptr, *d_w = nullptr;
  float *d_y = nullptr;
  int *d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_y, prob.output_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  MatmulLogitsCandidate baseline = make_logits_candidate_default();
  RunResult base_run =
      run_logits_variant(prob, d_y, d_x, d_w, d_pos, stream, opts, baseline);

  std::cout << "baseline (" << baseline.label << "): " << base_run.ms
            << " ms | " << base_run.tflops << " TFLOPS\n";

  static const auto candidates = build_logits_candidates();

  for (const auto &cand : candidates) {
    RunResult trial =
        run_logits_variant(prob, d_y, d_x, d_w, d_pos, stream, opts, cand);
    ErrStats err = compare_vectors(trial.output, base_run.output);
    const bool ok = accuracy_ok(err, 5e-2, 5e-3, 5e-3);
    const double speedup = trial.ms > 0.0 ? base_run.ms / trial.ms : 0.0;
    const bool has_speedup = speedup >= 1.0;
    if (!ok) {
      if (has_speedup) {
        std::cout << "  [discard] " << cand.label << " | " << trial.ms
                  << " ms | " << trial.tflops << " TFLOPS | speedup x"
                  << speedup << " | accuracy fail (max_abs=" << err.max_abs
                  << ", max_rel=" << err.max_rel << ", l2_rel="
                  << err.l2_rel << ")\n";
      }
      continue;
    }
    if (has_speedup) {
      std::cout << "  [faster] " << cand.label << " | " << trial.ms << " ms | "
                << trial.tflops << " TFLOPS | speedup x" << speedup << "\n";
    }
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_pos));
}

static void tune_mlp1(const BenchmarkSetting &setting, const TimingOptions &opts,
                      hipStream_t stream) {
  const ModelSpec &spec = setting.spec;
  const int B = setting.batch_size;
  const int H = spec.hidden_size;
  const int E = spec.num_experts;
  const int IM = spec.intermediate_size;
  const int K = spec.experts_per_token;

  MoeAssignments assigns = build_moe_assignments(B, E, K);

  const int total_assign = static_cast<int>(assigns.batches.size());
  MLP1Problem prob{B,
                   E,
                   H,
                   IM,
                   assigns.max_count,
                   total_assign,
                   (size_t)K * (size_t)B * (size_t)IM,
                   4.0 * static_cast<double>(H) * static_cast<double>(IM) *
                       static_cast<double>(total_assign)};

  std::cout << "\n== MLP1 | Model " << spec.name << " | B=" << B
            << " ==\n";

  std::mt19937 rng(777);
  std::vector<bf16_t> h_x((size_t)B * H);
  fill_random_bf16(h_x, rng);

  const size_t stride = matmul_packed_elems(2 * IM, H);
  std::vector<bf16_t> h_w((size_t)E * stride);
  for (int e = 0; e < E; ++e)
    fill_random_packed_matmul(h_w.data() + (size_t)e * stride, 2 * IM, H, rng);

  std::vector<bf16_t> h_bias((size_t)E * 2 * IM);
  fill_random_bf16(h_bias, rng);

  std::vector<int> h_pos(B, 0);

  bf16_t *d_x = nullptr, *d_w = nullptr, *d_bias = nullptr, *d_gate = nullptr;
  uint16_t *d_batches = nullptr;
  uint8_t *d_slots = nullptr;
  int *d_offsets = nullptr;
  int *d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_bias, h_bias.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_gate, prob.output_elems * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_batches, assigns.batches.size() * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&d_slots, assigns.slots.size() * sizeof(uint8_t)));
  HIP_CHECK(hipMalloc(&d_offsets, assigns.offsets.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_batches, assigns.batches.data(),
                      assigns.batches.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_slots, assigns.slots.data(),
                      assigns.slots.size() * sizeof(uint8_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_offsets, assigns.offsets.data(),
                      assigns.offsets.size() * sizeof(int),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  MLP1Candidate baseline = make_mlp1_candidate_default();
  RunResult base_run = run_mlp1_variant(prob, d_gate, d_x, d_w, stride, d_bias,
                                        d_batches, d_slots, d_offsets, 0,
                                        spec.swiglu_limit, d_pos, stream, opts,
                                        baseline);

  std::cout << "baseline (" << baseline.label << "): " << base_run.ms
            << " ms | " << base_run.tflops << " TFLOPS\n";

  static const auto candidates = build_mlp1_candidates();

  for (const auto &cand : candidates) {
    RunResult trial = run_mlp1_variant(prob, d_gate, d_x, d_w, stride, d_bias,
                                       d_batches, d_slots, d_offsets, 0,
                                       spec.swiglu_limit, d_pos, stream, opts,
                                       cand);
    ErrStats err = compare_vectors(trial.output, base_run.output);
    const bool ok = accuracy_ok(err, 1e-1, 1e-2, 1e-2);
    const double speedup = trial.ms > 0.0 ? base_run.ms / trial.ms : 0.0;
    const bool has_speedup = speedup >= 1.0;
    if (!ok) {
      if (has_speedup) {
        std::cout << "  [discard] " << cand.label << " | " << trial.ms
                  << " ms | " << trial.tflops << " TFLOPS | speedup x"
                  << speedup << " | accuracy fail (max_abs=" << err.max_abs
                  << ", max_rel=" << err.max_rel << ", l2_rel="
                  << err.l2_rel << ")\n";
      }
      continue;
    }
    if (has_speedup) {
      std::cout << "  [faster] " << cand.label << " | " << trial.ms << " ms | "
                << trial.tflops << " TFLOPS | speedup x" << speedup << "\n";
    }
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_bias));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_batches));
  HIP_CHECK(hipFree(d_slots));
  HIP_CHECK(hipFree(d_offsets));
  HIP_CHECK(hipFree(d_pos));
}

static void tune_mlp2(const BenchmarkSetting &setting, const TimingOptions &opts,
                      hipStream_t stream) {
  const ModelSpec &spec = setting.spec;
  const int B = setting.batch_size;
  const int H = spec.hidden_size;
  const int E = spec.num_experts;
  const int IM = spec.intermediate_size;
  const int K = spec.experts_per_token;

  MoeAssignments assigns = build_moe_assignments(B, E, K);

  const int total_assign = static_cast<int>(assigns.batches.size());
  MLP2Problem prob{B,
                   E,
                   H,
                   IM,
                   assigns.max_count,
                   total_assign,
                   (size_t)B * H,
                   2.0 * static_cast<double>(IM) * static_cast<double>(H) *
                       static_cast<double>(total_assign)};

  std::cout << "\n== MLP2 | Model " << spec.name << " | B=" << B
            << " ==\n";

  std::mt19937 rng(314159);
  std::vector<bf16_t> h_gate((size_t)K * B * IM);
  fill_random_bf16(h_gate, rng);

  const size_t stride = matmul_packed_elems(H, IM);
  std::vector<bf16_t> h_w((size_t)E * stride);
  for (int e = 0; e < E; ++e)
    fill_random_packed_matmul(h_w.data() + (size_t)e * stride, H, IM, rng);

  std::vector<bf16_t> h_bias((size_t)E * H);
  fill_random_bf16(h_bias, rng);

  std::vector<float> h_topk((size_t)B * K);
  fill_random_float(h_topk, rng, 0.0f, 1.0f);

  std::vector<int> h_pos(B, 0);

  float *d_out = nullptr;
  bf16_t *d_gate = nullptr, *d_w = nullptr, *d_bias = nullptr;
  float *d_topk = nullptr;
  uint16_t *d_batches = nullptr;
  uint8_t *d_slots = nullptr;
  int *d_offsets = nullptr;
  int *d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_out, prob.output_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, h_gate.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_bias, h_bias.size() * sizeof(bf16_t)));
  HIP_CHECK(hipMalloc(&d_topk, h_topk.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_batches, assigns.batches.size() * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&d_slots, assigns.slots.size() * sizeof(uint8_t)));
  HIP_CHECK(hipMalloc(&d_offsets, assigns.offsets.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(bf16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_batches, assigns.batches.data(),
                      assigns.batches.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_slots, assigns.slots.data(),
                      assigns.slots.size() * sizeof(uint8_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_offsets, assigns.offsets.data(),
                      assigns.offsets.size() * sizeof(int),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  MLP2Candidate baseline = make_mlp2_candidate_default();
  RunResult base_run = run_mlp2_variant(prob, d_out, d_gate, d_w, stride,
                                        d_bias, d_batches, d_slots, d_offsets,
                                        d_topk, 0, d_pos, stream, opts,
                                        baseline);

  std::cout << "baseline (" << baseline.label << "): " << base_run.ms
            << " ms | " << base_run.tflops << " TFLOPS\n";

  static const auto candidates = build_mlp2_candidates();

  for (const auto &cand : candidates) {
    RunResult trial = run_mlp2_variant(prob, d_out, d_gate, d_w, stride, d_bias,
                                       d_batches, d_slots, d_offsets, d_topk, 0,
                                       d_pos, stream, opts, cand);
    ErrStats err = compare_vectors(trial.output, base_run.output);
    const bool ok = accuracy_ok(err, 1e-1, 1e-2, 1e-2);
    const double speedup = trial.ms > 0.0 ? base_run.ms / trial.ms : 0.0;
    const bool has_speedup = speedup >= 1.0;
    if (!ok) {
      if (has_speedup) {
        std::cout << "  [discard] " << cand.label << " | " << trial.ms
                  << " ms | " << trial.tflops << " TFLOPS | speedup x"
                  << speedup << " | accuracy fail (max_abs=" << err.max_abs
                  << ", max_rel=" << err.max_rel << ", l2_rel="
                  << err.l2_rel << ")\n";
      }
      continue;
    }
    if (has_speedup) {
      std::cout << "  [faster] " << cand.label << " | " << trial.ms << " ms | "
                << trial.tflops << " TFLOPS | speedup x" << speedup << "\n";
    }
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_bias));
  HIP_CHECK(hipFree(d_topk));
  HIP_CHECK(hipFree(d_batches));
  HIP_CHECK(hipFree(d_slots));
  HIP_CHECK(hipFree(d_offsets));
  HIP_CHECK(hipFree(d_pos));
}

int main(int argc, char **argv) {
  TimingOptions opts;
  if (argc >= 2)
    opts.iters = std::max(10, std::atoi(argv[1]));
  if (argc >= 3)
    opts.warmup = std::max(3, std::atoi(argv[2]));

  print_device_info();
  std::cout << "Timing iterations: " << opts.iters
            << ", warmup: " << opts.warmup << "\n";

  hipStream_t stream = nullptr;

  for (const auto &setting : g_settings) {
    tune_matmul_qkv(setting, opts, stream);
    tune_matmul_att(setting, opts, stream);
    tune_matmul_logits(setting, opts, stream);
    tune_mlp1(setting, opts, stream);
    tune_mlp2(setting, opts, stream);
  }

  return 0;
}
