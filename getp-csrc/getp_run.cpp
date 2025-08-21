// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include <hip/hip_runtime.h>

#ifndef GETP_RUN
#define GETP_RUN

// Macro for error checking
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU memory pointers
static float *d_x, *d_t, *d_tb, *d_tb2;
static float *d_router_score, *d_topk_v, *d_mlp1_out;
static int *d_topk_i;
static float *d_gate, *d_up, *d_gate_up, *d_e_agg;
static float *d_qkv, *d_q, *d_k, *d_v;
static float *d_key_cache, *d_value_cache;
static float *d_att, *d_logits, *d_mask;
static float *d_cos_vals, *d_sin_vals;

// Weight pointers on GPU
static float *d_token_embedding_table, *d_rms_attn_w, *d_rms_ffn_w;
static float *d_w_qkv, *d_w_o, *d_b_qkv, *d_b_o, *d_attn_sinks;
static float *d_w_router, *d_b_router;
static float *d_w_mlp1, *d_w_mlp2, *d_b_mlp1, *d_b_mlp2;
static float *d_rms_out_w, *d_out;

static Config *h_config;

// Custom GPU kernels
__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Calculate sum of squares using shared memory
    __shared__ double shared_ss[256];
    int tid = threadIdx.x;
    shared_ss[tid] = 0.0;
    
    // Sum squares across all threads
    for (int i = tid; i < size; i += blockDim.x) {
        shared_ss[tid] += x[i] * x[i];
    }
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            shared_ss[tid] += shared_ss[tid + stride];
        }
        __syncthreads();
    }
    
    double ss = shared_ss[0] / size + 1e-5;
    ss = 1.0 / sqrt(ss);
    
    if (idx < size) {
        o[idx] = weight[idx] * (ss * x[idx]);
    }
}

__global__ void softmax_kernel(float *x, int size) {
    int tid = threadIdx.x;
    
    // Find max value using shared memory
    __shared__ float shared_max[256];
    shared_max[tid] = (tid < size) ? x[tid] : -INFINITY;
    
    for (int i = tid + blockDim.x; i < size; i += blockDim.x) {
        shared_max[tid] = fmaxf(shared_max[tid], x[i]);
    }
    __syncthreads();
    
    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared_max[0];
    
    // Exp and sum using shared memory
    __shared__ double shared_sum[256];
    shared_sum[tid] = 0.0;
    
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        shared_sum[tid] += x[i];
    }
    __syncthreads();
    
    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    double sum = shared_sum[0];
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {
    // n := in_features, d := out_features
    // W (d,n) @ x (n,) -> xout (d,)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[row * n + j] * x[j];
        }
        xout[row] = val;
    }
}

__global__ void add_bias_kernel(float *output, float *bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += bias[idx];
    }
}

__global__ void copy_embedding_kernel(float *dst, float *src, int token, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        dst[idx] = src[token * hidden_dim + idx];
    }
}

__global__ void copy_qkv_kernel(float *q, float *k, float *v, float *qkv, 
                                int n_attn_heads, int n_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_size = n_attn_heads * head_dim;
    int k_size = n_kv_heads * head_dim;
    
    if (idx < q_size) {
        q[idx] = qkv[idx];
    } else if (idx < q_size + k_size) {
        k[idx - q_size] = qkv[idx];
    } else if (idx < q_size + k_size + k_size) {
        v[idx - q_size - k_size] = qkv[idx];
    }
}

__global__ void compute_cos_sin_kernel(float *cos_vals, float *sin_vals, int pos, 
                                      float rope_theta, int head_dim, float scaling_factor,
                                      float initial_context_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_half = head_dim / 2;
    
    if (idx < d_half) {
        float freq = powf(rope_theta, (2.0f * idx) / head_dim);
        float inv_freq;
        
        if (scaling_factor > 1.0f) {
            inv_freq = 1.0f / (scaling_factor * freq);
        } else {
            inv_freq = 1.0f / freq;
        }
        
        float val = pos * inv_freq;
        float concentration = (scaling_factor > 1.0f) ? (0.1f * logf(scaling_factor) + 1.0f) : 1.0f;
        
        cos_vals[idx] = cosf(val) * concentration;
        sin_vals[idx] = sinf(val) * concentration;
    }
}

__global__ void apply_rotary_emb_kernel(float *x, float *cos, float *sin, 
                                       int n_heads, int head_dim) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half = head_dim / 2;
    
    if (h < n_heads && i < half) {
        float x1 = x[h * head_dim + i];
        float x2 = x[h * head_dim + half + i];
        
        float c = cos[i];
        float s = sin[i];
        
        x[h * head_dim + i] = x1 * c - x2 * s;
        x[h * head_dim + half + i] = x2 * c + x1 * s;
    }
}

__global__ void attention_scores_kernel(float *att, float *q, float *key_cache, 
                                       int h, int pos, int head_dim, int kv_dim, 
                                       int n_kv_heads, int seq_len, float *mask) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t <= pos) {
        int kv_mul = (n_kv_heads > 0) ? (h / n_kv_heads) : 0;
        float *k = key_cache + t * kv_dim + kv_mul * head_dim;
        
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q[h * head_dim + i] * k[i];
        }
        score /= sqrtf(head_dim);
        
        if (mask != NULL) {
            score += mask[pos * seq_len + t];
        }
        
        att[h * seq_len + t] = score;
    }
}

__global__ void attention_values_kernel(float *tb, float *att, float *value_cache,
                                       int h, int pos, int head_dim, int kv_dim,
                                       int n_kv_heads, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < head_dim) {
        float result = 0.0f;
        int kv_mul = (n_kv_heads > 0) ? (h / n_kv_heads) : 0;
        
        for (int t = 0; t <= pos; t++) {
            float *v = value_cache + t * kv_dim + kv_mul * head_dim;
            float a = att[h * seq_len + t];
            result += a * v[i];
        }
        
        tb[h * head_dim + i] = result;
    }
}

__global__ void residual_connection_kernel(float *x, float *residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += residual[idx];
    }
}

__global__ void topk_kernel(float *topk_values, int *topk_indices, float *router_score,
                           int num_experts, int experts_per_token) {
    // Simple selection sort for top-k
    for (int k = 0; k < experts_per_token; k++) {
        float max_val = -INFINITY;
        int max_idx = 0;
        
        for (int i = 0; i < num_experts; i++) {
            bool already_selected = false;
            for (int j = 0; j < k; j++) {
                if (topk_indices[j] == i) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && router_score[i] > max_val) {
                max_val = router_score[i];
                max_idx = i;
            }
        }
        
        topk_values[k] = max_val;
        topk_indices[k] = max_idx;
    }
}

__global__ void swiglu_kernel(float *gate_up, float *gate, float *up, 
                             int intermediate_dim, float swiglu_limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < intermediate_dim) {
        float val = gate[idx];
        float up_val = up[idx];
        
        // Clamping
        val = fminf(fmaxf(val, -swiglu_limit), swiglu_limit);
        up_val = fminf(fmaxf(up_val, -swiglu_limit), swiglu_limit);
        
        // SiLU activation
        const float alpha = 1.702f;
        val *= (1.0f / (1.0f + expf(-alpha * val)));
        val *= (up_val + 1.0f);
        
        gate_up[idx] = val;
    }
}

__global__ void split_gate_up_kernel(float *gate, float *up, float *mlp1_out, 
                                     int intermediate_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < intermediate_dim) {
        gate[idx] = mlp1_out[2 * idx];
        up[idx] = mlp1_out[2 * idx + 1];
    }
}

__global__ void weighted_sum_kernel(float *e_agg, float *expert_out, float weight, 
                                   int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_dim) {
        e_agg[idx] += expert_out[idx] * weight;
    }
}

__global__ void memset_zero_kernel(float *ptr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ptr[idx] = 0.0f;
    }
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
    h_config = &transformer->config;
    
    // Initialize HIP
    HIP_CHECK(hipSetDevice(0));
    
    int hidden_dim = h_config->hidden_dim;
    int vocab_size = h_config->vocab_size;
    int n_layers = h_config->n_layers;
    int n_experts = h_config->n_experts;
    int intermediate_dim = h_config->intermediate_dim;
    int seq_len = h_config->seq_len;
    int head_dim = h_config->head_dim;
    int n_attn_heads = h_config->n_attn_heads;
    int n_kv_heads = h_config->n_kv_heads;
    int kv_dim = head_dim * n_kv_heads;
    
    // Allocate GPU memory for activations
    HIP_CHECK(hipMalloc(&d_x, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_t, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tb, head_dim * n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tb2, hidden_dim * sizeof(float)));
    
    HIP_CHECK(hipMalloc(&d_router_score, n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_topk_v, h_config->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_topk_i, h_config->experts_per_token * sizeof(int)));
    
    HIP_CHECK(hipMalloc(&d_mlp1_out, 2 * intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_up, intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate_up, intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_e_agg, hidden_dim * sizeof(float)));
    
    HIP_CHECK(hipMalloc(&d_qkv, head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q, n_attn_heads * head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k, n_kv_heads * head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v, n_kv_heads * head_dim * sizeof(float)));
    
    HIP_CHECK(hipMalloc(&d_key_cache, n_layers * seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_value_cache, n_layers * seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_att, n_attn_heads * seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
    
    HIP_CHECK(hipMalloc(&d_cos_vals, (head_dim / 2) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sin_vals, (head_dim / 2) * sizeof(float)));
    
    if (h_config->sliding_window > 0) {
        HIP_CHECK(hipMalloc(&d_mask, seq_len * seq_len * sizeof(float)));
        // Initialize mask on GPU
        float *h_mask = (float*)malloc(seq_len * seq_len * sizeof(float));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (i - j >= h_config->sliding_window) {
                    h_mask[i * seq_len + j] = -INFINITY;
                } else {
                    h_mask[i * seq_len + j] = 0.0f;
                }
            }
        }
        HIP_CHECK(hipMemcpy(d_mask, h_mask, seq_len * seq_len * sizeof(float), hipMemcpyHostToDevice));
        free(h_mask);
    }
    
    // Allocate and copy weights to GPU
    TransformerWeights *w = &transformer->weights;
    
    HIP_CHECK(hipMalloc(&d_token_embedding_table, vocab_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_token_embedding_table, w->token_embedding_table, 
              vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_rms_attn_w, n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_rms_attn_w, w->rms_attn_w, 
              n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_rms_ffn_w, n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_rms_ffn_w, w->rms_ffn_w, 
              n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    size_t qkv_size = n_layers * hidden_dim * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
    HIP_CHECK(hipMalloc(&d_w_qkv, qkv_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w_qkv, w->w_qkv, qkv_size * sizeof(float), hipMemcpyHostToDevice));
    
    size_t o_size = n_layers * (head_dim * n_attn_heads) * hidden_dim;
    HIP_CHECK(hipMalloc(&d_w_o, o_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w_o, w->w_o, o_size * sizeof(float), hipMemcpyHostToDevice));
    
    size_t b_qkv_size = n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
    HIP_CHECK(hipMalloc(&d_b_qkv, b_qkv_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b_qkv, w->b_qkv, b_qkv_size * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_b_o, n_layers * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b_o, w->b_o, n_layers * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_attn_sinks, n_layers * n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_attn_sinks, w->attn_sinks, n_layers * n_attn_heads * sizeof(float), hipMemcpyHostToDevice));
    
    size_t router_size = n_layers * hidden_dim * n_experts;
    HIP_CHECK(hipMalloc(&d_w_router, router_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w_router, w->w_router, router_size * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_b_router, n_layers * n_experts * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b_router, w->b_router, n_layers * n_experts * sizeof(float), hipMemcpyHostToDevice));
    
    size_t mlp1_size = n_layers * n_experts * 2 * intermediate_dim * hidden_dim;
    HIP_CHECK(hipMalloc(&d_w_mlp1, mlp1_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w_mlp1, w->w_mlp1, mlp1_size * sizeof(float), hipMemcpyHostToDevice));
    
    size_t mlp2_size = n_layers * n_experts * hidden_dim * intermediate_dim;
    HIP_CHECK(hipMalloc(&d_w_mlp2, mlp2_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w_mlp2, w->w_mlp2, mlp2_size * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b_mlp1, w->b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_b_mlp2, w->b_mlp2, n_layers * n_experts * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_rms_out_w, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_rms_out_w, w->rms_out_w, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&d_out, vocab_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_out, w->out, vocab_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
    // Free GPU memory
    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_t)); HIP_CHECK(hipFree(d_tb)); HIP_CHECK(hipFree(d_tb2));
    HIP_CHECK(hipFree(d_router_score)); HIP_CHECK(hipFree(d_topk_v)); HIP_CHECK(hipFree(d_topk_i));
    HIP_CHECK(hipFree(d_mlp1_out)); HIP_CHECK(hipFree(d_gate)); HIP_CHECK(hipFree(d_up)); HIP_CHECK(hipFree(d_gate_up)); HIP_CHECK(hipFree(d_e_agg));
    HIP_CHECK(hipFree(d_qkv)); HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_k)); HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_key_cache)); HIP_CHECK(hipFree(d_value_cache)); HIP_CHECK(hipFree(d_att)); HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_cos_vals)); HIP_CHECK(hipFree(d_sin_vals));
    
    if (d_mask) HIP_CHECK(hipFree(d_mask));
    
    // Free weight memory
    HIP_CHECK(hipFree(d_token_embedding_table)); HIP_CHECK(hipFree(d_rms_attn_w)); HIP_CHECK(hipFree(d_rms_ffn_w));
    HIP_CHECK(hipFree(d_w_qkv)); HIP_CHECK(hipFree(d_w_o)); HIP_CHECK(hipFree(d_b_qkv)); HIP_CHECK(hipFree(d_b_o)); HIP_CHECK(hipFree(d_attn_sinks));
    HIP_CHECK(hipFree(d_w_router)); HIP_CHECK(hipFree(d_b_router));
    HIP_CHECK(hipFree(d_w_mlp1)); HIP_CHECK(hipFree(d_w_mlp2)); HIP_CHECK(hipFree(d_b_mlp1)); HIP_CHECK(hipFree(d_b_mlp2));
    HIP_CHECK(hipFree(d_rms_out_w)); HIP_CHECK(hipFree(d_out));
}

float *gpu_forward(Transformer *transformer, int token, int pos) {
    Config *p = h_config;
    
    int hidden_dim = p->hidden_dim;
    int head_dim = p->head_dim;
    int n_attn_heads = p->n_attn_heads;
    int n_kv_heads = p->n_kv_heads;
    int kv_dim = head_dim * n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    int seq_len = p->seq_len;
    
    dim3 block(256);
    dim3 grid((hidden_dim + block.x - 1) / block.x);
    
    // Copy token embedding to x
    copy_embedding_kernel<<<grid, block>>>(d_x, d_token_embedding_table, token, hidden_dim);
    
    // Forward through all layers
    for (int l = 0; l < p->n_layers; l++) {
        // RMSNorm for attention
        rmsnorm_kernel<<<grid, block>>>(d_t, d_x, d_rms_attn_w + l * hidden_dim, hidden_dim);
        
        // QKV projection
        int qkv_out_dim = (n_attn_heads + 2 * n_kv_heads) * head_dim;
        dim3 qkv_grid((qkv_out_dim + block.x - 1) / block.x);
        matmul_kernel<<<qkv_grid, block>>>(d_qkv, d_t, d_w_qkv + l * hidden_dim * qkv_out_dim, hidden_dim, qkv_out_dim);
        
        // Add bias
        add_bias_kernel<<<qkv_grid, block>>>(d_qkv, d_b_qkv + l * qkv_out_dim, qkv_out_dim);
        
        // Split QKV
        copy_qkv_kernel<<<qkv_grid, block>>>(d_q, d_k, d_v, d_qkv, n_attn_heads, n_kv_heads, head_dim);
        
        // Copy K,V to cache
        int loff = l * seq_len * kv_dim;
        HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * kv_dim, d_k, kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_value_cache + loff + pos * kv_dim, d_v, kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
        
        // RoPE
        dim3 rope_grid(((head_dim / 2) + block.x - 1) / block.x);
        compute_cos_sin_kernel<<<rope_grid, block>>>(d_cos_vals, d_sin_vals, pos,
                                                    p->rope_theta, head_dim, p->rope_scaling_factor,
                                                    p->initial_context_length);
        
        dim3 rope_apply_grid(n_attn_heads, (head_dim / 2 + 31) / 32);
        apply_rotary_emb_kernel<<<rope_apply_grid, 32>>>(d_q, d_cos_vals, d_sin_vals, n_attn_heads, head_dim);
        rope_apply_grid = dim3(n_kv_heads, (head_dim / 2 + 31) / 32);
        apply_rotary_emb_kernel<<<rope_apply_grid, 32>>>(d_k, d_cos_vals, d_sin_vals, n_kv_heads, head_dim);
        
        // Update cache with rotated K
        HIP_CHECK(hipMemcpy(d_key_cache + loff + pos * kv_dim, d_k, kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
        
        // Multi-head attention
        for (int h = 0; h < n_attn_heads; h++) {
            // Compute attention scores
            dim3 att_grid((pos + 2 + block.x - 1) / block.x);
            attention_scores_kernel<<<att_grid, block>>>(d_att, d_q, d_key_cache + loff,
                                                        h, pos, head_dim, kv_dim, n_kv_heads, seq_len,
                                                        (p->sliding_window > 0 && l % 2 == 0) ? d_mask : nullptr);
            
            // Add attention sink
            float sink_val;
            HIP_CHECK(hipMemcpy(&sink_val, d_attn_sinks + l * n_attn_heads + h, sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(d_att + h * seq_len + pos + 1, &sink_val, sizeof(float), hipMemcpyHostToDevice));
            
            // Softmax
            softmax_kernel<<<1, block>>>(d_att + h * seq_len, pos + 2);
            
            // Compute attention values
            dim3 val_grid((head_dim + block.x - 1) / block.x);
            attention_values_kernel<<<val_grid, block>>>(d_tb, d_att, d_value_cache + loff,
                                                        h, pos, head_dim, kv_dim, n_kv_heads, seq_len);
        }
        
        // Output projection
        dim3 o_grid((hidden_dim + block.x - 1) / block.x);
        matmul_kernel<<<o_grid, block>>>(d_tb2, d_tb, d_w_o + l * (head_dim * n_attn_heads) * hidden_dim, 
                                        head_dim * n_attn_heads, hidden_dim);
        
        // Add bias and residual
        add_bias_kernel<<<grid, block>>>(d_tb2, d_b_o + l * hidden_dim, hidden_dim);
        residual_connection_kernel<<<grid, block>>>(d_x, d_tb2, hidden_dim);
        
        // FFN RMSNorm
        rmsnorm_kernel<<<grid, block>>>(d_t, d_x, d_rms_ffn_w + l * hidden_dim, hidden_dim);
        
        // Router computation
        dim3 expert_grid((n_experts + block.x - 1) / block.x);
        matmul_kernel<<<expert_grid, block>>>(d_router_score, d_t, d_w_router + l * hidden_dim * n_experts, 
                                             hidden_dim, n_experts);
        
        // Add router bias
        add_bias_kernel<<<expert_grid, block>>>(d_router_score, d_b_router + l * n_experts, n_experts);
        
        // Top-k selection
        topk_kernel<<<1, 1>>>(d_topk_v, d_topk_i, d_router_score, n_experts, p->experts_per_token);
        
        // Softmax on top-k values
        softmax_kernel<<<1, block>>>(d_topk_v, p->experts_per_token);
        
        // Initialize aggregation buffer
        dim3 zero_grid((hidden_dim + block.x - 1) / block.x);
        memset_zero_kernel<<<zero_grid, block>>>(d_e_agg, hidden_dim);
        
        // Process each expert
        float *h_topk_v = (float*)malloc(p->experts_per_token * sizeof(float));
        int *h_topk_i = (int*)malloc(p->experts_per_token * sizeof(int));
        HIP_CHECK(hipMemcpy(h_topk_v, d_topk_v, p->experts_per_token * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_topk_i, d_topk_i, p->experts_per_token * sizeof(int), hipMemcpyDeviceToHost));
        
        for (int e = 0; e < n_experts; e++) {
            float expert_w = 0.0f;
            bool in_topk = false;
            
            // Check if expert is in top-k
            for (int idx = 0; idx < p->experts_per_token; idx++) {
                if (h_topk_i[idx] == e) {
                    in_topk = true;
                    expert_w = h_topk_v[idx];
                    break;
                }
            }
            
            if (in_topk) {
                // MLP1 (gate_up projection)
                int expert_offset = l * n_experts + e;
                dim3 mlp1_grid((2 * intermediate_dim + block.x - 1) / block.x);
                matmul_kernel<<<mlp1_grid, block>>>(d_mlp1_out, d_t, 
                                                   d_w_mlp1 + expert_offset * (2 * intermediate_dim) * hidden_dim, 
                                                   hidden_dim, 2 * intermediate_dim);
                
                // Add MLP1 bias
                add_bias_kernel<<<mlp1_grid, block>>>(d_mlp1_out, 
                                                     d_b_mlp1 + expert_offset * (2 * intermediate_dim), 
                                                     2 * intermediate_dim);
                
                // Split gate and up
                dim3 split_mlp_grid((intermediate_dim + block.x - 1) / block.x);
                split_gate_up_kernel<<<split_mlp_grid, block>>>(d_gate, d_up, d_mlp1_out, intermediate_dim);
                
                // SwiGLU activation
                swiglu_kernel<<<split_mlp_grid, block>>>(d_gate_up, d_gate, d_up, intermediate_dim, p->swiglu_limit);
                
                // MLP2 (down projection)
                matmul_kernel<<<grid, block>>>(d_tb2, d_gate_up, 
                                              d_w_mlp2 + expert_offset * hidden_dim * intermediate_dim, 
                                              intermediate_dim, hidden_dim);
                
                // Add MLP2 bias
                add_bias_kernel<<<grid, block>>>(d_tb2, d_b_mlp2 + expert_offset * hidden_dim, hidden_dim);
                
                // Weighted aggregation
                weighted_sum_kernel<<<grid, block>>>(d_e_agg, d_tb2, expert_w, hidden_dim);
            }
        }
        
        free(h_topk_v);
        free(h_topk_i);
        
        // Residual connection
        residual_connection_kernel<<<grid, block>>>(d_x, d_e_agg, hidden_dim);
    }
    
    // Final RMSNorm
    rmsnorm_kernel<<<grid, block>>>(d_x, d_x, d_rms_out_w, hidden_dim);
    
    // Output projection to logits
    dim3 logits_grid((p->vocab_size + block.x - 1) / block.x);
    matmul_kernel<<<logits_grid, block>>>(d_logits, d_x, d_out, hidden_dim, p->vocab_size);
    
    return d_logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
    const char *empty_prompt = "";
    if (input_seq == NULL) {
        input_seq = empty_prompt;
    }

    // Encode the prompt
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
    encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
           transformer->config.initial_context_length);
    
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // Main generation loop
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    while (pos < steps) {
        // Forward pass on GPU
        float *d_logits = gpu_forward(transformer, token, pos);
        
        // Copy logits back to CPU for sampling
        float *h_logits = (float*)malloc(transformer->config.vocab_size * sizeof(float));
        HIP_CHECK(hipMemcpy(h_logits, d_logits, transformer->config.vocab_size * sizeof(float), hipMemcpyDeviceToHost));
        
        // Advance position
        pos++;
        
        if (pos < num_prompt_tokens) {
            // Force next prompt token
            next = prompt_tokens[pos];
        } else {
            // Sample next token
            next = sample(sampler, h_logits);
            // Save output token
            output_tokens[pos - num_prompt_tokens] = next;
        }
        
        // Check for EOS tokens
        if (next == 199999 || next == 200002) {
            break;
        }

        // Print token (can be removed for pure throughput)
        const char *piece = decode_piece(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);

        token = next;
        free(h_logits);
    }

    printf("\n");

    // Mark end of sequence
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
        num_token_out += simple_getp_generate(transformer, tokenizer, sampler, 
                                             input_seq, output_tokens, requests->max_seq_len);
    }
    
    return num_token_out;
}

#endif // GETP_RUN