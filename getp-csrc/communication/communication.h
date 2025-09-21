#ifndef GETP_COMMUNICATION_H
#define GETP_COMMUNICATION_H

#include "../common/defines.h"
#include <hip/hip_runtime.h>
#include <vector>

struct ExpertDistribution {
    int num_devices;
    int experts_per_device;
    int remainder_experts;
    std::vector<int> device_expert_start;
    std::vector<int> device_expert_count;
};

struct CommunicationContext {
    int num_devices;
    int current_device_id;
    ExpertDistribution expert_dist;
    
    // GPU buffers for inter-device communication
    std::vector<float*> d_send_buffers;
    std::vector<float*> d_recv_buffers;
    std::vector<size_t> send_buffer_sizes;
    std::vector<size_t> recv_buffer_sizes;
    
    // Events for synchronization
    std::vector<hipEvent_t> send_events;
    std::vector<hipEvent_t> recv_events;
    
    // Workspace for expert routing
    int* d_expert_device_map;  // Maps expert_id -> device_id
    int* d_local_expert_map;   // Maps expert_id -> local_expert_id on device
    
    // Batch metadata for communication
    int* d_token_expert_assignments;    // [batch_size * experts_per_token] 
    int* d_token_expert_weights_idx;    // indices for weight lookup
    float* d_token_expert_weights;      // expert weights for aggregation
    
    CommunicationContext() : num_devices(0), current_device_id(0),
                           d_expert_device_map(nullptr), d_local_expert_map(nullptr),
                           d_token_expert_assignments(nullptr), d_token_expert_weights_idx(nullptr),
                           d_token_expert_weights(nullptr) {}
};

// Expert distribution functions
void init_expert_distribution(ExpertDistribution& dist, int num_experts, int num_devices);
int get_expert_device(const ExpertDistribution& dist, int expert_id);
int get_local_expert_id(const ExpertDistribution& dist, int expert_id);

// Communication context management
void init_communication_context(CommunicationContext& ctx, int num_devices, int device_id, int num_experts);
void cleanup_communication_context(CommunicationContext& ctx);

// Main communication functions for getp_run.cpp
void gather_expert_inputs(CommunicationContext& ctx, 
                         const float* d_input,           // [batch_size, hidden_dim]
                         const int* d_expert_assignments, // [batch_size * experts_per_token]
                         const float* d_expert_weights,   // [batch_size * experts_per_token] 
                         int batch_size, int hidden_dim, int experts_per_token,
                         float** d_gathered_inputs,       // output: gathered inputs for local experts
                         int* gathered_counts);           // output: counts per local expert

void scatter_expert_outputs(CommunicationContext& ctx,
                           const float** d_expert_outputs,  // [num_local_experts][expert_batch_size, hidden_dim]
                           const int* expert_counts,        // [num_local_experts]
                           int hidden_dim,
                           float* d_output);                // [batch_size, hidden_dim] - aggregated output

// Utility functions
void ensure_communication_buffers(CommunicationContext& ctx, size_t max_buffer_size);
void synchronize_communication(CommunicationContext& ctx);

#endif // GETP_COMMUNICATION_H