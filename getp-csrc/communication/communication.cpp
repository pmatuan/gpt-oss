#include "communication.h"
#include <algorithm>
#include <cmath>

void init_expert_distribution(ExpertDistribution& dist, int num_experts, int num_devices) {
    dist.num_devices = num_devices;
    dist.experts_per_device = num_experts / num_devices;
    dist.remainder_experts = num_experts % num_devices;
    
    dist.device_expert_start.resize(num_devices);
    dist.device_expert_count.resize(num_devices);
    
    int current_start = 0;
    for (int dev = 0; dev < num_devices; ++dev) {
        dist.device_expert_start[dev] = current_start;
        dist.device_expert_count[dev] = dist.experts_per_device;
        if (dev < dist.remainder_experts) {
            dist.device_expert_count[dev]++;
        }
        current_start += dist.device_expert_count[dev];
    }
}

int get_expert_device(const ExpertDistribution& dist, int expert_id) {
    for (int dev = 0; dev < dist.num_devices; ++dev) {
        int start = dist.device_expert_start[dev];
        int count = dist.device_expert_count[dev];
        if (expert_id >= start && expert_id < start + count) {
            return dev;
        }
    }
    return -1; // Invalid expert_id
}

int get_local_expert_id(const ExpertDistribution& dist, int expert_id) {
    int device = get_expert_device(dist, expert_id);
    if (device == -1) return -1;
    return expert_id - dist.device_expert_start[device];
}

void init_communication_context(CommunicationContext& ctx, int num_devices, int device_id, int num_experts) {
    ctx.num_devices = num_devices;
    ctx.current_device_id = device_id;
    
    init_expert_distribution(ctx.expert_dist, num_experts, num_devices);
    
    // Initialize communication buffers
    ctx.d_send_buffers.resize(num_devices, nullptr);
    ctx.d_recv_buffers.resize(num_devices, nullptr);
    ctx.send_buffer_sizes.resize(num_devices, 0);
    ctx.recv_buffer_sizes.resize(num_devices, 0);
    
    // Initialize events
    ctx.send_events.resize(num_devices);
    ctx.recv_events.resize(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        HIP_CHECK(hipEventCreate(&ctx.send_events[i]));
        HIP_CHECK(hipEventCreate(&ctx.recv_events[i]));
    }
    
    // Allocate expert mapping arrays
    HIP_CHECK(hipMalloc(&ctx.d_expert_device_map, num_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&ctx.d_local_expert_map, num_experts * sizeof(int)));
    
    // Populate expert mappings on host then copy to device
    std::vector<int> h_expert_device_map(num_experts);
    std::vector<int> h_local_expert_map(num_experts);
    
    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        h_expert_device_map[expert_id] = get_expert_device(ctx.expert_dist, expert_id);
        h_local_expert_map[expert_id] = get_local_expert_id(ctx.expert_dist, expert_id);
    }
    
    HIP_CHECK(hipMemcpy(ctx.d_expert_device_map, h_expert_device_map.data(),
                        num_experts * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ctx.d_local_expert_map, h_local_expert_map.data(),
                        num_experts * sizeof(int), hipMemcpyHostToDevice));
}

void cleanup_communication_context(CommunicationContext& ctx) {
    // Free send/recv buffers
    for (int i = 0; i < ctx.num_devices; ++i) {
        if (ctx.d_send_buffers[i]) {
            HIP_CHECK(hipFree(ctx.d_send_buffers[i]));
        }
        if (ctx.d_recv_buffers[i]) {
            HIP_CHECK(hipFree(ctx.d_recv_buffers[i]));
        }
        HIP_CHECK(hipEventDestroy(ctx.send_events[i]));
        HIP_CHECK(hipEventDestroy(ctx.recv_events[i]));
    }
    
    // Free expert mapping arrays
    if (ctx.d_expert_device_map) {
        HIP_CHECK(hipFree(ctx.d_expert_device_map));
        ctx.d_expert_device_map = nullptr;
    }
    if (ctx.d_local_expert_map) {
        HIP_CHECK(hipFree(ctx.d_local_expert_map));
        ctx.d_local_expert_map = nullptr;
    }
    
    // Free batch metadata
    if (ctx.d_token_expert_assignments) {
        HIP_CHECK(hipFree(ctx.d_token_expert_assignments));
        ctx.d_token_expert_assignments = nullptr;
    }
    if (ctx.d_token_expert_weights_idx) {
        HIP_CHECK(hipFree(ctx.d_token_expert_weights_idx));
        ctx.d_token_expert_weights_idx = nullptr;
    }
    if (ctx.d_token_expert_weights) {
        HIP_CHECK(hipFree(ctx.d_token_expert_weights));
        ctx.d_token_expert_weights = nullptr;
    }
}

void gather_expert_inputs(CommunicationContext& ctx, 
                         const float* d_input,
                         const int* d_expert_assignments,
                         const float* d_expert_weights, 
                         int batch_size, int hidden_dim, int experts_per_token,
                         float** d_gathered_inputs,
                         int* gathered_counts) {
    
    // Implementation depends on the communication mechanism
    // This is where the actual gather logic will be implemented
    // The mechanism (all-to-all, point-to-point, etc.) will be implemented here
    
    // For now, this is a placeholder that will be filled with the actual
    // communication implementation based on the chosen strategy
    
    // Example structure:
    // 1. Determine which tokens need which experts
    // 2. Group tokens by target device 
    // 3. Send data to appropriate devices
    // 4. Receive data from other devices
    // 5. Organize received data for local expert processing
}

void scatter_expert_outputs(CommunicationContext& ctx,
                           const float** d_expert_outputs,
                           const int* expert_counts,
                           int hidden_dim,
                           float* d_output) {
    
    // Implementation depends on the communication mechanism
    // This is where the actual scatter logic will be implemented
    // The mechanism (all-to-all, point-to-point, etc.) will be implemented here
    
    // For now, this is a placeholder that will be filled with the actual
    // communication implementation based on the chosen strategy
    
    // Example structure:
    // 1. Prepare expert outputs for sending back to origin devices
    // 2. Send data back to requesting devices
    // 3. Receive data from other devices
    // 4. Aggregate received expert outputs 
    // 5. Apply expert weights and sum for final output
}

void ensure_communication_buffers(CommunicationContext& ctx, size_t max_buffer_size) {
    for (int i = 0; i < ctx.num_devices; ++i) {
        if (ctx.send_buffer_sizes[i] < max_buffer_size) {
            if (ctx.d_send_buffers[i]) {
                HIP_CHECK(hipFree(ctx.d_send_buffers[i]));
            }
            HIP_CHECK(hipMalloc(&ctx.d_send_buffers[i], max_buffer_size));
            ctx.send_buffer_sizes[i] = max_buffer_size;
        }
        
        if (ctx.recv_buffer_sizes[i] < max_buffer_size) {
            if (ctx.d_recv_buffers[i]) {
                HIP_CHECK(hipFree(ctx.d_recv_buffers[i]));
            }
            HIP_CHECK(hipMalloc(&ctx.d_recv_buffers[i], max_buffer_size));
            ctx.recv_buffer_sizes[i] = max_buffer_size;
        }
    }
}

void synchronize_communication(CommunicationContext& ctx) {
    for (int i = 0; i < ctx.num_devices; ++i) {
        HIP_CHECK(hipEventSynchronize(ctx.send_events[i]));
        HIP_CHECK(hipEventSynchronize(ctx.recv_events[i]));
    }
}