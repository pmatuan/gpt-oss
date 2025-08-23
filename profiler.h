#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Include HIP headers for GPU profiling
#include <hip/hip_runtime.h>

struct FunctionStats {
    std::string name;
    double total_time_ms;
    int call_count;
    double average_time_ms;
    
    FunctionStats() : total_time_ms(0.0), call_count(0), average_time_ms(0.0) {}
    
    void add_call(double time_ms) {
        total_time_ms += time_ms;
        call_count++;
        average_time_ms = total_time_ms / call_count;
    }
};

class Profiler {
private:
    std::unordered_map<std::string, FunctionStats> stats;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::unordered_map<std::string, hipEvent_t> gpu_start_events;
    std::unordered_map<std::string, hipEvent_t> gpu_end_events;
    
public:
    void start_timing(const std::string& function_name) {
        start_times[function_name] = std::chrono::high_resolution_clock::now();
    }
    
    void end_timing(const std::string& function_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto it = start_times.find(function_name);
        if (it != start_times.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second);
            double time_ms = duration.count() / 1000.0;
            
            stats[function_name].name = function_name;
            stats[function_name].add_call(time_ms);
            
            start_times.erase(it);
        }
    }
    
    void start_gpu_timing(const std::string& kernel_name) {
        hipEvent_t start_event, end_event;
        if (hipEventCreate(&start_event) != hipSuccess ||
            hipEventCreate(&end_event) != hipSuccess) {
            return; // Skip profiling on error
        }
        
        gpu_start_events[kernel_name] = start_event;
        gpu_end_events[kernel_name] = end_event;
        
        (void)hipEventRecord(start_event, 0); // Cast to void to suppress warning
    }
    
    void end_gpu_timing(const std::string& kernel_name) {
        auto start_it = gpu_start_events.find(kernel_name);
        auto end_it = gpu_end_events.find(kernel_name);
        
        if (start_it != gpu_start_events.end() && end_it != gpu_end_events.end()) {
            (void)hipEventRecord(end_it->second, 0);
            (void)hipEventSynchronize(end_it->second);
            
            float gpu_time_ms;
            if (hipEventElapsedTime(&gpu_time_ms, start_it->second, end_it->second) == hipSuccess) {
                stats[kernel_name].name = kernel_name;
                stats[kernel_name].add_call(gpu_time_ms);
            }
            
            (void)hipEventDestroy(start_it->second);
            (void)hipEventDestroy(end_it->second);
            gpu_start_events.erase(start_it);
            gpu_end_events.erase(end_it);
        }
    }
    
    void print_summary() const {
        if (stats.empty()) {
            std::cout << "\n=== PROFILING SUMMARY ===\n";
            std::cout << "No profiling data collected.\n";
            return;
        }
        
        // Convert to vector for sorting
        std::vector<FunctionStats> sorted_stats;
        for (const auto& pair : stats) {
            sorted_stats.push_back(pair.second);
        }
        
        // Sort by average time (descending)
        std::sort(sorted_stats.begin(), sorted_stats.end(), 
                  [](const FunctionStats& a, const FunctionStats& b) {
                      return a.average_time_ms > b.average_time_ms;
                  });
        
        std::cout << "\n=== PROFILING SUMMARY ===\n";
        std::cout << std::left << std::setw(40) << "Function" 
                  << std::right << std::setw(12) << "Total (ms)" 
                  << std::setw(10) << "Calls" 
                  << std::setw(15) << "Average (ms)" << "\n";
        std::cout << std::string(77, '-') << "\n";
        
        for (const auto& stat : sorted_stats) {
            std::cout << std::left << std::setw(40) << stat.name
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << stat.total_time_ms
                      << std::setw(10) << stat.call_count
                      << std::setw(15) << std::fixed << std::setprecision(3) << stat.average_time_ms << "\n";
        }
        std::cout << std::string(77, '=') << "\n";
    }
};

// Global profiler instance
extern Profiler g_profiler;

// RAII profiling helper
class ScopedProfiler {
private:
    std::string function_name;
public:
    ScopedProfiler(const std::string& name) : function_name(name) {
        g_profiler.start_timing(function_name);
    }
    
    ~ScopedProfiler() {
        g_profiler.end_timing(function_name);
    }
};

// Convenience macros
#define PROFILE_FUNCTION() ScopedProfiler _prof(__func__)
#define PROFILE_SCOPE(name) ScopedProfiler _prof(name)

// GPU Kernel profiling macros
#define PROFILE_GPU_KERNEL_START(name) g_profiler.start_gpu_timing(name)
#define PROFILE_GPU_KERNEL_END(name) g_profiler.end_gpu_timing(name)

// Wrapper macro for profiling a single kernel launch
#define PROFILE_KERNEL_LAUNCH(kernel_name, ...) \
    do { \
        PROFILE_GPU_KERNEL_START(kernel_name); \
        __VA_ARGS__; \
        PROFILE_GPU_KERNEL_END(kernel_name); \
    } while(0)

#endif // PROFILER_H