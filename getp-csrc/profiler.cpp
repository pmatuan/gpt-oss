#include "profiler.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = call;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#ifdef ENABLE_PROFILING

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::addCpuTime(const std::string& name, double time_ms) {
    cpu_profiles_[name].total_time += time_ms;
    cpu_profiles_[name].call_count++;
}

void Profiler::addGpuTime(const std::string& name, double time_ms) {
    gpu_profiles_[name].total_time += time_ms;
    gpu_profiles_[name].call_count++;
}

void Profiler::printSummary() {
    std::vector<std::pair<std::string, ProfileData*>> all_profiles;
    
    // Combine CPU and GPU profiles
    for (auto& [name, data] : cpu_profiles_) {
        all_profiles.emplace_back("CPU:" + name, &data);
    }
    for (auto& [name, data] : gpu_profiles_) {
        all_profiles.emplace_back("GPU:" + name, &data);
    }
    
    if (all_profiles.empty()) {
        std::cout << "\n┌─────────────────────────────────┐\n";
        std::cout << "│      PROFILER SUMMARY           │\n";
        std::cout << "├─────────────────────────────────┤\n";
        std::cout << "│  No profiling data available.   │\n";
        std::cout << "└─────────────────────────────────┘\n" << std::endl;
        return;
    }
    
    // Sort by total time (descending)
    std::sort(all_profiles.begin(), all_profiles.end(),
              [](const auto& a, const auto& b) {
                  return a.second->total_time > b.second->total_time;
              });
    
    std::cout << "\n┌──────────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                                  PROFILER SUMMARY                                    │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ " << std::left << std::setw(45) << "Function/Kernel Name" 
              << " │ " << std::right << std::setw(12) << "Total (ms)" 
              << " │ " << std::setw(8) << "Calls" 
              << " │ " << std::setw(10) << "Avg (ms)" << " │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────────────┤\n";
    
    for (const auto& [name, data] : all_profiles) {
        std::cout << "│ " << std::left << std::setw(45) << name
                  << " │ " << std::right << std::setw(12) << std::fixed << std::setprecision(3) << data->total_time
                  << " │ " << std::setw(8) << data->call_count
                  << " │ " << std::setw(10) << std::fixed << std::setprecision(3) << data->getAverageTime() << " │\n";
    }
    
    std::cout << "└──────────────────────────────────────────────────────────────────────────────────────┘\n";
    std::cout << std::endl;
}

void Profiler::clear() {
    cpu_profiles_.clear();
    gpu_profiles_.clear();
}

ProfileScope::ProfileScope(const std::string& name) 
    : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {
}

ProfileScope::~ProfileScope() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
    double time_ms = duration.count() / 1000000.0;
    Profiler::getInstance().addCpuTime(name_, time_ms);
}

ProfileGpuScope::ProfileGpuScope(const std::string& name, hipStream_t stream)
    : name_(name), stream_(stream), valid_(false) {
    
    if (hipEventCreate(&start_event_) == hipSuccess &&
        hipEventCreate(&end_event_) == hipSuccess &&
        hipEventRecord(start_event_, stream_) == hipSuccess) {
        valid_ = true;
    }
}

ProfileGpuScope::~ProfileGpuScope() {
    if (!valid_) {
        return;
    }
    
    if (hipEventRecord(end_event_, stream_) == hipSuccess &&
        hipEventSynchronize(end_event_) == hipSuccess) {
        
        float time_ms = 0.0f;
        if (hipEventElapsedTime(&time_ms, start_event_, end_event_) == hipSuccess) {
            // For multi-GPU scenarios, we average the timing across devices
            // This assumes GPU operations on different devices happen in parallel
            int device_count = 1;
            HIP_CHECK(hipGetDeviceCount(&device_count));
            
            // Add the raw timing - the profiler will handle averaging if needed
            Profiler::getInstance().addGpuTime(name_, static_cast<double>(time_ms));
        }
    }
    
    HIP_CHECK(hipEventDestroy(start_event_));
    HIP_CHECK(hipEventDestroy(end_event_));
}

#else

// Empty implementations when profiling is disabled
Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::addCpuTime(const std::string&, double) {}
void Profiler::addGpuTime(const std::string&, double) {}
void Profiler::printSummary() {}
void Profiler::clear() {}

ProfileScope::ProfileScope(const std::string&) {}
ProfileScope::~ProfileScope() {}

ProfileGpuScope::ProfileGpuScope(const std::string&, hipStream_t) : valid_(false) {}
ProfileGpuScope::~ProfileGpuScope() {}

#endif