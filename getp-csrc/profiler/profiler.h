#ifndef PROFILER_H
#define PROFILER_H

#include <hip/hip_runtime.h>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_PROFILING
#define PROFILE_SCOPE(name) ProfileScope _prof(name)
#define PROFILE_GPU_SCOPE(name, stream) ProfileGpuScope _prof_gpu(name, stream)
#define PROFILER_PRINT_SUMMARY() Profiler::getInstance().printSummary()
#else
#define PROFILE_SCOPE(name) do {} while(0)
#define PROFILE_GPU_SCOPE(name, stream) do {} while(0)
#define PROFILER_PRINT_SUMMARY() do {} while(0)
#endif

struct ProfileData {
    double total_time;
    int call_count;
    
    ProfileData() : total_time(0.0), call_count(0) {}
    
    double getAverageTime() const {
        return call_count > 0 ? total_time / call_count : 0.0;
    }
};

class Profiler {
public:
    static Profiler& getInstance();
    
    void addCpuTime(const std::string& name, double time_ms);
    void addGpuTime(const std::string& name, double time_ms);
    void printSummary();
    void clear();
    
private:
    Profiler() = default;
    std::unordered_map<std::string, ProfileData> cpu_profiles_;
    std::unordered_map<std::string, ProfileData> gpu_profiles_;
};

class ProfileScope {
public:
    explicit ProfileScope(const std::string& name);
    ~ProfileScope();
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class ProfileGpuScope {
public:
    explicit ProfileGpuScope(const std::string& name, hipStream_t stream = 0);
    ~ProfileGpuScope();
    
private:
    std::string name_;
    hipEvent_t start_event_;
    hipEvent_t end_event_;
    hipStream_t stream_;
    bool valid_;
};

#endif // PROFILER_H