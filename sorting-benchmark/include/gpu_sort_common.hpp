#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>  // for std::forward

// ============================================================
// Error-checking macro
// ============================================================

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(_err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// ============================================================
// Device memory allocation helpers (declared here, defined in .cu)
// ============================================================

template <typename T>
T* device_malloc(std::size_t n);

template <typename T>
void device_free(T* ptr);

template <typename T>
void copy_to_device(T* d_dst, const T* h_src, std::size_t n);

template <typename T>
void copy_to_host(T* h_dst, const T* d_src, std::size_t n);


// ============================================================
// Timing utilities (templates so they work with lambdas)
// ============================================================

struct TimingResult {
    float kernel_ms;
    float total_ms;
};

// Measure only the "kernel section" callable
template <typename KernelLauncher>
float cuda_time_ms(KernelLauncher&& kernel_launcher) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    std::forward<KernelLauncher>(kernel_launcher)();
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

// Measure H2D + kernel + D2H
template <typename KernelLauncher, typename H2D, typename D2H>
TimingResult cuda_measure(KernelLauncher&& kernel_launcher,
                          H2D&& h2d,
                          D2H&& d2h) {
    TimingResult t{0.0f, 0.0f};

    cudaEvent_t e_start, e_end;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_end));

    CUDA_CHECK(cudaEventRecord(e_start));

    // Host -> Device
    std::forward<H2D>(h2d)();

    // Kernel timing
    t.kernel_ms = cuda_time_ms(std::forward<KernelLauncher>(kernel_launcher));

    // Device -> Host
    std::forward<D2H>(d2h)();

    CUDA_CHECK(cudaEventRecord(e_end));
    CUDA_CHECK(cudaEventSynchronize(e_end));
    CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, e_start, e_end));

    CUDA_CHECK(cudaEventDestroy(e_start));
    CUDA_CHECK(cudaEventDestroy(e_end));

    return t;
}


// ============================================================
// Grid/block helpers
// ============================================================

inline dim3 make_grid(std::size_t n, int threads) {
    return dim3((n + threads - 1) / threads);
}
