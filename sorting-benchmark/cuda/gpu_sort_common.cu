#include "gpu_sort_common.hpp"
#include <cuda_runtime.h>

// ============================================================
// Device Memory
// ============================================================

template <typename T>
T* device_malloc(std::size_t n) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return ptr;
}

template <typename T>
void device_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template <typename T>
void copy_to_device(T* d_dst, const T* h_src, std::size_t n) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(T* h_dst, const T* d_src, std::size_t n) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// Explicit instantiations (add more types if needed)
template uint64_t* device_malloc<uint64_t>(std::size_t);
template void device_free<uint64_t>(uint64_t*);
template void copy_to_device<uint64_t>(uint64_t*, const uint64_t*, std::size_t);
template void copy_to_host<uint64_t>(uint64_t*, const uint64_t*, std::size_t);
