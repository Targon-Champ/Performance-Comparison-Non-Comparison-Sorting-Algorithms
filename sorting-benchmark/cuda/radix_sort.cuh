// cuda/radix_sort.cuh
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// Optional: standalone histogram pass for debugging/analysis
void radix_histogram_pass_uint64(
    const std::uint64_t* d_in,
    std::size_t          N,
    int                  byte_index,
    std::vector<std::uint32_t>& out_global_hist,
    cudaStream_t         stream
);

// Main GPU radix sort (LSD, base 256, 64-bit keys)
void radix_sort_uint64_gpu(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   stream = 0   // default = legacy behavior
);
