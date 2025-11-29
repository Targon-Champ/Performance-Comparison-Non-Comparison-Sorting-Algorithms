// cuda/bucket_sort_radix_gpu.cuh
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int BUCKET_RADIX_BUCKETS = 1024; // or whatever value you picked

void bucket_sort_radix_uint64_gpu(
    uint64_t* d_in,
    uint64_t* d_out,
    size_t    N,
    int       blocks,
    int       threads,
    double&   kernel_ms,
    double&   total_ms
);

// (optional, but if referenced from other translation units, declare it here)
__global__ void shared_memory_sort_kernel(
    const uint64_t* d_in,
    uint64_t*       d_out,
    int             count
);
