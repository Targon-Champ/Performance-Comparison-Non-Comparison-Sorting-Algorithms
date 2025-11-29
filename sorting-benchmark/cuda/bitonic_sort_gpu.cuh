// cuda/bitonic_sort_gpu.cuh
#pragma once

#include <cstdint>
#include <cstddef>

// Global bitonic sort for uint64_t (ascending).
// Will pad to next power-of-two internally using UINT64_MAX.
void bitonic_sort_uint64_gpu(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms
);
