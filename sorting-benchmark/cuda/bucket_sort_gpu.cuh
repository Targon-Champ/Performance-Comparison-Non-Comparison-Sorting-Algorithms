// cuda/bucket_sort_gpu.cuh
#pragma once

#include <cstdint>
#include <cstddef>

// GPU Bucket Sort for uint64_t (ascending).
//
// Design:
//  - Computes min/max on GPU.
//  - Chooses a reasonable number of buckets based on N (capped).
//  - Buckets via GPU histogram + prefix sums + scatter.
//  - Sorts *within* each bucket on GPU (one thread per bucket, insertion sort).
//
// NOTE: This is correct and stable but not asymptotically optimal.
//       Good for research & distribution-wise behavior analysis.
void bucket_sort_uint64_gpu(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms
);
