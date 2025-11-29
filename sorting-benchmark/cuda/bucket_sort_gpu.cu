#include "bucket_sort_gpu.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::terminate();                                                 \
        }                                                                     \
    } while (0)
#endif

// -----------------------------------------------------------------------------
// 0. Min/max reduction kernel
// -----------------------------------------------------------------------------
__global__ void reduce_minmax_kernel_uint64(
    const std::uint64_t* __restrict__ d_in,
    std::size_t                        N,
    std::uint64_t*                     d_block_mins,
    std::uint64_t*                     d_block_maxs
) {
    extern __shared__ std::uint64_t s_data[];
    std::uint64_t* s_mins = s_data;
    std::uint64_t* s_maxs = s_data + blockDim.x;

    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    std::uint64_t local_min = 0xFFFFFFFFFFFFFFFFull;
    std::uint64_t local_max = 0;

    while (idx < N) {
        std::uint64_t v = d_in[idx];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
        idx += stride;
    }

    s_mins[threadIdx.x] = local_min;
    s_maxs[threadIdx.x] = local_max;
    __syncthreads();

    // In-block reduction
    for (unsigned offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (s_mins[threadIdx.x + offset] < s_mins[threadIdx.x]) {
                s_mins[threadIdx.x] = s_mins[threadIdx.x + offset];
            }
            if (s_maxs[threadIdx.x + offset] > s_maxs[threadIdx.x]) {
                s_maxs[threadIdx.x] = s_maxs[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_block_mins[blockIdx.x] = s_mins[0];
        d_block_maxs[blockIdx.x] = s_maxs[0];
    }
}

// -----------------------------------------------------------------------------
// 1. Bucket histogram kernel
// -----------------------------------------------------------------------------
__global__ void bucket_hist_kernel_uint64(
    const std::uint64_t* __restrict__ d_in,
    std::uint32_t*       __restrict__ d_block_hist, // [blocks x B]
    std::size_t                          N,
    int                                  B,
    std::uint64_t                        minv,
    std::uint64_t                        bucket_width
) {
    extern __shared__ std::uint32_t s_hist[]; // B bins

    // Init shared histogram
    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Contiguous chunk assigned to this block
    std::size_t chunk = (N + gridDim.x - 1) / gridDim.x; // ceil(N / gridDim.x)
    std::size_t start = static_cast<std::size_t>(blockIdx.x) * chunk;
    std::size_t end   = start + chunk;
    if (start >= N) {
        // write zeros for this block
        for (int i = threadIdx.x; i < B; i += blockDim.x) {
            d_block_hist[blockIdx.x * B + i] = 0u;
        }
        return;
    }
    if (end > N) end = N;

    // Threads cooperate over [start, end)
    for (std::size_t idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
        std::uint64_t key = d_in[idx];
        std::uint64_t shifted = key - minv;
        std::uint64_t bin64 = shifted / bucket_width;
        int bin = static_cast<int>(bin64);
        if (bin < 0) bin = 0;
        if (bin >= B) bin = B - 1;
        atomicAdd(&s_hist[bin], 1u);  // shared-mem atomic
    }

    __syncthreads();

    // Write per-block histogram to global memory
    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        d_block_hist[blockIdx.x * B + i] = s_hist[i];
    }
}

// -----------------------------------------------------------------------------
// 2. Bucket scatter kernel (stable within each bucket)
// -----------------------------------------------------------------------------
__global__ void bucket_scatter_kernel_uint64(
    const std::uint64_t* __restrict__ d_in,
    std::uint64_t*       __restrict__ d_out,
    const std::uint32_t* __restrict__ d_block_offsets, // [blocks x B]
    std::size_t                          N,
    int                                  B,
    std::uint64_t                        minv,
    std::uint64_t                        bucket_width
) {
    extern __shared__ std::uint32_t s_mem[];
    std::uint32_t* s_offsets = s_mem;          // B
    std::uint32_t* s_counts  = s_mem + B;      // B

    if (threadIdx.x == 0) {
        // Contiguous chunk assigned to this block
        std::size_t chunk = (N + gridDim.x - 1) / gridDim.x;
        std::size_t start = static_cast<std::size_t>(blockIdx.x) * chunk;
        std::size_t end   = start + chunk;
        if (start >= N) return;
        if (end > N) end = N;

        // Load per-bin base offsets for this block and reset counts
        for (int bin = 0; bin < B; ++bin) {
            s_offsets[bin] = d_block_offsets[blockIdx.x * B + bin];
            s_counts[bin]  = 0u;
        }

        // Process elements in index order (stability)
        for (std::size_t idx = start; idx < end; ++idx) {
            std::uint64_t key = d_in[idx];
            std::uint64_t shifted = key - minv;
            std::uint64_t bin64 = shifted / bucket_width;
            int bin = static_cast<int>(bin64);
            if (bin < 0) bin = 0;
            if (bin >= B) bin = B - 1;

            std::uint32_t pos = s_offsets[bin] + s_counts[bin];
            s_counts[bin]++;

            d_out[pos] = key;
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Final in-bucket sort kernel (insertion sort per bucket)
// -----------------------------------------------------------------------------
__global__ void bucket_finalize_sort_kernel_uint64(
    std::uint64_t*       d_data,
    const std::uint32_t* d_bucket_offsets, // [B]
    const std::uint32_t* d_bucket_sizes,   // [B]
    int                  B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    std::uint32_t start = d_bucket_offsets[b];
    std::uint32_t size  = d_bucket_sizes[b];
    if (size <= 1) return;

    // Simple insertion sort inside [start, start+size)
    for (std::uint32_t i = start + 1; i < start + size; ++i) {
        std::uint64_t key = d_data[i];
        std::uint32_t j = i;
        while (j > start && d_data[j - 1] > key) {
            d_data[j] = d_data[j - 1];
            --j;
        }
        d_data[j] = key;
    }
}

// -----------------------------------------------------------------------------
// 4. Public API: bucket_sort_uint64_gpu
// -----------------------------------------------------------------------------
void bucket_sort_uint64_gpu(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms
) {
    kernel_ms = 0.0;
    total_ms  = 0.0;

    if (N <= 1) {
        if (N == 1) {
            CUDA_CHECK(cudaMemcpy(d_out, d_in, sizeof(std::uint64_t),
                                  cudaMemcpyDeviceToDevice));
        }
        return;
    }

    if (threads <= 0 || blocks <= 0) {
        std::cerr << "[bucket_sort_uint64_gpu] blocks/threads should be > 0\n";
        throw std::runtime_error("Invalid blocks/threads in bucket_sort_uint64_gpu");
    }

    // -----------------------------
    // 4.1 Compute min/max on GPU
    // -----------------------------
    const int threads_mm = 256;
    int blocks_mm = static_cast<int>((N + threads_mm - 1) / threads_mm);
    const int maxBlocks_mm = 1024;
    if (blocks_mm > maxBlocks_mm) blocks_mm = maxBlocks_mm;
    if (blocks_mm <= 0) blocks_mm = 1;

    std::uint64_t* d_block_mins = nullptr;
    std::uint64_t* d_block_maxs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_mins, blocks_mm * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_block_maxs, blocks_mm * sizeof(std::uint64_t)));

    std::size_t shmem_mm = 2 * threads_mm * sizeof(std::uint64_t);
    reduce_minmax_kernel_uint64<<<blocks_mm, threads_mm, shmem_mm>>>(
        d_in,
        N,
        d_block_mins,
        d_block_maxs
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::uint64_t> h_block_mins(blocks_mm);
    std::vector<std::uint64_t> h_block_maxs(blocks_mm);
    CUDA_CHECK(cudaMemcpy(h_block_mins.data(), d_block_mins,
                          blocks_mm * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_maxs.data(), d_block_maxs,
                          blocks_mm * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_maxs));

    std::uint64_t minv = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maxv = 0;
    for (int i = 0; i < blocks_mm; ++i) {
        if (h_block_mins[i] < minv) minv = h_block_mins[i];
        if (h_block_maxs[i] > maxv) maxv = h_block_maxs[i];
    }

    // If all values equal, we're done: just copy
    if (minv == maxv) {
        CUDA_CHECK(cudaMemcpy(d_out, d_in, N * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToDevice));
        return;
    }

    // -----------------------------
    // 4.2 Choose number of buckets B
    // -----------------------------
    const int MAX_BUCKETS = 2048;
    int B = 1024;
    if (N < static_cast<std::size_t>(B)) {
        B = static_cast<int>(N);
    }
    if (B > MAX_BUCKETS) B = MAX_BUCKETS;
    if (B < 1) B = 1;

    std::uint64_t range = (maxv - minv) + 1ull;
    std::uint64_t bucket_width = (range + static_cast<std::uint64_t>(B) - 1ull)
                                 / static_cast<std::uint64_t>(B);
    if (bucket_width == 0) bucket_width = 1;

    // -----------------------------
    // 4.3 Allocate device & host buffers for hist + offsets
    // -----------------------------
    std::uint32_t* d_block_hist    = nullptr; // [blocks x B]
    std::uint32_t* d_block_offsets = nullptr; // [blocks x B]
    CUDA_CHECK(cudaMalloc(&d_block_hist,    blocks * B * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, blocks * B * sizeof(std::uint32_t)));

    std::vector<std::uint32_t> h_block_hist(blocks * B);
    std::vector<std::uint32_t> hist(B);
    std::vector<std::uint32_t> bucket_offsets(B);
    std::vector<std::uint32_t> h_block_offsets(blocks * B);

    // For final in-bucket sort
    std::vector<std::uint32_t> bucket_sizes(B);

    // We bucket directly into d_out (no ping-pong needed)
    std::uint64_t* d_src = d_in;
    std::uint64_t* d_dst = d_out;

    // Shared memory sizes
    std::size_t shmem_hist_bytes    = static_cast<std::size_t>(B) * sizeof(std::uint32_t);
    std::size_t shmem_scatter_bytes = 2 * static_cast<std::size_t>(B) * sizeof(std::uint32_t);

    // Timing
    cudaEvent_t ev_kernel_start, ev_kernel_end;
    CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_end));

    auto t_total_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));

    // -----------------------------
    // 4.4 Histogram per block
    // -----------------------------
    bucket_hist_kernel_uint64<<<blocks, threads, shmem_hist_bytes>>>(
        d_src,
        d_block_hist,
        N,
        B,
        minv,
        bucket_width
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        h_block_hist.data(),
        d_block_hist,
        blocks * B * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost
    ));

    // -----------------------------
    // 4.5 Build global hist & per-block offsets on host
    // -----------------------------
    std::fill(hist.begin(), hist.end(), 0u);
    for (int b = 0; b < blocks; ++b) {
        const std::uint32_t* row = &h_block_hist[b * B];
        for (int bin = 0; bin < B; ++bin) {
            hist[bin] += row[bin];
        }
    }

    // Global bucket offsets (exclusive prefix sum)
    std::uint32_t running = 0;
    for (int bin = 0; bin < B; ++bin) {
        bucket_offsets[bin] = running;
        running += hist[bin];
    }

    // Safety: running must equal N
    if (running != static_cast<std::uint32_t>(N)) {
        std::cerr << "[bucket_sort_uint64_gpu] WARNING: total hist != N ("
                  << running << " vs " << N << ")\n";
    }

    // Per-block offsets
    for (int bin = 0; bin < B; ++bin) {
        std::uint32_t offset = bucket_offsets[bin];
        for (int b = 0; b < blocks; ++b) {
            std::size_t idx = static_cast<std::size_t>(b) * B + bin;
            h_block_offsets[idx] = offset;
            offset += h_block_hist[idx];
        }
        bucket_sizes[bin] = hist[bin];
    }

    CUDA_CHECK(cudaMemcpy(
        d_block_offsets,
        h_block_offsets.data(),
        blocks * B * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice
    ));

    // -----------------------------
    // 4.6 Scatter into buckets on device
    // -----------------------------
    bucket_scatter_kernel_uint64<<<blocks, threads, shmem_scatter_bytes>>>(
        d_src,
        d_dst,
        d_block_offsets,
        N,
        B,
        minv,
        bucket_width
    );
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------
    // 4.7 Final in-bucket sort on device
    // -----------------------------
    std::uint32_t* d_bucket_offsets = nullptr;
    std::uint32_t* d_bucket_sizes   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bucket_offsets, B * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_sizes,   B * sizeof(std::uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_bucket_offsets, bucket_offsets.data(),
                          B * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bucket_sizes, bucket_sizes.data(),
                          B * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

    const int threads_finalize = 128;
    int blocks_finalize = (B + threads_finalize - 1) / threads_finalize;
    bucket_finalize_sort_kernel_uint64<<<blocks_finalize, threads_finalize>>>(
        d_dst,
        d_bucket_offsets,
        d_bucket_sizes,
        B
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_bucket_offsets));
    CUDA_CHECK(cudaFree(d_bucket_sizes));

    // -----------------------------
    // 4.8 Finish timing & cleanup
    // -----------------------------
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    float kernel_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_f, ev_kernel_start, ev_kernel_end));
    kernel_ms = static_cast<double>(kernel_ms_f);

    CUDA_CHECK(cudaEventDestroy(ev_kernel_start));
    CUDA_CHECK(cudaEventDestroy(ev_kernel_end));

    CUDA_CHECK(cudaFree(d_block_hist));
    CUDA_CHECK(cudaFree(d_block_offsets));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_total_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start
    ).count();
}
