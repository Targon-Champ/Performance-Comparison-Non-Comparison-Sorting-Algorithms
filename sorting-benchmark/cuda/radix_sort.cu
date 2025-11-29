// cuda/radix_sort.cu

#include "radix_sort.cuh"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

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

// ============================================================================
// 1. Histogram kernel (per-block 256-bin histograms in shared memory)
// ============================================================================

__global__ void histogram_pass_kernel_uint64(
    const std::uint64_t* __restrict__ d_in,
    std::uint32_t*       __restrict__ d_block_hist, // [blocks x 256]
    std::size_t                          N,
    int                                  byte_index  // 0..7
) {
    extern __shared__ std::uint32_t s_hist[]; // 256 bins

    // Initialize shared histogram
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Contiguous chunk assigned to this block
    std::size_t chunk = (N + gridDim.x - 1) / gridDim.x; // ceil(N / gridDim.x)
    std::size_t start = static_cast<std::size_t>(blockIdx.x) * chunk;
    std::size_t end   = start + chunk;
    if (start >= N) {
        // Nothing to do for this block
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            d_block_hist[blockIdx.x * 256 + i] = 0u;
        }
        return;
    }
    if (end > N) end = N;

    const int shift = byte_index * 8;

    // Threads cooperate over [start, end)
    for (std::size_t idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
        std::uint64_t key  = d_in[idx];
        std::uint32_t byte = static_cast<std::uint32_t>((key >> shift) & 0xFFull);
        atomicAdd(&s_hist[byte], 1u);  // shared-mem atomic
    }

    __syncthreads();

    // Write per-block histogram to global memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        d_block_hist[blockIdx.x * 256 + i] = s_hist[i];
    }
}


// ============================================================================
// 2. Standalone histogram helper (for debugging / analysis)
//    (Same as earlier; uses the kernel + host reduction)
// ============================================================================

void radix_histogram_pass_uint64(
    const std::uint64_t* d_in,
    std::size_t          N,
    int                  byte_index,
    std::vector<std::uint32_t>& out_global_hist,
    cudaStream_t         stream
) {
    if (byte_index < 0 || byte_index > 7) {
        throw std::invalid_argument("byte_index must be in [0,7]");
    }

    const int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);
    const int maxBlocks = 2048;
    if (blocks > maxBlocks) blocks = maxBlocks;
    if (blocks == 0) {
        out_global_hist.assign(256, 0u);
        return;
    }

    std::uint32_t* d_block_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_hist, blocks * 256 * sizeof(std::uint32_t)));

    const std::size_t shmem_bytes = 256 * sizeof(std::uint32_t);

    histogram_pass_kernel_uint64<<<blocks, threads, shmem_bytes, stream>>>(
        d_in, d_block_hist, N, byte_index
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::uint32_t> h_block_hist(blocks * 256);
    CUDA_CHECK(cudaMemcpyAsync(
        h_block_hist.data(),
        d_block_hist,
        blocks * 256 * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    out_global_hist.assign(256, 0u);
    for (int b = 0; b < blocks; ++b) {
        const std::uint32_t* row = &h_block_hist[b * 256];
        for (int bin = 0; bin < 256; ++bin) {
            out_global_hist[bin] += row[bin];
        }
    }

    CUDA_CHECK(cudaFree(d_block_hist));
}

// ============================================================================
// 3. Scatter kernel
//
// Uses per-block base offsets (from host) + shared-memory counters
// so that each element gets a unique global index without global atomics.
// ============================================================================

__global__ void scatter_pass_kernel_uint64(
    const std::uint64_t* __restrict__ d_in,
    std::uint64_t*       __restrict__ d_out,
    const std::uint32_t* __restrict__ d_block_offsets, // [blocks x 256]
    std::size_t                          N,
    int                                  byte_index
) {
    extern __shared__ std::uint32_t s_mem[];
    std::uint32_t* s_offsets = s_mem;          // 256
    std::uint32_t* s_counts  = s_mem + 256;    // 256

    // Only thread 0 does the work for this block to ensure stability
    if (threadIdx.x == 0) {
        // Contiguous chunk assigned to this block
        std::size_t chunk = (N + gridDim.x - 1) / gridDim.x;
        std::size_t start = static_cast<std::size_t>(blockIdx.x) * chunk;
        std::size_t end   = start + chunk;
        if (start >= N) return;
        if (end > N) end = N;

        // 1) Load per-bin base offsets for this block and reset counts
        for (int bin = 0; bin < 256; ++bin) {
            s_offsets[bin] = d_block_offsets[blockIdx.x * 256 + bin];
            s_counts[bin]  = 0u;
        }

        const int shift = byte_index * 8;

        // 2) Process elements in *index order* over [start, end)
        for (std::size_t idx = start; idx < end; ++idx) {
            std::uint64_t key = d_in[idx];
            std::uint32_t bin = static_cast<std::uint32_t>((key >> shift) & 0xFFull);

            std::uint32_t pos = s_offsets[bin] + s_counts[bin];
            s_counts[bin]++;

            d_out[pos] = key;
        }
    }
}



// ============================================================================
// 4. Full radix sort implementation (LSD, base 256, 64-bit keys)
// ============================================================================

void radix_sort_uint64_gpu(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   stream
) {
    kernel_ms = 0.0;
    total_ms  = 0.0;

    if (N <= 1) {
        // Nothing to do
        return;
    }

    if (threads <= 0 || blocks <= 0) {
        std::cerr << "[radix_sort_uint64_gpu] blocks/threads should be > 0 "
                     "(main harness is expected to choose them).\n";
        throw std::runtime_error("Invalid blocks/threads in radix_sort_uint64_gpu");
    }

    // Allocate device buffers for per-block histograms and offsets
    std::uint32_t* d_block_hist    = nullptr; // [blocks x 256]
    std::uint32_t* d_block_offsets = nullptr; // [blocks x 256]

    CUDA_CHECK(cudaMalloc(&d_block_hist,    blocks * 256 * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, blocks * 256 * sizeof(std::uint32_t)));

    // Host-side buffers
    std::vector<std::uint32_t> h_block_hist(blocks * 256);
    std::vector<std::uint32_t> hist(256);
    std::vector<std::uint32_t> prefix(256);
    std::vector<std::uint32_t> h_block_offsets(blocks * 256);

    // Timing setup
    cudaEvent_t ev_kernel_start, ev_kernel_end;
    CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_end));

    auto t_total_start = std::chrono::high_resolution_clock::now();

    std::uint64_t* d_src = d_in;
    std::uint64_t* d_dst = d_out;

    const std::size_t shmem_hist_bytes    = 256 * sizeof(std::uint32_t);
    const std::size_t shmem_scatter_bytes = 2 * 256 * sizeof(std::uint32_t);

    // 8 passes for 64-bit keys, 8 bits per pass
    const int num_passes = 8;

    for (int pass = 0; pass < num_passes; ++pass) {
        int byte_index = pass; // LSB-first

        CUDA_CHECK(cudaEventRecord(ev_kernel_start, stream));

        // --------------------------------------------------------------------
        // 4.1 Histogram per block on device
        // --------------------------------------------------------------------
        histogram_pass_kernel_uint64<<<blocks, threads, shmem_hist_bytes, stream>>>(
            d_src,
            d_block_hist,
            N,
            byte_index
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy per-block histograms to host (async on this stream)
        CUDA_CHECK(cudaMemcpyAsync(
            h_block_hist.data(),
            d_block_hist,
            blocks * 256 * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost,
            stream
        ));

        // Wait for histogram copy to complete before using h_block_hist
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --------------------------------------------------------------------
        // 4.2 Build global histogram and per-block offsets on host
        // --------------------------------------------------------------------
        // Global histogram
        std::fill(hist.begin(), hist.end(), 0u);
        for (int b = 0; b < blocks; ++b) {
            const std::uint32_t* row = &h_block_hist[b * 256];
            for (int bin = 0; bin < 256; ++bin) {
                hist[bin] += row[bin];
            }
        }

        // Exclusive prefix sum over bins -> global start offset of each bin
        std::uint32_t running = 0;
        for (int bin = 0; bin < 256; ++bin) {
            prefix[bin] = running;
            running += hist[bin];
        }

        // Per-block offsets: for each bin, walk blocks in order and accumulate
        for (int bin = 0; bin < 256; ++bin) {
            std::uint32_t offset = prefix[bin];
            for (int b = 0; b < blocks; ++b) {
                std::size_t idx = static_cast<std::size_t>(b) * 256 + bin;
                h_block_offsets[idx] = offset;
                offset += h_block_hist[idx];
            }
        }

        // Copy block offsets to device (async on this stream)
        CUDA_CHECK(cudaMemcpyAsync(
            d_block_offsets,
            h_block_offsets.data(),
            blocks * 256 * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice,
            stream
        ));

        // --------------------------------------------------------------------
        // 4.3 Scatter on device using block offsets
        // --------------------------------------------------------------------
        scatter_pass_kernel_uint64<<<blocks, threads, shmem_scatter_bytes, stream>>>(
            d_src,
            d_dst,
            d_block_offsets,
            N,
            byte_index
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_kernel_end, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

        float pass_kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pass_kernel_ms, ev_kernel_start, ev_kernel_end));
        kernel_ms += static_cast<double>(pass_kernel_ms);

        // Ping-pong buffers
        std::swap(d_src, d_dst);
    }

    // After an even number of passes (8), data ends up back in d_out,
    // because we started with (d_src = d_in, d_dst = d_out) and swapped every pass.

    auto t_total_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    CUDA_CHECK(cudaEventDestroy(ev_kernel_start));
    CUDA_CHECK(cudaEventDestroy(ev_kernel_end));

    CUDA_CHECK(cudaFree(d_block_hist));
    CUDA_CHECK(cudaFree(d_block_offsets));
}
