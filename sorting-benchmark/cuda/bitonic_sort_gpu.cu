// cuda/bitonic_sort_gpu.cu

#include "bitonic_sort_gpu.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <limits>

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

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

__host__ __device__
inline std::uint32_t next_power_of_two(std::uint32_t x) {
    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Pad kernel: for indices >= N, set pad value.
__global__ void pad_kernel_uint64(
    std::uint64_t* data,
    std::size_t    N,
    std::size_t    Npad,
    std::uint64_t  pad_value
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= Npad) return;
    if (i >= N) {
        data[i] = pad_value;
    }
}

// One bitonic step: compare-and-swap with partner index i ^ j
__global__ void bitonic_step_kernel_uint64(
    std::uint64_t* data,
    std::size_t    N,
    std::uint32_t  j,
    std::uint32_t  k
) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    std::size_t ixj = i ^ j;
    if (ixj > i && ixj < N) {
        bool ascending = ((i & k) == 0);
        std::uint64_t a = data[i];
        std::uint64_t b = data[ixj];

        if (ascending) {
            if (a > b) {
                data[i]   = b;
                data[ixj] = a;
            }
        } else {
            if (a < b) {
                data[i]   = b;
                data[ixj] = a;
            }
        }
    }
}

// ------------------------------------------------------------
// Public API: bitonic_sort_uint64_gpu
// ------------------------------------------------------------

void bitonic_sort_uint64_gpu(
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
        // Trivial; just copy
        CUDA_CHECK(cudaMemcpy(d_out, d_in, N * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToDevice));
        return;
    }

    // We restrict to Npad <= 2^32 (totally fine for your project sizes)
    if (N > static_cast<std::size_t>(0x7FFFFFFF)) {
        throw std::runtime_error("bitonic_sort_uint64_gpu: N too large for 32-bit indexing");
    }

    std::uint32_t N32    = static_cast<std::uint32_t>(N);
    std::uint32_t Npad32 = next_power_of_two(N32);
    std::size_t   Npad   = static_cast<std::size_t>(Npad32);

    // Choose launch config for the bitonic network.
    // We must ensure blocks * threads >= Npad.
    int threadsPerBlock = (threads > 0) ? threads : 256;
    int blocksPerGrid   = (blocks  > 0) ? blocks  :
                          static_cast<int>((Npad + threadsPerBlock - 1) / threadsPerBlock);

    std::size_t totalThreads = static_cast<std::size_t>(blocksPerGrid) * threadsPerBlock;
    if (totalThreads < Npad) {
        blocksPerGrid = static_cast<int>((Npad + threadsPerBlock - 1) / threadsPerBlock);
    }

    // Working buffer on device (size Npad)
    std::uint64_t* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, Npad * sizeof(std::uint64_t)));

    // Copy original data (N elements) to d_work
    CUDA_CHECK(cudaMemcpy(d_work, d_in, N * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToDevice));

    // Pad remaining elements with UINT64_MAX so they float to the end.
    std::uint64_t pad_value = std::numeric_limits<std::uint64_t>::max();
    pad_kernel_uint64<<<blocksPerGrid, threadsPerBlock>>>(
        d_work, N, Npad, pad_value
    );
    CUDA_CHECK(cudaGetLastError());

    // Timing setup
    cudaEvent_t ev_kernel_start, ev_kernel_end;
    CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_end));

    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Record start of kernel region
    CUDA_CHECK(cudaEventRecord(ev_kernel_start));

    // Bitonic network: O(N log^2 N)
    // k: size of subsequences being merged (2, 4, 8, ..., Npad)
    for (std::uint32_t k = 2; k <= Npad32; k <<= 1) {
        // j: distance of comparison partner (k/2, k/4, ..., 1)
        for (std::uint32_t j = k >> 1; j > 0; j >>= 1) {
            bitonic_step_kernel_uint64<<<blocksPerGrid, threadsPerBlock>>>(
                d_work,
                Npad,
                j,
                k
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Record end of kernel region
    CUDA_CHECK(cudaEventRecord(ev_kernel_end));
    CUDA_CHECK(cudaEventSynchronize(ev_kernel_end));

    float kernel_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_f, ev_kernel_start, ev_kernel_end));
    kernel_ms = static_cast<double>(kernel_ms_f);

    // Copy only the first N sorted elements to d_out
    CUDA_CHECK(cudaMemcpy(d_out, d_work, N * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaEventDestroy(ev_kernel_start));
    CUDA_CHECK(cudaEventDestroy(ev_kernel_end));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_total_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
}
