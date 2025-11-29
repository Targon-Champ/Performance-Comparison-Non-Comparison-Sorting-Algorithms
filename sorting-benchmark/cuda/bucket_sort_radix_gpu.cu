#include "bucket_sort_radix_gpu.cuh"
#include "radix_sort.cuh"    // radix_sort_uint64_gpu declaration

#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <vector>

// We assume BUCKET_RADIX_BUCKETS is a power of two (e.g., 256, 512, 1024).
// Derive BUCKET_RADIX_BITS from that macro to avoid mismatches.

constexpr int compute_bucket_bits(int buckets) {
    int bits = 0;
    int v = buckets;
    while (v > 1) {
        v >>= 1;
        ++bits;
    }
    return bits;
}

constexpr int BUCKET_RADIX_BITS = compute_bucket_bits(BUCKET_RADIX_BUCKETS);
static_assert((1 << BUCKET_RADIX_BITS) == BUCKET_RADIX_BUCKETS,
              "BUCKET_RADIX_BUCKETS must be a power of two");

// ============================================================================
// 1. Bucket histogram kernel
//    Each block accumulates a local histogram in shared memory, then merges
//    into a single global histogram d_hist[B] (small).
// ============================================================================

__global__ void bucketHistogramKernel(const uint64_t* d_in,
                                      unsigned int*   d_hist,
                                      size_t          N) {
    extern __shared__ unsigned int local_hist[]; // size = BUCKET_RADIX_BUCKETS

    const int B = BUCKET_RADIX_BUCKETS;
    int tid     = threadIdx.x;

    // Initialize shared histogram
    for (int b = tid; b < B; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    // Strided loop over input
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    while (idx < N) {
        uint64_t key = d_in[idx];
        // Take top BUCKET_RADIX_BITS bits as bucket index
        unsigned int bucket =
            static_cast<unsigned int>((key >> (64 - BUCKET_RADIX_BITS)) &
                                      (BUCKET_RADIX_BUCKETS - 1));
        // Shared-memory atomic (cheap)
        atomicAdd(&local_hist[bucket], 1u);
        idx += stride;
    }
    __syncthreads();

    // Merge local histogram into global histogram (once per bucket per block)
    for (int b = tid; b < B; b += blockDim.x) {
        unsigned int count = local_hist[b];
        if (count > 0) {
            atomicAdd(&d_hist[b], count);
        }
    }
}

// ============================================================================
// 2. Prefix-sum kernel (exclusive scan over BUCKET_RADIX_BUCKETS bins)
// ============================================================================

__global__ void prefixSumKernel(const unsigned int* d_hist,
                                unsigned int*       d_prefix,
                                int                 B) {
    __shared__ unsigned int temp[BUCKET_RADIX_BUCKETS];
    int tid = threadIdx.x;

    // Load histogram into shared memory
    if (tid < B) {
        temp[tid] = d_hist[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Hillis–Steele inclusive scan
    for (int offset = 1; offset < B; offset <<= 1) {
        unsigned int val = 0;
        if (tid >= offset && tid < B) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset && tid < B) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Convert to exclusive scan
    if (tid < B) {
        if (tid == 0) {
            d_prefix[0] = 0;
        } else {
            d_prefix[tid] = temp[tid - 1];
        }
    }
}

// ============================================================================
// 3. Warp-aggregated scatter into buckets using prefix offsets
//    - Each warp groups threads by bucket
//    - One atomicAdd per (warp, bucket_group) instead of per element
// ============================================================================

__global__ void bucketScatterKernelWarpAgg(const uint64_t*   d_in,
                                           uint64_t*         d_temp,
                                           const unsigned int* d_prefix,
                                           unsigned int*     d_bucketOffset,
                                           size_t            N) {
    const int lane = threadIdx.x & 31;
    const unsigned int fullMask = 0xffffffffu;

    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    while (idx < N) {
        // Load key & bucket for this iteration
        uint64_t key = d_in[idx];
        unsigned int bucket =
            static_cast<unsigned int>((key >> (64 - BUCKET_RADIX_BITS)) &
                                      (BUCKET_RADIX_BUCKETS - 1));

        // Active lanes in this warp for this iteration
        unsigned int activeMask = __activemask();

        // We'll partition active lanes into groups by bucket value
        unsigned int remaining = activeMask;

        while (remaining) {
            // Pick a leader (lowest set bit in remaining)
            int leader = __ffs(remaining) - 1;

            // Broadcast the leader's bucket to the warp
            unsigned int leaderBucket =
                __shfl_sync(activeMask, bucket, leader);

            // All lanes with this bucket join the group
            unsigned int groupMask = __ballot_sync(
                activeMask,
                (bucket == leaderBucket)
            );

            // How many threads in this group?
            int groupCount = __popc(groupMask);

            // Leader does a single atomicAdd for the whole group
            unsigned int base = 0;
            if (lane == leader) {
                base = atomicAdd(&d_bucketOffset[leaderBucket],
                                 static_cast<unsigned int>(groupCount));
            }

            // Broadcast base to all lanes in group
            base = __shfl_sync(groupMask, base, leader);

            // Compute each lane's rank within the group
            unsigned int laneMask = groupMask & ((1u << lane) - 1u);
            int rank = __popc(laneMask);

            // Lanes in the group write their keys
            if (groupMask & (1u << lane)) {
                unsigned int destIndex =
                    d_prefix[leaderBucket] + base + static_cast<unsigned int>(rank);
                d_temp[destIndex] = key;
            }

            // Remove this group from remaining lanes
            remaining &= ~groupMask;
        }

        idx += stride;
    }
}

// ============================================================================
// 4. Shared-memory bitonic sort for small buckets (<= 32 elements)
// ============================================================================

__global__ void shared_memory_sort_kernel(const uint64_t* d_in,
                                          uint64_t*       d_out,
                                          int             count) {
    __shared__ uint64_t sdata[32];
    int tid = threadIdx.x;

    uint64_t val = (tid < count) ? d_in[tid] : 0xFFFFFFFFFFFFFFFFull;
    sdata[tid]   = val;
    __syncthreads();

    // Bitonic sort network for 32 elements
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ix     = tid;
            int ix_xor = ix ^ j;
            if (ix_xor > ix) {
                bool up = ((ix & k) == 0);
                uint64_t a = sdata[ix];
                uint64_t b = sdata[ix_xor];
                if ((up && a > b) || (!up && a < b)) {
                    sdata[ix]     = b;
                    sdata[ix_xor] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < count) {
        d_out[tid] = sdata[tid];
    }
}

// ============================================================================
// 5. Hybrid bucket + radix sort (bucket_radix)
//    - Global histogram (shared per block -> global d_hist[B])
//    - Device prefix over d_hist -> d_prefix[B]
//    - Warp-aggregated scatter (few atomics, coalesced writes)
//    - Small buckets: shared-memory bitonic sort
//    - Larger buckets: GPU radix sort per bucket with stream pool
// ============================================================================

void bucket_sort_radix_uint64_gpu(
    uint64_t* d_in,
    uint64_t* d_out,
    size_t    N,
    int       blocks,
    int       threads,
    double&   kernel_ms,
    double&   total_ms
) {
    assert(d_in != nullptr && d_out != nullptr);
    kernel_ms = 0.0;
    total_ms  = 0.0;

    if (N <= 1) {
        if (N == 1) {
            cudaMemcpy(d_out, d_in, sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        }
        return;
    }

    // ----------------- Timing setup -----------------
    auto t_start = std::chrono::high_resolution_clock::now();
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    // ----------------- Launch config -----------------
    int THREADS = (threads > 0) ? threads : 256;
    int BLOCKS  = (blocks  > 0) ? blocks  : 0;

    if (BLOCKS <= 0) {
        BLOCKS = static_cast<int>((N + THREADS - 1) / THREADS);
        // reasonable clamp to avoid too many blocks
        if (BLOCKS > 256) BLOCKS = 256;
        if (BLOCKS < 1)   BLOCKS = 1;
    }

    const int B = BUCKET_RADIX_BUCKETS;

    // ----------------- Histogram & prefix on device -----------------
    unsigned int* d_hist   = nullptr;
    unsigned int* d_prefix = nullptr;
    cudaMalloc(&d_hist,   B * sizeof(unsigned int));
    cudaMalloc(&d_prefix, B * sizeof(unsigned int));
    cudaMemset(d_hist, 0, B * sizeof(unsigned int));

    // Histogram over buckets
    size_t histShmem = B * sizeof(unsigned int);
    bucketHistogramKernel<<<BLOCKS, THREADS, histShmem>>>(d_in, d_hist, N);
    cudaDeviceSynchronize();

    // Prefix sum over bucket counts
    int prefixThreads = (B < 32) ? 32 : B;
    prefixSumKernel<<<1, prefixThreads>>>(d_hist, d_prefix, B);
    cudaDeviceSynchronize();

    // ----------------- Scatter into temporary array (warp aggregated) -----------------
    uint64_t* d_temp = nullptr;
    cudaMalloc(&d_temp, N * sizeof(uint64_t));

    // Reuse d_hist as per-bucket offset counter (now used by warp-aggregated scatter)
    cudaMemset(d_hist, 0, B * sizeof(unsigned int));

    bucketScatterKernelWarpAgg<<<BLOCKS, THREADS>>>(
        d_in,
        d_temp,
        d_prefix,
        d_hist,
        N
    );
    cudaDeviceSynchronize();

    // ----------------- Fetch bucket sizes to host -----------------
    std::vector<unsigned int> host_hist(B);
    cudaMemcpy(host_hist.data(),
               d_hist,
               B * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // ----------------- Stream pool for parallel bucket sorting -----------------
    const int STREAM_COUNT = 16;
    cudaStream_t streams[STREAM_COUNT];
    for (int s = 0; s < STREAM_COUNT; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    // We'll accumulate offsets as we go through buckets in order.
    size_t offset = 0;

    for (int b = 0; b < B; ++b) {
        unsigned int bucketSize = host_hist[b];
        if (bucketSize == 0) {
            // No elements in this bucket
            continue;
        }

        int sid = b % STREAM_COUNT;

        uint64_t* bucket_in  = d_temp + offset;
        uint64_t* bucket_out = d_out  + offset;

        if (bucketSize <= 32) {
            // Very small bucket: use shared-memory bitonic sort
            shared_memory_sort_kernel<<<1, 32, 0, streams[sid]>>>(
                bucket_in,
                bucket_out,
                static_cast<int>(bucketSize)
            );
        } else {
            // Larger bucket: call GPU radix sort on the bucket's range in its own stream
            double dummy_kernel_ms = 0.0;
            double dummy_total_ms  = 0.0;

            // Choose local grid for this bucket (avoid using too many "empty" blocks)
            int localThreads = THREADS;
            int localBlocks  = static_cast<int>((bucketSize + localThreads - 1) / localThreads);
            if (localBlocks <= 0) localBlocks = 1;
            if (localBlocks > BLOCKS) localBlocks = BLOCKS;

            radix_sort_uint64_gpu(
                bucket_in,
                bucket_out,
                bucketSize,
                localBlocks,
                localThreads,
                dummy_kernel_ms,
                dummy_total_ms,
                streams[sid]
            );
        }

        offset += bucketSize;
    }

    // Wait for all bucket sorts to complete and destroy streams
    for (int s = 0; s < STREAM_COUNT; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    // ----------------- End timing -----------------
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    kernel_ms = static_cast<double>(ms);

    auto t_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ----------------- Cleanup -----------------
    cudaFree(d_temp);
    cudaFree(d_prefix);
    cudaFree(d_hist);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}
