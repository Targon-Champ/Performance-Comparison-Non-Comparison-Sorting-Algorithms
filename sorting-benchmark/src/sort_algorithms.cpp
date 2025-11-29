#include "sort_algorithms.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <limits>
#ifdef USE_OPENMP
#include <omp.h>
#endif

// For now, everything just calls std::sort so the harness works.
// We will replace these one by one with real implementations.

// ====================== COUNTING SORT (SEQ) ======================

void counting_sort_uint64_seq(std::vector<std::uint64_t>& a) {
    if (a.empty()) return;

    // Find min and max to support arbitrary ranges
    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv < minv) {
        throw std::runtime_error("Counting sort: invalid range (max < min)");
    }

    std::uint64_t range = maxv - minv + 1;

    // HARD CAP: avoid insane allocations on full 32-bit range
    // 1e8 * 8 bytes ≈ 800 MB of counts; okay for research nodes,
    // but 4e9 would be ~32 GB and blow up.
    const std::uint64_t MAX_RANGE = 100000000ULL;  // 1e8
    if (range > MAX_RANGE) {
        throw std::runtime_error(
            "Counting sort range too large: " + std::to_string(range)
        );
    }

    std::vector<std::size_t> count(static_cast<std::size_t>(range), 0);

    // Histogram
    for (std::uint64_t x : a) {
        std::uint64_t idx = x - minv;
        ++count[static_cast<std::size_t>(idx)];
    }

    // Write back in sorted order
    std::size_t out_idx = 0;
    for (std::uint64_t offset = 0; offset < range; ++offset) {
        std::size_t c = count[static_cast<std::size_t>(offset)];
        std::uint64_t value = minv + offset;
        for (std::size_t i = 0; i < c; ++i) {
            a[out_idx++] = value;
        }
    }
}


// ====================== COUNTING SORT (OMP) ======================

void counting_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    // Fallback: no OpenMP available, use sequential version
    counting_sort_uint64_seq(a);
#else
    if (a.empty()) return;

    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv < minv) {
        throw std::runtime_error("Counting sort (OMP): invalid range (max < min)");
    }

    std::uint64_t range = maxv - minv + 1;
    const std::uint64_t MAX_RANGE = 100000000ULL;
    if (range > MAX_RANGE) {
        throw std::runtime_error(
            "Counting sort (OMP) range too large: " + std::to_string(range)
        );
    }

    std::size_t R = static_cast<std::size_t>(range);

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    // Determine T actually used
    int T = 1;
    #pragma omp parallel
    {
        #pragma omp single
        T = omp_get_num_threads();
    }

    // local_counts: T × R
    std::vector<std::size_t> local_counts(static_cast<std::size_t>(T) * R, 0);

    // 1) Parallel histogram building
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::size_t offset = static_cast<std::size_t>(tid) * R;
        std::size_t* lc = local_counts.data() + offset;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < a.size(); ++i) {
            std::uint64_t x = a[i];
            std::uint64_t idx = x - minv;
            ++lc[static_cast<std::size_t>(idx)];
        }
    }

    // 2) Reduce local_counts into a single global count
    std::vector<std::size_t> count(R, 0);
    for (std::size_t r = 0; r < R; ++r) {
        std::size_t sum = 0;
        for (int t = 0; t < T; ++t) {
            sum += local_counts[static_cast<std::size_t>(t) * R + r];
        }
        count[r] = sum;
    }

    // 3) Rebuild array sequentially
    std::size_t out_idx = 0;
    for (std::size_t r = 0; r < R; ++r) {
        std::size_t c = count[r];
        std::uint64_t value = minv + static_cast<std::uint64_t>(r);
        for (std::size_t i = 0; i < c; ++i) {
            a[out_idx++] = value;
        }
    }
#endif
}


// =============================================================
// 64-bit LSD Radix Sort (base 256) — SEQUENTIAL
// =============================================================
//
// Stable, 8-pass LSD radix sort with 256 buckets per pass.
// Requires an auxiliary buffer of size N, reused each pass.
// Very fast for large arrays and full 64-bit domains.
//
void radix_sort_uint64_seq(std::vector<std::uint64_t>& a) {
    const std::size_t N = a.size();
    if (N <= 1) return;

    std::vector<std::uint64_t> temp(N);

    // There are 8 bytes in a uint64:
    // pass 0 sorts by byte 0 (least significant)
    // pass 7 sorts by byte 7 (most significant)
    for (int byte = 0; byte < 8; ++byte) {
        std::size_t count[256] = {0};

        // Extract shift amount
        const int shift = byte * 8;

        // 1. Build histogram for this byte
        for (std::size_t i = 0; i < N; ++i) {
            unsigned int b = (a[i] >> shift) & 0xFFu;
            ++count[b];
        }

        // 2. Prefix sum for stable distribution
        std::size_t prefix[256];
        prefix[0] = 0;
        for (int b = 1; b < 256; ++b) {
            prefix[b] = prefix[b - 1] + count[b - 1];
        }

        // 3. Scatter into temp using prefix positions
        for (std::size_t i = 0; i < N; ++i) {
            unsigned int b = (a[i] >> shift) & 0xFFu;
            std::size_t pos = prefix[b]++;
            temp[pos] = a[i];
        }

        // 4. Copy back for next pass
        a.swap(temp);
    }
}


// =============================================================
// 64-bit LSD Radix Sort (base 256) — OPENMP PARALLEL
// =============================================================
//
// Parallel strategy:
//  - For each byte (8 passes total):
//    * Each thread processes a fixed chunk of the array
//      and builds a 256-bin local histogram.
//    * We reduce local histograms into a global histogram.
//    * We compute global prefix positions for each bucket.
//    * For each thread and bucket, we precompute a starting
//      offset = global_prefix[b] + sum(local_counts of previous threads).
//    * Each thread processes its chunk again, scattering into
//      a shared temp buffer using its own per-bucket offsets.
//    * Swap a <-> temp for the next pass.
//
// This is stable and scales well for large N.
//
void radix_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    // Fallback: no OpenMP support compiled in; use sequential version.
    radix_sort_uint64_seq(a);
#else
    const std::size_t N = a.size();
    if (N <= 1) {
        return;
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    // Cap threads to N to avoid a silly number of mostly idle threads.
    if (static_cast<std::size_t>(num_threads) > N) {
        num_threads = static_cast<int>(N);
    }

    omp_set_num_threads(num_threads);

    std::vector<std::uint64_t> temp(N);

    int T = 1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            T = omp_get_num_threads();
        }
    }

    // Chunk size used for both histogram and scatter so that
    // each thread sees the same indices in both phases.
    const std::size_t chunk_size = (N + static_cast<std::size_t>(T) - 1) / static_cast<std::size_t>(T);

    // Per-pass buffers: local histograms and per-thread offsets
    std::vector<std::size_t> local_counts(static_cast<std::size_t>(T) * 256);
    std::vector<std::size_t> thread_offsets(static_cast<std::size_t>(T) * 256);

    // 8 passes over the 8 bytes of uint64_t
    for (int byte = 0; byte < 8; ++byte) {
        const int shift = byte * 8;

        // Zero local_counts
        std::fill(local_counts.begin(), local_counts.end(), 0);

        // 1) Build per-thread local histograms
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::size_t start = static_cast<std::size_t>(tid) * chunk_size;
            std::size_t end   = start + chunk_size;
            if (end > N) end = N;

            std::size_t* lc = &local_counts[static_cast<std::size_t>(tid) * 256];

            for (std::size_t i = start; i < end; ++i) {
                unsigned int b = static_cast<unsigned int>((a[i] >> shift) & 0xFFu);
                ++lc[b];
            }
        }

        // 2) Reduce to global histogram
        std::size_t global_count[256] = {0};
        for (int t = 0; t < T; ++t) {
            const std::size_t* lc = &local_counts[static_cast<std::size_t>(t) * 256];
            for (int b = 0; b < 256; ++b) {
                global_count[b] += lc[b];
            }
        }

        // 3) Compute global prefix sums for each bucket
        std::size_t global_prefix[256];
        global_prefix[0] = 0;
        for (int b = 1; b < 256; ++b) {
            global_prefix[b] = global_prefix[b - 1] + global_count[b - 1];
        }

        // 4) Compute per-thread starting offsets for each bucket
        for (int b = 0; b < 256; ++b) {
            std::size_t running = global_prefix[b];
            for (int t = 0; t < T; ++t) {
                std::size_t idx = static_cast<std::size_t>(t) * 256 + static_cast<std::size_t>(b);
                thread_offsets[idx] = running;
                running += local_counts[idx];
            }
        }

        // 5) Scatter into temp in parallel using per-thread offsets
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::size_t start = static_cast<std::size_t>(tid) * chunk_size;
            std::size_t end   = start + chunk_size;
            if (end > N) end = N;

            std::size_t* offs = &thread_offsets[static_cast<std::size_t>(tid) * 256];

            for (std::size_t i = start; i < end; ++i) {
                unsigned int b = static_cast<unsigned int>((a[i] >> shift) & 0xFFu);
                std::size_t pos = offs[b]++;
                temp[pos] = a[i];
            }
        }

        // 6) Swap for next pass
        a.swap(temp);
    }
#endif
}


// =============================================================
// Bucket Sort (uint64) — SEQUENTIAL
// =============================================================
//
// Strategy:
//  - Compute min/max.
//  - Choose B = min(N, MAX_BUCKETS) buckets.
//  - Map each element into a bucket using a linear mapping over [min, max].
//  - Sort each bucket with std::sort.
//  - Concatenate buckets.
//
// This is well-suited for roughly uniform distributions.
// No hard range cap is needed (only B buckets).
//
void bucket_sort_uint64_seq(std::vector<std::uint64_t>& a) {
    const std::size_t N = a.size();
    if (N <= 1) return;

    // 1. Find min and max
    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv <= minv) {
        // All elements equal or single element
        return;
    }

    std::uint64_t range = maxv - minv;
    // 2. Choose number of buckets
    const std::size_t MAX_BUCKETS = 100000;  // safety cap
    std::size_t B = N;
    if (B > MAX_BUCKETS) B = MAX_BUCKETS;
    if (B == 0) B = 1;

    std::vector<std::vector<std::uint64_t>> buckets(B);

    // Precompute scaling factor for mapping values to buckets.
    // scale = B / (range + 1)
    const long double scale = static_cast<long double>(B) /
                              (static_cast<long double>(range) + 1.0L);

    // 3. Distribute elements into buckets
    for (std::uint64_t x : a) {
        std::uint64_t offset = x - minv;
        std::size_t idx = static_cast<std::size_t>(
            static_cast<long double>(offset) * scale
        );
        if (idx >= B) idx = B - 1;  // guard against rounding
        buckets[idx].push_back(x);
    }

    // 4. Sort each bucket
    for (std::size_t i = 0; i < B; ++i) {
        auto &bucket = buckets[i];
        if (!bucket.empty()) {
            std::sort(bucket.begin(), bucket.end());
        }
    }

    // 5. Concatenate back into 'a'
    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < B; ++i) {
        const auto &bucket = buckets[i];
        for (std::uint64_t x : bucket) {
            a[out_idx++] = x;
        }
    }

    // Sanity (debug) check: out_idx should equal N, but we don't throw.
    // if (out_idx != N) {
    //     throw std::runtime_error("Bucket sort: output size mismatch");
    // }
}


// =============================================================
// Bucket Sort (uint64) — OPENMP PARALLEL
// =============================================================
//
// Parallelization strategy:
//  - Same bucketing as sequential (done sequentially for simplicity).
//  - Parallelize sorting of each bucket with std::sort in each thread.
//  - Concatenation done sequentially (cheap compared to sorting).
//
void bucket_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    bucket_sort_uint64_seq(a);
#else
    const std::size_t N = a.size();
    if (N <= 1) return;

    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv <= minv) {
        // All elements equal or single element
        return;
    }

    std::uint64_t range = maxv - minv;
    const std::size_t MAX_BUCKETS = 100000;
    std::size_t B = N;
    if (B > MAX_BUCKETS) B = MAX_BUCKETS;
    if (B == 0) B = 1;

    std::vector<std::vector<std::uint64_t>> buckets(B);

    const long double scale = static_cast<long double>(B) /
                              (static_cast<long double>(range) + 1.0L);

    // 1. Distribute elements into buckets (sequential)
    for (std::uint64_t x : a) {
        std::uint64_t offset = x - minv;
        std::size_t idx = static_cast<std::size_t>(
            static_cast<long double>(offset) * scale
        );
        if (idx >= B) idx = B - 1;
        buckets[idx].push_back(x);
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    // 2. Parallel sort buckets
    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(B); ++ii) {
        std::size_t i = static_cast<std::size_t>(ii);
        auto &bucket = buckets[i];
        if (!bucket.empty()) {
            std::sort(bucket.begin(), bucket.end());
        }
    }

    // 3. Concatenate back into 'a' (sequential)
    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < B; ++i) {
        const auto &bucket = buckets[i];
        for (std::uint64_t x : bucket) {
            a[out_idx++] = x;
        }
    }
#endif
}


// =============================================================
// Pigeonhole Sort (uint64) — SEQUENTIAL
// =============================================================
//
// Classic pigeonhole: create one "hole" per possible key in range,
// place each element into its hole, then flatten holes in order.
// This is only practical when (max - min + 1) is small.
//
// We enforce a MAX_RANGE cap to avoid insane memory usage.
// For your research, this helps demonstrate "algorithm not applicable"
// on wide-range datasets.
//
void pigeonhole_sort_uint64_seq(std::vector<std::uint64_t>& a) {
    const std::size_t N = a.size();
    if (N <= 1) return;

    // 1. Find min and max
    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv < minv) {
        throw std::runtime_error("Pigeonhole sort: invalid range (max < min)");
    }

    std::uint64_t range = maxv - minv + 1;

    // Pigeonhole is more memory-heavy than counting: we allocate
    // a vector< vector<uint64_t> > with size=range, plus all elements.
    // Keep MAX_RANGE reasonably small (e.g. 1e6) for safety.
    const std::uint64_t MAX_RANGE = 1000000ULL;  // 1e6 holes
    if (range > MAX_RANGE) {
        throw std::runtime_error(
            "Pigeonhole sort range too large: " + std::to_string(range)
        );
    }

    const std::size_t R = static_cast<std::size_t>(range);

    // 2. Create holes
    std::vector<std::vector<std::uint64_t>> holes(R);

    // 3. Place elements into holes
    for (std::uint64_t x : a) {
        std::size_t idx = static_cast<std::size_t>(x - minv);
        holes[idx].push_back(x);
    }

    // 4. Flatten holes back into 'a' in ascending order
    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < R; ++i) {
        const auto& bucket = holes[i];
        for (std::uint64_t x : bucket) {
            a[out_idx++] = x;
        }
    }
}


// =============================================================
// Pigeonhole Sort (uint64) — OPENMP PARALLEL
// =============================================================
//
// Strategy:
//  - Same range check as sequential.
//  - Bucket fill is done sequentially (simpler & avoids per-thread
//    huge 2D structures).
//  - Flatten step is parallelized across holes using prefix sums.
//
// This still shows interesting behavior in benchmarks, and the
// main point is comparing small-range performance vs failure on
// wide-range datasets.
//
void pigeonhole_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    pigeonhole_sort_uint64_seq(a);
#else
    const std::size_t N = a.size();
    if (N <= 1) return;

    auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    std::uint64_t minv = *min_it;
    std::uint64_t maxv = *max_it;

    if (maxv < minv) {
        throw std::runtime_error("Pigeonhole sort (OMP): invalid range (max < min)");
    }

    std::uint64_t range = maxv - minv + 1;
    const std::uint64_t MAX_RANGE = 1000000ULL;  // 1e6 holes
    if (range > MAX_RANGE) {
        throw std::runtime_error(
            "Pigeonhole sort (OMP) range too large: " + std::to_string(range)
        );
    }

    const std::size_t R = static_cast<std::size_t>(range);

    // Bucket storage
    std::vector<std::vector<std::uint64_t>> holes(R);

    // --- Bucket fill (sequential for simplicity) ---
    for (std::uint64_t x : a) {
        std::size_t idx = static_cast<std::size_t>(x - minv);
        holes[idx].push_back(x);
    }

    // --- Prefix sums of bucket sizes ---
    std::vector<std::size_t> offsets(R, 0);
    std::size_t running = 0;
    for (std::size_t i = 0; i < R; ++i) {
        offsets[i] = running;
        running += holes[i].size();
    }

    if (running != N) {
        throw std::runtime_error("Pigeonhole sort (OMP): total bucket size mismatch");
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    // --- Parallel flatten: each thread copies some buckets ---
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(R); ++ii) {
        std::size_t i = static_cast<std::size_t>(ii);
        const auto& bucket = holes[i];
        std::size_t start = offsets[i];
        for (std::size_t j = 0; j < bucket.size(); ++j) {
            a[start + j] = bucket[j];
        }
    }
#endif
}


// Compute next power of two >= n (for n > 0).
static std::size_t next_power_of_two(std::size_t n) {
    if (n <= 1) return 1;
    // Round up to next power of two.
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    n |= n >> 32;
#endif
    ++n;
    return n;
}

// =============================================================
// Bitonic Sort (uint64) — SEQUENTIAL
// =============================================================
//
// We pad to the next power-of-two with a sentinel (UINT64_MAX)
// that is > all real keys (safe for current dataset ranges).
//
void bitonic_sort_uint64_seq(std::vector<std::uint64_t>& a) {
    const std::size_t N = a.size();
    if (N <= 1) return;

    const std::size_t N_pad = next_power_of_two(N);

    // Copy into padded array with sentinel values
    std::vector<std::uint64_t> arr(N_pad, std::numeric_limits<std::uint64_t>::max());
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = a[i];
    }

    // Standard iterative bitonic sort (ascending)
    for (std::size_t k = 2; k <= N_pad; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {
            for (std::size_t i = 0; i < N_pad; ++i) {
                std::size_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    auto &x = arr[i];
                    auto &y = arr[ixj];
                    if (ascending) {
                        if (y < x) std::swap(x, y);
                    } else {
                        if (x < y) std::swap(x, y);
                    }
                }
            }
        }
    }

    // Copy back only real elements
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = arr[i];
    }
}


// =============================================================
// Bitonic Sort (uint64) — OPENMP PARALLEL
// =============================================================
//
// Parallelization: we parallelize the innermost loop over i.
// The compare–swap network structure (k, j) remains sequential.
//
void bitonic_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    bitonic_sort_uint64_seq(a);
#else
    const std::size_t N = a.size();
    if (N <= 1) return;

    const std::size_t N_pad = next_power_of_two(N);

    std::vector<std::uint64_t> arr(N_pad, std::numeric_limits<std::uint64_t>::max());
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = a[i];
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    for (std::size_t k = 2; k <= N_pad; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {

            #pragma omp parallel for schedule(static)
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(N_pad); ++ii) {
                std::size_t i = static_cast<std::size_t>(ii);
                std::size_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    auto &x = arr[i];
                    auto &y = arr[ixj];
                    if (ascending) {
                        if (y < x) std::swap(x, y);
                    } else {
                        if (x < y) std::swap(x, y);
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = arr[i];
    }
#endif
}


// Signed support via mapping (to be implemented properly later)
static inline std::uint64_t int64_to_uint64_key(std::int64_t x) {
    return static_cast<std::uint64_t>(x) ^ (1ull << 63);
}

static inline std::int64_t uint64_to_int64_key(std::uint64_t x) {
    return static_cast<std::int64_t>(x ^ (1ull << 63));
}

// =============================================================
// Signed int64 sorting via uint64 radix — SEQUENTIAL
// =============================================================
//
// 1. Map each int64 to uint64 by flipping sign bit.
// 2. Run uint64 radix sort.
// 3. Map back.
// This gives correct ascending ordering on int64.
//
void sort_int64_via_uint64_radix_seq(std::vector<std::int64_t>& a) {
    const std::size_t N = a.size();
    if (N <= 1) return;

    std::vector<std::uint64_t> tmp(N);
    for (std::size_t i = 0; i < N; ++i) {
        tmp[i] = int64_to_uint64_key(a[i]);
    }

    radix_sort_uint64_seq(tmp);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = uint64_to_int64_key(tmp[i]);
    }
}


// =============================================================
// Signed int64 sorting via uint64 radix — OPENMP
// =============================================================
//
// Same mapping, but using the OMP radix implementation.
//
void sort_int64_via_uint64_radix_omp(std::vector<std::int64_t>& a, int num_threads) {
#ifndef USE_OPENMP
    (void)num_threads;
    sort_int64_via_uint64_radix_seq(a);
#else
    const std::size_t N = a.size();
    if (N <= 1) return;

    std::vector<std::uint64_t> tmp(N);

    // Mapping step can be parallelized, but it's cheap; keep it simple.
    for (std::size_t i = 0; i < N; ++i) {
        tmp[i] = int64_to_uint64_key(a[i]);
    }

    radix_sort_uint64_omp(tmp, num_threads);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = uint64_to_int64_key(tmp[i]);
    }
#endif
}
