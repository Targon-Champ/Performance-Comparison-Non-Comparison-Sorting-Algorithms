// cuda/main_gpu_bench.cu
//
// Generic GPU sorting benchmark harness.
// Ready for multiple algorithms (Radix, Bitonic, Bucket, etc.).
//
// Usage:
//   gpu_sort_bench <dataset.bin> <algorithm> [blocks] [threads] [verify] [repeats] [csv_path]
//
// Example:
//   gpu_sort_bench data/uniform_N1e6.bin radix 0 0 1 5 results.csv
//
//   - blocks = 0  -> auto-tune based on N
//   - threads = 0 -> default 256
//   - verify = 1  -> check sortedness on host
//   - repeats = 5 -> run algorithm 5 times and average times
//   - csv_path    -> optional; if given, append metrics row to this file
//

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <cmath>

#include "radix_sort.cuh"
#include "bitonic_sort_gpu.cuh"
#include "bucket_sort_gpu.cuh"
#include "bucket_sort_radix_gpu.cuh"

// -----------------------------------------------------------------------------
// CUDA error check
// -----------------------------------------------------------------------------
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
// Utility: load uint64_t dataset from a binary file
// -----------------------------------------------------------------------------
std::vector<std::uint64_t> load_binary_uint64(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open dataset file: " + path);
    }

    std::streamsize file_size = ifs.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Failed to get file size for: " + path);
    }
    if (file_size % static_cast<std::streamsize>(sizeof(std::uint64_t)) != 0) {
        throw std::runtime_error("File size is not a multiple of sizeof(uint64_t): " + path);
    }

    std::size_t N = static_cast<std::size_t>(file_size / sizeof(std::uint64_t));
    std::vector<std::uint64_t> data(N);

    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(reinterpret_cast<char*>(data.data()), file_size)) {
        throw std::runtime_error("Failed to read dataset from: " + path);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Utility: check sortedness (ascending) on host
// -----------------------------------------------------------------------------
bool is_sorted_ascending(const std::vector<std::uint64_t>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i - 1] > v[i]) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Algorithm function type (includes a stream parameter)
// -----------------------------------------------------------------------------
using GpuSortFn = void(*)(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   stream
);

struct GpuSortAlgorithm {
    std::string name;   // algo name: radix / bitonic / bucket / bucket_radix
    GpuSortFn   fn;
};

// -----------------------------------------------------------------------------
// Small wrapper entry points to match GpuSortFn
// -----------------------------------------------------------------------------

// radix_sort_uint64_gpu has a stream parameter already; just forward it.
void radix_sort_entry(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   stream
) {
    radix_sort_uint64_gpu(d_in, d_out, N, blocks, threads,
                          kernel_ms, total_ms, stream);
}

// bitonic_sort_uint64_gpu: no stream argument; ignore 'stream'.
void bitonic_sort_entry(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   /*stream*/
) {
    bitonic_sort_uint64_gpu(d_in, d_out, N, blocks, threads,
                            kernel_ms, total_ms);
}

// bucket_sort_uint64_gpu: no stream; ignore 'stream'.
void bucket_sort_entry(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   /*stream*/
) {
    bucket_sort_uint64_gpu(d_in, d_out, N, blocks, threads,
                           kernel_ms, total_ms);
}

// bucket_sort_radix_uint64_gpu manages its own internal streams; ignore 'stream'.
void bucket_radix_sort_entry(
    std::uint64_t* d_in,
    std::uint64_t* d_out,
    std::size_t    N,
    int            blocks,
    int            threads,
    double&        kernel_ms,
    double&        total_ms,
    cudaStream_t   /*stream*/
) {
    bucket_sort_radix_uint64_gpu(d_in, d_out, N, blocks, threads,
                                 kernel_ms, total_ms);
}

// -----------------------------------------------------------------------------
// Algorithm registry
// -----------------------------------------------------------------------------
std::unordered_map<std::string, GpuSortAlgorithm> build_algorithm_registry() {
    std::unordered_map<std::string, GpuSortAlgorithm> registry;

    registry.emplace("radix", GpuSortAlgorithm{
        "radix",
        &radix_sort_entry
    });

    registry.emplace("bitonic", GpuSortAlgorithm{
        "bitonic",
        &bitonic_sort_entry
    });

    registry.emplace("bucket", GpuSortAlgorithm{
        "bucket",
        &bucket_sort_entry
    });

    registry.emplace("bucket_radix", GpuSortAlgorithm{
        "bucket_radix",
        &bucket_radix_sort_entry
    });

    return registry;
}

// -----------------------------------------------------------------------------
// GPU info
// -----------------------------------------------------------------------------
void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Using device " << device << ": " << prop.name << "\n";
    std::cout << "  SMs                : " << prop.multiProcessorCount << "\n";
    std::cout << "  Global memory (MB) : "
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0)
              << "\n";
    std::cout << "  Compute capability : " << prop.major << "." << prop.minor << "\n";
    std::cout << std::string(60, '-') << "\n";
}

// -----------------------------------------------------------------------------
// Auto-tune grid / block if user passes 0
// -----------------------------------------------------------------------------
void choose_launch_config(
    std::size_t N,
    int& blocks,
    int& threads
) {
    if (threads <= 0) {
        threads = 256; // default
    }

    if (blocks <= 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        int maxBlocksBySM = prop.multiProcessorCount * 4; // 4 blocks/SM
        std::size_t needed = (N + static_cast<std::size_t>(threads) - 1) / threads;
        blocks = static_cast<int>(std::min<std::size_t>(needed, maxBlocksBySM));
        if (blocks <= 0) blocks = 1;
    }
}

// -----------------------------------------------------------------------------
// Dataset metadata parsing from filename
// e.g. exponential_n1000000_min0_max4294967295_dtypeuint64_seed42_lam1.0.bin
// -----------------------------------------------------------------------------
struct DatasetMeta {
    std::string dist;
    std::string dtype;
};

DatasetMeta parse_dataset_meta(const std::string& dataset_path) {
    DatasetMeta meta;
    meta.dist  = "unknown";
    meta.dtype = "unknown";

    // Extract filename
    std::string fname = dataset_path;
    auto pos = fname.find_last_of("/\\");
    if (pos != std::string::npos) {
        fname = fname.substr(pos + 1);
    }

    // dist = substring from start until "_n"
    auto npos = fname.find("_n");
    if (npos != std::string::npos && npos > 0) {
        meta.dist = fname.substr(0, npos);
    }

    // dtype = substring after "_dtype" until next '_' or '.'
    auto dpos = fname.find("_dtype");
    if (dpos != std::string::npos) {
        auto start = dpos + 6; // length of "_dtype"
        auto end   = fname.find('_', start);
        if (end == std::string::npos) {
            end = fname.find('.', start);
        }
        if (end != std::string::npos && end > start) {
            meta.dtype = fname.substr(start, end - start);
        } else {
            meta.dtype = fname.substr(start);
        }
    }

    return meta;
}

// -----------------------------------------------------------------------------
// Benchmark single run for one algorithm on one dataset
// -----------------------------------------------------------------------------
struct BenchmarkResult {
    std::string algorithm;
    std::string impl;         // e.g., "gpu"
    std::string dataset_path;
    std::string dist;
    std::string dtype;

    std::size_t N;

    int blocks;
    int threads;

    int    repeats;
    double kernel_ms_avg;
    double total_ms_avg;
    double kernel_ms_std;
    double total_ms_std;

    bool   verified;
    bool   sorted_ok;
};

BenchmarkResult run_benchmark(
    const GpuSortAlgorithm& algo,
    const std::string&      dataset_path,
    int                     blocks,
    int                     threads,
    bool                    verify,
    int                     repeats
) {
    // 1) Load dataset
    std::vector<std::uint64_t> h_in = load_binary_uint64(dataset_path);
    std::size_t N = h_in.size();
    if (N == 0) {
        throw std::runtime_error("Dataset is empty: " + dataset_path);
    }

    // Parse metadata from filename
    DatasetMeta meta = parse_dataset_meta(dataset_path);

    // 2) Choose launch config if needed
    choose_launch_config(N, blocks, threads);

    std::cout << "Dataset: " << dataset_path << "\n";
    std::cout << "  N       : " << N << "\n";
    std::cout << "  Algo    : " << algo.name << "\n";
    std::cout << "  Blocks  : " << blocks << "\n";
    std::cout << "  Threads : " << threads << "\n";
    std::cout << "  Repeats : " << repeats << "\n";
    std::cout << std::string(60, '-') << "\n";

    // 3) Device allocations
    std::uint64_t* d_in  = nullptr;
    std::uint64_t* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(std::uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));

    // 4) Benchmark loop
    std::vector<double> kernel_times;
    std::vector<double> total_times;
    kernel_times.reserve(repeats);
    total_times.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        // Re-copy input each run to avoid sorted input bias
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                              N * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice));

        double kernel_ms = 0.0;
        double total_ms  = 0.0;

        // Call algorithm with stream = 0 (default stream)
        algo.fn(d_in, d_out, N, blocks, threads, kernel_ms, total_ms, 0);

        kernel_times.push_back(kernel_ms);
        total_times.push_back(total_ms);

        std::cout << "Run " << (r + 1)
                  << ": kernel_ms = " << kernel_ms
                  << ", total_ms = " << total_ms << "\n";
    }

    // 5) Copy result back and optionally verify
    std::vector<std::uint64_t> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          N * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));

    bool sorted_ok = true;
    if (verify) {
        sorted_ok = is_sorted_ascending(h_out);
        std::cout << "Sorted check: " << (sorted_ok ? "OK" : "FAILED") << "\n";
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // 6) Aggregate statistics
    auto stats = [](const std::vector<double>& xs) {
        double mean = 0.0;
        double var  = 0.0;
        if (!xs.empty()) {
            mean = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
            if (xs.size() > 1) {
                for (double x : xs) {
                    double diff = x - mean;
                    var += diff * diff;
                }
                var /= static_cast<double>(xs.size() - 1);
            }
        }
        return std::pair<double,double>(mean, std::sqrt(var));
    };

    auto [k_mean, k_std] = stats(kernel_times);
    auto [t_mean, t_std] = stats(total_times);

    BenchmarkResult res;
    res.algorithm     = algo.name;
    res.impl          = "gpu";
    res.dataset_path  = dataset_path;
    res.dist          = meta.dist;
    res.dtype         = meta.dtype;
    res.N             = N;
    res.blocks        = blocks;
    res.threads       = threads;
    res.repeats       = repeats;
    res.kernel_ms_avg = k_mean;
    res.total_ms_avg  = t_mean;
    res.kernel_ms_std = k_std;
    res.total_ms_std  = t_std;
    res.verified      = verify;
    res.sorted_ok     = sorted_ok;

    return res;
}

// -----------------------------------------------------------------------------
// Pretty-print + CSV summary
// -----------------------------------------------------------------------------
void print_result_human(const BenchmarkResult& r) {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "GPU SORT BENCH RESULT\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Algorithm : " << r.algorithm << "\n";
    std::cout << "Dataset   : " << r.dataset_path << "\n";
    std::cout << "N         : " << r.N << "\n";
    std::cout << "Blocks    : " << r.blocks << "\n";
    std::cout << "Threads   : " << r.threads << "\n";
    std::cout << "Repeats   : " << r.repeats << "\n";
    std::cout << "Verified  : " << (r.verified ? "yes" : "no") << "\n";
    if (r.verified) {
        std::cout << "Sorted OK : " << (r.sorted_ok ? "yes" : "NO") << "\n";
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Kernel ms : " << r.kernel_ms_avg << " ± " << r.kernel_ms_std << "\n";
    std::cout << "Total ms  : " << r.total_ms_avg << " ± " << r.total_ms_std << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_result_csv_header(std::ostream& os) {
    os << "dist,algo,impl,dtype,"
          "N,threads,ms,s_per_elem,ns_per_elem,mels_per_s,"
          "bytes,GB,GBps,"
          "sorted,match_ref,error,"
          "blocks,repeats,kernel_ms_avg,kernel_ms_std,total_ms_avg,total_ms_std,verified,"
          "dataset\n";
}

void print_result_csv_row(std::ostream& os, const BenchmarkResult& r) {
    // Use total_ms_avg as main "ms"
    double ms          = r.total_ms_avg;
    double seconds     = ms / 1000.0;
    double s_per_elem  = (r.N > 0) ? (seconds / static_cast<double>(r.N)) : 0.0;
    double ns_per_elem = s_per_elem * 1e9;
    double mels_per_s  = (seconds > 0.0)
                         ? (static_cast<double>(r.N) / 1e6) / seconds
                         : 0.0;

    // uint64_t elements: 8 bytes each
    double bytes = static_cast<double>(r.N) * 8.0;
    double GB    = bytes / 1e9;            // decimal GB
    double GBps  = (seconds > 0.0) ? (GB / seconds) : 0.0;

    int sorted_flag     = (r.verified && r.sorted_ok) ? 1 : 0;
    int match_ref_flag  = sorted_flag;     // no separate reference yet
    double error_metric = 0.0;             // placeholder

    os << r.dist << ","
       << r.algorithm << ","
       << r.impl << ","
       << r.dtype << ",";

    os << r.N << ","
       << r.threads << ",";

    os << std::fixed << std::setprecision(6)
       << ms << ","
       << s_per_elem << ","
       << ns_per_elem << ","
       << mels_per_s << ","
       << bytes << ","
       << GB << ","
       << GBps << ",";

    os << sorted_flag << ","
       << match_ref_flag << ","
       << error_metric << ",";

    os << r.blocks << ","
       << r.repeats << ","
       << r.kernel_ms_avg << ","
       << r.kernel_ms_std << ","
       << r.total_ms_avg << ","
       << r.total_ms_std << ","
       << (r.verified ? 1 : 0) << ",";

    os << r.dataset_path << "\n";
}

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  " << argv[0]
                      << " <dataset.bin> <algorithm> [blocks] [threads] [verify] [repeats] [csv_path]\n\n"
                      << "Example:\n"
                      << "  " << argv[0]
                      << " data/uniform_N1e6.bin radix 0 0 1 5 results.csv\n\n"
                      << "Algorithms:\n"
                      << "  radix        (Radix Sort)\n"
                      << "  bitonic      (Global bitonic network, padded to power-of-two)\n"
                      << "  bucket       (GPU bucket sort)\n"
                      << "  bucket_radix (Bucket + per-bucket radix hybrid)\n";
            return 1;
        }

        std::string dataset_path = argv[1];
        std::string algo_name    = argv[2];

        int blocks  = 0;
        int threads = 0;
        bool verify = true;
        int repeats = 3;

        if (argc >= 4) blocks  = std::stoi(argv[3]);
        if (argc >= 5) threads = std::stoi(argv[4]);
        if (argc >= 6) verify  = (std::stoi(argv[5]) != 0);
        if (argc >= 7) repeats = std::stoi(argv[6]);
        if (repeats <= 0) repeats = 1;

        std::string csv_path;
        bool have_csv = false;
        if (argc >= 8) {
            csv_path = argv[7];
            have_csv = true;
        }

        // Build algorithm registry and look up requested algorithm
        auto registry = build_algorithm_registry();
        auto it = registry.find(algo_name);
        if (it == registry.end()) {
            std::cerr << "Unknown algorithm: " << algo_name << "\n";
            std::cerr << "Available algorithms:\n";
            for (const auto& kv : registry) {
                std::cerr << "  " << kv.first << "\n";
            }
            return 1;
        }
        const GpuSortAlgorithm& algo = it->second;

        print_device_info();

        BenchmarkResult result = run_benchmark(
            algo,
            dataset_path,
            blocks,
            threads,
            verify,
            repeats
        );

        // Human-readable metrics to stdout
        print_result_human(result);

        // CSV output to file (if provided)
        if (have_csv) {
            bool need_header = false;

            {
                std::ifstream ifs(csv_path, std::ios::binary | std::ios::ate);
                if (!ifs.good()) {
                    // File does not exist or cannot be opened: assume new -> header
                    need_header = true;
                } else {
                    auto sz = ifs.tellg();
                    if (sz == 0) {
                        need_header = true;
                    }
                }
            }

            std::ofstream ofs(csv_path, std::ios::app);
            if (!ofs) {
                std::cerr << "Failed to open CSV file for append: " << csv_path << "\n";
                return 1;
            }

            if (need_header) {
                print_result_csv_header(ofs);
            }
            print_result_csv_row(ofs, result);
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    }
}
