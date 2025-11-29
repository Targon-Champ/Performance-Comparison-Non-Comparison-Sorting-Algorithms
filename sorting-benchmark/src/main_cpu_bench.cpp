#include "sort_common.hpp"
#include "sort_algorithms.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Helper: print metrics to stdout and (optionally) append CSV row.
static void output_metrics_and_csv(
    const std::string& dist,
    const std::string& csv_path,
    const std::string& algo_str,
    const std::string& impl_str,
    const std::string& dtype_str,
    std::size_t        N,
    int                threads,
    double             ms,
    bool               sorted,
    bool               have_ref,
    bool               match_ref,
    const std::string& error_msg)
{
    // Derived metrics
    const double seconds = ms / 1000.0;
    const double elems   = static_cast<double>(N);

    double s_per_elem  = (seconds > 0.0 && elems > 0.0) ? (seconds / elems) : 0.0;
    double ns_per_elem = s_per_elem * 1e9;  // seconds → nanoseconds
    double mels_per_s  = (seconds > 0.0) ? (elems / seconds) / 1e6 : 0.0;

    // Assume each element is 8 bytes (int64/uint64)
    const double bytes = elems * 8.0;
    const double GB    = bytes / 1e9;
    const double GBps  = (seconds > 0.0) ? (GB / seconds) : 0.0;

    std::string match_ref_str;
    if (!have_ref) {
        match_ref_str = "na";
    } else {
        match_ref_str = match_ref ? "true" : "false";
    }

    // ---- stdout line ----
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "dist=" << dist
              << "algo=" << algo_str
              << " impl=" << impl_str
              << " dtype=" << dtype_str
              << " N=" << N
              << " threads=" << threads
              << " ms=" << ms
              << " s_per_elem=" << s_per_elem
              << " ns_per_elem=" << ns_per_elem
              << " mels_per_s=" << mels_per_s
              << " bytes=" << static_cast<long long>(bytes)
              << " GB=" << GB
              << " GBps=" << GBps
              << " sorted=" << (sorted ? "true" : "false")
              << " match_ref=" << match_ref_str
              << " error=\"" << error_msg << "\"\n";

    // ---- optional CSV output ----
    if (!csv_path.empty()) {
        bool need_header = false;
        {
            std::ifstream in(csv_path, std::ios::ate);
            // Need header if file doesn't exist OR exists but is empty
            if (!in.good() || in.tellg() == 0) {
                need_header = true;
            }
        }


        std::ofstream out(csv_path, std::ios::app);
        if (!out) {
            std::cerr << "Warning: failed to open CSV file for writing: "
                      << csv_path << "\n";
            return;
        }

        if (need_header) {
            out << "dist,algo,impl,dtype,N,threads,ms,s_per_elem,ns_per_elem,"
                   "mels_per_s,bytes,GB,GBps,sorted,match_ref,error\n";
        }

        // Very simple CSV; if you expect commas/quotes in error_msg, you
        // might want more robust escaping.
        out << dist << ','
            << algo_str << ','
            << impl_str << ','
            << dtype_str << ','
            << N << ','
            << threads << ','
            << ms << ','
            << s_per_elem << ','
            << ns_per_elem << ','
            << mels_per_s << ','
            << bytes << ','
            << GB << ','
            << GBps << ','
            << (sorted ? "true" : "false") << ','
            << match_ref_str << ','
            << '"' << error_msg << '"'  // quote the error field
            << "\n";
    }
}
static bool check_if_exists(
    const std::string& dist,
    const std::string& csv_path,
    const std::string& algo_str,
    const std::string& impl_str,
    const std::string& dtype_str,
    std::size_t        N
){
    std::ifstream in(csv_path);
    if (!in.good()) {
        return false;  // CSV does not exist → row cannot exist
    }

    std::string line;
    // Skip header (first line)
    if (!std::getline(in, line)) {
        return false; // Empty file → no entries
    }

    // Construct the key string we want to find in CSV
    // CSV columns (with dist included) should look like:
    // dist,algo,impl,dtype,N,threads,...
    //
    // So we match:
    //   dist + "," + algo_str + "," + impl_str + "," + dtype_str + "," + N + ","
    //
    // The trailing comma ensures "n10000" doesn't match "n100000".
    const std::string key =
        dist + "," +
        algo_str + "," +
        impl_str + "," +
        dtype_str + "," +
        std::to_string(N) + ",";

    // Scan CSV for a matching line
    while (std::getline(in, line)) {
        if (line.rfind(key, 0) == 0) {
            // The line starts with our exact key
            return true;
        }
    }

    return false;  // Not found
}


int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " --algo=<radix|counting|bucket|pigeonhole|bitonic>"
                  << " --impl=<seq|omp>"
                  << " --dtype=<int64|uint64>"
                  << " --input=<file.bin>"
                  << " --n=<N>"
                  << " --threads=<T>"
                  << " [--check]"
                  << " [--csv=<metrics.csv>]"
                  << " [--repeats=<R>]"
                  << " [--warmup=<W>]\n";
        return 1;
    }

    std::string algo_str, impl_str, dtype_str, input_path;
    std::string csv_path;
    std::size_t N       = 0;
    int         threads = 1;
    bool        check   = false;
    int         repeats = 1;
    int         warmup  = 0;

    // ----------------- Parse CLI -----------------
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--algo=", 0) == 0) {
            algo_str = arg.substr(7);
        } else if (arg.rfind("--impl=", 0) == 0) {
            impl_str = arg.substr(7);
        } else if (arg.rfind("--dtype=", 0) == 0) {
            dtype_str = arg.substr(8);
        } else if (arg.rfind("--input=", 0) == 0) {
            input_path = arg.substr(8);
        } else if (arg.rfind("--n=", 0) == 0) {
            N = std::stoull(arg.substr(4));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads = std::stoi(arg.substr(10));
        } else if (arg == "--check") {
            check = true;
        } else if (arg.rfind("--csv=", 0) == 0) {
            csv_path = arg.substr(6);
        } else if (arg.rfind("--repeats=", 0) == 0) {
            repeats = std::stoi(arg.substr(10));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            warmup = std::stoi(arg.substr(9));
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return 1;
        }
    }

    if (repeats < 1) repeats = 1;
    if (warmup < 0) warmup = 0;

    if (algo_str.empty() || impl_str.empty() || dtype_str.empty()
        || input_path.empty() || N == 0) {
        std::cerr << "Error: missing required arguments.\n";
        return 1;
    }
    std::string dist = input_path.substr(input_path.find_last_of("/\\") + 1);
    dist = dist.substr(0, dist.find("_n"));


    Algorithm algo;
    Impl      impl;
    KeyDtype  dtype;

    try {
        algo  = algorithm_from_string(algo_str);
        impl  = impl_from_string(impl_str);
        dtype = dtype_from_string(dtype_str);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    // Common outcome flags
    bool        best_sorted = false;
    bool        best_match  = false;
    double      ms_best     = 0.0;
    std::string error_msg   = "none";
    if (check_if_exists(dist, csv_path, algo_str, impl_str, dtype_str, N)) {
        std::cout << "Skipping existing entry for dist=" << dist
                << " algo="   << algo_str
                << " impl="   << impl_str
                << " dtype="  << dtype_str
                << " N="      << N << "\n";
        return 0;
    }


    // ============================================================
    // UNSIGNED uint64_t BRANCH
    // ============================================================
    if (dtype == KeyDtype::UInt64) {
        std::vector<std::uint64_t> original;

        try {
            original = load_uint64_bin(input_path, N);
        } catch (const std::exception& e) {
            error_msg = std::string("load_failed: ") + e.what();
            output_metrics_and_csv(dist,csv_path,
                                   algo_str, impl_str, "uint64",
                                   N, threads,
                                   0.0,     // ms
                                   false,   // sorted
                                   check,   // have_ref
                                   false,   // match_ref
                                   error_msg);
            return 1;
        }

        std::vector<std::uint64_t> ref;
        if (check) {
            ref = original;
            std::sort(ref.begin(), ref.end());
        }

        // Warmup runs (not measured)
        for (int w = 0; w < warmup; ++w) {
            std::vector<std::uint64_t> data = original;
            try {
                switch (algo) {
                case Algorithm::Counting:
                    if (impl == Impl::Seq) counting_sort_uint64_seq(data);
                    else                   counting_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Radix:
                    if (impl == Impl::Seq) radix_sort_uint64_seq(data);
                    else                   radix_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Bucket:
                    if (impl == Impl::Seq) bucket_sort_uint64_seq(data);
                    else                   bucket_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Pigeonhole:
                    if (impl == Impl::Seq) pigeonhole_sort_uint64_seq(data);
                    else                   pigeonhole_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Bitonic:
                    if (impl == Impl::Seq) bitonic_sort_uint64_seq(data);
                    else                   bitonic_sort_uint64_omp(data, threads);
                    break;
                }
            } catch (const std::exception& e) {
                error_msg = e.what();
                // If warmup fails, no point in continuing.
                output_metrics_and_csv(dist, csv_path,
                                       algo_str, impl_str, "uint64",
                                       N, threads,
                                       0.0,
                                       false,
                                       check,
                                       false,
                                       error_msg);
                return 1;
            }
        }

        bool any_success = false;

        // Measured runs
        for (int r = 0; r < repeats; ++r) {
            std::vector<std::uint64_t> data = original;

            double ms_run = 0.0;
            bool   sorted_run = false;
            bool   match_run  = false;

            try {
                auto t0 = std::chrono::high_resolution_clock::now();

                switch (algo) {
                case Algorithm::Counting:
                    if (impl == Impl::Seq) counting_sort_uint64_seq(data);
                    else                   counting_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Radix:
                    if (impl == Impl::Seq) radix_sort_uint64_seq(data);
                    else                   radix_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Bucket:
                    if (impl == Impl::Seq) bucket_sort_uint64_seq(data);
                    else                   bucket_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Pigeonhole:
                    if (impl == Impl::Seq) pigeonhole_sort_uint64_seq(data);
                    else                   pigeonhole_sort_uint64_omp(data, threads);
                    break;
                case Algorithm::Bitonic:
                    if (impl == Impl::Seq) bitonic_sort_uint64_seq(data);
                    else                   bitonic_sort_uint64_omp(data, threads);
                    break;
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                ms_run = std::chrono::duration<double, std::milli>(t1 - t0).count();

                sorted_run = is_sorted_uint64(data);
                if (check) {
                    match_run = (data == ref);
                } else {
                    match_run = sorted_run;
                }

            } catch (const std::exception& e) {
                error_msg = e.what();
                // Continue to next run; if all runs fail we'll handle later.
                continue;
            }

            if (!any_success || ms_run < ms_best) {
                any_success = true;
                ms_best     = ms_run;
                best_sorted = sorted_run;
                best_match  = match_run;
            }
        }

        if (!any_success) {
            if (error_msg == "none") {
                error_msg = "all_runs_failed";
            }
            output_metrics_and_csv(dist, csv_path,
                                   algo_str, impl_str, "uint64",
                                   N, threads,
                                   0.0,
                                   false,
                                   check,
                                   false,
                                   error_msg);
            return 1;
        }

        output_metrics_and_csv(dist, csv_path,
                               algo_str, impl_str, "uint64",
                               N, threads,
                               ms_best,
                               best_sorted,
                               check,
                               best_match,
                               error_msg);

        return (error_msg == "none") ? 0 : 1;
    }

    // ============================================================
    // SIGNED int64_t BRANCH
    // ============================================================
    else {
        std::vector<std::int64_t> original;

        try {
            original = load_int64_bin(input_path, N);
        } catch (const std::exception& e) {
            error_msg = std::string("load_failed: ") + e.what();
            output_metrics_and_csv(dist, csv_path,
                                   algo_str, impl_str, "int64",
                                   N, threads,
                                   0.0,
                                   false,
                                   check,
                                   false,
                                   error_msg);
            return 1;
        }

        if (algo != Algorithm::Radix) {
            error_msg = "Signed int64 supported only for algo=radix in this harness.";
            output_metrics_and_csv(dist, csv_path,
                                   algo_str, impl_str, "int64",
                                   N, threads,
                                   0.0,
                                   false,
                                   check,
                                   false,
                                   error_msg);
            return 1;
        }

        std::vector<std::int64_t> ref;
        if (check) {
            ref = original;
            std::sort(ref.begin(), ref.end());
        }

        // Warmup runs
        for (int w = 0; w < warmup; ++w) {
            std::vector<std::int64_t> data = original;
            try {
                if (impl == Impl::Seq) {
                    sort_int64_via_uint64_radix_seq(data);
                } else {
                    sort_int64_via_uint64_radix_omp(data, threads);
                }
            } catch (const std::exception& e) {
                error_msg = e.what();
                output_metrics_and_csv(dist, csv_path,
                                       algo_str, impl_str, "int64",
                                       N, threads,
                                       0.0,
                                       false,
                                       check,
                                       false,
                                       error_msg);
                return 1;
            }
        }

        bool any_success = false;
        for (int r = 0; r < repeats; ++r) {
            std::vector<std::int64_t> data = original;

            double ms_run    = 0.0;
            bool   sorted_run = false;
            bool   match_run  = false;

            try {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (impl == Impl::Seq) {
                    sort_int64_via_uint64_radix_seq(data);
                } else {
                    sort_int64_via_uint64_radix_omp(data, threads);
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                ms_run = std::chrono::duration<double, std::milli>(t1 - t0).count();

                sorted_run = is_sorted_int64(data);
                if (check) {
                    match_run = (data == ref);
                } else {
                    match_run = sorted_run;
                }

            } catch (const std::exception& e) {
                error_msg = e.what();
                continue;
            }

            if (!any_success || ms_run < ms_best) {
                any_success = true;
                ms_best     = ms_run;
                best_sorted = sorted_run;
                best_match  = match_run;
            }
        }

        if (!any_success) {
            if (error_msg == "none") {
                error_msg = "all_runs_failed";
            }
            output_metrics_and_csv(dist, csv_path,
                                   algo_str, impl_str, "int64",
                                   N, threads,
                                   0.0,
                                   false,
                                   check,
                                   false,
                                   error_msg);
            return 1;
        }

        output_metrics_and_csv(dist, csv_path,
                               algo_str, impl_str, "int64",
                               N, threads,
                               ms_best,
                               best_sorted,
                               check,
                               best_match,
                               error_msg);

        return (error_msg == "none") ? 0 : 1;
    }
}
