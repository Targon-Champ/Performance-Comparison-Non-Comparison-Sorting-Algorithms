#include "sort_common.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

Algorithm algorithm_from_string(const std::string& s) {
    if (s == "radix")      return Algorithm::Radix;
    if (s == "counting")   return Algorithm::Counting;
    if (s == "bucket")     return Algorithm::Bucket;
    if (s == "pigeonhole") return Algorithm::Pigeonhole;
    if (s == "bitonic")    return Algorithm::Bitonic;
    throw std::invalid_argument("Unknown algorithm: " + s);
}

Impl impl_from_string(const std::string& s) {
    if (s == "seq") return Impl::Seq;
    if (s == "omp") return Impl::Omp;
    throw std::invalid_argument("Unknown implementation: " + s);
}

KeyDtype dtype_from_string(const std::string& s) {
    if (s == "int64")  return KeyDtype::Int64;
    if (s == "uint64") return KeyDtype::UInt64;
    throw std::invalid_argument("Unknown dtype: " + s);
}

template <typename T>
static std::vector<T> load_bin_typed(const std::string& path, std::size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));

    if (!f) {
        throw std::runtime_error(
            "Failed to read " + std::to_string(n) + " elements from " + path
        );
    }

    return v;
}

std::vector<std::uint64_t> load_uint64_bin(const std::string& path, std::size_t n) {
    return load_bin_typed<std::uint64_t>(path, n);
}

std::vector<std::int64_t> load_int64_bin(const std::string& path, std::size_t n) {
    return load_bin_typed<std::int64_t>(path, n);
}

bool is_sorted_uint64(const std::vector<std::uint64_t>& v) {
    if (v.size() < 2) return true;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] < v[i - 1]) return false;
    }
    return true;
}

bool is_sorted_int64(const std::vector<std::int64_t>& v) {
    if (v.size() < 2) return true;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] < v[i - 1]) return false;
    }
    return true;
}
