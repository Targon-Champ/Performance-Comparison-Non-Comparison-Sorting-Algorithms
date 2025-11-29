#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

enum class KeyDtype {
    Int64,
    UInt64
};

enum class Algorithm {
    Radix,
    Counting,
    Bucket,
    Pigeonhole,
    Bitonic
};

enum class Impl {
    Seq,
    Omp
};

// String → enum mapping
Algorithm algorithm_from_string(const std::string& s);
Impl      impl_from_string(const std::string& s);
KeyDtype  dtype_from_string(const std::string& s);

// Generic I/O helpers for binary .bin files of raw 64-bit ints
std::vector<std::uint64_t> load_uint64_bin(const std::string& path, std::size_t n);
std::vector<std::int64_t>  load_int64_bin (const std::string& path, std::size_t n);

// Sortedness checks
bool is_sorted_uint64(const std::vector<std::uint64_t>& v);
bool is_sorted_int64 (const std::vector<std::int64_t>& v);
