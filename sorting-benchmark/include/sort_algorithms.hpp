#pragma once

#include <cstdint>
#include <vector>

// UNSIGNED uint64_t variants
void counting_sort_uint64_seq(std::vector<std::uint64_t>& a);
void counting_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads);

void radix_sort_uint64_seq(std::vector<std::uint64_t>& a);
void radix_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads);

void bucket_sort_uint64_seq(std::vector<std::uint64_t>& a);
void bucket_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads);

void pigeonhole_sort_uint64_seq(std::vector<std::uint64_t>& a);
void pigeonhole_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads);

void bitonic_sort_uint64_seq(std::vector<std::uint64_t>& a);
void bitonic_sort_uint64_omp(std::vector<std::uint64_t>& a, int num_threads);

// SIGNED: implemented by mapping to uint64, then re-mapping (to be filled later)
void sort_int64_via_uint64_radix_seq(std::vector<std::int64_t>& a);
void sort_int64_via_uint64_radix_omp(std::vector<std::int64_t>& a, int num_threads);
