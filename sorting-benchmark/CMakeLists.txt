cmake_minimum_required(VERSION 3.16)

# Enable both C++ and CUDA languages
project(sorting_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75;90)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable extra warnings (optional but recommended)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP)

# ---- CPU benchmark ----
add_executable(cpu_sort_bench
    src/main_cpu_bench.cpp
    src/sort_common.cpp
    src/sort_algorithms.cpp
)

target_include_directories(cpu_sort_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(cpu_sort_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cpu_sort_bench PRIVATE USE_OPENMP)
endif()

# ---- GPU benchmark ----
add_executable(gpu_sort_bench
    cuda/main_gpu_bench.cu
    cuda/gpu_sort_common.cu
    cuda/radix_sort.cu
    cuda/bitonic_sort_gpu.cu
    cuda/bucket_sort_gpu.cu
    cuda/bucket_sort_radix_gpu.cu
)

set_target_properties(gpu_sort_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gpu_sort_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)



