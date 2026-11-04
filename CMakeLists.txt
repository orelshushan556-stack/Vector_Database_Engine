cmake_minimum_required(VERSION 3.25)
project(vector_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(/W4 /arch:AVX2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -mavx2 -mfma -O3)
endif()

# ==============================================================================
# Core Vector Engine Library
# ==============================================================================
add_library(engine_core
        src/distance/metrics.cpp
        src/distance/simd_metrics.cpp
        src/index/flat_index.cpp
        src/index/hnsw_index.cpp
        src/index/hnsw_index_sq8.cpp
        src/index/quantized_flat_index.cpp
        src/quantization/scalar_quantizer.cpp
        src/storage/serializer.cpp
)

target_include_directories(engine_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(vector_engine_app examples/main.cpp)
target_link_libraries(vector_engine_app PRIVATE engine_core)

# ==============================================================================
# Testing Framework (GoogleTest)
# ==============================================================================
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_distance)

add_executable(test_flat_index tests/test_flat_index.cpp)
target_link_libraries(test_flat_index PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_flat_index)

add_executable(test_quantization tests/test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_quantization)

add_executable(test_quantized_flat_index tests/test_quantized_flat_index.cpp)
target_link_libraries(test_quantized_flat_index PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_quantized_flat_index)

add_executable(test_serialization tests/test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_serialization)

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_hnsw)

add_executable(test_hnsw_sq8 tests/test_hnsw_sq8.cpp)
target_link_libraries(test_hnsw_sq8 PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_hnsw_sq8)

# ==============================================================================
# Academic SIFT-10K Benchmark Target
# ==============================================================================
add_executable(bench_sift benchmarks/bench_sift.cpp)
target_link_libraries(bench_sift PRIVATE engine_core)

# ==============================================================================
# Standalone Vector Database Network Service (nexus_server)
# ==============================================================================
file(GLOB SERVER_SOURCES "src/server/*.cpp")

add_executable(nexus_server ${SERVER_SOURCES})

target_include_directories(nexus_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nexus_server PRIVATE engine_core)

if (WIN32)
    target_link_libraries(nexus_server PRIVATE ws2_32)
endif()
# Target: test_wal
add_executable(test_wal tests/test_wal.cpp src/storage/wal_manager.cpp)
target_include_directories(test_wal PRIVATE include)
target_link_libraries(test_wal PRIVATE GTest::gtest_main)
add_test(NAME test_wal COMMAND test_wal)
