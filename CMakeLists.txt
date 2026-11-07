cmake_minimum_required(VERSION 3.25)
project(vector_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(/W4 /arch:AVX2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -mavx2 -mfma -O3)
endif()

# ==============================================================================
# Sanitizers (GCC / Clang only; ASan and TSan are mutually exclusive)
# ==============================================================================
option(ENABLE_ASAN "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if (ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ENABLE_ASAN and ENABLE_TSAN cannot be enabled together")
endif()

if (ENABLE_ASAN OR ENABLE_TSAN)
    if (MSVC)
        message(FATAL_ERROR "Sanitizer options require GCC or Clang")
    endif()
    if (ENABLE_ASAN)
        set(NEXUS_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
    else()
        set(NEXUS_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
    endif()
    add_compile_options(${NEXUS_SANITIZER_FLAGS})
    add_link_options(${NEXUS_SANITIZER_FLAGS})
endif()

# ==============================================================================
# Engine Core Library
# ==============================================================================
add_library(engine_core
        src/core/vector_engine.cpp
        src/distance/metrics.cpp
        src/distance/simd_metrics.cpp
        src/index/flat_index.cpp
        src/index/hnsw_index.cpp
        src/index/hnsw_index_sq8.cpp
        src/index/quantized_flat_index.cpp
        src/quantization/scalar_quantizer.cpp
        src/storage/serializer.cpp
        src/storage/wal_manager.cpp
)

target_include_directories(engine_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(vector_engine_app examples/main.cpp)
target_link_libraries(vector_engine_app PRIVATE engine_core)

add_executable(bench_sift benchmarks/bench_sift.cpp)
target_link_libraries(bench_sift PRIVATE engine_core)

# ==============================================================================
# Benchmark suite (benchmarks/README.md)
# Instrumentation toggles are compile-time only and default OFF; with all of them OFF the
# engine is compiled exactly as without them. Use a separate build directory per variant.
# ==============================================================================
option(NEXUS_COUNT_DISTANCES "Count HNSW distance evaluations (benchmark instrumentation)" OFF)
option(NEXUS_DISABLE_PREFETCH "Disable software prefetch in HNSW search (ablation)" OFF)
option(NEXUS_WAL_NO_FSYNC "Skip the per-record WAL fsync (write-path benchmark only, NOT durable)" OFF)
option(NEXUS_BENCH_HNSWLIB "Fetch hnswlib and build the hnswlib comparison harness" OFF)
option(NEXUS_BUILD_TESTS "Build the GoogleTest suite" ON)

foreach(_toggle NEXUS_COUNT_DISTANCES NEXUS_DISABLE_PREFETCH NEXUS_WAL_NO_FSYNC)
    if (${_toggle})
        target_compile_definitions(engine_core PUBLIC ${_toggle})
        message(STATUS "Benchmark toggle enabled: ${_toggle}")
    endif()
endforeach()

add_executable(nexus_bench benchmarks/nexus_bench.cpp)
target_link_libraries(nexus_bench PRIVATE engine_core)
if (WIN32)
    target_link_libraries(nexus_bench PRIVATE psapi)
endif()

if (NEXUS_BENCH_HNSWLIB)
    include(FetchContent)
    FetchContent_Declare(
            hnswlib
            URL https://github.com/nmslib/hnswlib/archive/refs/tags/v0.8.0.tar.gz
            # Header-only: download only, do not add hnswlib's own CMake project.
            SOURCE_SUBDIR do-not-add-subdirectory
    )
    FetchContent_MakeAvailable(hnswlib)
    add_executable(hnswlib_bench benchmarks/hnswlib_bench.cpp)
    target_include_directories(hnswlib_bench SYSTEM PRIVATE ${hnswlib_SOURCE_DIR})
    if (NEXUS_COUNT_DISTANCES)
        target_compile_definitions(hnswlib_bench PRIVATE NEXUS_COUNT_DISTANCES)
    endif()
    if (WIN32)
        target_link_libraries(hnswlib_bench PRIVATE psapi)
    endif()
endif()

# ==============================================================================
# Standalone Vector Database Network Service (nexus_server)
# ==============================================================================
set(SERVER_SOURCES
        src/server/ConceptRegistry.cpp
        src/server/HttpRequestParser.cpp
        src/server/HttpResponse.cpp
        src/server/HttpServer.cpp
        src/server/Router.cpp
        src/server/StaticFileHandler.cpp
        src/server/WorkerThreadPool.cpp
        src/server/main.cpp
)

add_executable(nexus_server ${SERVER_SOURCES})

target_include_directories(nexus_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nexus_server PRIVATE engine_core)

if (WIN32)
    target_link_libraries(nexus_server PRIVATE ws2_32)
endif()

if (NEXUS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()