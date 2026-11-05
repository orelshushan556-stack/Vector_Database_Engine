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

enable_testing()
add_subdirectory(tests)