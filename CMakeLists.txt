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

add_library(engine_core
        src/distance/metrics.cpp
        src/distance/simd_metrics.cpp
        src/index/flat_index.cpp
        src/index/hnsw_index.cpp
        src/quantization/scalar_quantizer.cpp
        src/storage/serializer.cpp
)

target_include_directories(engine_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(vector_engine_app examples/main.cpp)
target_link_libraries(vector_engine_app PRIVATE engine_core)