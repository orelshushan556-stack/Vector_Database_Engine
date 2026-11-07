# Builds the identical fix_bench harness against an arbitrary engine source tree, so two git
# revisions can be compared with byte-identical benchmark code:
#   cmake -S benchmarks/fix_compare -B build-fix-before -DENGINE_ROOT=<worktree of old commit>
cmake_minimum_required(VERSION 3.25)
project(fix_compare LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT ENGINE_ROOT)
    message(FATAL_ERROR "set -DENGINE_ROOT=<path to an engine source tree>")
endif()
if (MSVC)
    add_compile_options(/arch:AVX2 /permissive-)
else()
    add_compile_options(-mavx2 -mfma -O3)
endif()
file(GLOB ENGINE_SOURCES
        ${ENGINE_ROOT}/src/distance/*.cpp
        ${ENGINE_ROOT}/src/index/*.cpp
        ${ENGINE_ROOT}/src/quantization/*.cpp
        ${ENGINE_ROOT}/src/storage/*.cpp)
add_library(engine_rev STATIC ${ENGINE_SOURCES})
target_include_directories(engine_rev PUBLIC ${ENGINE_ROOT}/include)
add_executable(fix_bench fix_bench.cpp)
target_include_directories(fix_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fix_bench PRIVATE engine_rev)
if (WIN32)
    target_link_libraries(fix_bench PRIVATE psapi)
endif()
