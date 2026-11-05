include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_distance)

add_executable(test_flat_index test_flat_index.cpp)
target_link_libraries(test_flat_index PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_flat_index)

add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_quantization)

add_executable(test_quantized_flat_index test_quantized_flat_index.cpp)
target_link_libraries(test_quantized_flat_index PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_quantized_flat_index)

add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_serialization)

add_executable(test_hnsw test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_hnsw)

add_executable(test_hnsw_sq8 test_hnsw_sq8.cpp)
target_link_libraries(test_hnsw_sq8 PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_hnsw_sq8)

add_executable(test_wal test_wal.cpp)
target_link_libraries(test_wal PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_wal)

add_executable(test_crc32 test_crc32.cpp)
target_link_libraries(test_crc32 PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_crc32)

add_executable(test_vector_engine test_vector_engine.cpp)
target_link_libraries(test_vector_engine PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_vector_engine)

add_executable(test_hardening
        test_hardening.cpp
        ../src/server/HttpRequestParser.cpp
        ../src/server/ConceptRegistry.cpp)
target_include_directories(test_hardening PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/server)
target_link_libraries(test_hardening PRIVATE engine_core GTest::gtest_main)
gtest_discover_tests(test_hardening)

set(SERVER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/server)
add_executable(test_http_server
        test_http_server.cpp
        ${SERVER_SRC_DIR}/HttpServer.cpp
        ${SERVER_SRC_DIR}/HttpRequestParser.cpp
        ${SERVER_SRC_DIR}/HttpResponse.cpp
        ${SERVER_SRC_DIR}/Router.cpp
        ${SERVER_SRC_DIR}/StaticFileHandler.cpp
        ${SERVER_SRC_DIR}/WorkerThreadPool.cpp
        ${SERVER_SRC_DIR}/ConceptRegistry.cpp)
target_include_directories(test_http_server PRIVATE ${SERVER_SRC_DIR})
target_link_libraries(test_http_server PRIVATE engine_core GTest::gtest_main)
if (WIN32)
    target_link_libraries(test_http_server PRIVATE ws2_32)
endif()
gtest_discover_tests(test_http_server)
