#include <gtest/gtest.h>
#include "engine/core/vector_engine.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/storage/serializer.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

class VectorEngineTest : public ::testing::Test {
protected:
    const std::string test_wal_path = "engine_test.wal";
    const std::string test_snapshot_path = "engine_test_snapshot.bin";

    void SetUp() override {
        cleanup();
    }

    void TearDown() override {
        cleanup();
    }

    void cleanup() {
        if (fs::exists(test_wal_path)) {
            fs::remove(test_wal_path);
        }
        if (fs::exists(test_snapshot_path)) {
            fs::remove(test_snapshot_path);
        }
    }
};

TEST_F(VectorEngineTest, InsertsAndSearchesSuccessfully) {
    auto index = std::make_unique<engine::FlatIndex>(3, engine::MetricType::L2);
    engine::VectorEngine engine(std::move(index), test_wal_path);

    std::vector<float> v1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> v3 = {0.0f, 0.0f, 1.0f};

    engine.insert(1, v1);
    engine.insert(2, v2);
    engine.insert(3, v3);

    EXPECT_EQ(engine.size(), 3);
    EXPECT_EQ(engine.dimension(), 3);
    EXPECT_EQ(engine.metric(), engine::MetricType::L2);

    std::vector<float> query = {0.9f, 0.1f, 0.0f};
    auto results = engine.search(query, 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 1);
}

TEST_F(VectorEngineTest, RecoversFromWalOnStartup) {
    std::vector<float> v1 = {1.0f, 2.0f};
    std::vector<float> v2 = {3.0f, 4.0f};

    {
        auto index = std::make_unique<engine::FlatIndex>(2, engine::MetricType::L2);
        engine::VectorEngine engine(std::move(index), test_wal_path);
        engine.insert(10, v1);
        engine.insert(20, v2);
        EXPECT_EQ(engine.size(), 2);
    }

    {
        auto fresh_index = std::make_unique<engine::FlatIndex>(2, engine::MetricType::L2);
        engine::VectorEngine recovered_engine(std::move(fresh_index), test_wal_path);

        EXPECT_EQ(recovered_engine.size(), 2);

        std::vector<float> query = {1.0f, 2.0f};
        auto results = recovered_engine.search(query, 1);

        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].id, 10);
        EXPECT_NEAR(results[0].distance, 0.0f, 1e-5f);
    }
}

TEST_F(VectorEngineTest, CreatesCheckpointAndTruncatesWal) {
    std::vector<float> v1 = {0.5f, 0.5f};
    std::vector<float> v2 = {1.5f, 1.5f};

    {
        auto index = std::make_unique<engine::FlatIndex>(2, engine::MetricType::L2);
        engine::VectorEngine engine(std::move(index), test_wal_path);
        engine.insert(100, v1);
        engine.insert(200, v2);

        EXPECT_GT(fs::file_size(test_wal_path), 0);

        engine.create_checkpoint(test_snapshot_path);

        EXPECT_TRUE(fs::exists(test_snapshot_path));
        EXPECT_EQ(fs::file_size(test_wal_path), 0);
    }

    {
        auto loaded_index = engine::Serializer::load(test_snapshot_path);
        ASSERT_NE(loaded_index, nullptr);
        EXPECT_EQ(loaded_index->size(), 2);
        EXPECT_EQ(loaded_index->dimension(), 2);
    }
}

TEST_F(VectorEngineTest, SupportsConcurrentReadsAndWrites) {
    auto index = std::make_unique<engine::FlatIndex>(4, engine::MetricType::L2);
    engine::VectorEngine engine(std::move(index), test_wal_path);

    std::vector<float> initial_vec = {0.1f, 0.2f, 0.3f, 0.4f};
    engine.insert(1, initial_vec);

    std::atomic<bool> running{true};
    std::atomic<size_t> search_count{0};

    auto reader = [&]() {
        std::vector<float> q = {0.1f, 0.2f, 0.3f, 0.4f};
        while (running.load(std::memory_order_relaxed)) {
            auto res = engine.search(q, 1);
            if (!res.empty()) {
                search_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::thread t1(reader);
    std::thread t2(reader);

    for (uint64_t i = 2; i <= 50; ++i) {
        std::vector<float> v = {
            static_cast<float>(i),
            static_cast<float>(i * 2),
            static_cast<float>(i * 3),
            static_cast<float>(i * 4)
        };
        engine.insert(i, v);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    running.store(false, std::memory_order_relaxed);
    t1.join();
    t2.join();

    EXPECT_EQ(engine.size(), 50);
    EXPECT_GT(search_count.load(), 0);
}
