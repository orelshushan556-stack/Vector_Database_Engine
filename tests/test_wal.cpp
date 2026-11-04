#include <gtest/gtest.h>
#include "engine/storage/wal_manager.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class WalManagerBasicTest : public ::testing::Test {
protected:
    const std::string test_wal_path = "test_run.wal";

    void SetUp() override {
        if (fs::exists(test_wal_path)) {
            fs::remove(test_wal_path);
        }
    }

    void TearDown() override {
        if (fs::exists(test_wal_path)) {
            fs::remove(test_wal_path);
        }
    }
};

TEST_F(WalManagerBasicTest, CreatesFileAndOpensSuccessfully) {
    {
        engine::storage::WalManager wal(test_wal_path);
        EXPECT_TRUE(wal.is_open());
    }
    EXPECT_TRUE(fs::exists(test_wal_path));
}

TEST_F(WalManagerBasicTest, SyncToDiskExecutesWithoutErrors) {
    engine::storage::WalManager wal(test_wal_path);
    EXPECT_TRUE(wal.is_open());
    EXPECT_NO_THROW(wal.sync());
}

TEST_F(WalManagerBasicTest, ThrowsOnInvalidPath) {
    EXPECT_THROW({
        engine::storage::WalManager wal("non_existing_dir_xyz/test.wal");
    }, std::runtime_error);
}

TEST_F(WalManagerBasicTest, AppendInsertWritesExpectedBytes) {
    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
        wal.append_insert(42, vec);
    }

    const size_t expected_size = 1 + 4 + 8 + 4 + (4 * sizeof(float)) + 1;
    EXPECT_EQ(fs::file_size(test_wal_path), expected_size);
}

TEST_F(WalManagerBasicTest, ReplayRestoresAppendedRecords) {
    std::vector<float> vec1 = {1.1f, 2.2f, 3.3f};
    std::vector<float> vec2 = {4.4f, 5.5f, 6.6f};

    {
        engine::storage::WalManager wal(test_wal_path);
        wal.append_insert(100, vec1);
        wal.append_insert(200, vec2);
    }

    std::vector<uint64_t> replayed_ids;
    std::vector<std::vector<float>> replayed_vectors;

    {
        engine::storage::WalManager wal(test_wal_path);
        wal.replay([&](uint64_t id, std::span<const float> vec) {
            replayed_ids.push_back(id);
            replayed_vectors.emplace_back(vec.begin(), vec.end());
        });
    }

    ASSERT_EQ(replayed_ids.size(), 2);
    EXPECT_EQ(replayed_ids[0], 100);
    EXPECT_EQ(replayed_ids[1], 200);

    ASSERT_EQ(replayed_vectors.size(), 2);
    EXPECT_EQ(replayed_vectors[0], vec1);
    EXPECT_EQ(replayed_vectors[1], vec2);
}

TEST_F(WalManagerBasicTest, ReplayHandlesEmptyFileGracefully) {
    engine::storage::WalManager wal(test_wal_path);
    size_t call_count = 0;
    wal.replay([&](uint64_t, std::span<const float>) {
        call_count++;
    });
    EXPECT_EQ(call_count, 0);
}

TEST_F(WalManagerBasicTest, ReplayDetectsDataCorruption) {
    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
        wal.append_insert(50, vec);
    }

    {
        std::fstream file(test_wal_path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(20);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0xFF;
        file.seekp(20);
        file.write(&byte, 1);
    }

    {
        engine::storage::WalManager wal(test_wal_path);
        EXPECT_THROW({
            wal.replay([](uint64_t, std::span<const float>) {});
        }, std::runtime_error);
    }
}
