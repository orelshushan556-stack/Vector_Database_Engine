#include <gtest/gtest.h>
#include "engine/storage/wal_manager.hpp"
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <span>
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

TEST_F(WalManagerBasicTest, ReplayTruncatesTornTail) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    {
        engine::storage::WalManager wal(test_wal_path);
        wal.append_insert(1, vec);
        wal.append_insert(2, vec);
    }
    const auto full_size = std::filesystem::file_size(test_wal_path);
    std::filesystem::resize_file(test_wal_path, full_size - 5);

    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<uint64_t> ids;
        EXPECT_NO_THROW(wal.replay([&](uint64_t id, std::span<const float>) { ids.push_back(id); }));
        ASSERT_EQ(ids.size(), 1u);
        EXPECT_EQ(ids[0], 1u);

        wal.append_insert(3, vec);
    }
    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<uint64_t> ids;
        wal.replay([&](uint64_t id, std::span<const float>) { ids.push_back(id); });
        ASSERT_EQ(ids.size(), 2u);
        EXPECT_EQ(ids[1], 3u);
    }
}

TEST_F(WalManagerBasicTest, CrashMidWriteRecoversAllCommittedRecordsAndAcceptsNewAppends) {
    constexpr std::size_t kDim = 4;
    constexpr std::size_t kRecords = 10;
    constexpr std::size_t kRecordSize = 18 + kDim * sizeof(float);

    auto make_vec = [](uint64_t id) {
        return std::vector<float>{static_cast<float>(id), 1.0f, 2.0f, 3.0f};
    };

    {
        engine::storage::WalManager wal(test_wal_path);
        for (uint64_t id = 0; id < kRecords; ++id) {
            wal.append_insert(id, make_vec(id));
        }
    }
    ASSERT_EQ(fs::file_size(test_wal_path), kRecords * kRecordSize);

    // Simulate a crash in the middle of an append: the header and half of the
    // vector payload reach the disk, the rest and the end marker do not.
    {
        std::ofstream out(test_wal_path, std::ios::binary | std::ios::app);
        const uint8_t op = 0x01;
        const uint32_t crc = 0xDEADBEEF;
        const uint64_t id = 999;
        const uint32_t dim = static_cast<uint32_t>(kDim);
        const float partial[2] = {7.0f, 8.0f};
        out.write(reinterpret_cast<const char*>(&op), sizeof(op));
        out.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(partial), sizeof(partial));
    }
    ASSERT_GT(fs::file_size(test_wal_path), kRecords * kRecordSize);

    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<uint64_t> ids;
        ASSERT_NO_THROW(wal.replay([&](uint64_t id, std::span<const float> v) {
            ASSERT_EQ(v.size(), kDim);
            EXPECT_EQ(v[0], static_cast<float>(id));
            ids.push_back(id);
        }));

        ASSERT_EQ(ids.size(), kRecords);
        for (uint64_t i = 0; i < kRecords; ++i) {
            EXPECT_EQ(ids[i], i);
        }

        // The torn tail must be gone and the file must end on a record boundary.
        EXPECT_EQ(fs::file_size(test_wal_path), kRecords * kRecordSize);

        wal.append_insert(1000, make_vec(1000));
    }

    {
        engine::storage::WalManager wal(test_wal_path);
        std::vector<uint64_t> ids;
        ASSERT_NO_THROW(wal.replay([&](uint64_t id, std::span<const float>) { ids.push_back(id); }));
        ASSERT_EQ(ids.size(), kRecords + 1);
        EXPECT_EQ(ids.back(), 1000u);
        EXPECT_EQ(fs::file_size(test_wal_path), (kRecords + 1) * kRecordSize);
    }
}
