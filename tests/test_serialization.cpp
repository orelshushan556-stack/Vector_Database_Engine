#include <gtest/gtest.h>
#include <sstream>
#include <filesystem>
#include <vector>
#include "engine/storage/serializer.hpp"
#include "engine/index/flat_index.hpp"

using namespace engine;

class SerializerTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir = std::filesystem::temp_directory_path() / "serializer_tests";
        std::filesystem::create_directories(temp_dir);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
    }

    std::filesystem::path temp_dir;
};

TEST_F(SerializerTest, RoundTripInMemoryExactMatch) {
    FlatIndex original(3, MetricType::L2);
    original.add(101, {1.0f, 2.0f, 3.0f});
    original.add(102, {4.0f, 5.0f, 6.0f});

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    Serializer::serialize(original, stream);

    FlatIndex loaded = Serializer::deserialize(stream);

    EXPECT_EQ(loaded.dimension(), original.dimension());
    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.metric(), original.metric());

    auto orig_ids = original.raw_ids();
    auto loaded_ids = loaded.raw_ids();
    ASSERT_EQ(orig_ids.size(), loaded_ids.size());
    for (size_t i = 0; i < orig_ids.size(); ++i) {
        EXPECT_EQ(orig_ids[i], loaded_ids[i]);
    }

    auto orig_storage = original.raw_storage();
    auto loaded_storage = loaded.raw_storage();
    ASSERT_EQ(orig_storage.size(), loaded_storage.size());
    for (size_t i = 0; i < orig_storage.size(); ++i) {
        EXPECT_FLOAT_EQ(orig_storage[i], loaded_storage[i]);
    }

    std::vector<float> query = {1.0f, 2.0f, 3.1f};
    auto orig_res = original.search(query, 1);
    auto loaded_res = loaded.search(query, 1);
    ASSERT_EQ(orig_res.size(), 1);
    ASSERT_EQ(loaded_res.size(), 1);
    EXPECT_EQ(orig_res[0].id, loaded_res[0].id);
    EXPECT_FLOAT_EQ(orig_res[0].distance, loaded_res[0].distance);
}

TEST_F(SerializerTest, FileSystemSaveAndLoad) {
    FlatIndex original(4, MetricType::Cosine);
    original.add(1, {0.1f, 0.2f, 0.3f, 0.4f});
    original.add(2, {0.5f, 0.6f, 0.7f, 0.8f});

    std::filesystem::path file_path = temp_dir / "test_index.bin";

    Serializer::save_flat_index(original, file_path);
    EXPECT_TRUE(std::filesystem::exists(file_path));

    FlatIndex loaded = Serializer::load_flat_index(file_path);

    EXPECT_EQ(loaded.dimension(), 4);
    EXPECT_EQ(loaded.size(), 2);
    EXPECT_EQ(loaded.metric(), MetricType::Cosine);
}

TEST_F(SerializerTest, EmptyIndexHandling) {
    FlatIndex original(128, MetricType::L2);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    Serializer::serialize(original, stream);

    FlatIndex loaded = Serializer::deserialize(stream);

    EXPECT_EQ(loaded.dimension(), 128);
    EXPECT_EQ(loaded.size(), 0);
    EXPECT_EQ(loaded.metric(), MetricType::L2);
    EXPECT_TRUE(loaded.raw_ids().empty());
    EXPECT_TRUE(loaded.raw_storage().empty());
}

TEST_F(SerializerTest, RejectsInvalidMagicNumber) {
    FileHeader bad_header;
    bad_header.magic = 0x11223344;
    bad_header.version = 1;
    bad_header.metric = 0;
    bad_header.dimension = 4;
    bad_header.count = 0;

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&bad_header), sizeof(bad_header));

    EXPECT_THROW(Serializer::deserialize(stream), std::runtime_error);
}

TEST_F(SerializerTest, RejectsUnsupportedVersion) {
    FileHeader bad_header;
    bad_header.magic = 0x56454349;
    bad_header.version = 2;
    bad_header.metric = 0;
    bad_header.dimension = 4;
    bad_header.count = 0;

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&bad_header), sizeof(bad_header));

    EXPECT_THROW(Serializer::deserialize(stream), std::runtime_error);
}

TEST_F(SerializerTest, RejectsInvalidMetricType) {
    FileHeader bad_header;
    bad_header.magic = 0x56454349;
    bad_header.version = 1;
    bad_header.metric = 999;
    bad_header.dimension = 4;
    bad_header.count = 0;

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&bad_header), sizeof(bad_header));

    EXPECT_THROW(Serializer::deserialize(stream), std::runtime_error);
}

TEST_F(SerializerTest, RejectsTruncatedStream) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    char partial_data[8] = {0};
    stream.write(partial_data, sizeof(partial_data));

    EXPECT_THROW(Serializer::deserialize(stream), std::runtime_error);
}

TEST_F(SerializerTest, RejectsTruncatedPayload) {
    FileHeader header;
    header.magic = 0x56454349;
    header.version = 1;
    header.metric = 0;
    header.dimension = 2;
    header.count = 10;

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

    VectorId single_id = 1;
    stream.write(reinterpret_cast<const char*>(&single_id), sizeof(single_id));

    EXPECT_THROW(Serializer::deserialize(stream), std::runtime_error);
}

TEST_F(SerializerTest, NonExistentFileThrowsException) {
    std::filesystem::path non_existent = temp_dir / "does_not_exist.bin";
    EXPECT_THROW(Serializer::load_flat_index(non_existent), std::runtime_error);
}