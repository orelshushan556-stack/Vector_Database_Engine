#include <gtest/gtest.h>
#include "engine/index/quantized_flat_index.hpp"
#include <vector>
#include <stdexcept>

namespace engine {
namespace {

TEST(QuantizedFlatIndexTest, ThrowsBeforeTraining) {
    QuantizedFlatIndex index(4);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_FALSE(index.is_trained());
    EXPECT_THROW(index.add(vec), std::runtime_error);
    EXPECT_THROW(index.add_batch(vec), std::runtime_error);
    EXPECT_THROW(std::ignore = index.search(vec, 1), std::runtime_error);
}

TEST(QuantizedFlatIndexTest, ThrowsOnDimensionMismatchInAdd) {
    QuantizedFlatIndex index(3);
    std::vector<float> train_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    index.train(train_data);

    std::vector<float> bad_vec = {1.0f, 2.0f};
    EXPECT_THROW(index.add(bad_vec), std::invalid_argument);
}

TEST(QuantizedFlatIndexTest, ThrowsOnDimensionMismatchInSearch) {
    QuantizedFlatIndex index(3);
    std::vector<float> train_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    index.train(train_data);

    std::vector<float> bad_query = {1.0f, 2.0f};
    EXPECT_THROW(std::ignore = index.search(bad_query, 2), std::invalid_argument);
}

TEST(QuantizedFlatIndexTest, HandlesEmptyIndexAndZeroK) {
    QuantizedFlatIndex index(3);
    std::vector<float> train_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    index.train(train_data);

    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    auto results_empty = index.search(query, 5);
    EXPECT_TRUE(results_empty.empty());

    index.add(query);
    auto results_zero_k = index.search(query, 0);
    EXPECT_TRUE(results_zero_k.empty());
}

TEST(QuantizedFlatIndexTest, AddBatchAndBoundsChecking) {
    QuantizedFlatIndex index(2);
    std::vector<float> train_data = {0.0f, 0.0f, 10.0f, 10.0f};
    index.train(train_data);

    std::vector<float> batch = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    index.add_batch(batch);

    EXPECT_EQ(index.size(), 3);
    EXPECT_NO_THROW(std::ignore = index.get_quantized_vector(0));
    EXPECT_NO_THROW(std::ignore = index.get_quantized_vector(2));
    EXPECT_THROW(std::ignore = index.get_quantized_vector(3), std::out_of_range);
}

TEST(QuantizedFlatIndexTest, PerformsSearchAndMaintainsOrder) {
    QuantizedFlatIndex index(2);
    std::vector<float> train_data = {
        0.0f, 0.0f,
        10.0f, 10.0f
    };
    index.train(train_data);

    std::vector<float> v0 = {0.0f, 0.0f};
    std::vector<float> v1 = {2.0f, 0.0f};
    std::vector<float> v2 = {8.0f, 0.0f};

    index.add(v0);
    index.add(v1);
    index.add(v2);

    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(query, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 0);
    EXPECT_EQ(results[1].id, 1);
    EXPECT_LT(results[0].distance, results[1].distance);
}

TEST(QuantizedFlatIndexTest, HandlesKGreaterThanSize) {
    QuantizedFlatIndex index(2);
    std::vector<float> train_data = {0.0f, 0.0f, 1.0f, 1.0f};
    index.train(train_data);

    std::vector<float> v0 = {0.5f, 0.5f};
    index.add(v0);

    std::vector<float> query = {0.5f, 0.5f};
    auto results = index.search(query, 10);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 0);
}

TEST(QuantizedFlatIndexTest, ClearsIndex) {
    QuantizedFlatIndex index(2);
    std::vector<float> train_data = {0.0f, 0.0f, 1.0f, 1.0f};
    index.train(train_data);

    std::vector<float> v0 = {0.5f, 0.5f};
    index.add(v0);
    EXPECT_EQ(index.size(), 1);

    index.clear();
    EXPECT_EQ(index.size(), 0);
    EXPECT_TRUE(index.empty());
}

}
}