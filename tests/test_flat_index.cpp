#include <gtest/gtest.h>
#include "engine/index/flat_index.hpp"
#include <vector>
#include <stdexcept>
#include <tuple>

namespace engine {
namespace {

TEST(FlatIndexTest, ThrowsOnDimensionMismatchInAdd) {
    FlatIndex index(3, MetricType::L2);
    std::vector<float> vec = {1.0f, 2.0f};

    EXPECT_THROW(index.add(1, VectorView(vec.data(), vec.size())), std::invalid_argument);
}

TEST(FlatIndexTest, ThrowsOnDimensionMismatchInSearch) {
    FlatIndex index(3, MetricType::L2);
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    index.add(1, VectorView(vec.data(), vec.size()));

    std::vector<float> query = {1.0f, 2.0f};
    EXPECT_THROW({ std::ignore = index.search(VectorView(query.data(), query.size()), 1); }, std::invalid_argument);
}

TEST(FlatIndexTest, HandlesEmptyIndexAndZeroK) {
    FlatIndex index(3, MetricType::L2);
    std::vector<float> query = {1.0f, 2.0f, 3.0f};

    auto results_empty = index.search(VectorView(query.data(), query.size()), 5);
    EXPECT_TRUE(results_empty.empty());

    index.add(1, VectorView(query.data(), query.size()));
    auto results_zero_k = index.search(VectorView(query.data(), query.size()), 0);
    EXPECT_TRUE(results_zero_k.empty());
}

TEST(FlatIndexTest, PerformsL2SearchAndMaintainsOrder) {
    FlatIndex index(2, MetricType::L2);

    std::vector<float> v1 = {1.0f, 0.0f};
    std::vector<float> v2 = {2.0f, 0.0f};
    std::vector<float> v3 = {5.0f, 0.0f};

    index.add(101, VectorView(v1.data(), v1.size()));
    index.add(102, VectorView(v2.data(), v2.size()));
    index.add(103, VectorView(v3.data(), v3.size()));

    std::vector<float> query = {1.0f, 0.0f};
    auto results = index.search(VectorView(query.data(), query.size()), 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 101);
    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);

    EXPECT_EQ(results[1].id, 102);
    EXPECT_FLOAT_EQ(results[1].distance, 1.0f);
}

TEST(FlatIndexTest, PerformsDotProductSearchAndSignCorrection) {
    FlatIndex index(2, MetricType::DotProduct);

    std::vector<float> v1 = {1.0f, 1.0f};
    std::vector<float> v2 = {3.0f, 3.0f};
    std::vector<float> v3 = {2.0f, 2.0f};

    index.add(1, VectorView(v1.data(), v1.size()));
    index.add(2, VectorView(v2.data(), v2.size()));
    index.add(3, VectorView(v3.data(), v3.size()));

    std::vector<float> query = {1.0f, 1.0f};
    auto results = index.search(VectorView(query.data(), query.size()), 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 2);
    EXPECT_FLOAT_EQ(results[0].distance, 6.0f);

    EXPECT_EQ(results[1].id, 3);
    EXPECT_FLOAT_EQ(results[1].distance, 4.0f);

    EXPECT_EQ(results[2].id, 1);
    EXPECT_FLOAT_EQ(results[2].distance, 2.0f);
}

TEST(FlatIndexTest, HandlesKGreaterThanSize) {
    FlatIndex index(2, MetricType::L2);

    std::vector<float> v1 = {0.0f, 0.0f};
    index.add(1, VectorView(v1.data(), v1.size()));

    std::vector<float> query = {0.0f, 0.0f};
    auto results = index.search(VectorView(query.data(), query.size()), 10);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 1);
}

TEST(FlatIndexTest, ClearsIndexState) {
    FlatIndex index(2, MetricType::L2);

    std::vector<float> v1 = {1.0f, 1.0f};
    index.add(1, VectorView(v1.data(), v1.size()));
    EXPECT_EQ(index.size(), 1);

    index.clear();
    EXPECT_EQ(index.size(), 0);

    std::vector<float> query = {1.0f, 1.0f};
    auto results = index.search(VectorView(query.data(), query.size()), 1);
    EXPECT_TRUE(results.empty());
}

} // namespace
} // namespace engine