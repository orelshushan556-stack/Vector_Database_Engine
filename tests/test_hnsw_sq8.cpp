#include <gtest/gtest.h>
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace engine {
namespace {

std::vector<float> make_random_vec(std::size_t dim, std::mt19937& rng, float min_v = -2.0f, float max_v = 2.0f) {
    std::uniform_real_distribution<float> dist(min_v, max_v);
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

ScalarQuantizer build_trained_quantizer(std::size_t dim, std::size_t count, std::mt19937& rng) {
    ScalarQuantizer q(dim);
    std::vector<float> train_data;
    train_data.reserve(dim * count);
    for (std::size_t i = 0; i < count; ++i) {
        auto vec = make_random_vec(dim, rng);
        train_data.insert(train_data.end(), vec.begin(), vec.end());
    }
    q.train(train_data);
    return q;
}

TEST(HNSWIndexSQ8Test, GuardClausesAndValidation) {
    ScalarQuantizer untrained(4);
    EXPECT_THROW(HNSWIndexSQ8(4, MetricType::L2, untrained), std::runtime_error);

    std::mt19937 rng(11);
    ScalarQuantizer trained_dim4 = build_trained_quantizer(4, 50, rng);
    EXPECT_THROW(HNSWIndexSQ8(8, MetricType::L2, trained_dim4), std::invalid_argument);

    HNSWIndexSQ8 index(4, MetricType::L2, trained_dim4);
    std::vector<float> q = {0.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_TRUE(index.empty());
    auto empty_res = index.search(VectorView(q.data(), q.size()), 5);
    EXPECT_TRUE(empty_res.empty());

    std::vector<float> bad_dim = {1.0f, 2.0f};
    EXPECT_THROW(index.add(1, VectorView(bad_dim.data(), bad_dim.size())), std::invalid_argument);
}

TEST(HNSWIndexSQ8Test, SingleVectorExactSearch) {
    const std::size_t dim = 4;
    std::mt19937 rng(22);
    ScalarQuantizer q = build_trained_quantizer(dim, 100, rng);

    HNSWIndexSQ8 index(dim, MetricType::L2, q, 16, 200, 50, 2.0f);
    std::vector<float> v = {0.5f, 1.2f, -0.8f, 0.1f};

    index.add(10, VectorView(v.data(), v.size()));
    EXPECT_EQ(index.size(), 1);
    EXPECT_FALSE(index.empty());

    auto results = index.search(VectorView(v.data(), v.size()), 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 10);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-5f);
}

TEST(HNSWIndexSQ8Test, QuantizedRecallAgainstFlatWithReRanking) {
    const std::size_t dim = 16;
    const std::size_t num_vectors = 300;
    const std::size_t num_queries = 25;
    const std::size_t k = 10;

    std::mt19937 rng(999);
    ScalarQuantizer quantizer = build_trained_quantizer(dim, num_vectors, rng);

    FlatIndex flat(dim, MetricType::L2);
    HNSWIndexSQ8 sq8_index(dim, MetricType::L2, quantizer, 16, 200, 50, 2.5f);

    rng.seed(1234);
    for (std::size_t i = 0; i < num_vectors; ++i) {
        auto vec = make_random_vec(dim, rng);
        flat.add(static_cast<VectorId>(i), VectorView(vec.data(), vec.size()));
        sq8_index.add(static_cast<VectorId>(i), VectorView(vec.data(), vec.size()));
    }

    double total_recall = 0.0;
    for (std::size_t q = 0; q < num_queries; ++q) {
        auto query = make_random_vec(dim, rng);
        auto gt = flat.search(VectorView(query.data(), query.size()), k);
        auto sq8_res = sq8_index.search(VectorView(query.data(), query.size()), k);

        std::unordered_set<VectorId> gt_set;
        for (const auto& r : gt) {
            gt_set.insert(r.id);
        }

        std::size_t matches = 0;
        for (const auto& r : sq8_res) {
            if (gt_set.contains(r.id)) {
                ++matches;
            }
        }
        total_recall += static_cast<double>(matches) / static_cast<double>(k);
    }

    double avg_recall = total_recall / static_cast<double>(num_queries);
    EXPECT_GE(avg_recall, 0.85);
}

TEST(HNSWIndexSQ8Test, ReRankFactorScaling) {
    const std::size_t dim = 8;
    std::mt19937 rng(456);
    ScalarQuantizer quantizer = build_trained_quantizer(dim, 100, rng);

    HNSWIndexSQ8 sq8_index(dim, MetricType::L2, quantizer, 16, 100, 30, 1.0f);
    for (std::size_t i = 0; i < 50; ++i) {
        auto v = make_random_vec(dim, rng);
        sq8_index.add(static_cast<VectorId>(i), VectorView(v.data(), v.size()));
    }

    std::vector<float> q = make_random_vec(dim, rng);

    sq8_index.set_re_rank_factor(1.0f);
    auto res_factor1 = sq8_index.search(VectorView(q.data(), q.size()), 5);
    EXPECT_EQ(res_factor1.size(), 5);

    sq8_index.set_re_rank_factor(4.0f);
    auto res_factor4 = sq8_index.search(VectorView(q.data(), q.size()), 5);
    EXPECT_EQ(res_factor4.size(), 5);
}

TEST(HNSWIndexSQ8Test, ClampingAndExtremeValues) {
    const std::size_t dim = 4;
    std::mt19937 rng(333);
    ScalarQuantizer quantizer = build_trained_quantizer(dim, 50, rng);

    HNSWIndexSQ8 sq8_index(dim, MetricType::L2, quantizer, 16, 100, 30);
    std::vector<float> normal_vec = {0.1f, -0.2f, 0.5f, -0.1f};
    sq8_index.add(1, VectorView(normal_vec.data(), normal_vec.size()));

    std::vector<float> extreme_query = {50.0f, -100.0f, 25.0f, -80.0f};
    auto results = sq8_index.search(VectorView(extreme_query.data(), extreme_query.size()), 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(std::isnan(results[0].distance));
    EXPECT_FALSE(std::isinf(results[0].distance));
    EXPECT_GT(results[0].distance, 0.0f);
}

TEST(HNSWIndexSQ8Test, PersistenceAndUniversalLoader) {
    const std::size_t dim = 8;
    std::mt19937 rng(555);
    ScalarQuantizer quantizer = build_trained_quantizer(dim, 100, rng);

    HNSWIndexSQ8 original(dim, MetricType::L2, quantizer, 16, 100, 50, 2.0f);
    for (std::size_t i = 0; i < 40; ++i) {
        auto v = make_random_vec(dim, rng);
        original.add(static_cast<VectorId>(i + 500), VectorView(v.data(), v.size()));
    }

    std::vector<float> query = make_random_vec(dim, rng);
    auto baseline = original.search(VectorView(query.data(), query.size()), 5);

    const std::filesystem::path file_path = "temp_hnsw_sq8_test.bin";
    Serializer::save_hnsw_sq8_index(original, file_path);
    ASSERT_TRUE(std::filesystem::exists(file_path));

    HNSWIndexSQ8 loaded = Serializer::load_hnsw_sq8_index(file_path);
    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.dimension(), original.dimension());
    EXPECT_EQ(loaded.metric(), original.metric());
    EXPECT_FLOAT_EQ(loaded.re_rank_factor(), original.re_rank_factor());

    auto loaded_res = loaded.search(VectorView(query.data(), query.size()), 5);
    ASSERT_EQ(loaded_res.size(), baseline.size());
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(loaded_res[i].id, baseline[i].id);
        EXPECT_FLOAT_EQ(loaded_res[i].distance, baseline[i].distance);
    }

    auto universal_loaded = Serializer::load(file_path);
    ASSERT_NE(universal_loaded, nullptr);
    EXPECT_EQ(universal_loaded->size(), original.size());

    auto universal_res = universal_loaded->search(VectorView(query.data(), query.size()), 5);
    ASSERT_EQ(universal_res.size(), baseline.size());
    EXPECT_EQ(universal_res[0].id, baseline[0].id);

    std::filesystem::remove(file_path);
}

TEST(HNSWIndexSQ8Test, ConcurrentSearchStress) {
    const std::size_t dim = 16;
    std::mt19937 rng(888);
    ScalarQuantizer quantizer = build_trained_quantizer(dim, 100, rng);

    HNSWIndexSQ8 index(dim, MetricType::L2, quantizer, 16, 100, 50);
    for (std::size_t i = 0; i < 80; ++i) {
        auto v = make_random_vec(dim, rng);
        index.add(static_cast<VectorId>(i), VectorView(v.data(), v.size()));
    }

    auto q = make_random_vec(dim, rng);
    auto baseline = index.search(VectorView(q.data(), q.size()), 5);

    const int num_threads = 8;
    const int queries_per_thread = 50;
    std::vector<std::future<bool>> futures;

    for (int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&]() {
            for (int i = 0; i < queries_per_thread; ++i) {
                auto res = index.search(VectorView(q.data(), q.size()), 5);
                if (res.size() != baseline.size()) return false;
                for (std::size_t j = 0; j < baseline.size(); ++j) {
                    if (res[j].id != baseline[j].id || std::fabs(res[j].distance - baseline[j].distance) > 1e-5f) {
                        return false;
                    }
                }
            }
            return true;
        }));
    }

    for (auto& f : futures) {
        EXPECT_TRUE(f.get());
    }
}

} // namespace
} // namespace engine