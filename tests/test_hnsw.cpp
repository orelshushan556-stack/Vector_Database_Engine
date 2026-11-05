#include <gtest/gtest.h>
#include "engine/index/index_base.hpp"
#include "engine/index/hnsw_index.hpp"
#include "engine/index/flat_index.hpp"
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

// Gaussian clusters with wide separation: the layout that exposed unreachable regions
// in graphs built with nearest-replacement pruning.
std::vector<float> make_clustered_dataset(std::size_t dim, std::size_t count, std::size_t clusters,
                                          std::mt19937& rng) {
    std::normal_distribution<float> center_dist(0.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (auto& c : centers) {
        for (auto& v : c) v = center_dist(rng);
    }
    std::vector<float> data(count * dim);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& c = centers[i % clusters];
        for (std::size_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + noise(rng);
    }
    return data;
}

double recall_at_k(const IndexBase& approx, const IndexBase& exact, const std::vector<float>& queries,
                   std::size_t dim, std::size_t k) {
    const std::size_t nq = queries.size() / dim;
    double total = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
        VectorView qv(queries.data() + q * dim, dim);
        std::unordered_set<VectorId> truth;
        for (const auto& r : exact.search(qv, k)) truth.insert(r.id);
        std::size_t hit = 0;
        for (const auto& r : approx.search(qv, k)) hit += truth.contains(r.id) ? 1 : 0;
        total += static_cast<double>(hit) / static_cast<double>(k);
    }
    return total / static_cast<double>(nq);
}

std::vector<float> make_random_vec(std::size_t dim, std::mt19937& rng, float min_v = -1.0f, float max_v = 1.0f) {
    std::uniform_real_distribution<float> dist(min_v, max_v);
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

TEST(HNSWIndexTest, EmptyIndexAndInvalidArguments) {
    HNSWIndex index(4, MetricType::L2);
    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f};

    auto res_empty = index.search(VectorView(q.data(), q.size()), 5);
    EXPECT_TRUE(res_empty.empty());

    std::vector<float> bad_dim_vec = {1.0f, 2.0f};
    EXPECT_THROW(index.add(1, VectorView(bad_dim_vec.data(), bad_dim_vec.size())), std::invalid_argument);

    index.add(1, VectorView(q.data(), q.size()));

    auto res_zero_k = index.search(VectorView(q.data(), q.size()), 0);
    EXPECT_TRUE(res_zero_k.empty());

    auto res_k_large = index.search(VectorView(q.data(), q.size()), 10);
    ASSERT_EQ(res_k_large.size(), 1);
    EXPECT_EQ(res_k_large[0].id, 1);
}

TEST(HNSWIndexTest, SingleNodeLifecycle) {
    HNSWIndex index(3, MetricType::L2);
    std::vector<float> v = {0.5f, -0.5f, 1.0f};

    index.add(42, VectorView(v.data(), v.size()));
    ASSERT_EQ(index.size(), 1);

    auto results = index.search(VectorView(v.data(), v.size()), 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, 42);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-5f);
}

TEST(HNSWIndexTest, HandlesDuplicateVectors) {
    HNSWIndex index(2, MetricType::L2, 16, 100, 50);
    std::vector<float> v = {2.0f, 3.0f};

    index.add(101, VectorView(v.data(), v.size()));
    index.add(102, VectorView(v.data(), v.size()));
    index.add(103, VectorView(v.data(), v.size()));

    EXPECT_EQ(index.size(), 3);

    auto results = index.search(VectorView(v.data(), v.size()), 3);
    ASSERT_EQ(results.size(), 3);
    for (const auto& r : results) {
        EXPECT_NEAR(r.distance, 0.0f, 1e-5f);
    }
}

TEST(HNSWIndexTest, MetricOrderingL2AndDotProduct) {
    HNSWIndex index_dp(2, MetricType::DotProduct, 16, 100, 50);
    std::vector<float> v1 = {1.0f, 1.0f};
    std::vector<float> v2 = {3.0f, 3.0f};
    std::vector<float> v3 = {2.0f, 2.0f};

    index_dp.add(1, VectorView(v1.data(), v1.size()));
    index_dp.add(2, VectorView(v2.data(), v2.size()));
    index_dp.add(3, VectorView(v3.data(), v3.size()));

    std::vector<float> query = {1.0f, 1.0f};
    auto results = index_dp.search(VectorView(query.data(), query.size()), 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 2);
    EXPECT_FLOAT_EQ(results[0].distance, -6.0f);
    EXPECT_EQ(results[1].id, 3);
    EXPECT_FLOAT_EQ(results[1].distance, -4.0f);
    EXPECT_EQ(results[2].id, 1);
    EXPECT_FLOAT_EQ(results[2].distance, -2.0f);
}

TEST(HNSWIndexTest, HighRecallAgainstFlatIndex) {
    const std::size_t dim = 16;
    const std::size_t num_vectors = 200;
    const std::size_t num_queries = 20;
    const std::size_t k = 10;

    std::mt19937 rng(42);
    FlatIndex flat(dim, MetricType::L2);
    HNSWIndex hnsw(dim, MetricType::L2, 16, 200, 100);

    for (std::size_t i = 0; i < num_vectors; ++i) {
        auto vec = make_random_vec(dim, rng);
        flat.add(static_cast<VectorId>(i), VectorView(vec.data(), vec.size()));
        hnsw.add(static_cast<VectorId>(i), VectorView(vec.data(), vec.size()));
    }

    double total_recall = 0.0;
    for (std::size_t q = 0; q < num_queries; ++q) {
        auto query = make_random_vec(dim, rng);
        auto gt = flat.search(VectorView(query.data(), query.size()), k);
        auto hnsw_res = hnsw.search(VectorView(query.data(), query.size()), k);

        std::unordered_set<VectorId> gt_set;
        for (const auto& r : gt) {
            gt_set.insert(r.id);
        }

        std::size_t matches = 0;
        for (const auto& r : hnsw_res) {
            if (gt_set.contains(r.id)) {
                ++matches;
            }
        }
        total_recall += static_cast<double>(matches) / static_cast<double>(k);
    }

    double avg_recall = total_recall / static_cast<double>(num_queries);
    EXPECT_GE(avg_recall, 0.90);
}

TEST(HNSWIndexTest, BinaryPersistenceAndUniversalLoader) {
    const std::size_t dim = 8;
    HNSWIndex original(dim, MetricType::L2, 16, 100, 50);

    std::mt19937 rng(123);
    for (std::size_t i = 0; i < 50; ++i) {
        auto v = make_random_vec(dim, rng);
        original.add(static_cast<VectorId>(i + 100), VectorView(v.data(), v.size()));
    }

    std::vector<float> query = make_random_vec(dim, rng);
    auto baseline = original.search(VectorView(query.data(), query.size()), 5);

    const std::filesystem::path file_path = "temp_hnsw_test.bin";
    Serializer::save_hnsw_index(original, file_path);
    ASSERT_TRUE(std::filesystem::exists(file_path));

    HNSWIndex loaded = Serializer::load_hnsw_index(file_path);
    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.dimension(), original.dimension());
    EXPECT_EQ(loaded.metric(), original.metric());

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

TEST(HNSWIndexTest, ConcurrentSearchStress) {
    const std::size_t dim = 16;
    HNSWIndex index(dim, MetricType::L2, 16, 100, 50);

    std::mt19937 rng(777);
    for (std::size_t i = 0; i < 100; ++i) {
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

TEST(HNSWIndexTest, ClearAndRebuild) {
    HNSWIndex index(2, MetricType::L2);
    std::vector<float> v = {1.0f, 1.0f};
    index.add(1, VectorView(v.data(), v.size()));
    EXPECT_EQ(index.size(), 1);

    index.clear();
    EXPECT_EQ(index.size(), 0);

    auto res = index.search(VectorView(v.data(), v.size()), 1);
    EXPECT_TRUE(res.empty());

    index.add(2, VectorView(v.data(), v.size()));
    EXPECT_EQ(index.size(), 1);
    auto res_after = index.search(VectorView(v.data(), v.size()), 1);
    ASSERT_EQ(res_after.size(), 1);
    EXPECT_EQ(res_after[0].id, 2);
}

TEST(HNSWIndexTest, RecallIsStableAcrossRandomBuilds) {
    constexpr std::size_t dim = 32, n = 4000, clusters = 20, k = 10;
    std::mt19937 data_rng(7);
    const auto base = make_clustered_dataset(dim, n + 100, clusters, data_rng);

    FlatIndex flat(dim, MetricType::L2);
    for (std::size_t i = 0; i < n; ++i) {
        flat.add(i, VectorView(base.data() + i * dim, dim));
    }
    const std::vector<float> queries(base.begin() + n * dim, base.end());

    // Each build draws its own random node levels; every one must reach high recall.
    for (int build = 0; build < 6; ++build) {
        HNSWIndex hnsw(dim, MetricType::L2, 16, 200, 64);
        for (std::size_t i = 0; i < n; ++i) {
            hnsw.add(i, VectorView(base.data() + i * dim, dim));
        }
        EXPECT_GE(recall_at_k(hnsw, flat, queries, dim, k), 0.95) << "build " << build;
    }
}

} // namespace
} // namespace engine