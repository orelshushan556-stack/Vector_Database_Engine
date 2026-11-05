#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <memory>
#include <stdexcept>
#include <span>

#include "engine/types.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/index/quantized_flat_index.hpp"
#include "engine/index/hnsw_index.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"

using namespace engine;

namespace {

// Binary .fvecs reader (dimension + raw floats)
std::vector<float> load_fvecs(const std::string& path, std::size_t& count, std::size_t& dim) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int32_t d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (!file) return {};

    dim = static_cast<std::size_t>(d);
    file.seekg(0, std::ios::end);
    std::size_t file_size = file.tellg();
    std::size_t vector_bytes = sizeof(int32_t) + dim * sizeof(float);
    count = file_size / vector_bytes;
    file.seekg(0, std::ios::beg);

    std::vector<float> data(count * dim);
    for (std::size_t i = 0; i < count; ++i) {
        int32_t cur_dim = 0;
        file.read(reinterpret_cast<char*>(&cur_dim), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(data.data() + (i * dim)), dim * sizeof(float));
    }
    return data;
}

// Binary .ivecs reader (Ground Truth k-NN ids)
std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<std::vector<uint32_t>> gt;
    while (file.peek() != EOF) {
        int32_t k = 0;
        if (!file.read(reinterpret_cast<char*>(&k), sizeof(int32_t))) break;
        std::vector<int32_t> neighbors(k);
        file.read(reinterpret_cast<char*>(neighbors.data()), k * sizeof(int32_t));

        std::vector<uint32_t> u_neighbors(k);
        for (int i = 0; i < k; ++i) {
            u_neighbors[i] = static_cast<uint32_t>(neighbors[i]);
        }
        gt.push_back(std::move(u_neighbors));
    }
    return gt;
}

double compute_recall(const std::vector<uint32_t>& gt_knn,
                      const std::vector<SearchResult>& results,
                      std::size_t k) {
    if (gt_knn.empty() || results.empty() || k == 0) return 0.0;
    std::unordered_set<uint32_t> gt_set;
    for (std::size_t i = 0; i < std::min(k, gt_knn.size()); ++i) {
        gt_set.insert(gt_knn[i]);
    }
    std::size_t matches = 0;
    for (std::size_t i = 0; i < std::min(k, results.size()); ++i) {
        if (gt_set.contains(static_cast<uint32_t>(results[i].id))) {
            ++matches;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(k);
}

} // namespace

int main() {
    std::cout << "========================================================================\n";
    std::cout << " Official Academic Benchmark: SIFT-10K (INRIA Texmex)\n";
    std::cout << " Metric: L2 Distance | Vector Dimension: 128 | Top-K: 10\n";
    std::cout << "========================================================================\n\n";

    std::size_t base_count = 0, dim = 0, query_count = 0, qdim = 0;
    std::cout << "[*] Loading SIFT binary datasets from disk... " << std::flush;
    auto base_data = load_fvecs("data/siftsmall/siftsmall_base.fvecs", base_count, dim);
    auto query_data = load_fvecs("data/siftsmall/siftsmall_query.fvecs", query_count, qdim);
    auto ground_truth = load_ivecs("data/siftsmall/siftsmall_groundtruth.ivecs");
    std::cout << "Done.\n";
    std::cout << "    Base Vectors:  " << base_count << " (128-dim)\n";
    std::cout << "    Query Vectors: " << query_count << "\n";
    std::cout << "    Ground Truth:  " << ground_truth.size() << " precomputed queries\n\n";

    constexpr std::size_t K = 10;
    auto get_query = [&](std::size_t q) -> VectorView {
        return VectorView(query_data.data() + (q * dim), dim);
    };

    constexpr std::size_t REPEATS = 20;

    struct Measurement {
        double recall;
        double mean_us;
        double p50_us;
        double p99_us;
    };

    // Runs every query REPEATS times (after one warm-up pass), timing each query
    // individually. Recall is computed from the final pass.
    auto measure = [&](auto&& search_fn) {
        double recall = 0.0;
        for (std::size_t q = 0; q < query_count; ++q) {
            recall += compute_recall(ground_truth[q], search_fn(get_query(q)), K);
        }

        std::vector<double> samples;
        samples.reserve(query_count * REPEATS);
        for (std::size_t r = 0; r < REPEATS; ++r) {
            for (std::size_t q = 0; q < query_count; ++q) {
                const auto s = std::chrono::steady_clock::now();
                auto res = search_fn(get_query(q));
                const auto e = std::chrono::steady_clock::now();
                if (res.empty()) {
                    throw std::runtime_error("benchmark query returned no results");
                }
                samples.push_back(std::chrono::duration<double, std::micro>(e - s).count());
            }
        }
        std::sort(samples.begin(), samples.end());

        double sum = 0.0;
        for (double v : samples) sum += v;

        Measurement m{};
        m.recall = recall / static_cast<double>(query_count);
        m.mean_us = sum / static_cast<double>(samples.size());
        m.p50_us = samples[samples.size() / 2];
        m.p99_us = samples[std::min(samples.size() - 1, static_cast<std::size_t>(static_cast<double>(samples.size()) * 0.99))];
        return m;
    };

    std::cout << "========================================================================\n";
    std::cout << std::left
              << std::setw(22) << "Index Configuration"
              << std::setw(11) << "Recall@10"
              << std::setw(11) << "P50 (us)"
              << std::setw(11) << "P99 (us)"
              << std::setw(11) << "Mean (us)"
              << std::setw(12) << "QPS (1 core)"
              << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    auto print_row = [](const std::string& name, const Measurement& m) {
        std::cout << std::left << std::setw(22) << name
                  << std::fixed << std::setprecision(4) << std::setw(11) << m.recall
                  << std::setprecision(2)
                  << std::setw(11) << m.p50_us
                  << std::setw(11) << m.p99_us
                  << std::setw(11) << m.mean_us
                  << std::setprecision(0) << std::setw(12) << (1e6 / m.mean_us)
                  << "\n";
    };

    FlatIndex flat(dim, MetricType::L2);
    for (std::size_t i = 0; i < base_count; ++i) {
        flat.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    print_row("FlatIndex (Exact)", measure([&](VectorView q) { return flat.search(q, K); }));

    QuantizedFlatIndex qflat(dim);
    qflat.train(base_data);
    for (std::size_t i = 0; i < base_count; ++i) {
        qflat.add(VectorView(base_data.data() + (i * dim), dim));
    }
    print_row("QuantizedFlat (SQ8)", measure([&](VectorView q) { return qflat.search(q, K); }));

    const std::vector<std::size_t> ef_values = {32, 64, 100, 150};

    HNSWIndex hnsw(dim, MetricType::L2, 16, 200, 50);
    for (std::size_t i = 0; i < base_count; ++i) {
        hnsw.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    for (auto ef : ef_values) {
        hnsw.set_ef_search(ef);
        print_row("HNSW (ef=" + std::to_string(ef) + ")",
                  measure([&](VectorView q) { return hnsw.search(q, K); }));
    }

    ScalarQuantizer quantizer(dim);
    quantizer.train(base_data);
    HNSWIndexSQ8 sq8(dim, MetricType::L2, quantizer, 16, 200, 50, 5.0f);
    for (std::size_t i = 0; i < base_count; ++i) {
        sq8.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    for (auto ef : ef_values) {
        sq8.set_ef_search(ef);
        print_row("HNSW-SQ8 (ef=" + std::to_string(ef) + ")",
                  measure([&](VectorView q) { return sq8.search(q, K); }));
    }

    std::cout << "========================================================================\n";
    std::cout << "Single thread; " << query_count << " queries x " << REPEATS
              << " passes per row; HNSW M=16 efConstruction=200; SQ8 re-rank factor 5.0\n";
    return 0;
}
