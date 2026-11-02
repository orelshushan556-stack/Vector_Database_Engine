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

    // 1. FlatIndex Baseline
    std::cout << "[*] Running FlatIndex (Exact L2)... " << std::flush;
    FlatIndex flat(dim, MetricType::L2);
    for (std::size_t i = 0; i < base_count; ++i) {
        flat.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    double flat_recall = 0.0;
    for (std::size_t q = 0; q < query_count; ++q) {
        auto res = flat.search(get_query(q), K);
        flat_recall += compute_recall(ground_truth[q], res, K);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double flat_lat = std::chrono::duration<double, std::micro>(t1 - t0).count() / query_count;
    std::cout << "Done (" << flat_lat << " us/query | Recall@10: " << (flat_recall / query_count) << ")\n";

    // 2. QuantizedFlatIndex
    std::cout << "[*] Running QuantizedFlat (SQ8 ADC)... " << std::flush;
    QuantizedFlatIndex qflat(dim);
    qflat.train(base_data);
    for (std::size_t i = 0; i < base_count; ++i) {
        qflat.add(VectorView(base_data.data() + (i * dim), dim));
    }
    t0 = std::chrono::high_resolution_clock::now();
    double qflat_recall = 0.0;
    for (std::size_t q = 0; q < query_count; ++q) {
        auto res = qflat.search(get_query(q), K);
        qflat_recall += compute_recall(ground_truth[q], res, K);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double qflat_lat = std::chrono::duration<double, std::micro>(t1 - t0).count() / query_count;
    std::cout << "Done (" << qflat_lat << " us/query | Recall@10: " << (qflat_recall / query_count) << ")\n\n";

    // 3. HNSW Parameter Sweep (Pareto Curve)
    std::cout << "[*] Building HNSW (M=16, ef_c=200)... " << std::flush;
    HNSWIndex hnsw(dim, MetricType::L2, 16, 200, 50);
    for (std::size_t i = 0; i < base_count; ++i) {
        hnsw.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    std::cout << "Done.\n";

    // 4. HNSW-SQ8 Building
    std::cout << "[*] Building HNSW-SQ8 (M=16, ef_c=200, Re-rank 2.0x)... " << std::flush;
    ScalarQuantizer quantizer(dim);
    quantizer.train(base_data);
    HNSWIndexSQ8 sq8(dim, MetricType::L2, quantizer, 16, 200, 50, 5.0f);
    for (std::size_t i = 0; i < base_count; ++i) {
        sq8.add(static_cast<VectorId>(i), VectorView(base_data.data() + (i * dim), dim));
    }
    std::cout << "Done.\n\n";

    // Print SIFT Benchmark Results
    std::cout << "========================================================================\n";
    std::cout << std::left 
              << std::setw(20) << "Index Configuration"
              << std::setw(15) << "Latency (us)"
              << std::setw(18) << "QPS (Single-Core)"
              << std::setw(15) << "Recall@10"
              << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    auto print_row = [](const std::string& name, double lat_us, double recall) {
        std::cout << std::left 
                  << std::setw(20) << name
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << lat_us
                  << std::setprecision(0)
                  << std::setw(18) << ((1.0 / lat_us) * 1e6)
                  << std::setprecision(4)
                  << std::setw(15) << recall
                  << "\n";
    };

    print_row("FlatIndex (Exact)", flat_lat, flat_recall / query_count);
    print_row("QuantizedFlat (SQ8)", qflat_lat, qflat_recall / query_count);

    // Sweep ef_search for HNSW
    const std::vector<std::size_t> ef_values = {32, 64, 100, 150};
    for (auto ef : ef_values) {
        hnsw.set_ef_search(ef);
        t0 = std::chrono::high_resolution_clock::now();
        double recall = 0.0;
        for (std::size_t q = 0; q < query_count; ++q) {
            auto res = hnsw.search(get_query(q), K);
            recall += compute_recall(ground_truth[q], res, K);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double, std::micro>(t1 - t0).count() / query_count;
        print_row("HNSW (ef=" + std::to_string(ef) + ")", lat, recall / query_count);
    }

    // Sweep ef_search for HNSW-SQ8
    for (auto ef : ef_values) {
        sq8.set_ef_search(ef);
        t0 = std::chrono::high_resolution_clock::now();
        double recall = 0.0;
        for (std::size_t q = 0; q < query_count; ++q) {
            auto res = sq8.search(get_query(q), K);
            recall += compute_recall(ground_truth[q], res, K);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double, std::micro>(t1 - t0).count() / query_count;
        print_row("HNSW-SQ8 (ef=" + std::to_string(ef) + ")", lat, recall / query_count);
    }

    std::cout << "========================================================================\n\n";
    return 0;
}