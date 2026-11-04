#include "HttpServer.hpp"
#include "engine/core/vector_engine.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    Nexus Vector Database Engine Server (C++)    " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::unique_ptr<engine::IndexBase> index;

    constexpr std::string_view default_index_path = "index.bin";
    constexpr std::string_view wal_path = "nexus.wal";

    if (std::filesystem::exists(default_index_path)) {
        try {
            std::cout << "[*] Loading serialized index snapshot from: " << default_index_path << " ..." << std::endl;
            index = engine::Serializer::load(default_index_path);
            std::cout << "[+] Index loaded successfully (" << index->size() << " vectors, dim=" << index->dimension() << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[!] Failed to load index: " << e.what() << std::endl;
        }
    }

    if (!index) {
        std::cout << "[*] Initializing demo HNSW-SQ8 index (dim=128, SIFT compatible)..." << std::endl;
        constexpr size_t dim = 128;
        constexpr size_t num_vectors = 1000;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> training_data(100 * dim);
        std::ranges::generate(training_data, [&]() { return dist(rng); });

        engine::ScalarQuantizer quantizer(dim);
        quantizer.train(training_data);

        auto hnsw_sq8 = std::make_unique<engine::HNSWIndexSQ8>(
            dim,
            engine::MetricType::L2,
            std::move(quantizer),
            16,
            200
        );

        std::vector<float> vec(dim);
        for (size_t i = 0; i < num_vectors; ++i) {
            std::ranges::generate(vec, [&]() { return dist(rng); });
            hnsw_sq8->add(static_cast<uint64_t>(i), vec);
        }

        index = std::move(hnsw_sq8);
        std::cout << "[+] Demo index ready with " << index->size() << " vectors." << std::endl;
    }

    std::cout << "[*] Initializing VectorEngine with WAL recovery: " << wal_path << " ..." << std::endl;
    auto engine = std::make_shared<engine::VectorEngine>(std::move(index), std::string(wal_path));
    std::cout << "[+] VectorEngine ready. Total live vectors: " << engine->size() << std::endl;

    HttpServer server(8080, engine, 4);
    server.start();

    return 0;
}
