#include "HttpServer.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <vector>

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    Nexus Vector Database Engine Server (C++)    " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::shared_ptr<engine::IndexBase> index;

    const std::string default_index_path = "index.bin";
    if (std::filesystem::exists(default_index_path)) {
        try {
            std::cout << "[*] Loading serialized index from: " << default_index_path << " ..." << std::endl;
            index = engine::Serializer::load(default_index_path);
            std::cout << "[+] Index loaded successfully (" << index->size() << " vectors, dim=" << index->dimension() << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[!] Failed to load index: " << e.what() << std::endl;
        }
    }

    if (!index) {
        std::cout << "[*] Initializing demo HNSW-SQ8 index (dim=128, SIFT compatible)..." << std::endl;
        const size_t dim = 128;
        const size_t num_vectors = 1000;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> training_data(100 * dim);
        for (auto& v : training_data) v = dist(rng);

        engine::ScalarQuantizer quantizer(dim);
        quantizer.train(training_data);

        auto hnsw_sq8 = std::make_shared<engine::HNSWIndexSQ8>(
            dim,
            engine::MetricType::L2,
            std::move(quantizer),
            16,
            200
        );

        for (size_t i = 0; i < num_vectors; ++i) {
            std::vector<float> vec(dim);
            for (auto& v : vec) v = dist(rng);
            hnsw_sq8->add(static_cast<uint32_t>(i), vec);
        }

        index = hnsw_sq8;
        std::cout << "[+] Demo index ready with " << index->size() << " vectors." << std::endl;
    }

    HttpServer server(8080, index, 4);
    server.start();

    return 0;
}
