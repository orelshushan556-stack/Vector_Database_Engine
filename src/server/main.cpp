#include "HttpServer.hpp"
#include "ConceptRegistry.hpp"
#include "engine/core/vector_engine.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    Nexus Vector Database Engine Server (C++)    " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::unique_ptr<engine::IndexBase> index;

    const std::string default_index_path = "index.bin";
    const std::string wal_path = "nexus.wal";

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
        std::cout << "[*] Initializing HNSW-SQ8 index with 100 Technical Architecture Concepts..." << std::endl;
        const size_t dim = ConceptRegistry::DIMENSION;

        // אימון הקוונטייזר
        std::vector<float> train_data;
        train_data.reserve(ConceptRegistry::TOTAL_CONCEPTS * dim);
        for (uint32_t id = 1; id <= ConceptRegistry::TOTAL_CONCEPTS; ++id) {
            auto span = ConceptRegistry::instance().get_vector(id);
            train_data.insert(train_data.end(), span.begin(), span.end());
        }

        engine::ScalarQuantizer quantizer(dim);
        quantizer.train(train_data);

        auto hnsw_sq8 = std::make_unique<engine::HNSWIndexSQ8>(
            dim,
            engine::MetricType::L2,
            std::move(quantizer),
            16,
            200
        );

        ConceptRegistry::instance().populate_index(*hnsw_sq8);
        index = std::move(hnsw_sq8);
        std::cout << "[+] Ready: " << index->size() << " technical concepts indexed across 4 architecture branches." << std::endl;
    }

    std::cout << "[*] Initializing VectorEngine with WAL recovery: " << wal_path << " ..." << std::endl;
    auto engine = std::make_shared<engine::VectorEngine>(std::move(index), wal_path);
    std::cout << "[+] VectorEngine ready. Total live vectors: " << engine->size() << std::endl;

    HttpServer server(8080, engine, 4);
    server.start();

    return 0;
}
