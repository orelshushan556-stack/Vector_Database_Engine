#pragma once

#include "engine/index/index_base.hpp"
#include <string_view>
#include <string>
#include <vector>
#include <array>
#include <span>

struct ConceptMetadata {
    uint32_t id;
    uint32_t branch; // 0: SIMD, 1: HNSW, 2: WAL, 3: Concurrency
    std::string_view label;
    std::string_view category;
    std::string_view description;
    std::string_view code_ref;
};

class ConceptRegistry {
public:
    static constexpr size_t TOTAL_CONCEPTS = 100;
    static constexpr size_t DIMENSION = 128;

    static ConceptRegistry& instance() noexcept {
        static ConceptRegistry registry;
        return registry;
    }

    [[nodiscard]] const std::string& get_precompiled_json() const noexcept {
        return precompiled_json_;
    }

    [[nodiscard]] const ConceptMetadata* get_concept(uint32_t id) const noexcept {
        if (id == 0 || id > TOTAL_CONCEPTS) return nullptr;
        return &concepts_[id - 1];
    }

    [[nodiscard]] std::span<const float> get_vector(uint32_t id) const noexcept {
        if (id == 0 || id > TOTAL_CONCEPTS) return {};
        return {vectors_[id - 1].data(), DIMENSION};
    }

    void populate_index(engine::IndexBase& index) const;

private:
    ConceptRegistry();
    ~ConceptRegistry() = default;
    ConceptRegistry(const ConceptRegistry&) = delete;
    ConceptRegistry& operator=(const ConceptRegistry&) = delete;

    std::array<ConceptMetadata, TOTAL_CONCEPTS> concepts_;
    std::vector<std::vector<float>> vectors_;
    std::string precompiled_json_;

    void init_concepts();
    void generate_semantic_vectors();
    void precompile_json();
};
