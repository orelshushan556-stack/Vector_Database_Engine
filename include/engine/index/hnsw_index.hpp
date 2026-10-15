#pragma once

#include "engine/index/index_base.hpp"
#include "engine/types.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace engine {

    class HNSWIndex : public IndexBase {
    public:
        HNSWIndex(std::size_t dimension, MetricType metric, std::size_t m = 16, std::size_t ef_construction = 200, std::size_t ef_search = 50);
        ~HNSWIndex() override = default;

        void add(VectorId id, VectorView vector) override;
        [[nodiscard]] std::vector<SearchResult> search(VectorView query, std::size_t k) const override;

        [[nodiscard]] std::size_t size() const noexcept override;
        [[nodiscard]] std::size_t dimension() const noexcept override;
        [[nodiscard]] MetricType metric() const noexcept override;

        void set_ef_search(std::size_t ef) noexcept;
        [[nodiscard]] std::size_t ef_search() const noexcept;
        [[nodiscard]] std::size_t m() const noexcept;
        [[nodiscard]] std::size_t ef_construction() const noexcept;

        void reserve(std::size_t capacity);
        void clear();

    private:
        [[nodiscard]] int get_random_level();

        [[nodiscard]] std::span<uint32_t> get_links(uint32_t internal_id, int level) noexcept;
        [[nodiscard]] std::span<const uint32_t> get_links(uint32_t internal_id, int level) const noexcept;

        [[nodiscard]] std::priority_queue<std::pair<float, uint32_t>> search_layer(
            VectorView query,
            uint32_t enter_point,
            std::size_t ef,
            int level) const;

        std::size_t dimension_;
        MetricType metric_;
        std::size_t m_;
        std::size_t max_m0_;
        std::size_t ef_construction_;
        std::size_t ef_search_;
        double mult_;

        int max_level_{-1};
        int64_t enter_point_{-1};

        AlignedVector<float> storage_;
        std::vector<VectorId> ids_;
        std::vector<int> levels_;
        std::vector<uint32_t> layer0_graph_;
        std::vector<std::vector<uint32_t>> upper_graph_;

        mutable std::vector<uint32_t> visited_tags_;
        mutable uint32_t current_visited_tag_{0};

        std::mt19937 rng_;
        std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    };

}