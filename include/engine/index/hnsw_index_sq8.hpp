#pragma once

#include "engine/index/index_base.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/types.hpp"

#include <vector>
#include <span>
#include <cstdint>
#include <cstddef>
#include <random>
#include <iosfwd>
#include <utility>

namespace engine {

    class HNSWIndexSQ8 : public IndexBase {
    public:
        HNSWIndexSQ8(std::size_t dimension,
                     MetricType metric,
                     ScalarQuantizer quantizer,
                     std::size_t m = 16,
                     std::size_t ef_construction = 200,
                     std::size_t ef_search = 50,
                     float re_rank_factor = 2.0f);

        ~HNSWIndexSQ8() override = default;

        void add(VectorId id, VectorView vector) override;
        void add(VectorView vector) { add(static_cast<VectorId>(count_), vector); }

        [[nodiscard]] std::vector<SearchResult> search(VectorView query, std::size_t k) const override;

        [[nodiscard]] std::size_t dimension() const noexcept override;
        [[nodiscard]] std::size_t size() const noexcept override;
        [[nodiscard]] MetricType metric() const noexcept override;

        [[nodiscard]] bool empty() const noexcept;
        void reserve(std::size_t capacity);
        void clear() noexcept;

        void save(std::ostream& os) const;
        void load(std::istream& is);

        void set_ef_search(std::size_t ef) noexcept;
        [[nodiscard]] std::size_t ef_search() const noexcept;

        void set_re_rank_factor(float factor) noexcept;
        [[nodiscard]] float re_rank_factor() const noexcept;

        [[nodiscard]] const ScalarQuantizer& quantizer() const noexcept;

    private:
        [[nodiscard]] float compute_quantized_distance(VectorView query, uint32_t node_id) const;
        [[nodiscard]] float compute_exact_distance(VectorView query, uint32_t node_id) const;

        [[nodiscard]] std::span<const uint8_t> get_quantized_vector(uint32_t node_id) const;
        [[nodiscard]] VectorView get_raw_vector(uint32_t node_id) const;

        [[nodiscard]] int get_random_level();
        
        [[nodiscard]] std::vector<std::pair<float, uint32_t>> search_layer(
            VectorView query,
            const std::vector<uint32_t>& enter_points,
            std::size_t num_candidates,
            int level) const;

        void add_link(uint32_t src, uint32_t dst, int level);
        [[nodiscard]] std::span<const uint32_t> get_links(uint32_t node_id, int level) const;
        [[nodiscard]] std::span<uint32_t> get_links(uint32_t node_id, int level);

        ScalarQuantizer quantizer_;

        std::vector<uint8_t> quantized_storage_;
        AlignedVector<float> raw_storage_;

        AlignedVector<uint32_t> layer0_graph_;
        std::vector<std::vector<uint32_t>> upper_graph_;
        std::vector<int> levels_;
        std::vector<VectorId> ids_;

        mutable std::vector<uint16_t> visited_tags_;
        mutable uint16_t current_visited_tag_{0};

        std::size_t dimension_;
        MetricType metric_;
        std::size_t m_;
        std::size_t max_m_;
        std::size_t max_m0_;
        std::size_t ef_construction_;
        std::size_t ef_search_;
        float re_rank_factor_{2.0f};
        double mult_;

        int max_level_{-1};
        uint32_t enter_point_{0};
        std::size_t count_{0};

        std::mt19937 rng_{std::random_device{}()};
    };

}
