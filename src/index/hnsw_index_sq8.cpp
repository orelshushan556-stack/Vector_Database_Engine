#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/distance/simd_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <iostream>

namespace engine {

    HNSWIndexSQ8::HNSWIndexSQ8(std::size_t dimension,
                               MetricType metric,
                               ScalarQuantizer quantizer,
                               std::size_t m,
                               std::size_t ef_construction,
                               std::size_t ef_search,
                               float re_rank_factor)
        : quantizer_(std::move(quantizer)),
          dimension_(dimension),
          metric_(metric),
          m_(m),
          max_m_(m),
          max_m0_(2 * m),
          ef_construction_(ef_construction),
          ef_search_(ef_search),
          re_rank_factor_(re_rank_factor),
          mult_(1.0 / std::log(static_cast<double>(m))) {
        if (!quantizer_.is_trained()) {
            throw std::runtime_error("ScalarQuantizer must be trained before creating HNSWIndexSQ8.");
        }
        if (quantizer_.dimension() != dimension_) {
            throw std::invalid_argument("Quantizer dimension does not match index dimension.");
        }
    }

    void HNSWIndexSQ8::add(VectorId id, VectorView vector) {
    }

    std::vector<SearchResult> HNSWIndexSQ8::search(VectorView query, std::size_t k) const {
        return {};
    }

    std::size_t HNSWIndexSQ8::dimension() const noexcept {
        return dimension_;
    }

    std::size_t HNSWIndexSQ8::size() const noexcept {
        return count_;
    }

    MetricType HNSWIndexSQ8::metric() const noexcept {
        return metric_;
    }

    bool HNSWIndexSQ8::empty() const noexcept {
        return count_ == 0;
    }

    void HNSWIndexSQ8::reserve(std::size_t capacity) {
    }

    void HNSWIndexSQ8::clear() noexcept {
    }

    void HNSWIndexSQ8::save(std::ostream& os) const {
    }

    void HNSWIndexSQ8::load(std::istream& is) {
    }

    void HNSWIndexSQ8::set_ef_search(std::size_t ef) noexcept {
        ef_search_ = ef;
    }

    std::size_t HNSWIndexSQ8::ef_search() const noexcept {
        return ef_search_;
    }

    void HNSWIndexSQ8::set_re_rank_factor(float factor) noexcept {
        re_rank_factor_ = factor;
    }

    float HNSWIndexSQ8::re_rank_factor() const noexcept {
        return re_rank_factor_;
    }

    const ScalarQuantizer& HNSWIndexSQ8::quantizer() const noexcept {
        return quantizer_;
    }

    float HNSWIndexSQ8::compute_quantized_distance(VectorView query, uint32_t node_id) const {
        const auto target = get_quantized_vector(node_id);
        const auto& scales = quantizer_.scale_values();
        const auto& mins = quantizer_.min_values();
        return quantized_l2_distance_sq_avx2(query, target, scales, mins);
    }

    float HNSWIndexSQ8::compute_exact_distance(VectorView query, uint32_t node_id) const {
        const auto target = get_raw_vector(node_id);

        switch (metric_) {
            case MetricType::L2:
                return l2_distance_squared_avx2(query, target);
            case MetricType::Cosine:
                return cosine_distance_avx2(query, target);
            case MetricType::DotProduct:
                return -dot_product_avx2(query, target);
        }

        return 0.0f;
    }

    std::span<const uint8_t> HNSWIndexSQ8::get_quantized_vector(uint32_t node_id) const {
        const std::size_t offset = static_cast<std::size_t>(node_id) * dimension_;
        return {quantized_storage_.data() + offset, dimension_};
    }

    VectorView HNSWIndexSQ8::get_raw_vector(uint32_t node_id) const {
        const std::size_t offset = static_cast<std::size_t>(node_id) * dimension_;
        return {raw_storage_.data() + offset, dimension_};
    }

    int HNSWIndexSQ8::get_random_level() {
        return 0;
    }

    std::vector<std::pair<float, uint32_t>> HNSWIndexSQ8::search_layer(
        VectorView query,
        const std::vector<uint32_t>& enter_points,
        std::size_t num_candidates,
        int level) const {
        return {};
    }

    void HNSWIndexSQ8::add_link(uint32_t src, uint32_t dst, int level) {
    }

    std::span<const uint32_t> HNSWIndexSQ8::get_links(uint32_t node_id, int level) const {
        return {};
    }

    std::span<uint32_t> HNSWIndexSQ8::get_links(uint32_t node_id, int level) {
        return {};
    }

} // namespace engine
