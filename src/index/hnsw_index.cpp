#include "engine/index/hnsw_index.hpp"
#include "engine/distance/simd_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace engine {

    HNSWIndex::HNSWIndex(std::size_t dimension, MetricType metric, std::size_t m, std::size_t ef_construction, std::size_t ef_search)
        : dimension_(dimension),
          metric_(metric),
          m_(m),
          max_m0_(2 * m),
          ef_construction_(ef_construction),
          ef_search_(ef_search),
          mult_(1.0 / std::log(static_cast<double>(m))),
          rng_(std::random_device{}()) {
    }

    void HNSWIndex::add(VectorId id, VectorView vector) {
    }

    std::vector<SearchResult> HNSWIndex::search(VectorView query, std::size_t k) const {
        return {};
    }

    std::size_t HNSWIndex::size() const noexcept {
        return ids_.size();
    }

    std::size_t HNSWIndex::dimension() const noexcept {
        return dimension_;
    }

    MetricType HNSWIndex::metric() const noexcept {
        return metric_;
    }

    void HNSWIndex::set_ef_search(std::size_t ef) noexcept {
        ef_search_ = ef;
    }

    std::size_t HNSWIndex::ef_search() const noexcept {
        return ef_search_;
    }

    std::size_t HNSWIndex::m() const noexcept {
        return m_;
    }

    std::size_t HNSWIndex::ef_construction() const noexcept {
        return ef_construction_;
    }

    void HNSWIndex::reserve(std::size_t capacity) {
    }

    void HNSWIndex::clear() {
    }

    int HNSWIndex::get_random_level() {
        double r = uniform_dist_(rng_);
        while (r == 0.0) {
            r = uniform_dist_(rng_);
        }

        return static_cast<int>(-std::log(r) * mult_);
    }

    std::span<uint32_t> HNSWIndex::get_links(uint32_t internal_id, int level) noexcept {
        if (level == 0) {
            const std::size_t offset = static_cast<std::size_t>(internal_id) * (max_m0_ + 1);
            uint32_t degree = layer0_graph_[offset];
            return {layer0_graph_.data() + offset + 1, degree};
        } else {
            auto& node_links = upper_graph_[internal_id];
            const std::size_t offset = static_cast<std::size_t>(level - 1) * (m_ + 1);
            uint32_t degree = node_links[offset];
            return {node_links.data() + offset + 1, degree};
        }
    }

    std::span<const uint32_t> HNSWIndex::get_links(uint32_t internal_id, int level) const noexcept {
        if (level == 0) {
            const std::size_t offset = static_cast<std::size_t>(internal_id) * (max_m0_ + 1);
            uint32_t degree = layer0_graph_[offset];
            return {layer0_graph_.data() + offset + 1, degree};
        } else {
            const auto& node_links = upper_graph_[internal_id];
            const std::size_t offset = static_cast<std::size_t>(level - 1) * (m_ + 1);
            uint32_t degree = node_links[offset];
            return {node_links.data() + offset + 1, degree};
        }
    }
std::priority_queue<std::pair<float, uint32_t>> HNSWIndex::search_layer(
    VectorView query,
    uint32_t enter_point,
    std::size_t ef,
    int level) const {

    using DistPair = std::pair<float, uint32_t>;

    std::vector<DistPair> top_container;
    top_container.reserve(ef + 1);
    std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>> top_candidates(std::less<DistPair>{}, std::move(top_container));

    std::vector<DistPair> cand_container;
    cand_container.reserve(ef * 2);
    std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>> candidates(std::greater<DistPair>{}, std::move(cand_container));

    if (visited_tags_.size() < ids_.size()) {
        visited_tags_.resize(ids_.size(), 0);
    }

    ++current_visited_tag_;
    if (current_visited_tag_ == 0) {
        std::fill(visited_tags_.begin(), visited_tags_.end(), 0);
        current_visited_tag_ = 1;
    }

    auto compute_dist = [&](uint32_t internal_id) noexcept -> float {
        VectorView target(&storage_[static_cast<std::size_t>(internal_id) * dimension_], dimension_);
        switch (metric_) {
            case MetricType::L2:
                return l2_distance_squared_avx2(query, target);
            case MetricType::Cosine:
                return cosine_distance_avx2(query, target);
            case MetricType::DotProduct:
                return -dot_product_avx2(query, target);
        }
        return 0.0f;
    };

    float ep_dist = compute_dist(enter_point);
    visited_tags_[enter_point] = current_visited_tag_;
    candidates.emplace(ep_dist, enter_point);
    top_candidates.emplace(ep_dist, enter_point);

    while (!candidates.empty()) {
        auto [curr_dist, curr_node] = candidates.top();
        candidates.pop();

        float worst_dist = top_candidates.top().first;
        if (curr_dist > worst_dist && top_candidates.size() >= ef) {
            break;
        }

        for (uint32_t neighbor_id : get_links(curr_node, level)) {
            if (visited_tags_[neighbor_id] != current_visited_tag_) {
                visited_tags_[neighbor_id] = current_visited_tag_;

                float dist = compute_dist(neighbor_id);

                if (top_candidates.size() < ef || dist < top_candidates.top().first) {
                    candidates.emplace(dist, neighbor_id);
                    top_candidates.emplace(dist, neighbor_id);

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }

    return top_candidates;
}

}



