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
        if (vector.size() != dimension_) {
            throw std::invalid_argument("Vector dimension does not match index dimension.");
        }

        const auto cur_internal_id = static_cast<uint32_t>(count_);
        const int new_level = get_random_level();

        ids_.push_back(id);
        levels_.push_back(new_level);
        raw_storage_.insert(raw_storage_.end(), vector.begin(), vector.end());

        const std::size_t q_offset = static_cast<std::size_t>(cur_internal_id) * dimension_;
        quantized_storage_.resize(q_offset + dimension_);
        quantizer_.encode(vector, std::span<uint8_t>(quantized_storage_.data() + q_offset, dimension_));

        layer0_graph_.resize(layer0_graph_.size() + max_m0_ + 1, 0);
        upper_graph_.emplace_back();
        if (new_level > 0) {
            upper_graph_.back().resize(static_cast<std::size_t>(new_level) * (max_m_ + 1), 0);
        }

        ++count_;

        if (max_level_ == -1) {
            enter_point_ = cur_internal_id;
            max_level_ = new_level;
            return;
        }

        uint32_t curr_ep = enter_point_;
        for (int lc = max_level_; lc > new_level; --lc) {
            auto candidates = search_layer(vector, {curr_ep}, 1, lc);
            if (!candidates.empty()) {
                curr_ep = candidates.front().second;
            }
        }

        const int start_level = std::min(max_level_, new_level);
        for (int lc = start_level; lc >= 0; --lc) {
            auto candidates = search_layer(vector, {curr_ep}, ef_construction_, lc);
            const std::size_t target_m = (lc == 0) ? max_m0_ : max_m_;
            const std::size_t num_neighbors = std::min(candidates.size(), target_m);

            if (!candidates.empty()) {
                curr_ep = candidates.front().second;
            }

            for (std::size_t i = 0; i < num_neighbors; ++i) {
                uint32_t neighbor = candidates[i].second;
                add_link(cur_internal_id, neighbor, lc);
                add_link(neighbor, cur_internal_id, lc);
            }
        }

        if (new_level > max_level_) {
            max_level_ = new_level;
            enter_point_ = cur_internal_id;
        }
    }

    std::vector<SearchResult> HNSWIndexSQ8::search(VectorView query, std::size_t k) const {
        if (query.size() != dimension_) {
            throw std::invalid_argument("Query vector dimension does not match index dimension.");
        }

        if (empty() || k == 0) {
            return {};
        }

        uint32_t curr_ep = enter_point_;
        for (int lc = max_level_; lc > 0; --lc) {
            auto candidates = search_layer(query, {curr_ep}, 1, lc);
            if (!candidates.empty()) {
                curr_ep = candidates.front().second;
            }
        }

        const float effective_factor = std::max(1.0f, re_rank_factor_);
        const auto candidates_to_rerank = static_cast<std::size_t>(std::ceil(static_cast<float>(k) * effective_factor));
        const std::size_t ef = std::max(ef_search_, candidates_to_rerank);

        auto top_candidates = search_layer(query, {curr_ep}, ef, 0);

        const std::size_t pool_size = std::min(top_candidates.size(), candidates_to_rerank);
        std::vector<SearchResult> results;
        results.reserve(pool_size);

        for (std::size_t i = 0; i < pool_size; ++i) {
            uint32_t node_id = top_candidates[i].second;
            float exact_dist = compute_exact_distance(query, node_id);
            results.push_back(SearchResult{ids_[node_id], exact_dist});
        }

        std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });

        if (results.size() > k) {
            results.resize(k);
        }

        if (metric_ == MetricType::DotProduct) {
            for (auto& res : results) {
                res.distance = -res.distance;
            }
        }

        return results;
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
        ids_.reserve(capacity);
        levels_.reserve(capacity);
        raw_storage_.reserve(capacity * dimension_);
        quantized_storage_.reserve(capacity * dimension_);
        layer0_graph_.reserve(capacity * (max_m0_ + 1));
        upper_graph_.reserve(capacity);
        visited_tags_.reserve(capacity);
    }

    void HNSWIndexSQ8::clear() noexcept {
        quantized_storage_.clear();
        raw_storage_.clear();
        layer0_graph_.clear();
        upper_graph_.clear();
        levels_.clear();
        ids_.clear();
        visited_tags_.clear();
        current_visited_tag_ = 0;
        max_level_ = -1;
        enter_point_ = 0;
        count_ = 0;
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
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        double r = uniform_dist(rng_);
        while (r == 0.0) {
            r = uniform_dist(rng_);
        }
        return static_cast<int>(-std::log(r) * mult_);
    }

    std::vector<std::pair<float, uint32_t>> HNSWIndexSQ8::search_layer(
        VectorView query,
        const std::vector<uint32_t>& enter_points,
        std::size_t num_candidates,
        int level) const {

        using DistPair = std::pair<float, uint32_t>;

        std::vector<DistPair> top_container;
        top_container.reserve(num_candidates + 1);
        std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>> top_candidates(
            std::less<DistPair>{}, std::move(top_container));

        std::vector<DistPair> cand_container;
        cand_container.reserve(num_candidates * 2);
        std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>> candidates(
            std::greater<DistPair>{}, std::move(cand_container));

        if (visited_tags_.size() < ids_.size()) {
            visited_tags_.resize(ids_.size(), 0);
        }
        ++current_visited_tag_;
        if (current_visited_tag_ == 0) {
            std::fill(visited_tags_.begin(), visited_tags_.end(), 0);
            current_visited_tag_ = 1;
        }

        for (uint32_t ep : enter_points) {
            visited_tags_[ep] = current_visited_tag_;
            float dist = compute_quantized_distance(query, ep);
            top_candidates.emplace(dist, ep);
            candidates.emplace(dist, ep);
        }

        while (!candidates.empty()) {
            auto [curr_dist, curr_node] = candidates.top();
            candidates.pop();

            float worst_dist = top_candidates.top().first;
            if (curr_dist > worst_dist && top_candidates.size() >= num_candidates) {
                break;
            }

            for (uint32_t neighbor_id : get_links(curr_node, level)) {
                if (visited_tags_[neighbor_id] != current_visited_tag_) {
                    visited_tags_[neighbor_id] = current_visited_tag_;

                    float dist = compute_quantized_distance(query, neighbor_id);

                    if (top_candidates.size() < num_candidates || dist < top_candidates.top().first) {
                        candidates.emplace(dist, neighbor_id);
                        top_candidates.emplace(dist, neighbor_id);

                        if (top_candidates.size() > num_candidates) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        std::vector<DistPair> results;
        results.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            results.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(results.begin(), results.end());

        return results;
    }

    void HNSWIndexSQ8::add_link(uint32_t src, uint32_t dst, int level) {
        std::size_t offset = 0;
        uint32_t* graph_data = nullptr;
        const std::size_t max_links = (level == 0) ? max_m0_ : max_m_;

        if (level == 0) {
            offset = static_cast<std::size_t>(src) * (max_m0_ + 1);
            graph_data = layer0_graph_.data();
        } else {
            offset = static_cast<std::size_t>(level - 1) * (max_m_ + 1);
            graph_data = upper_graph_[src].data();
        }

        uint32_t& degree = graph_data[offset];
        VectorView src_vec = get_raw_vector(src);

        if (degree < max_links) {
            graph_data[offset + 1 + degree] = dst;
            degree++;
        } else {
            float max_dist = -1.0f;
            uint32_t worst_idx = 0;
            float dist_new = compute_quantized_distance(src_vec, dst);

            for (uint32_t i = 0; i < degree; ++i) {
                uint32_t neighbor = graph_data[offset + 1 + i];
                float dist = compute_quantized_distance(src_vec, neighbor);
                if (dist > max_dist) {
                    max_dist = dist;
                    worst_idx = i;
                }
            }

            if (dist_new < max_dist) {
                graph_data[offset + 1 + worst_idx] = dst;
            }
        }
    }

    std::span<const uint32_t> HNSWIndexSQ8::get_links(uint32_t node_id, int level) const {
        if (level == 0) {
            const std::size_t offset = static_cast<std::size_t>(node_id) * (max_m0_ + 1);
            uint32_t degree = layer0_graph_[offset];
            return {layer0_graph_.data() + offset + 1, degree};
        } else {
            const auto& node_links = upper_graph_[node_id];
            const std::size_t offset = static_cast<std::size_t>(level - 1) * (max_m_ + 1);
            uint32_t degree = node_links[offset];
            return {node_links.data() + offset + 1, degree};
        }
    }

    std::span<uint32_t> HNSWIndexSQ8::get_links(uint32_t node_id, int level) {
        if (level == 0) {
            const std::size_t offset = static_cast<std::size_t>(node_id) * (max_m0_ + 1);
            uint32_t degree = layer0_graph_[offset];
            return {layer0_graph_.data() + offset + 1, degree};
        } else {
            auto& node_links = upper_graph_[node_id];
            const std::size_t offset = static_cast<std::size_t>(level - 1) * (max_m_ + 1);
            uint32_t degree = node_links[offset];
            return {node_links.data() + offset + 1, degree};
        }
    }

}