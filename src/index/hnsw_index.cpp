#include "engine/index/hnsw_index.hpp"
#include "engine/distance/simd_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

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
        if (vector.size() != dimension_) {
            throw std::invalid_argument("Vector dimension does not match index dimension.");
        }

        auto cur_internal_id = static_cast<uint32_t>(ids_.size());
        int new_level = get_random_level();

        ids_.push_back(id);
        storage_.insert(storage_.end(), vector.begin(), vector.end());
        levels_.push_back(new_level);
        layer0_graph_.resize(layer0_graph_.size() + max_m0_ + 1);
        upper_graph_.emplace_back();

        if (new_level > 0) {
            upper_graph_.back().resize(static_cast<std::size_t>(new_level) * (m_ + 1));
        }

        if (enter_point_ == -1) {
            enter_point_ = static_cast<int64_t>(cur_internal_id);
            max_level_ = new_level;
            return;
        }

        auto curr_ep = static_cast<uint32_t>(enter_point_);
        for (int lc = max_level_; lc > new_level; --lc) {
            auto candidates = search_layer(vector, curr_ep, 1, lc);
            curr_ep = candidates.top().second;
        }

        const int start_level = std::min(max_level_, new_level);

        auto compute_dist = [&](uint32_t id1, uint32_t id2) noexcept -> float {
            VectorView v1(&storage_[static_cast<std::size_t>(id1) * dimension_], dimension_);
            VectorView v2(&storage_[static_cast<std::size_t>(id2) * dimension_], dimension_);
            switch (metric_) {
                case MetricType::L2:
                    return l2_distance_squared_avx2(v1, v2);
                case MetricType::Cosine:
                    return cosine_distance_avx2(v1, v2);
                case MetricType::DotProduct:
                    return -dot_product_avx2(v1, v2);
            }
            return 0.0f;
        };

        auto add_link = [&](uint32_t src, uint32_t dst, int level, std::size_t max_links) {
            std::size_t offset = 0;
            uint32_t* graph_data = nullptr;

            if (level == 0) {
                offset = static_cast<std::size_t>(src) * (max_m0_ + 1);
                graph_data = layer0_graph_.data();
            } else {
                offset = static_cast<std::size_t>(level - 1) * (m_ + 1);
                graph_data = upper_graph_[src].data();
            }

            uint32_t& degree = graph_data[offset];

            if (degree < max_links) {
                graph_data[offset + 1 + degree] = dst;
                degree++;
            } else {
                float max_dist = std::numeric_limits<float>::lowest();
                uint32_t worst_idx = 0;
                float dist_new = compute_dist(src, dst);

                for (uint32_t i = 0; i < degree; ++i) {
                    uint32_t neighbor = graph_data[offset + 1 + i];
                    float dist = compute_dist(src, neighbor);
                    if (dist > max_dist) {
                        max_dist = dist;
                        worst_idx = i;
                    }
                }

                if (dist_new < max_dist) {
                    graph_data[offset + 1 + worst_idx] = dst;
                }
            }
        };

        for (int lc = start_level; lc >= 0; --lc) {
            auto can = search_layer(vector, curr_ep, ef_construction_, lc);
            const std::size_t target_m = (lc == 0) ? max_m0_ : m_;

            while (can.size() > target_m) {
                can.pop();
            }

            std::vector<uint32_t> neighbors;
            neighbors.reserve(can.size());

            while (!can.empty()) {
                curr_ep = can.top().second;
                neighbors.push_back(curr_ep);
                can.pop();
            }

            for (uint32_t neighbor : neighbors) {
                add_link(cur_internal_id, neighbor, lc, target_m);
                add_link(neighbor, cur_internal_id, lc, target_m);
            }
        }

        if (new_level > max_level_) {
            max_level_ = new_level;
            enter_point_ = cur_internal_id;
        }
    }

    std::vector<SearchResult> HNSWIndex::search(VectorView query, std::size_t k) const {
        if (query.size() != dimension_) {
            throw std::invalid_argument("Query vector dimension does not match index dimension.");
        }

        if (ids_.empty() || k == 0) {
            return {};
        }

        auto curr_ep = static_cast<uint32_t>(enter_point_);

        for (int lc = max_level_; lc > 0; --lc) {
            auto candidates = search_layer(query, curr_ep, 1, lc);
            curr_ep = candidates.top().second;
        }

        auto top_candidates = search_layer(query, curr_ep, std::max(ef_search_, k), 0);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        std::vector<SearchResult> results;
        results.reserve(top_candidates.size());

        while (!top_candidates.empty()) {
            auto [dist, node_id] = top_candidates.top();
            top_candidates.pop();

            if (metric_ == MetricType::DotProduct) {
                dist = -dist;
            }

            results.push_back(SearchResult{ids_[node_id], dist});
        }

        std::reverse(results.begin(), results.end());

        return results;
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
        ids_.reserve(capacity);
        storage_.reserve(capacity * dimension_);
        levels_.reserve(capacity);
        layer0_graph_.reserve(capacity * (max_m0_ + 1));
        upper_graph_.reserve(capacity);
        visited_tags_.reserve(capacity);
    }

    void HNSWIndex::clear() {
        ids_.clear();
        storage_.clear();
        levels_.clear();
        layer0_graph_.clear();
        upper_graph_.clear();
        visited_tags_.clear();

        enter_point_ = -1;
        max_level_ = -1;
        current_visited_tag_ = 0;
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

    void HNSWIndex::save(const std::string& file_path) const {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing: " + file_path);
        }

        auto write_val = [&out](const auto& val) {
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        };

        auto write_vec = [&out, &write_val](const auto& vec) {
            const std::size_t sz = vec.size();
            write_val(sz);
            if (sz > 0) {
                out.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(vec[0]));
            }
        };

        write_val(dimension_);
        write_val(metric_);
        write_val(m_);
        write_val(max_m0_);
        write_val(ef_construction_);
        write_val(ef_search_);
        write_val(mult_);
        write_val(max_level_);
        write_val(enter_point_);

        write_vec(ids_);
        write_vec(storage_);
        write_vec(levels_);
        write_vec(layer0_graph_);

        const std::size_t upper_sz = upper_graph_.size();
        write_val(upper_sz);
        for (const auto& inner : upper_graph_) {
            write_vec(inner);
        }
    }

    void HNSWIndex::load(const std::string& file_path) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file for reading: " + file_path);
        }

        auto read_val = [&in](auto& val) {
            in.read(reinterpret_cast<char*>(&val), sizeof(val));
        };

        auto read_vec = [&in, &read_val](auto& vec) {
            std::size_t sz = 0;
            read_val(sz);
            vec.resize(sz);
            if (sz > 0) {
                in.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(vec[0]));
            }
        };

        read_val(dimension_);
        read_val(metric_);
        read_val(m_);
        read_val(max_m0_);
        read_val(ef_construction_);
        read_val(ef_search_);
        read_val(mult_);
        read_val(max_level_);
        read_val(enter_point_);

        read_vec(ids_);
        read_vec(storage_);
        read_vec(levels_);
        read_vec(layer0_graph_);

        std::size_t upper_sz = 0;
        read_val(upper_sz);
        upper_graph_.resize(upper_sz);
        for (auto& inner : upper_graph_) {
            read_vec(inner);
        }

        current_visited_tag_ = 0;
        visited_tags_.assign(ids_.size(), 0);
    }

}