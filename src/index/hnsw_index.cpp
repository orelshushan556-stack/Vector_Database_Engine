#include "engine/index/hnsw_index.hpp"
#include "engine/distance/simd_metrics.hpp"
#include "engine/storage/serializer.hpp"
#include "hnsw_common.hpp"

#include <algorithm>
#include <cmath>
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
      rng_(std::random_device{}()) {}

std::size_t HNSWIndex::size() const noexcept { return ids_.size(); }
std::size_t HNSWIndex::dimension() const noexcept { return dimension_; }
MetricType HNSWIndex::metric() const noexcept { return metric_; }
void HNSWIndex::set_ef_search(std::size_t ef) noexcept { ef_search_ = ef; }
std::size_t HNSWIndex::ef_search() const noexcept { return ef_search_; }
std::size_t HNSWIndex::m() const noexcept { return m_; }
std::size_t HNSWIndex::ef_construction() const noexcept { return ef_construction_; }

void HNSWIndex::reserve(std::size_t capacity) {
    storage_.reserve(capacity * dimension_);
    ids_.reserve(capacity);
    levels_.reserve(capacity);
    layer0_graph_.reserve(capacity * (max_m0_ + 1));
    upper_graph_.reserve(capacity);
}

void HNSWIndex::clear() {
    storage_.clear();
    ids_.clear();
    levels_.clear();
    layer0_graph_.clear();
    upper_graph_.clear();
    max_level_ = -1;
    enter_point_ = -1;
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

    const std::size_t vec_bytes = dimension_ * sizeof(float);

    return hnsw_detail::search_layer_impl(
        std::span<const uint32_t>(&enter_point, 1),
        ef,
        ids_.size(),
        [&](uint32_t node) { return get_links(node, level); },
        [&](uint32_t node) { return distance_to(query, node); },
        [&](uint32_t node) {
            hnsw_detail::prefetch_bytes(&storage_[static_cast<std::size_t>(node) * dimension_], vec_bytes);
        });
}

float HNSWIndex::distance_to(VectorView query, uint32_t internal_id) const {
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

    auto add_link = [&](uint32_t src, uint32_t dst, int level, std::size_t max_links) {
        uint32_t* graph_data = nullptr;
        std::size_t offset = 0;

        if (level == 0) {
            offset = static_cast<std::size_t>(src) * (max_m0_ + 1);
            graph_data = layer0_graph_.data();
        } else {
            offset = static_cast<std::size_t>(level - 1) * (m_ + 1);
            graph_data = upper_graph_[src].data();
        }

        const VectorView src_vec(&storage_[static_cast<std::size_t>(src) * dimension_], dimension_);
        hnsw_detail::add_link_impl(graph_data, offset, max_links, dst,
                                   [&](uint32_t node) { return distance_to(src_vec, node); });
    };

    for (int lc = start_level; lc >= 0; --lc) {
        const std::size_t target_m = (lc == 0) ? max_m0_ : m_;
        const auto candidates = hnsw_detail::drain_ascending(
            search_layer(vector, curr_ep, ef_construction_, lc));

        if (!candidates.empty()) {
            curr_ep = candidates.front().second;
        }

        hnsw_detail::connect_new_node(cur_internal_id, candidates, target_m,
                                      [&](uint32_t src, uint32_t dst) { add_link(src, dst, lc, target_m); });
    }

    if (new_level > max_level_) {
        max_level_ = new_level;
        enter_point_ = cur_internal_id;
    }
}

std::vector<SearchResult> HNSWIndex::search(VectorView query, std::size_t k) const {
    if (enter_point_ == -1 || k == 0 || query.size() != dimension_) {
        return {};
    }

    uint32_t curr_ep = static_cast<uint32_t>(enter_point_);
    float curr_dist = distance_to(query, curr_ep);

    hnsw_detail::greedy_descend(
        curr_ep, curr_dist, max_level_, 0,
        [&](uint32_t node, int lc) { return get_links(node, lc); },
        [&](uint32_t node) { return distance_to(query, node); });

    std::size_t ef = std::max(ef_search_, k);
    auto top_candidates = search_layer(query, curr_ep, ef, 0);

    std::vector<SearchResult> results;
    results.reserve(top_candidates.size());

    while (!top_candidates.empty()) {
        auto [dist, internal_id] = top_candidates.top();
        top_candidates.pop();
        results.push_back({ids_[internal_id], dist});
    }

    std::reverse(results.begin(), results.end());
    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

void HNSWIndex::save(const std::string& file_path) const {
    Serializer::save_hnsw_index(*this, file_path);
}

void HNSWIndex::load(const std::string& file_path) {
    *this = Serializer::load_hnsw_index(file_path);
}

} // namespace engine
