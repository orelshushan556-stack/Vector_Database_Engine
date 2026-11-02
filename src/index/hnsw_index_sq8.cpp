#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/distance/simd_metrics.hpp"
#include "engine/storage/serializer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace engine {

namespace {
    struct ThreadLocalVisited {
        std::vector<uint32_t> tags;
        uint32_t current_tag{0};
    };

    thread_local ThreadLocalVisited tl_visited;
}

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

std::size_t HNSWIndexSQ8::dimension() const noexcept { return dimension_; }
std::size_t HNSWIndexSQ8::size() const noexcept { return count_; }
MetricType HNSWIndexSQ8::metric() const noexcept { return metric_; }
bool HNSWIndexSQ8::empty() const noexcept { return count_ == 0; }
void HNSWIndexSQ8::set_ef_search(std::size_t ef) noexcept { ef_search_ = ef; }
std::size_t HNSWIndexSQ8::ef_search() const noexcept { return ef_search_; }
void HNSWIndexSQ8::set_re_rank_factor(float factor) noexcept { re_rank_factor_ = factor; }
float HNSWIndexSQ8::re_rank_factor() const noexcept { return re_rank_factor_; }
const ScalarQuantizer& HNSWIndexSQ8::quantizer() const noexcept { return quantizer_; }

void HNSWIndexSQ8::reserve(std::size_t capacity) {
    ids_.reserve(capacity);
    levels_.reserve(capacity);
    raw_storage_.reserve(capacity * dimension_);
    quantized_storage_.reserve(capacity * dimension_);
    layer0_graph_.reserve(capacity * (max_m0_ + 1));
    upper_graph_.reserve(capacity);
}

void HNSWIndexSQ8::clear() noexcept {
    quantized_storage_.clear();
    raw_storage_.clear();
    layer0_graph_.clear();
    upper_graph_.clear();
    levels_.clear();
    ids_.clear();
    max_level_ = -1;
    enter_point_ = 0;
    count_ = 0;
}

int HNSWIndexSQ8::get_random_level() {
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    double r = uniform_dist(rng_);
    while (r == 0.0) {
        r = uniform_dist(rng_);
    }
    return static_cast<int>(-std::log(r) * mult_);
}

std::span<const uint8_t> HNSWIndexSQ8::get_quantized_vector(uint32_t node_id) const {
    const std::size_t offset = static_cast<std::size_t>(node_id) * dimension_;
    return {quantized_storage_.data() + offset, dimension_};
}

VectorView HNSWIndexSQ8::get_raw_vector(uint32_t node_id) const {
    const std::size_t offset = static_cast<std::size_t>(node_id) * dimension_;
    return {raw_storage_.data() + offset, dimension_};
}

float HNSWIndexSQ8::compute_quantized_distance(VectorView query, uint32_t node_id) const {
    const auto target = get_quantized_vector(node_id);
    const auto& scales = quantizer_.scale_values();
    const auto& mins = quantizer_.min_values();
    switch (metric_) {
        case MetricType::L2:
            return quantized_l2_distance_sq_avx2(query, target, scales, mins);
        case MetricType::Cosine: {
            auto raw = get_raw_vector(node_id);
            return cosine_distance_avx2(query, raw);
        }
        case MetricType::DotProduct: {
            auto raw = get_raw_vector(node_id);
            return -dot_product_avx2(query, raw);
        }
    }
    return 0.0f;
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

std::vector<std::pair<float, uint32_t>> HNSWIndexSQ8::search_layer(
    VectorView query,
    const std::vector<uint32_t>& enter_points,
    std::size_t num_candidates,
    int level) const {

    using DistPair = std::pair<float, uint32_t>;

    std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>> candidates;
    std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>> top_results;

    if (tl_visited.tags.size() < count_) {
        tl_visited.tags.resize(count_, 0);
    }

    ++tl_visited.current_tag;
    if (tl_visited.current_tag == 0) {
        std::fill(tl_visited.tags.begin(), tl_visited.tags.end(), 0);
        tl_visited.current_tag = 1;
    }

    for (uint32_t ep : enter_points) {
        float d = compute_quantized_distance(query, ep);
        tl_visited.tags[ep] = tl_visited.current_tag;
        candidates.emplace(d, ep);
        top_results.emplace(d, ep);
    }

    while (!candidates.empty()) {
        auto [c_dist, c_node] = candidates.top();
        candidates.pop();

        if (c_dist > top_results.top().first && top_results.size() >= num_candidates) {
            break;
        }

        for (uint32_t neighbor : get_links(c_node, level)) {
            if (tl_visited.tags[neighbor] != tl_visited.current_tag) {
                tl_visited.tags[neighbor] = tl_visited.current_tag;

                float d = compute_quantized_distance(query, neighbor);

                if (top_results.size() < num_candidates || d < top_results.top().first) {
                    candidates.emplace(d, neighbor);
                    top_results.emplace(d, neighbor);

                    if (top_results.size() > num_candidates) {
                        top_results.pop();
                    }
                }
            }
        }
    }

    std::vector<DistPair> results;
    results.reserve(top_results.size());
    while (!top_results.empty()) {
        results.push_back(top_results.top());
        top_results.pop();
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
        float max_dist = std::numeric_limits<float>::lowest();
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

void HNSWIndexSQ8::add(VectorId id, VectorView vector) {
    if (vector.size() != dimension_) {
        throw std::invalid_argument("Vector dimension does not match index dimension");
    }

    uint32_t new_id = static_cast<uint32_t>(count_);
    ids_.push_back(id);
    raw_storage_.insert(raw_storage_.end(), vector.begin(), vector.end());

    std::size_t q_offset = quantized_storage_.size();
    quantized_storage_.resize(q_offset + dimension_);
    quantizer_.encode(vector, std::span<uint8_t>(quantized_storage_.data() + q_offset, dimension_));

    int new_level = get_random_level();
    levels_.push_back(new_level);

    layer0_graph_.resize((new_id + 1) * (max_m0_ + 1), 0);
    if (new_level > 0) {
        upper_graph_.emplace_back(static_cast<std::size_t>(new_level) * (max_m_ + 1), 0);
    } else {
        upper_graph_.emplace_back();
    }

    count_++;

    if (new_id == 0) {
        enter_point_ = 0;
        max_level_ = new_level;
        return;
    }

    uint32_t curr_ep = enter_point_;
    float curr_dist = compute_quantized_distance(vector, curr_ep);

    for (int lc = max_level_; lc > new_level; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t neighbor : get_links(curr_ep, lc)) {
                float dist = compute_quantized_distance(vector, neighbor);
                if (dist < curr_dist) {
                    curr_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    int start_level = std::min(max_level_, new_level);
    std::vector<uint32_t> enter_points = {curr_ep};

    for (int lc = start_level; lc >= 0; --lc) {
        auto candidates = search_layer(vector, enter_points, ef_construction_, lc);
        enter_points.clear();
        for (const auto& [d, node] : candidates) {
            enter_points.push_back(node);
        }

        std::size_t target_m = (lc == 0) ? max_m0_ : max_m_;
        if (candidates.size() > target_m) {
            candidates.resize(target_m);
        }

        for (const auto& [d, neighbor] : candidates) {
            add_link(new_id, neighbor, lc);
            add_link(neighbor, new_id, lc);
        }
    }

    if (new_level > max_level_) {
        max_level_ = new_level;
        enter_point_ = new_id;
    }
}

std::vector<SearchResult> HNSWIndexSQ8::search(VectorView query, std::size_t k) const {
    if (count_ == 0 || k == 0 || query.size() != dimension_) {
        return {};
    }

    uint32_t curr_ep = enter_point_;
    float curr_dist = compute_quantized_distance(query, curr_ep);

    for (int lc = max_level_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t neighbor : get_links(curr_ep, lc)) {
                float dist = compute_quantized_distance(query, neighbor);
                if (dist < curr_dist) {
                    curr_dist = dist;
                    curr_ep = neighbor;
                    changed = true;
                }
            }
        }
    }

    std::size_t num_candidates = static_cast<std::size_t>(static_cast<float>(std::max(ef_search_, k)) * re_rank_factor_);
    auto candidates = search_layer(query, {curr_ep}, num_candidates, 0);

    std::vector<SearchResult> re_ranked;
    re_ranked.reserve(candidates.size());
    for (const auto& [approx_dist, node_id] : candidates) {
        float exact_dist = compute_exact_distance(query, node_id);
        re_ranked.push_back({ids_[node_id], exact_dist});
    }

    std::sort(re_ranked.begin(), re_ranked.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });

    if (re_ranked.size() > k) {
        re_ranked.resize(k);
    }

    return re_ranked;
}

void HNSWIndexSQ8::save(std::ostream& os) const {
    Serializer::serialize(*this, os);
}

void HNSWIndexSQ8::load(std::istream& is) {
    *this = Serializer::deserialize_hnsw_sq8(is);
}

} // namespace engine
