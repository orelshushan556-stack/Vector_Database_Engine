#pragma once

// Graph traversal and maintenance primitives shared by HNSWIndex and HNSWIndexSQ8.
// The index classes supply distance, link-access and prefetch policies as callables.

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace engine::hnsw_detail {

using DistPair = std::pair<float, uint32_t>;
using MaxHeap = std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>>;
using MinHeap = std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>>;

struct VisitedList {
    std::vector<uint32_t> tags;
    uint32_t current_tag{0};

    void begin(std::size_t node_count) {
        if (tags.size() < node_count) {
            tags.resize(node_count, 0);
        }
        ++current_tag;
        if (current_tag == 0) {
            std::fill(tags.begin(), tags.end(), 0);
            current_tag = 1;
        }
    }

    [[nodiscard]] bool visited(uint32_t id) const { return tags[id] == current_tag; }
    void mark(uint32_t id) { tags[id] = current_tag; }
};

// search_layer is never re-entered on a thread, so one list per thread is enough
// for both index types.
inline VisitedList& thread_visited() {
    thread_local VisitedList list;
    return list;
}

// Prefetch up to four cache lines starting at p.
inline void prefetch_bytes(const void* p, std::size_t bytes) {
    constexpr std::size_t line = 64;
    constexpr std::size_t max_lines = 4;
    const char* c = static_cast<const char*>(p);
    const std::size_t lines = std::min((bytes + line - 1) / line, max_lines);
    for (std::size_t i = 0; i < lines; ++i) {
        _mm_prefetch(c + i * line, _MM_HINT_T0);
    }
}

// Best-first search on a single layer.
//   links(node)    -> span of neighbor ids of node on this layer
//   dist(node)     -> distance from the query to node
//   prefetch(node) -> issue cache prefetches for node's vector data
template <class LinksFn, class DistFn, class PrefetchFn>
MaxHeap search_layer_impl(std::span<const uint32_t> enter_points,
                          std::size_t ef,
                          std::size_t node_count,
                          LinksFn&& links,
                          DistFn&& dist,
                          PrefetchFn&& prefetch) {
    std::vector<DistPair> top_container;
    top_container.reserve(ef + 1);
    MaxHeap top(std::less<DistPair>{}, std::move(top_container));

    std::vector<DistPair> cand_container;
    cand_container.reserve(ef * 2);
    MinHeap candidates(std::greater<DistPair>{}, std::move(cand_container));

    VisitedList& visited = thread_visited();
    visited.begin(node_count);

    for (uint32_t ep : enter_points) {
        const float d = dist(ep);
        visited.mark(ep);
        candidates.emplace(d, ep);
        top.emplace(d, ep);
    }

    while (!candidates.empty()) {
        const auto [curr_dist, curr_node] = candidates.top();
        candidates.pop();

        if (curr_dist > top.top().first && top.size() >= ef) {
            break;
        }

        const auto neighbors = links(curr_node);

        // Issue all loads first so the memory latency of the pointer hops overlaps.
        for (uint32_t id : neighbors) {
            if (!visited.visited(id)) {
                prefetch(id);
            }
        }

        for (uint32_t id : neighbors) {
            if (visited.visited(id)) {
                continue;
            }
            visited.mark(id);

            const float d = dist(id);
            if (top.size() < ef || d < top.top().first) {
                candidates.emplace(d, id);
                top.emplace(d, id);
                if (top.size() > ef) {
                    top.pop();
                }
            }
        }
    }

    return top;
}

// Greedy hill-climb from ep down through layers (from_level .. stop_level + 1).
template <class LinksFn, class DistFn>
void greedy_descend(uint32_t& ep, float& ep_dist, int from_level, int stop_level,
                    LinksFn&& links, DistFn&& dist) {
    for (int lc = from_level; lc > stop_level; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t neighbor : links(ep, lc)) {
                const float d = dist(neighbor);
                if (d < ep_dist) {
                    ep_dist = d;
                    ep = neighbor;
                    changed = true;
                }
            }
        }
    }
}

// Drain a max-heap into a vector sorted by ascending distance.
inline std::vector<DistPair> drain_ascending(MaxHeap&& heap) {
    std::vector<DistPair> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Append dst to the adjacency block at graph_data[offset] (slot 0 holds the degree).
// When full, replace the farthest existing neighbor if dst is closer.
//   dist_from_src(node) -> distance between the block owner and node
template <class DistFromSrc>
void add_link_impl(uint32_t* graph_data, std::size_t offset, std::size_t max_links,
                   uint32_t dst, DistFromSrc&& dist_from_src) {
    uint32_t& degree = graph_data[offset];

    if (degree < max_links) {
        graph_data[offset + 1 + degree] = dst;
        ++degree;
        return;
    }

    float max_dist = std::numeric_limits<float>::lowest();
    uint32_t worst_idx = 0;
    const float dist_new = dist_from_src(dst);

    for (uint32_t i = 0; i < degree; ++i) {
        const float d = dist_from_src(graph_data[offset + 1 + i]);
        if (d > max_dist) {
            max_dist = d;
            worst_idx = i;
        }
    }

    if (dist_new < max_dist) {
        graph_data[offset + 1 + worst_idx] = dst;
    }
}

// Link new_id bidirectionally to the closest target_m of the ascending candidates.
template <class AddLinkFn>
void connect_new_node(uint32_t new_id, const std::vector<DistPair>& ascending,
                      std::size_t target_m, AddLinkFn&& add_link) {
    const std::size_t n = std::min(ascending.size(), target_m);
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t neighbor = ascending[i].second;
        add_link(new_id, neighbor);
        add_link(neighbor, new_id);
    }
}

} // namespace engine::hnsw_detail
