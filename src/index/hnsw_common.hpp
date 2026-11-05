#pragma once

// Graph traversal and maintenance primitives shared by HNSWIndex and HNSWIndexSQ8.
// The index classes supply distance, link-access and prefetch policies as callables.

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Neighbor selection heuristic (Algorithm 4 of the HNSW paper, with keepPrunedConnections).
// A candidate is kept only if it is closer to the base node than to every neighbor already
// kept; this spreads links across directions and retains the long-range edges that bridge
// clusters. If fewer than m candidates survive, pruned candidates fill the remaining slots
// in order of increasing distance, so a node never gets fewer links than are available.
//   ascending      -> candidates sorted by ascending distance to the base node
//   pair_dist(a,b) -> distance between two stored nodes
template <class PairDist>
std::vector<DistPair> select_neighbors_heuristic(const std::vector<DistPair>& ascending,
                                                 std::size_t m,
                                                 PairDist&& pair_dist) {
    if (ascending.size() <= m) {
        return ascending;
    }

    std::vector<DistPair> kept;
    kept.reserve(m);
    std::vector<DistPair> pruned;

    for (const DistPair& cand : ascending) {
        if (kept.size() >= m) {
            break;
        }
        bool diverse = true;
        for (const DistPair& chosen : kept) {
            if (pair_dist(cand.second, chosen.second) < cand.first) {
                diverse = false;
                break;
            }
        }
        if (diverse) {
            kept.push_back(cand);
        } else {
            pruned.push_back(cand);
        }
    }

    for (std::size_t i = 0; i < pruned.size() && kept.size() < m; ++i) {
        kept.push_back(pruned[i]);
    }
    return kept;
}

// Add dst to the adjacency block at graph_data[offset] (slot 0 holds the degree).
// If the block is full, the existing neighbors plus dst are re-selected with the diversity
// heuristic, so the owner keeps a well-spread set of links instead of silently dropping the
// new edge or only the farthest one.
//   dist_from_owner(node) -> distance between the block owner and node
//   pair_dist(a, b)       -> distance between two stored nodes
template <class DistFromOwner, class PairDist>
void add_link_impl(uint32_t* graph_data, std::size_t offset, std::size_t max_links,
                   uint32_t dst, DistFromOwner&& dist_from_owner, PairDist&& pair_dist) {
    uint32_t& degree = graph_data[offset];

    if (degree < max_links) {
        graph_data[offset + 1 + degree] = dst;
        ++degree;
        return;
    }

    std::vector<DistPair> pool;
    pool.reserve(static_cast<std::size_t>(degree) + 1);
    pool.emplace_back(dist_from_owner(dst), dst);
    for (uint32_t i = 0; i < degree; ++i) {
        const uint32_t node = graph_data[offset + 1 + i];
        pool.emplace_back(dist_from_owner(node), node);
    }
    std::sort(pool.begin(), pool.end());

    const std::vector<DistPair> kept = select_neighbors_heuristic(pool, max_links, pair_dist);
    degree = static_cast<uint32_t>(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
        graph_data[offset + 1 + i] = kept[i].second;
    }
}

// Link new_id bidirectionally to up to m neighbors chosen from the ascending candidates.
template <class PairDist, class AddLinkFn>
void connect_new_node(uint32_t new_id, const std::vector<DistPair>& ascending,
                      std::size_t m, PairDist&& pair_dist, AddLinkFn&& add_link) {
    for (const DistPair& chosen : select_neighbors_heuristic(ascending, m, pair_dist)) {
        add_link(new_id, chosen.second);
        add_link(chosen.second, new_id);
    }
}

} // namespace engine::hnsw_detail
