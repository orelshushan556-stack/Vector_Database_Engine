#include "engine/index/flat_index.hpp"
#include "engine/distance/simd_metrics.hpp"

#include <queue>
#include <algorithm>
#include <stdexcept>

namespace engine {

    FlatIndex::FlatIndex(std::size_t dimension, MetricType metric)
        : dimension_(dimension), metric_(metric) {
    }

    void FlatIndex::add(VectorId id, VectorView vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument("Vector dimension does not match index dimension.");
        }

        ids_.push_back(id);
        storage_.insert(storage_.end(), vector.begin(), vector.end());
    }

    std::vector<SearchResult> FlatIndex::search(VectorView query, std::size_t k) const {
        if (query.size() != dimension_) {
            throw std::invalid_argument("Vector dimension does not match index dimension.");
        }
        if (k == 0 || ids_.empty()) {
            return {};
        }

        auto comp = [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        };

        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)> heap(comp);

        for (std::size_t i = 0; i < ids_.size(); ++i) {
            VectorView target(&storage_[i * dimension_], dimension_);
            float dist = 0.0f;

            switch (metric_) {
                case MetricType::L2:
                    dist = l2_distance_squared_avx2(query, target);
                    break;
                case MetricType::Cosine:
                    dist = cosine_distance_avx2(query, target);
                    break;
                case MetricType::DotProduct:
                    dist = -dot_product_avx2(query, target);
                    break;
            }

            if (heap.size() < k) {
                heap.push(SearchResult{ids_[i], dist});
            } else if (dist < heap.top().distance) {
                heap.pop();
                heap.push(SearchResult{ids_[i], dist});
            }
        }

        std::vector<SearchResult> results;
        results.reserve(heap.size());

        while (!heap.empty()) {
            SearchResult item = heap.top();
            if (metric_ == MetricType::DotProduct) {
                item.distance = -item.distance;
            }
            results.push_back(item);
            heap.pop();
        }

        std::reverse(results.begin(), results.end());

        return results;
    }

    std::size_t FlatIndex::size() const noexcept {
        return ids_.size();
    }

    std::size_t FlatIndex::dimension() const noexcept {
        return dimension_;
    }

    MetricType FlatIndex::metric() const noexcept {
        return metric_;
    }

    void FlatIndex::reserve(std::size_t capacity) {
        ids_.reserve(capacity);
        storage_.reserve(capacity * dimension_);
    }

    void FlatIndex::clear() {
        ids_.clear();
        storage_.clear();
    }

}