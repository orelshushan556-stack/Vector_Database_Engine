#include "engine/index/quantized_flat_index.hpp"
#include "engine/distance/simd_metrics.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace engine {

    QuantizedFlatIndex::QuantizedFlatIndex(std::size_t dimension)
        : dimension_(dimension), quantizer_(dimension) {
    }

    void QuantizedFlatIndex::train(std::span<const float> training_data) {
        quantizer_.train(training_data);
    }

    void QuantizedFlatIndex::add(std::span<const float> vector) {
        if (!is_trained()) {
            throw std::runtime_error("QuantizedFlatIndex::add");
        }
        if (vector.size() != dimension_) {
            throw std::invalid_argument("QuantizedFlatIndex::add");
        }
        const std::size_t offset = storage_.size();
        storage_.resize(offset + dimension_);
        std::span<uint8_t> dst_span(storage_.data() + offset, dimension_);
        quantizer_.encode(vector, dst_span);
        count_++;
    }

    void QuantizedFlatIndex::add_batch(std::span<const float> vectors) {
        if (!is_trained()) {
            throw std::runtime_error("QuantizedFlatIndex::add_batch");
        }
        if (vectors.empty() || vectors.size() % dimension_ != 0) {
            throw std::invalid_argument("QuantizedFlatIndex::add_batch");
        }

        const std::size_t num_vectors = vectors.size() / dimension_;
        const std::size_t old_size = storage_.size();

        storage_.resize(old_size + vectors.size());

        for (std::size_t i = 0; i < num_vectors; ++i) {
            auto src = vectors.subspan(i * dimension_, dimension_);
            std::span<uint8_t> dst(storage_.data() + old_size + (i * dimension_), dimension_);
            quantizer_.encode(src, dst);
        }

        count_ += num_vectors;
    }

    std::vector<SearchResult> QuantizedFlatIndex::search(std::span<const float> query, std::size_t k) const {
        if (!is_trained()) {
            throw std::runtime_error("QuantizedFlatIndex::search");
        }
        if (query.size() != dimension_) {
            throw std::invalid_argument("QuantizedFlatIndex::search");
        }
        if (k == 0 || count_ == 0) {
            return {};
        }

        const auto& scales = quantizer_.scale_values();
        const auto& mins = quantizer_.min_values();

        auto comp = [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)> heap(comp);

        for (std::size_t i = 0; i < count_; ++i) {
            std::span<const uint8_t> target{storage_.data() + (i * dimension_), dimension_};
            float dist = quantized_l2_distance_sq_avx2(query, target, scales, mins);

            if (heap.size() < k) {
                heap.push(SearchResult{i, dist});
            } else if (dist < heap.top().distance) {
                heap.pop();
                heap.push(SearchResult{i, dist});
            }
        }

        std::vector<SearchResult> results(heap.size());
        for (std::size_t i = heap.size(); i > 0; --i) {
            results[i - 1] = heap.top();
            heap.pop();
        }

        return results;
    }

    std::size_t QuantizedFlatIndex::dimension() const noexcept {
        return dimension_;
    }

    std::size_t QuantizedFlatIndex::size() const noexcept {
        return count_;
    }

    bool QuantizedFlatIndex::empty() const noexcept {
        return count_ == 0;
    }

    bool QuantizedFlatIndex::is_trained() const noexcept {
        return quantizer_.is_trained();
    }

    void QuantizedFlatIndex::clear() noexcept {
        storage_.clear();
        count_ = 0;
    }

    const ScalarQuantizer& QuantizedFlatIndex::quantizer() const noexcept {
        return quantizer_;
    }

    std::span<const uint8_t> QuantizedFlatIndex::get_quantized_vector(std::size_t id) const {
        if (id >= count_) {
            throw std::out_of_range("QuantizedFlatIndex::get_quantized_vector");
        }
        return {storage_.data() + (id * dimension_), dimension_};
    }

}
