#include "engine/quantization/scalar_quantizer.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace engine {

    ScalarQuantizer::ScalarQuantizer(std::size_t dimension)
        : dimension_(dimension) {
    }

    void ScalarQuantizer::train(std::span<const float> training_data) {
        if (dimension_ == 0 || training_data.empty() || (training_data.size() % dimension_ != 0)) {
            throw std::invalid_argument("Invalid training data or dimension");
        }

        min_values_.resize(dimension_);
        scale_values_.resize(dimension_);
        std::vector<float> max_values(dimension_);

        const std::size_t num_vectors = training_data.size() / dimension_;

        for (std::size_t d = 0; d < dimension_; ++d) {
            min_values_[d] = training_data[d];
            max_values[d] = training_data[d];
        }

        for (std::size_t v = 1; v < num_vectors; ++v) {
            std::size_t d = 0;
            auto vec = training_data.subspan(v * dimension_, dimension_);
            for (d = 0; d + 7 < dimension_; d += 8) {
                __m256 cur_v = _mm256_loadu_ps(&vec[d]);
                __m256 current_min = _mm256_loadu_ps(&min_values_[d]);
                __m256 current_max = _mm256_loadu_ps(&max_values[d]);
                __m256 new_min = _mm256_min_ps(current_min, cur_v);
                __m256 new_max = _mm256_max_ps(current_max, cur_v);
                _mm256_storeu_ps(&min_values_[d], new_min);
                _mm256_storeu_ps(&max_values[d], new_max);
            }
            for (; d < dimension_; ++d) {
                min_values_[d] = std::min(min_values_[d], vec[d]);
                max_values[d] = std::max(max_values[d], vec[d]);
            }
        }

        for (std::size_t d = 0; d < dimension_; ++d) {
            const float range = max_values[d] - min_values_[d];
            if (range < 1e-7f) {
                scale_values_[d] = 1.0f;
            } else {
                scale_values_[d] = range / 255.0f;
            }
        }
        is_trained_ = true;
    }

    void ScalarQuantizer::encode(std::span<const float> src, std::span<uint8_t> dst) const {
        if (!is_trained_) {
            throw std::runtime_error("Quantizer is not trained");
        }
        if (src.size() != dimension_ || dst.size() != dimension_) {
            throw std::invalid_argument("Source and destination sizes must match quantizer dimension");
        }
        for (std::size_t d = 0; d < dimension_; ++d) {
            const float scaled = ((src[d] - min_values_[d]) / scale_values_[d]) + 0.5f;
            const float clamped = std::clamp(scaled, 0.0f, 255.0f);
            dst[d] = static_cast<uint8_t>(clamped);
        }
    }

    void ScalarQuantizer::decode(std::span<const uint8_t> src, std::span<float> dst) const {
        if (!is_trained_) {
            throw std::runtime_error("Quantizer is not trained");
        }
        if (src.size() != dimension_ || dst.size() != dimension_) {
            throw std::invalid_argument("Source and destination sizes must match quantizer dimension");
        }
        for (std::size_t d = 0; d < dimension_; ++d) {
            const float unscaled = min_values_[d] + (static_cast<float>(src[d]) * scale_values_[d]);
            dst[d] = unscaled;
        }
    }

    std::vector<uint8_t> ScalarQuantizer::encode(std::span<const float> src) const {
        std::vector<uint8_t> dst(dimension_);
        encode(src, dst);
        return dst;
    }

    std::vector<float> ScalarQuantizer::decode(std::span<const uint8_t> src) const {
        std::vector<float> dst(dimension_);
        decode(src, dst);
        return dst;
    }

    std::size_t ScalarQuantizer::dimension() const noexcept {
        return dimension_;
    }

    bool ScalarQuantizer::is_trained() const noexcept {
        return is_trained_;
    }

    const std::vector<float>& ScalarQuantizer::min_values() const noexcept {
        return min_values_;
    }

    const std::vector<float>& ScalarQuantizer::scale_values() const noexcept {
        return scale_values_;
    }

}