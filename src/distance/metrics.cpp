#include "engine/distance/metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace engine {

    float l2_distance_squared(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }

        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    float dot_product(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float prod = a[i] * b[i];
            sum += prod;
        }
        return sum;
    }

    float l2_norm(VectorView a) {
        if (a.empty()) {
            throw std::invalid_argument("Vector must be non-empty.");
        }
        float sum = 0.0f;
        for (float val : a) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    float cosine_distance(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }

        const float dot_p = dot_product(a, b);
        const float norm_a = l2_norm(a);
        const float norm_b = l2_norm(b);

        const float denom = norm_a * norm_b;
        if (denom == 0.0f) {
            throw std::invalid_argument("Cannot compute cosine distance for zero-norm vector.");
        }

        const float similarity = dot_p / denom;
        return 1.0f - similarity;
    }

}