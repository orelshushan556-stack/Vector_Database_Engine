#include "engine/distance/metrics.hpp"
#include "engine/distance/simd_metrics.hpp"
#include "engine/types.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) <= epsilon * (std::fabs(a) + std::fabs(b) + 1.0f);
}

engine::AlignedVector<float> generate_random_vector(std::size_t dim, float min_val = -10.0f, float max_val = 10.0f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    engine::AlignedVector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

void test_metrics_across_dimensions() {
    const std::vector<std::size_t> test_dims = {
        1, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 256, 512, 1024, 1536
    };

    for (std::size_t dim : test_dims) {
        auto a = generate_random_vector(dim);
        auto b = generate_random_vector(dim);

        engine::VectorView va{a.data(), a.size()};
        engine::VectorView vb{b.data(), b.size()};

        // 1. L2 Distance Squared
        float l2_scalar = engine::l2_distance_squared(va, vb);
        float l2_simd   = engine::l2_distance_squared_avx2(va, vb);
        assert(approx_equal(l2_scalar, l2_simd) && "L2 AVX2 mismatch!");

        // 2. Dot Product
        float dot_scalar = engine::dot_product(va, vb);
        float dot_simd   = engine::dot_product_avx2(va, vb);
        assert(approx_equal(dot_scalar, dot_simd) && "Dot Product AVX2 mismatch!");

        // 3. Cosine Distance
        float cos_scalar = engine::cosine_distance(va, vb);
        float cos_simd   = engine::cosine_distance_avx2(va, vb);
        assert(approx_equal(cos_scalar, cos_simd) && "Cosine Distance AVX2 mismatch!");
    }

    std::cout << "[PASS] All metric parity tests passed across dimensions.\n";
}

void test_edge_cases() {
    auto v1 = generate_random_vector(32);
    auto v2 = generate_random_vector(64);
    engine::VectorView v_empty{};
    engine::VectorView va{v1.data(), v1.size()};
    engine::VectorView vb{v2.data(), v2.size()};

    // Dimension mismatch
    bool threw = false;
    try {
        engine::l2_distance_squared_avx2(va, vb);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "Failed to throw on dimension mismatch (L2)!");

    threw = false;
    try {
        engine::dot_product_avx2(va, vb);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "Failed to throw on dimension mismatch (Dot)!");

    // Empty vector
    threw = false;
    try {
        engine::cosine_distance_avx2(v_empty, v_empty);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "Failed to throw on empty vector!");

    // Zero-norm vector in cosine
    engine::AlignedVector<float> zero_vec(32, 0.0f);
    engine::VectorView vz{zero_vec.data(), zero_vec.size()};
    threw = false;
    try {
        engine::cosine_distance_avx2(va, vz);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "Failed to throw on zero-norm vector (Cosine)!");

    std::cout << "[PASS] All edge case and validation tests passed.\n";
}

} // namespace

int main() {
    std::cout << "Running distance metric test suite...\n";
    test_metrics_across_dimensions();
    test_edge_cases();
    std::cout << "All distance tests completed successfully.\n";
    return 0;
}
