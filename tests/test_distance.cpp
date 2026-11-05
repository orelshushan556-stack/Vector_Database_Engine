#include <gtest/gtest.h>

#include "engine/distance/metrics.hpp"
#include "engine/distance/simd_metrics.hpp"
#include "engine/types.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
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

TEST(DistanceTest, SimdKernelsMatchScalarReferenceAcrossDimensions) {
    // Covers dimensions below, at, and just above every unroll boundary (8, 16, 32) plus
    // sizes that exercise the scalar tail.
    const std::vector<std::size_t> test_dims = {
        1, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 256, 512, 1024, 1536
    };

    for (std::size_t dim : test_dims) {
        SCOPED_TRACE("dimension " + std::to_string(dim));
        auto a = generate_random_vector(dim);
        auto b = generate_random_vector(dim);

        const engine::VectorView va{a.data(), a.size()};
        const engine::VectorView vb{b.data(), b.size()};

        EXPECT_TRUE(approx_equal(engine::l2_distance_squared(va, vb), engine::l2_distance_squared_avx2(va, vb)))
            << "L2 AVX2 mismatch";
        EXPECT_TRUE(approx_equal(engine::dot_product(va, vb), engine::dot_product_avx2(va, vb)))
            << "Dot product AVX2 mismatch";
        EXPECT_TRUE(approx_equal(engine::cosine_distance(va, vb), engine::cosine_distance_avx2(va, vb)))
            << "Cosine AVX2 mismatch";
    }
}

TEST(DistanceTest, KernelsRejectInvalidInput) {
    auto v1 = generate_random_vector(32);
    auto v2 = generate_random_vector(64);
    const engine::VectorView v_empty{};
    const engine::VectorView va{v1.data(), v1.size()};
    const engine::VectorView vb{v2.data(), v2.size()};

    EXPECT_THROW(static_cast<void>(engine::l2_distance_squared_avx2(va, vb)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(engine::dot_product_avx2(va, vb)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(engine::cosine_distance_avx2(v_empty, v_empty)), std::invalid_argument);

    engine::AlignedVector<float> zero_vec(32, 0.0f);
    const engine::VectorView vz{zero_vec.data(), zero_vec.size()};
    EXPECT_THROW(static_cast<void>(engine::cosine_distance_avx2(va, vz)), std::invalid_argument);
}

} // namespace
