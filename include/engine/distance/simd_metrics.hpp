#pragma once

#include "engine/types.hpp"
#include <immintrin.h>

namespace engine {

    inline __m256 reduce_accumulators(__m256 s0, __m256 s1, __m256 s2, __m256 s3) noexcept {
        __m256 s01 = _mm256_add_ps(s0, s1);
        __m256 s23 = _mm256_add_ps(s2, s3);
        return _mm256_add_ps(s01, s23);
    }

    float l2_distance_squared_avx2(VectorView a, VectorView b);
    float dot_product_avx2(VectorView a, VectorView b);
    float cosine_distance_avx2(VectorView a, VectorView b);

}