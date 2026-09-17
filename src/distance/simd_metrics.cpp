#include "engine/distance/simd_metrics.hpp"

#include <immintrin.h>
#include <cmath>
#include <stdexcept>

namespace engine {
    namespace {
        inline float horizontal_add(__m256 v) noexcept {
            __m128 low_part    = _mm256_castps256_ps128(v);
            __m128 high_part   = _mm256_extractf128_ps(v, 1);
            __m128 sum_r       = _mm_add_ps(low_part, high_part);

            __m128 high_to_low = _mm_movehl_ps(sum_r, sum_r);
            __m128 sum_2       = _mm_add_ps(high_to_low, sum_r);

            __m128 sum_1       = _mm_movehdup_ps(sum_2);
            __m128 final_sum   = _mm_add_ss(sum_2, sum_1);

            return _mm_cvtss_f32(final_sum);
        }
    }

    float l2_distance_squared_avx2(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }

        const std::size_t size = a.size();
        std::size_t i = 0;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (; i + 31 < size; i += 32) {
            __m256 a0 = _mm256_loadu_ps(&a[i]);
            __m256 b0 = _mm256_loadu_ps(&b[i]);
            __m256 diff0 = _mm256_sub_ps(a0, b0);
            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

            __m256 a1 = _mm256_loadu_ps(&a[i + 8]);
            __m256 b1 = _mm256_loadu_ps(&b[i + 8]);
            __m256 diff1 = _mm256_sub_ps(a1, b1);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

            __m256 a2 = _mm256_loadu_ps(&a[i + 16]);
            __m256 b2 = _mm256_loadu_ps(&b[i + 16]);
            __m256 diff2 = _mm256_sub_ps(a2, b2);
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

            __m256 a3 = _mm256_loadu_ps(&a[i + 24]);
            __m256 b3 = _mm256_loadu_ps(&b[i + 24]);
            __m256 diff3 = _mm256_sub_ps(a3, b3);
            sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        }

        __m256 sum = reduce_accumulators(sum0, sum1, sum2, sum3);

        for (; i + 7 < size; i += 8) {
            __m256 a_chunk = _mm256_loadu_ps(&a[i]);
            __m256 b_chunk = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(a_chunk, b_chunk);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        float total = horizontal_add(sum);

        for (; i < size; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }

        return total;
    }

    float dot_product_avx2(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        const std::size_t size = a.size();
        std::size_t i = 0;

        for (; i + 31 < size; i += 32) {
            __m256 a0 = _mm256_loadu_ps(&a[i]);
            __m256 b0 = _mm256_loadu_ps(&b[i]);
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);

            __m256 a1 = _mm256_loadu_ps(&a[i + 8]);
            __m256 b1 = _mm256_loadu_ps(&b[i + 8]);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);

            __m256 a2 = _mm256_loadu_ps(&a[i + 16]);
            __m256 b2 = _mm256_loadu_ps(&b[i + 16]);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);

            __m256 a3 = _mm256_loadu_ps(&a[i + 24]);
            __m256 b3 = _mm256_loadu_ps(&b[i + 24]);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }

        __m256 sum = reduce_accumulators(sum0, sum1, sum2, sum3);

        for (; i + 7 < size; i += 8) {
            __m256 a_chunk = _mm256_loadu_ps(&a[i]);
            __m256 b_chunk = _mm256_loadu_ps(&b[i]);
            sum = _mm256_fmadd_ps(a_chunk, b_chunk, sum);
        }

        float total = horizontal_add(sum);

        for (; i < size; ++i) {
            total += a[i] * b[i];
        }

        return total;
    }

    float cosine_distance_avx2(VectorView a, VectorView b) {
        if (a.empty() || a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be non-empty and have matching dimensions.");
        }

        __m256 dot0 = _mm256_setzero_ps();
        __m256 dot1 = _mm256_setzero_ps();
        __m256 na0  = _mm256_setzero_ps();
        __m256 na1  = _mm256_setzero_ps();
        __m256 nb0  = _mm256_setzero_ps();
        __m256 nb1  = _mm256_setzero_ps();

        const std::size_t size = a.size();
        std::size_t i = 0;

        for (; i + 15 < size; i += 16) {
            __m256 a0 = _mm256_loadu_ps(&a[i]);
            __m256 b0 = _mm256_loadu_ps(&b[i]);
            dot0 = _mm256_fmadd_ps(a0, b0, dot0);
            na0  = _mm256_fmadd_ps(a0, a0, na0);
            nb0  = _mm256_fmadd_ps(b0, b0, nb0);

            __m256 a1 = _mm256_loadu_ps(&a[i + 8]);
            __m256 b1 = _mm256_loadu_ps(&b[i + 8]);
            dot1 = _mm256_fmadd_ps(a1, b1, dot1);
            na1  = _mm256_fmadd_ps(a1, a1, na1);
            nb1  = _mm256_fmadd_ps(b1, b1, nb1);
        }

        __m256 sum_dot = _mm256_add_ps(dot0, dot1);
        __m256 sum_na  = _mm256_add_ps(na0, na1);
        __m256 sum_nb  = _mm256_add_ps(nb0, nb1);

        for (; i + 7 < size; i += 8) {
            __m256 a_chunk = _mm256_loadu_ps(&a[i]);
            __m256 b_chunk = _mm256_loadu_ps(&b[i]);
            sum_dot = _mm256_fmadd_ps(a_chunk, b_chunk, sum_dot);
            sum_na  = _mm256_fmadd_ps(a_chunk, a_chunk, sum_na);
            sum_nb  = _mm256_fmadd_ps(b_chunk, b_chunk, sum_nb);
        }

        float dot_val = horizontal_add(sum_dot);
        float na_val  = horizontal_add(sum_na);
        float nb_val  = horizontal_add(sum_nb);

        for (; i < size; ++i) {
            dot_val += a[i] * b[i];
            na_val  += a[i] * a[i];
            nb_val  += b[i] * b[i];
        }

        const float denom = std::sqrt(na_val * nb_val);
        if (denom == 0.0f) {
            throw std::invalid_argument("Cannot compute cosine distance for zero-norm vector.");
        }

        return 1.0f - (dot_val / denom);
    }
}