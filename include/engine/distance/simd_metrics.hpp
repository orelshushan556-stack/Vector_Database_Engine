#pragma once

#include "engine/types.hpp"

namespace engine {

    float l2_distance_squared_avx2(VectorView a, VectorView b);
    float dot_product_avx2(VectorView a, VectorView b);
    float cosine_distance_avx2(VectorView a, VectorView b);

}