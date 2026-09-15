#pragma once

#include "engine/types.hpp"

namespace engine {

    float l2_distance_squared(VectorView a, VectorView b);
    float dot_product(VectorView a, VectorView b);
    float cosine_distance(VectorView a, VectorView b);
    float L2_norm(VectorView a);

}