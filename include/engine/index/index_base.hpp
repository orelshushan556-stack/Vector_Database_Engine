#pragma once

#include "engine/types.hpp"
#include <vector>
#include <cstddef>

namespace engine {

    enum class MetricType {
        L2,
        DotProduct,
        Cosine
    };

    class IndexBase {
    public:
        virtual ~IndexBase() = default;

        virtual void add(VectorId id, VectorView vector) = 0;
        virtual std::vector<SearchResult> search(VectorView query, std::size_t k) const = 0;

        [[nodiscard]] virtual std::size_t size() const noexcept = 0;
        [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
        [[nodiscard]] virtual MetricType metric() const noexcept = 0;
    };

}