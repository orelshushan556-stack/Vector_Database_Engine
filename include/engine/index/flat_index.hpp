#pragma once

#include "engine/index/index_base.hpp"
#include "engine/types.hpp"

#include <span>
#include <vector>
#include <cstddef>

namespace engine {

    class FlatIndex : public IndexBase {
    public:
        FlatIndex(std::size_t dimension, MetricType metric);
        ~FlatIndex() override = default;

        void add(VectorId id, VectorView vector) override;
        [[nodiscard]] std::vector<SearchResult> search(VectorView query, std::size_t k) const override;

        [[nodiscard]] std::size_t size() const noexcept override;
        [[nodiscard]] std::size_t dimension() const noexcept override;
        [[nodiscard]] MetricType metric() const noexcept override;

        void reserve(std::size_t capacity);
        void clear();

        [[nodiscard]] std::span<const float> raw_storage() const noexcept;
        [[nodiscard]] std::span<const VectorId> raw_ids() const noexcept;
        void populate(std::vector<VectorId> ids, AlignedVector<float> storage);

    private:
        std::size_t dimension_;
        MetricType metric_;
        AlignedVector<float> storage_;
        std::vector<VectorId> ids_;
    };

}