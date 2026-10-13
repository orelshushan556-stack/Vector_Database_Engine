#pragma once

#include "engine/quantization/scalar_quantizer.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include "engine/types.hpp"
namespace engine {



    class QuantizedFlatIndex {
    public:
        explicit QuantizedFlatIndex(std::size_t dimension);

        void train(std::span<const float> training_data);

        void add(std::span<const float> vector);
        void add_batch(std::span<const float> vectors);

        [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

        [[nodiscard]] std::size_t dimension() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] bool is_trained() const noexcept;

        void clear() noexcept;

        [[nodiscard]] const ScalarQuantizer& quantizer() const noexcept;
        [[nodiscard]] std::span<const uint8_t> get_quantized_vector(std::size_t id) const;

    private:
        std::size_t dimension_;
        ScalarQuantizer quantizer_;
        std::vector<uint8_t> storage_;
        std::size_t count_{0};
    };

} // namespace engine