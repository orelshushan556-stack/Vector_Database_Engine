#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace engine {

    class ScalarQuantizer {
    public:
        explicit ScalarQuantizer(std::size_t dimension);

        void train(std::span<const float> training_data);

        void encode(std::span<const float> src, std::span<uint8_t> dst) const;
        void decode(std::span<const uint8_t> src, std::span<float> dst) const;

        [[nodiscard]] std::vector<uint8_t> encode(std::span<const float> src) const;
        [[nodiscard]] std::vector<float> decode(std::span<const uint8_t> src) const;

        [[nodiscard]] std::size_t dimension() const noexcept;
        [[nodiscard]] bool is_trained() const noexcept;

        [[nodiscard]] const std::vector<float>& min_values() const noexcept;
        [[nodiscard]] const std::vector<float>& scale_values() const noexcept;

    private:
        std::size_t dimension_;
        bool is_trained_{false};
        std::vector<float> min_values_;
        std::vector<float> scale_values_;
    };

} // namespace engine