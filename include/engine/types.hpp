#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <new>

namespace engine {

    using VectorId = uint64_t;
    using VectorView = std::span<const float>;
    using MutableVectorView = std::span<float>;

    struct SearchResult {
        VectorId id;
        float distance;

        constexpr auto operator<=>(const SearchResult& other) const = default;
    };

    template <typename T, std::size_t Alignment = 32>
    struct AlignedAllocator {
        using value_type = T;

        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() noexcept = default;

        template <typename U>
        constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }
            std::size_t size = sizeof(T) * n;
            void* ptr = ::operator new[](size, std::align_val_t(Alignment));
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
            ::operator delete[](p, std::align_val_t(Alignment));
        }

        template <typename U>
        bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
            return true;
        }
    };

    template <typename T = float>
    using AlignedVector = std::vector<T, AlignedAllocator<T, 32>>;

}