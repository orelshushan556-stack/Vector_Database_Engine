#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <span>

namespace engine::storage {

inline uint32_t calculate_crc32c(std::span<const uint8_t> data) {
    uint64_t crc = 0xFFFFFFFF;
    const uint8_t* ptr = data.data();
    size_t length = data.size();

    while (length >= 8) {
        uint64_t value;
        std::memcpy(&value, ptr, sizeof(uint64_t));
        crc = _mm_crc32_u64(crc, value);
        ptr += 8;
        length -= 8;
    }

    while (length > 0) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *ptr);
        ++ptr;
        --length;
    }

    return static_cast<uint32_t>(crc ^ 0xFFFFFFFF);
}

} // namespace engine::storage
