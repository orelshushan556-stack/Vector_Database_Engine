#include <gtest/gtest.h>
#include "engine/storage/crc32.hpp"
#include <cstdint>
#include <string>
#include <span>
#include <vector>

TEST(Crc32CTest, KnownStandardVector) {
    const std::string text = "123456789";
    std::span<const uint8_t> span(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    uint32_t crc = engine::storage::calculate_crc32c(span);
    EXPECT_EQ(crc, 0xe3069283);
}

TEST(Crc32CTest, EmptyDataReturnsZero) {
    std::vector<uint8_t> empty;
    uint32_t crc = engine::storage::calculate_crc32c(empty);
    EXPECT_EQ(crc, 0x00000000);
}

TEST(Crc32CTest, DetectsSingleBitCorruption) {
    std::vector<uint8_t> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    uint32_t crc_original = engine::storage::calculate_crc32c(data);
    
    data[5] ^= 0x01;
    uint32_t crc_corrupted = engine::storage::calculate_crc32c(data);
    
    EXPECT_NE(crc_original, crc_corrupted);
}
