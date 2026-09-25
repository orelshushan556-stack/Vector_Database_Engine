#pragma once

#include "engine/index/flat_index.hpp"
#include <cstdint>
#include <filesystem>
#include <istream>
#include <ostream>

namespace engine {

#pragma pack(push, 1)
    struct FileHeader {
        uint32_t magic{0x56454349}; // "VECI" (Vector Engine Chunk Index)
        uint32_t version{1};
        uint32_t metric{0};
        uint64_t dimension{0};
        uint64_t count{0};
    };
#pragma pack(pop)

    class Serializer {
    public:
        static void save_flat_index(const FlatIndex& index, const std::filesystem::path& file_path);
        static FlatIndex load_flat_index(const std::filesystem::path& file_path);

        static void serialize(const FlatIndex& index, std::ostream& out);
        static FlatIndex deserialize(std::istream& in);
    };

} // namespace engine