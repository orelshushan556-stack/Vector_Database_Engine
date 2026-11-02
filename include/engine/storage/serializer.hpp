#pragma once

#include "engine/index/index_base.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/index/hnsw_index.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"

#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>

namespace engine {

#pragma pack(push, 1)
    struct FileHeader {
        uint32_t magic{0x56454349}; // "VECI" (FlatIndex)
        uint32_t version{1};
        uint32_t metric{0};
        uint64_t dimension{0};
        uint64_t count{0};
    };

    struct HNSWHeader {
        uint32_t magic{0x484E5357}; // "HNSW" (HNSWIndex)
        uint32_t version{1};
        uint32_t metric{0};
        uint64_t dimension{0};
        uint64_t count{0};
        uint64_t m{0};
        uint64_t max_m0{0};
        uint64_t ef_construction{0};
        uint64_t ef_search{0};
        double mult{0.0};
        int32_t max_level{-1};
        int64_t enter_point{-1};
    };

    struct HNSWSQ8Header {
        uint32_t magic{0x53513848}; // "SQ8H" (HNSWIndexSQ8)
        uint32_t version{1};
        uint32_t metric{0};
        uint64_t dimension{0};
        uint64_t count{0};
        uint64_t m{0};
        uint64_t max_m{0};
        uint64_t max_m0{0};
        uint64_t ef_construction{0};
        uint64_t ef_search{0};
        float re_rank_factor{2.0f};
        double mult{0.0};
        int32_t max_level{-1};
        uint32_t enter_point{0};
    };
#pragma pack(pop)

    class Serializer {
    public:
        // Universal Polymorphic Reader & Writer
        static void save(const IndexBase& index, const std::filesystem::path& file_path);
        [[nodiscard]] static std::unique_ptr<IndexBase> load(const std::filesystem::path& file_path);
        [[nodiscard]] static std::unique_ptr<IndexBase> load(std::istream& in);

        // FlatIndex
        static void save_flat_index(const FlatIndex& index, const std::filesystem::path& file_path);
        static FlatIndex load_flat_index(const std::filesystem::path& file_path);
        static void serialize(const FlatIndex& index, std::ostream& out);
        static FlatIndex deserialize(std::istream& in);

        // HNSWIndex
        static void save_hnsw_index(const HNSWIndex& index, const std::filesystem::path& file_path);
        static HNSWIndex load_hnsw_index(const std::filesystem::path& file_path);
        static void serialize(const HNSWIndex& index, std::ostream& out);
        static HNSWIndex deserialize_hnsw(std::istream& in);

        // HNSWIndexSQ8
        static void save_hnsw_sq8_index(const HNSWIndexSQ8& index, const std::filesystem::path& file_path);
        static HNSWIndexSQ8 load_hnsw_sq8_index(const std::filesystem::path& file_path);
        static void serialize(const HNSWIndexSQ8& index, std::ostream& out);
        static HNSWIndexSQ8 deserialize_hnsw_sq8(std::istream& in);
    };

} // namespace engine
