#include "engine/storage/serializer.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif
#include <span>
#include <stdexcept>

namespace engine {

namespace {
    void fsync_file(const std::filesystem::path& path) {
        std::FILE* f = std::fopen(path.string().c_str(), "rb+");
        if (!f) {
            throw std::runtime_error("Cannot reopen snapshot for fsync: " + path.string());
        }
        std::fflush(f);
#if defined(_WIN32)
        const int rc = _commit(_fileno(f));
#else
        const int rc = fsync(fileno(f));
#endif
        std::fclose(f);
        if (rc != 0) {
            throw std::runtime_error("fsync failed for snapshot: " + path.string());
        }
    }

    constexpr uint32_t MAGIC_FLAT     = 0x56454349; // "VECI"
    constexpr uint32_t MAGIC_HNSW     = 0x484E5357; // "HNSW"
    constexpr uint32_t MAGIC_HNSW_SQ8 = 0x53513848; // "SQ8H"

    template <typename T>
    void write_pod(std::ostream& out, const T& val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    template <typename T>
    void read_pod(std::istream& in, T& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(T));
    }

    template <typename Vec>
    void write_vec(std::ostream& out, const Vec& vec) {
        std::span s(vec.data(), vec.size());
        uint64_t sz = s.size();
        write_pod(out, sz);
        if (!s.empty()) {
            out.write(reinterpret_cast<const char*>(s.data()), s.size_bytes());
        }
    }

    template <typename Vec>
    void read_vec(std::istream& in, Vec& vec) {
        uint64_t sz = 0;
        read_pod(in, sz);
        vec.resize(static_cast<std::size_t>(sz));
        if (sz > 0) {
            std::span s(vec.data(), vec.size());
            in.read(reinterpret_cast<char*>(s.data()), s.size_bytes());
        }
    }
}

// ============================================================================
// Universal Polymorphic Reader & Writer
// ============================================================================

void Serializer::save(const IndexBase& index, const std::filesystem::path& file_path) {
    std::filesystem::path tmp_path = file_path;
    tmp_path += ".tmp";

    try {
        if (const auto* flat = dynamic_cast<const FlatIndex*>(&index)) {
            save_flat_index(*flat, tmp_path);
        } else if (const auto* hnsw = dynamic_cast<const HNSWIndex*>(&index)) {
            save_hnsw_index(*hnsw, tmp_path);
        } else if (const auto* sq8 = dynamic_cast<const HNSWIndexSQ8*>(&index)) {
            save_hnsw_sq8_index(*sq8, tmp_path);
        } else {
            throw std::invalid_argument("Unsupported index type for universal serialization");
        }

        fsync_file(tmp_path);
        std::filesystem::rename(tmp_path, file_path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw;
    }
}

std::unique_ptr<IndexBase> Serializer::load(const std::filesystem::path& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + file_path.string());
    }
    return load(in);
}

std::unique_ptr<IndexBase> Serializer::load(std::istream& in) {
    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!in) {
        throw std::runtime_error("Failed to read magic number from stream");
    }

    in.seekg(-static_cast<std::streamoff>(sizeof(magic)), std::ios::cur);
    if (!in) {
        throw std::runtime_error("Failed to rewind stream after reading magic number");
    }

    switch (magic) {
        case MAGIC_FLAT:
            return std::make_unique<FlatIndex>(deserialize(in));
        case MAGIC_HNSW:
            return std::make_unique<HNSWIndex>(deserialize_hnsw(in));
        case MAGIC_HNSW_SQ8:
            return std::make_unique<HNSWIndexSQ8>(deserialize_hnsw_sq8(in));
        default:
            throw std::runtime_error("Unknown or corrupted binary index format (Magic mismatch)");
    }
}

// ============================================================================
// FlatIndex Serialization
// ============================================================================

void Serializer::save_flat_index(const FlatIndex& index, const std::filesystem::path& file_path) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + file_path.string());
    }

    serialize(index, out);

    if (!out) {
        throw std::runtime_error("Failed to write to file: " + file_path.string());
    }
}

FlatIndex Serializer::load_flat_index(const std::filesystem::path& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + file_path.string());
    }

    FlatIndex index = deserialize(in);

    if (!in) {
        throw std::runtime_error("Failed to read from file: " + file_path.string());
    }

    return index;
}

void Serializer::serialize(const FlatIndex& index, std::ostream& out) {
    FileHeader header;
    header.metric = static_cast<uint32_t>(index.metric());
    header.dimension = index.dimension();
    header.count = index.size();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    auto ids = index.raw_ids();
    if (!ids.empty()) {
        out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(ids[0]));
    }

    auto storage = index.raw_storage();
    if (!storage.empty()) {
        out.write(reinterpret_cast<const char*>(storage.data()), storage.size() * sizeof(storage[0]));
    }
}

FlatIndex Serializer::deserialize(std::istream& in) {
    FileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("Failed to read header from stream");
    }
    if (header.magic != MAGIC_FLAT) {
        throw std::runtime_error("Invalid binary index format (Magic number mismatch)");
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported index version");
    }
    if (header.metric > static_cast<uint32_t>(MetricType::Cosine)) {
        throw std::runtime_error("Invalid metric type in header");
    }

    std::vector<VectorId> ids(header.count);
    AlignedVector<float> storage(header.count * header.dimension);

    if (header.count > 0) {
        in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(ids[0]));
        in.read(reinterpret_cast<char*>(storage.data()), storage.size() * sizeof(storage[0]));
        if (!in) {
            throw std::runtime_error("Failed to read vector data from stream");
        }
    }

    FlatIndex index(header.dimension, static_cast<MetricType>(header.metric));
    index.populate(std::move(ids), std::move(storage));

    return index;
}

// ============================================================================
// HNSWIndex Serialization
// ============================================================================

void Serializer::save_hnsw_index(const HNSWIndex& index, const std::filesystem::path& file_path) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + file_path.string());
    }

    serialize(index, out);

    if (!out) {
        throw std::runtime_error("Failed to write HNSW index to file: " + file_path.string());
    }
}

HNSWIndex Serializer::load_hnsw_index(const std::filesystem::path& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + file_path.string());
    }

    HNSWIndex index = deserialize_hnsw(in);

    if (!in) {
        throw std::runtime_error("Failed to read HNSW index from file: " + file_path.string());
    }

    return index;
}

void Serializer::serialize(const HNSWIndex& index, std::ostream& out) {
    HNSWHeader header;
    header.metric = static_cast<uint32_t>(index.metric_);
    header.dimension = index.dimension_;
    header.count = index.ids_.size();
    header.m = index.m_;
    header.max_m0 = index.max_m0_;
    header.ef_construction = index.ef_construction_;
    header.ef_search = index.ef_search_;
    header.mult = index.mult_;
    header.max_level = index.max_level_;
    header.enter_point = index.enter_point_;

    write_pod(out, header);

    write_vec(out, index.ids_);
    write_vec(out, index.storage_);
    write_vec(out, index.levels_);
    write_vec(out, index.layer0_graph_);

    uint64_t upper_sz = index.upper_graph_.size();
    write_pod(out, upper_sz);
    for (const auto& inner : index.upper_graph_) {
        write_vec(out, inner);
    }
}

HNSWIndex Serializer::deserialize_hnsw(std::istream& in) {
    HNSWHeader header;
    read_pod(in, header);
    if (!in) {
        throw std::runtime_error("Failed to read HNSW header from stream");
    }
    if (header.magic != MAGIC_HNSW) {
        throw std::runtime_error("Invalid HNSW binary format (Magic number mismatch)");
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported HNSW index version");
    }
    if (header.metric > static_cast<uint32_t>(MetricType::Cosine)) {
        throw std::runtime_error("Invalid metric type in HNSW header");
    }

    HNSWIndex index(
        static_cast<std::size_t>(header.dimension),
        static_cast<MetricType>(header.metric),
        static_cast<std::size_t>(header.m),
        static_cast<std::size_t>(header.ef_construction),
        static_cast<std::size_t>(header.ef_search)
    );

    index.max_m0_ = static_cast<std::size_t>(header.max_m0);
    index.mult_ = header.mult;
    index.max_level_ = header.max_level;
    index.enter_point_ = header.enter_point;

    read_vec(in, index.ids_);
    read_vec(in, index.storage_);
    read_vec(in, index.levels_);
    read_vec(in, index.layer0_graph_);

    uint64_t upper_sz = 0;
    read_pod(in, upper_sz);
    index.upper_graph_.resize(static_cast<std::size_t>(upper_sz));
    for (auto& inner : index.upper_graph_) {
        read_vec(in, inner);
    }

    if (!in) {
        throw std::runtime_error("Failed to read HNSW graph data from stream");
    }

    return index;
}

// ============================================================================
// HNSWIndexSQ8 Serialization
// ============================================================================

void Serializer::save_hnsw_sq8_index(const HNSWIndexSQ8& index, const std::filesystem::path& file_path) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + file_path.string());
    }

    serialize(index, out);

    if (!out) {
        throw std::runtime_error("Failed to write HNSWIndexSQ8 to file: " + file_path.string());
    }
}

HNSWIndexSQ8 Serializer::load_hnsw_sq8_index(const std::filesystem::path& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + file_path.string());
    }

    HNSWIndexSQ8 index = deserialize_hnsw_sq8(in);

    if (!in) {
        throw std::runtime_error("Failed to read HNSWIndexSQ8 from file: " + file_path.string());
    }

    return index;
}

void Serializer::serialize(const HNSWIndexSQ8& index, std::ostream& out) {
    HNSWSQ8Header header;
    header.metric = static_cast<uint32_t>(index.metric_);
    header.dimension = index.dimension_;
    header.count = index.count_;
    header.m = index.m_;
    header.max_m = index.max_m_;
    header.max_m0 = index.max_m0_;
    header.ef_construction = index.ef_construction_;
    header.ef_search = index.ef_search_;
    header.re_rank_factor = index.re_rank_factor_;
    header.mult = index.mult_;
    header.max_level = index.max_level_;
    header.enter_point = index.enter_point_;

    write_pod(out, header);

    uint8_t trained = index.quantizer_.is_trained_ ? 1 : 0;
    write_pod(out, trained);
    write_vec(out, index.quantizer_.min_values_);
    write_vec(out, index.quantizer_.scale_values_);

    write_vec(out, index.ids_);
    write_vec(out, index.quantized_storage_);
    write_vec(out, index.raw_storage_);
    write_vec(out, index.levels_);
    write_vec(out, index.layer0_graph_);

    uint64_t upper_sz = index.upper_graph_.size();
    write_pod(out, upper_sz);
    for (const auto& inner : index.upper_graph_) {
        write_vec(out, inner);
    }
}

HNSWIndexSQ8 Serializer::deserialize_hnsw_sq8(std::istream& in) {
    HNSWSQ8Header header;
    read_pod(in, header);
    if (!in) {
        throw std::runtime_error("Failed to read HNSWIndexSQ8 header from stream");
    }
    if (header.magic != MAGIC_HNSW_SQ8) {
        throw std::runtime_error("Invalid HNSWIndexSQ8 binary format (Magic number mismatch)");
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported HNSWIndexSQ8 version");
    }
    if (header.metric > static_cast<uint32_t>(MetricType::Cosine)) {
        throw std::runtime_error("Invalid metric type in HNSWIndexSQ8 header");
    }

    uint8_t trained = 0;
    read_pod(in, trained);

    ScalarQuantizer quantizer(static_cast<std::size_t>(header.dimension));
    quantizer.is_trained_ = (trained != 0);
    read_vec(in, quantizer.min_values_);
    read_vec(in, quantizer.scale_values_);

    if (!quantizer.is_trained_) {
        throw std::runtime_error("Corrupted HNSWIndexSQ8 stream: Quantizer is not marked as trained");
    }

    HNSWIndexSQ8 index(
        static_cast<std::size_t>(header.dimension),
        static_cast<MetricType>(header.metric),
        std::move(quantizer),
        static_cast<std::size_t>(header.m),
        static_cast<std::size_t>(header.ef_construction),
        static_cast<std::size_t>(header.ef_search),
        header.re_rank_factor
    );

    index.max_m_ = static_cast<std::size_t>(header.max_m);
    index.max_m0_ = static_cast<std::size_t>(header.max_m0);
    index.mult_ = header.mult;
    index.max_level_ = header.max_level;
    index.enter_point_ = header.enter_point;
    index.count_ = static_cast<std::size_t>(header.count);

    read_vec(in, index.ids_);
    read_vec(in, index.quantized_storage_);
    read_vec(in, index.raw_storage_);
    read_vec(in, index.levels_);
    read_vec(in, index.layer0_graph_);

    uint64_t upper_sz = 0;
    read_pod(in, upper_sz);
    index.upper_graph_.resize(static_cast<std::size_t>(upper_sz));
    for (auto& inner : index.upper_graph_) {
        read_vec(in, inner);
    }

    if (!in) {
        throw std::runtime_error("Failed to read HNSWIndexSQ8 graph data from stream");
    }

    return index;
}

} // namespace engine
