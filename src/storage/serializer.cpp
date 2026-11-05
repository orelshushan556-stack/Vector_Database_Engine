#include "engine/storage/serializer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <span>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <system_error>

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

    // Replace dst with src atomically and make the directory entry durable.
    //  - POSIX: rename(2), then fsync on the parent directory descriptor.
    //  - Windows: MoveFileExW with MOVEFILE_WRITE_THROUGH, which returns only after
    //    the move has been flushed to disk (directories cannot be fsynced there).
    void durable_replace(const std::filesystem::path& src, const std::filesystem::path& dst) {
#if defined(_WIN32)
        if (!MoveFileExW(src.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Failed to atomically replace snapshot: " + dst.string());
        }
#else
        std::filesystem::rename(src, dst);

        std::filesystem::path parent = dst.parent_path();
        if (parent.empty()) {
            parent = ".";
        }
        const int dir_fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (dir_fd < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "Cannot open snapshot directory for fsync: " + parent.string());
        }
        const int rc = ::fsync(dir_fd);
        const int saved_errno = errno;
        ::close(dir_fd);
        if (rc != 0) {
            throw std::system_error(saved_errno, std::generic_category(),
                                    "fsync failed for snapshot directory: " + parent.string());
        }
#endif
    }

    constexpr uint32_t MAGIC_FLAT     = 0x56454349; // "VECI"
    constexpr uint32_t MAGIC_HNSW     = 0x484E5357; // "HNSW"
    constexpr uint32_t MAGIC_HNSW_SQ8 = 0x53513848; // "SQ8H"

    template <typename T>
    void write_pod(std::ostream& out, const T& val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    // Upper bounds for header fields read from an untrusted snapshot.
    constexpr uint64_t MAX_SNAPSHOT_DIM = 1u << 16;
    constexpr uint64_t MAX_SNAPSHOT_M = 1024;
    constexpr uint64_t MAX_SNAPSHOT_M0 = 4096;
    constexpr int MAX_SNAPSHOT_LEVEL = 128;

    [[noreturn]] void corrupt(const std::string& why) {
        throw std::runtime_error("Corrupted snapshot: " + why);
    }

    template <typename T>
    void read_pod(std::istream& in, T& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!in) {
            corrupt("unexpected end of data");
        }
    }

    // Reads exactly n elements. The container grows in bounded steps as data actually
    // arrives, so a corrupted length field cannot trigger a huge up-front allocation.
    template <typename Vec>
    void read_exact(std::istream& in, Vec& vec, uint64_t n) {
        using T = typename Vec::value_type;
        constexpr uint64_t kChunkElems = (1u << 20) / sizeof(T);

        vec.clear();
        uint64_t remaining = n;
        while (remaining > 0) {
            const auto step = static_cast<std::size_t>(std::min(remaining, kChunkElems));
            const std::size_t old_size = vec.size();
            vec.resize(old_size + step);
            in.read(reinterpret_cast<char*>(vec.data() + old_size),
                    static_cast<std::streamsize>(step * sizeof(T)));
            if (!in) {
                corrupt("unexpected end of data");
            }
            remaining -= step;
        }
    }

    // Structural checks that make graph traversal safe: every adjacency block must fit
    // its buffer, every degree must respect the layer capacity, and every neighbor id
    // must refer to an existing node.
    template <typename Levels, typename Layer0, typename Upper>
    void validate_graph(const Levels& levels, const Layer0& layer0, const Upper& upper,
                        std::size_t n, std::size_t m, std::size_t m0) {
        if (levels.size() != n || upper.size() != n) {
            corrupt("per-node arrays have inconsistent lengths");
        }
        if (layer0.size() != n * (m0 + 1)) {
            corrupt("layer-0 adjacency has the wrong size");
        }
        for (std::size_t i = 0; i < n; ++i) {
            const int level = levels[i];
            if (level < 0 || level > MAX_SNAPSHOT_LEVEL) {
                corrupt("node level out of range");
            }
            if (upper[i].size() != static_cast<std::size_t>(level) * (m + 1)) {
                corrupt("upper-layer adjacency has the wrong size");
            }

            const std::size_t base0 = i * (m0 + 1);
            const std::size_t degree0 = layer0[base0];
            if (degree0 > m0) {
                corrupt("layer-0 degree exceeds capacity");
            }
            for (std::size_t j = 0; j < degree0; ++j) {
                if (layer0[base0 + 1 + j] >= n) {
                    corrupt("layer-0 neighbor id out of range");
                }
            }

            for (int l = 0; l < level; ++l) {
                const std::size_t base = static_cast<std::size_t>(l) * (m + 1);
                const std::size_t degree = upper[i][base];
                if (degree > m) {
                    corrupt("upper-layer degree exceeds capacity");
                }
                for (std::size_t j = 0; j < degree; ++j) {
                    const std::size_t neighbor = upper[i][base + 1 + j];
                    if (neighbor >= n) {
                        corrupt("upper-layer neighbor id out of range");
                    }
                    // Traversal reads the neighbor's adjacency on this layer, so the
                    // neighbor must actually own that layer.
                    if (levels[neighbor] < l + 1) {
                        corrupt("upper-layer neighbor does not exist on that layer");
                    }
                }
            }
        }
    }

    void validate_hnsw_params(uint64_t dimension, uint64_t m, uint64_t m0) {
        if (dimension == 0 || dimension > MAX_SNAPSHOT_DIM) {
            corrupt("dimension out of range");
        }
        if (m < 2 || m > MAX_SNAPSHOT_M || m0 < m || m0 > MAX_SNAPSHOT_M0) {
            corrupt("graph parameters out of range");
        }
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
        read_exact(in, vec, sz);
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
        durable_replace(tmp_path, file_path);
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

    if (header.dimension == 0 || header.dimension > MAX_SNAPSHOT_DIM) {
        throw std::runtime_error("Corrupted snapshot: dimension out of range");
    }
    if (header.count > (UINT64_MAX / sizeof(float)) / header.dimension) {
        throw std::runtime_error("Corrupted snapshot: vector count out of range");
    }

    std::vector<VectorId> ids;
    AlignedVector<float> storage;
    read_exact(in, ids, header.count);
    read_exact(in, storage, header.count * header.dimension);

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

    validate_hnsw_params(header.dimension, header.m, header.max_m0);

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
    if (upper_sz != index.ids_.size()) {
        corrupt("upper-layer table length does not match node count");
    }
    index.upper_graph_.resize(static_cast<std::size_t>(upper_sz));
    for (auto& inner : index.upper_graph_) {
        read_vec(in, inner);
    }

    const std::size_t n = index.ids_.size();
    if (index.storage_.size() / header.dimension != n || index.storage_.size() % header.dimension != 0) {
        corrupt("vector storage does not match node count");
    }
    validate_graph(index.levels_, index.layer0_graph_, index.upper_graph_, n, index.m_, index.max_m0_);
    if (n == 0) {
        if (index.enter_point_ != -1 || index.max_level_ != -1) {
            corrupt("empty index has an entry point");
        }
    } else if (index.enter_point_ < 0 || static_cast<std::size_t>(index.enter_point_) >= n ||
               index.max_level_ != index.levels_[static_cast<std::size_t>(index.enter_point_)]) {
        corrupt("entry point is inconsistent with the graph");
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

    validate_hnsw_params(header.dimension, header.m, header.max_m0);
    if (header.max_m < 2 || header.max_m > MAX_SNAPSHOT_M) {
        corrupt("graph parameters out of range");
    }
    if (!std::isfinite(header.re_rank_factor) || header.re_rank_factor <= 0.0f) {
        corrupt("re-rank factor out of range");
    }

    ScalarQuantizer quantizer(static_cast<std::size_t>(header.dimension));
    quantizer.is_trained_ = (trained != 0);
    read_vec(in, quantizer.min_values_);
    read_vec(in, quantizer.scale_values_);

    if (!quantizer.is_trained_) {
        throw std::runtime_error("Corrupted HNSWIndexSQ8 stream: Quantizer is not marked as trained");
    }
    if (quantizer.min_values_.size() != header.dimension || quantizer.scale_values_.size() != header.dimension) {
        corrupt("quantizer parameters do not match the dimension");
    }
    for (std::size_t d = 0; d < quantizer.min_values_.size(); ++d) {
        if (!std::isfinite(quantizer.min_values_[d]) || !std::isfinite(quantizer.scale_values_[d]) ||
            quantizer.scale_values_[d] <= 0.0f) {
            corrupt("quantizer parameters are not usable");
        }
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
    if (upper_sz != index.ids_.size()) {
        corrupt("upper-layer table length does not match node count");
    }
    index.upper_graph_.resize(static_cast<std::size_t>(upper_sz));
    for (auto& inner : index.upper_graph_) {
        read_vec(in, inner);
    }

    const std::size_t n = index.ids_.size();
    if (index.count_ != n || index.quantized_storage_.size() / header.dimension != n ||
        index.quantized_storage_.size() % header.dimension != 0 ||
        index.raw_storage_.size() != index.quantized_storage_.size()) {
        corrupt("vector storage does not match node count");
    }
    validate_graph(index.levels_, index.layer0_graph_, index.upper_graph_, n, index.max_m_, index.max_m0_);
    if (n == 0) {
        if (index.max_level_ != -1) {
            corrupt("empty index has an entry point");
        }
    } else if (index.enter_point_ >= n ||
               index.max_level_ != index.levels_[index.enter_point_]) {
        corrupt("entry point is inconsistent with the graph");
    }

    return index;
}

} // namespace engine
