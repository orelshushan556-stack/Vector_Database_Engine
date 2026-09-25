#include "engine/storage/serializer.hpp"
#include <fstream>
#include <stdexcept>

namespace engine {

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
        out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(ids[0]));

        auto storage = index.raw_storage();
        out.write(reinterpret_cast<const char*>(storage.data()), storage.size() * sizeof(storage[0]));
    }

    FlatIndex Serializer::deserialize(std::istream& in) {
        FileHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in) {
            throw std::runtime_error("Failed to read header from stream");
        }
        if (header.magic != 0x56454349) {
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

        in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(ids[0]));
        in.read(reinterpret_cast<char*>(storage.data()), storage.size() * sizeof(storage[0]));
        if (!in) {
            throw std::runtime_error("Failed to read vector data from stream");
        }

        FlatIndex index(header.dimension, static_cast<MetricType>(header.metric));
        index.populate(std::move(ids), std::move(storage));

        return index;
    }
} // namespace engine