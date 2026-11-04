#include "engine/core/vector_engine.hpp"
#include "engine/storage/serializer.hpp"

#include <stdexcept>
#include <utility>

namespace engine {

VectorEngine::VectorEngine(std::unique_ptr<IndexBase> index, std::string wal_path)
    : index_(std::move(index)), wal_(std::move(wal_path)) {
    if (!index_) {
        throw std::invalid_argument("Index pointer cannot be null");
    }

    wal_.replay([this](uint64_t id, std::span<const float> vec) {
        index_->add(id, vec);
    });
}

void VectorEngine::insert(uint64_t id, std::span<const float> vector) {
    {
        std::lock_guard<std::mutex> wal_lock(wal_mutex_);
        wal_.append_insert(id, vector);
    }

    {
        std::unique_lock<std::shared_mutex> index_lock(index_rw_lock_);
        index_->add(id, vector);
    }
}

std::vector<SearchResult> VectorEngine::search(std::span<const float> query, std::size_t k) const {
    std::shared_lock<std::shared_mutex> lock(index_rw_lock_);
    return index_->search(query, k);
}

void VectorEngine::create_checkpoint(const std::filesystem::path& snapshot_path) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    std::shared_lock<std::shared_mutex> index_lock(index_rw_lock_);

    Serializer::save(*index_, snapshot_path);
    wal_.clear();
}

std::size_t VectorEngine::size() const {
    std::shared_lock<std::shared_mutex> lock(index_rw_lock_);
    return index_->size();
}

std::size_t VectorEngine::dimension() const {
    return index_->dimension();
}

MetricType VectorEngine::metric() const {
    return index_->metric();
}

} // namespace engine
