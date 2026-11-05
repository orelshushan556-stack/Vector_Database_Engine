#include "engine/core/vector_engine.hpp"
#include "engine/storage/serializer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine {

namespace {

// NaN or infinite components make distances unordered and would be persisted to the WAL.
void require_finite(std::span<const float> v, const char* what) {
    if (!std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); })) {
        throw std::invalid_argument(std::string(what) + " contains a non-finite value");
    }
}

} // namespace

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
    if (vector.size() != index_->dimension()) {
        throw std::invalid_argument("Vector dimension does not match engine dimension");
    }
    require_finite(vector, "Vector");

    // Hold the WAL mutex across both steps so a checkpoint cannot snapshot and
    // truncate the WAL between the append and the in-memory add.
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    wal_.append_insert(id, vector);

    std::unique_lock<std::shared_mutex> index_lock(index_rw_lock_);
    index_->add(id, vector);
}

std::vector<SearchResult> VectorEngine::search(std::span<const float> query, std::size_t k) const {
    if (query.size() != index_->dimension()) {
        throw std::invalid_argument("Query dimension does not match engine dimension");
    }
    require_finite(query, "Query");

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
