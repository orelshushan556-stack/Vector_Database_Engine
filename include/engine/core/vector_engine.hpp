#pragma once

#include "engine/index/index_base.hpp"
#include "engine/storage/wal_manager.hpp"
#include "engine/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

namespace engine {

class VectorEngine {
public:
    VectorEngine(std::unique_ptr<IndexBase> index, std::string wal_path);
    ~VectorEngine() = default;

    VectorEngine(const VectorEngine&) = delete;
    VectorEngine& operator=(const VectorEngine&) = delete;

    VectorEngine(VectorEngine&&) = delete;
    VectorEngine& operator=(VectorEngine&&) = delete;

    void insert(uint64_t id, std::span<const float> vector);
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;
    void create_checkpoint(const std::filesystem::path& snapshot_path);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t dimension() const;
    [[nodiscard]] MetricType metric() const;

private:
    std::unique_ptr<IndexBase> index_;
    storage::WalManager wal_;
    mutable std::shared_mutex index_rw_lock_;
    std::mutex wal_mutex_;
};

} // namespace engine
