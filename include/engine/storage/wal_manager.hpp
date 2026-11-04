#pragma once

#include <cstdint>
#include <string>
#include <span>
#include <vector>
#include <functional>
#include <cstdio>

namespace engine::storage {

enum class WalOpType : uint8_t {
    Insert = 0x01
};

struct WalRecord {
    WalOpType type;
    uint64_t id;
    std::vector<float> vector;
};

class WalManager {
public:
    explicit WalManager(std::string wal_path);
    ~WalManager();

    WalManager(const WalManager&) = delete;
    WalManager& operator=(const WalManager&) = delete;

    WalManager(WalManager&& other) noexcept;
    WalManager& operator=(WalManager&& other) noexcept;

    void append_insert(uint64_t id, std::span<const float> vector);
    void replay(const std::function<void(uint64_t, std::span<const float>)>& callback);
    void clear();
    void sync();
    bool is_open() const;

private:
    std::string wal_path_;
    FILE* file_{nullptr};
    std::vector<uint8_t> buffer_;

    void sync_to_disk();
};

} // namespace engine::storage
