#include "engine/storage/wal_manager.hpp"
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace engine::storage {

WalManager::WalManager(std::string wal_path)
    : wal_path_(std::move(wal_path)) {
    file_ = std::fopen(wal_path_.c_str(), "ab+");
    if (!file_) {
        throw std::runtime_error("Failed to open WAL file: " + wal_path_);
    }
}

WalManager::~WalManager() {
    if (file_) {
        sync_to_disk();
        std::fclose(file_);
        file_ = nullptr;
    }
}

WalManager::WalManager(WalManager&& other) noexcept
    : wal_path_(std::move(other.wal_path_)), file_(other.file_) {
    other.file_ = nullptr;
}

WalManager& WalManager::operator=(WalManager&& other) noexcept {
    if (this != &other) {
        if (file_) {
            sync_to_disk();
            std::fclose(file_);
        }
        wal_path_ = std::move(other.wal_path_);
        file_ = other.file_;
        other.file_ = nullptr;
    }
    return *this;
}

void WalManager::sync_to_disk() {
    if (!file_) {
        return;
    }

    std::fflush(file_);

#ifdef _WIN32
    int fd = _fileno(file_);
    if (fd != -1) {
        HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (handle != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(handle);
        }
    }
#else
    int fd = fileno(file_);
    if (fd != -1) {
        fdatasync(fd);
    }
#endif
}

void WalManager::sync() {
    sync_to_disk();
}

bool WalManager::is_open() const {
    return file_ != nullptr;
}

void WalManager::append_insert(uint64_t id, std::span<const float> vector) {
}

void WalManager::replay(const std::function<void(uint64_t, std::span<const float>)>& callback) {
}

void WalManager::clear() {
}

} // namespace engine::storage
