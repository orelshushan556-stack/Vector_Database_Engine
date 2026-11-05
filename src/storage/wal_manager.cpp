#include "engine/storage/wal_manager.hpp"
#include "engine/storage/crc32.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace engine::storage {

namespace {
constexpr uint32_t MAX_SUPPORTED_DIM = 4096;
}

WalManager::WalManager(std::string wal_path)
    : wal_path_(std::move(wal_path)) {
    file_ = std::fopen(wal_path_.c_str(), "ab+");
    if (!file_) {
        throw std::runtime_error("Failed to open WAL file: " + wal_path_);
    }
}

WalManager::~WalManager() {
    if (file_) {
        try {
            sync_to_disk();
        } catch (...) {
        }
        std::fclose(file_);
        file_ = nullptr;
    }
}

WalManager::WalManager(WalManager&& other) noexcept
    : wal_path_(std::move(other.wal_path_)),
      file_(std::exchange(other.file_, nullptr)),
      buffer_(std::move(other.buffer_)) {}

WalManager& WalManager::operator=(WalManager&& other) noexcept {
    if (this != &other) {
        if (file_) {
            try {
                sync_to_disk();
            } catch (...) {
            }
            std::fclose(file_);
        }
        wal_path_ = std::move(other.wal_path_);
        file_ = std::exchange(other.file_, nullptr);
        buffer_ = std::move(other.buffer_);
    }
    return *this;
}

void WalManager::sync_to_disk() {
    if (!file_) {
        throw std::runtime_error("WAL file is not open for syncing");
    }

    if (std::fflush(file_) != 0) {
        throw std::runtime_error("Failed to flush C runtime buffer");
    }

#if defined(_WIN32)
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
    if (!file_) {
        throw std::runtime_error("WAL file is not open for writing");
    }

    if (vector.empty() || vector.size() > MAX_SUPPORTED_DIM) {
        throw std::invalid_argument("Vector dimension is outside the supported WAL range");
    }

    const uint32_t dim = static_cast<uint32_t>(vector.size());
    const size_t vector_bytes_len = vector.size_bytes();
    const size_t payload_len = sizeof(id) + sizeof(dim) + vector_bytes_len;
    const size_t total_record_size = 1 + sizeof(uint32_t) + payload_len + 1;

    buffer_.resize(total_record_size);

    buffer_[0] = static_cast<uint8_t>(WalOpType::Insert);

    size_t cursor = 5;

    auto id_bytes = std::bit_cast<std::array<uint8_t, sizeof(id)>>(id);
    std::ranges::copy(id_bytes, buffer_.begin() + cursor);
    cursor += sizeof(id);

    auto dim_bytes = std::bit_cast<std::array<uint8_t, sizeof(dim)>>(dim);
    std::ranges::copy(dim_bytes, buffer_.begin() + cursor);
    cursor += sizeof(dim);

    std::span<const uint8_t> vector_bytes(
        reinterpret_cast<const uint8_t*>(vector.data()),
        vector_bytes_len
    );
    std::ranges::copy(vector_bytes, buffer_.begin() + cursor);

    buffer_.back() = 0xAA;

    std::span<const uint8_t> frame_span(buffer_);
    uint32_t crc = calculate_crc32c(frame_span.subspan(5, payload_len));

    auto crc_bytes = std::bit_cast<std::array<uint8_t, sizeof(crc)>>(crc);
    std::ranges::copy(crc_bytes, buffer_.begin() + 1);

    size_t written = std::fwrite(buffer_.data(), 1, buffer_.size(), file_);
    if (written != buffer_.size()) {
        throw std::runtime_error("Failed to write complete WAL record");
    }

    sync_to_disk();
}

void WalManager::replay(const std::function<void(uint64_t, std::span<const float>)>& callback) {
    if (file_) {
        std::fflush(file_);
    }

    std::FILE* read_file = std::fopen(wal_path_.c_str(), "rb");
    if (!read_file) {
        return;
    }

    struct FileCloser {
        std::FILE* f;
        ~FileCloser() {
            if (f) {
                std::fclose(f);
            }
        }
    } closer{read_file};

    constexpr size_t header_size = 1 + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);
    std::array<uint8_t, header_size> header_buf{};

    while (true) {
        size_t bytes_read = std::fread(header_buf.data(), 1, header_size, read_file);
        if (bytes_read == 0) {
            break;
        }
        if (bytes_read < header_size) {
            break;
        }

        if (header_buf[0] != static_cast<uint8_t>(WalOpType::Insert)) {
            throw std::runtime_error("Unknown WAL operation type encountered during replay");
        }

        std::array<uint8_t, 4> crc_bytes{};
        std::ranges::copy(header_buf.begin() + 1, header_buf.begin() + 5, crc_bytes.begin());
        uint32_t expected_crc = std::bit_cast<uint32_t>(crc_bytes);

        std::array<uint8_t, 8> id_bytes{};
        std::ranges::copy(header_buf.begin() + 5, header_buf.begin() + 13, id_bytes.begin());
        uint64_t id = std::bit_cast<uint64_t>(id_bytes);

        std::array<uint8_t, 4> dim_bytes{};
        std::ranges::copy(header_buf.begin() + 13, header_buf.begin() + 17, dim_bytes.begin());
        uint32_t dim = std::bit_cast<uint32_t>(dim_bytes);

        if (dim == 0 || dim > MAX_SUPPORTED_DIM) {
            std::cerr << "[WAL] Corrupted record: implausible dimension " << dim
                      << " (max " << MAX_SUPPORTED_DIM << "); aborting replay\n";
            throw std::runtime_error("Corrupted WAL record: invalid vector dimension");
        }

        const size_t vector_bytes_len = static_cast<size_t>(dim) * sizeof(float);
        const size_t remaining_record_len = vector_bytes_len + 1;

        const size_t full_payload_len = sizeof(id) + sizeof(dim) + vector_bytes_len;
        buffer_.resize(full_payload_len);

        std::ranges::copy(id_bytes, buffer_.begin());
        std::ranges::copy(dim_bytes, buffer_.begin() + sizeof(id));

        size_t payload_offset = sizeof(id) + sizeof(dim);
        size_t vector_read = std::fread(buffer_.data() + payload_offset, 1, vector_bytes_len, read_file);
        if (vector_read != vector_bytes_len) {
            break;
        }

        uint8_t end_byte = 0;
        size_t end_read = std::fread(&end_byte, 1, 1, read_file);
        if (end_read != 1 || end_byte != 0xAA) {
            throw std::runtime_error("Corrupted WAL record: invalid end byte");
        }

        uint32_t actual_crc = calculate_crc32c(std::span<const uint8_t>(buffer_));
        if (actual_crc != expected_crc) {
            throw std::runtime_error("Corrupted WAL record: CRC32-C mismatch detected");
        }

        std::span<const float> vec_span(
            reinterpret_cast<const float*>(buffer_.data() + payload_offset),
            dim
        );

        callback(id, vec_span);
    }
}

void WalManager::clear() {
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
    file_ = std::fopen(wal_path_.c_str(), "wb+");
    if (!file_) {
        throw std::runtime_error("Failed to truncate WAL file: " + wal_path_);
    }
}

} // namespace engine::storage
