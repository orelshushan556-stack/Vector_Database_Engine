#pragma once

#include "WorkerThreadPool.hpp"
#include "engine/core/vector_engine.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    using socket_t = SOCKET;
#else
    using socket_t = int;
#endif

class HttpServer {
public:
    explicit HttpServer(uint16_t port, std::shared_ptr<engine::VectorEngine> engine = nullptr, size_t num_threads = 4);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    HttpServer(HttpServer&&) = delete;
    HttpServer& operator=(HttpServer&&) = delete;

    void start();
    void stop();

private:
    uint16_t port_;
    std::shared_ptr<engine::VectorEngine> engine_;
    // stop() may be called from a different thread than start(), so both are atomic.
    std::atomic<socket_t> server_fd_{static_cast<socket_t>(-1)};
    std::atomic<bool> is_running_{false};
    WorkerThreadPool thread_pool_;

    void setup_socket();
    void handle_client(socket_t client_fd);
};
