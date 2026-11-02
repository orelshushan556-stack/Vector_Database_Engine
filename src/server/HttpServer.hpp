#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include "WorkerThreadPool.hpp"
#include "engine/index/index_base.hpp"

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    using socket_t = SOCKET;
#else
    using socket_t = int;
#endif

class HttpServer {
public:
    explicit HttpServer(uint16_t port, std::shared_ptr<engine::IndexBase> index = nullptr, size_t num_threads = 4);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    HttpServer(HttpServer&&) noexcept = default;
    HttpServer& operator=(HttpServer&&) noexcept = default;

    void start();
    void stop();

private:
    uint16_t port_;
    std::shared_ptr<engine::IndexBase> index_;
    socket_t server_fd_{static_cast<socket_t>(-1)};
    bool is_running_{false};
    WorkerThreadPool thread_pool_;

    void setup_socket();
    void handle_client(socket_t client_fd);
};
