#include "HttpServer.hpp"
#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "Router.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <exception>
#include <system_error>
#include <iostream>
#include <string>
#include <string_view>

#ifdef _WIN32
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define CLOSE_SOCKET(s) close(s)
#endif

HttpServer::HttpServer(uint16_t port, std::shared_ptr<engine::VectorEngine> engine, size_t num_threads)
    : port_(port), engine_(std::move(engine)), server_fd_(static_cast<socket_t>(-1)), is_running_(false), thread_pool_(num_threads) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
}

HttpServer::~HttpServer() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

void HttpServer::setup_socket() {
    const socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    if (server_fd == INVALID_SOCKET) {
#else
    if (server_fd < 0) {
#endif
        std::cerr << "Error creating socket" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options" << std::endl;
        CLOSE_SOCKET(server_fd);
        std::exit(EXIT_FAILURE);
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Error binding socket" << std::endl;
        CLOSE_SOCKET(server_fd);
        std::exit(EXIT_FAILURE);
    }

    if (listen(server_fd, SOMAXCONN) < 0) {
        std::cerr << "Error listening on socket" << std::endl;
        CLOSE_SOCKET(server_fd);
        std::exit(EXIT_FAILURE);
    }

    server_fd_.store(server_fd);
}

void HttpServer::start() {
    setup_socket();
    is_running_.store(true);
    std::cout << "[*] nexus_server online at http://localhost:" << port_ << std::endl;

    while (is_running_.load()) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
        SOCKET client_fd = accept(server_fd_.load(), reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd == INVALID_SOCKET) {
#else
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_.load(), reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
#endif
            if (!is_running_.load()) break;
            continue;
        }

        thread_pool_.enqueue([this, client_fd]() {
            handle_client(client_fd);
        });
    }
}

namespace {

constexpr size_t MAX_BODY_SIZE = 16 * 1024 * 1024;
constexpr size_t MAX_HEADER_SIZE = 16 * 1024;
constexpr int RECV_TIMEOUT_SECONDS = 10;

// A peer that disconnects mid-response must not raise SIGPIPE and kill the process.
#ifdef MSG_NOSIGNAL
constexpr int SEND_FLAGS = MSG_NOSIGNAL;
#else
constexpr int SEND_FLAGS = 0;
#endif

// Closes the client socket on every exit path, including exceptions.
struct SocketGuard {
    socket_t fd;
    explicit SocketGuard(socket_t f) : fd(f) {}
    ~SocketGuard() { CLOSE_SOCKET(fd); }
    SocketGuard(const SocketGuard&) = delete;
    SocketGuard& operator=(const SocketGuard&) = delete;
};

// send() may transfer fewer bytes than requested; loop until everything is written.
bool send_all(socket_t fd, std::string_view data) {
    constexpr size_t kMaxChunk = 1u << 30;
    while (!data.empty()) {
        const int sent = static_cast<int>(send(fd, data.data(), static_cast<int>(std::min(data.size(), kMaxChunk)), SEND_FLAGS));
        if (sent <= 0) {
            return false;
        }
        data.remove_prefix(static_cast<size_t>(sent));
    }
    return true;
}

void send_error(socket_t fd, int status, const char* reason, const char* message) {
    HttpResponse res(status, reason);
    res.set_header("Content-Type", "application/json");
    res.set_header("Connection", "close");
    res.set_body(std::string("{\"status\":\"error\",\"message\":\"") + message + "\"}");
    send_all(fd, res.to_string());
}

// Bounds how long an idle or stalled client can occupy a worker thread.
void set_recv_timeout(socket_t fd) {
#ifdef _WIN32
    const DWORD timeout_ms = RECV_TIMEOUT_SECONDS * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    timeval tv{};
    tv.tv_sec = RECV_TIMEOUT_SECONDS;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

} // namespace

void HttpServer::handle_client(socket_t client_fd) {
    const SocketGuard guard(client_fd);
    set_recv_timeout(client_fd);

    try {
        std::string raw_request;
        std::array<char, 4096> buffer{};
        size_t header_end_pos = std::string::npos;
        size_t content_length = 0;
        bool headers_parsed = false;
        bool complete = false;

        while (true) {
            const int bytes_read = static_cast<int>(recv(client_fd, buffer.data(), static_cast<int>(buffer.size()), 0));
            if (bytes_read <= 0) break;
            raw_request.append(buffer.data(), static_cast<size_t>(bytes_read));

            if (!headers_parsed) {
                const std::string_view req_view(raw_request);
                header_end_pos = req_view.find("\r\n\r\n");

                if (header_end_pos == std::string_view::npos) {
                    if (raw_request.size() > MAX_HEADER_SIZE) {
                        send_error(client_fd, 431, "Request Header Fields Too Large", "Request headers are too large.");
                        return;
                    }
                    continue;
                }

                headers_parsed = true;
                const std::string_view headers_view = req_view.substr(0, header_end_pos);

                auto cl_pos = headers_view.find("Content-Length:");
                if (cl_pos == std::string_view::npos) {
                    cl_pos = headers_view.find("content-length:");
                }

                if (cl_pos != std::string_view::npos) {
                    auto val_view = headers_view.substr(cl_pos + 15);
                    const auto first_digit = val_view.find_first_not_of(" \t");
                    if (first_digit != std::string_view::npos) {
                        val_view.remove_prefix(first_digit);
                        const auto parsed = std::from_chars(val_view.data(), val_view.data() + val_view.size(), content_length);
                        if (parsed.ec == std::errc::result_out_of_range) {
                            content_length = MAX_BODY_SIZE + 1;
                        }
                    }
                }

                if (header_end_pos > MAX_HEADER_SIZE) {
                    send_error(client_fd, 431, "Request Header Fields Too Large", "Request headers are too large.");
                    return;
                }
                if (content_length > MAX_BODY_SIZE) {
                    send_error(client_fd, 413, "Payload Too Large", "Request body exceeds the 16 MiB limit.");
                    return;
                }
            }

            const size_t body_received = raw_request.size() - (header_end_pos + 4);
            if (body_received > MAX_BODY_SIZE) {
                send_error(client_fd, 413, "Payload Too Large", "Request body exceeds the 16 MiB limit.");
                return;
            }
            if (body_received >= content_length) {
                complete = true;
                break;
            }
        }

        if (raw_request.empty()) {
            return;
        }
        if (!complete) {
            send_error(client_fd, 400, "Bad Request", "Incomplete request.");
            return;
        }

        const HttpRequest req = HttpRequestParser::parse(raw_request);
        const Router router(engine_);
        const HttpResponse res = router.route(req);
        send_all(client_fd, res.to_string());
    } catch (const std::exception& e) {
        std::cerr << "[HttpServer] Request failed: " << e.what() << std::endl;
        try {
            send_error(client_fd, 500, "Internal Server Error", "Internal server error.");
        } catch (...) {
        }
    }
}

void HttpServer::stop() {
    is_running_.store(false);

    // exchange() guarantees the listening socket is closed exactly once even if stop()
    // races with another caller or with the accept loop.
    const socket_t fd = server_fd_.exchange(static_cast<socket_t>(-1));
#ifdef _WIN32
    if (fd != INVALID_SOCKET) {
        shutdown(fd, SD_BOTH);
#else
    if (fd >= 0) {
        // On Linux, close() alone does not wake a thread blocked in accept(); shutdown() does.
        shutdown(fd, SHUT_RDWR);
#endif
        CLOSE_SOCKET(fd);
    }
}
