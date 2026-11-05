#include "HttpServer.hpp"
#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "Router.hpp"

#include <array>
#include <charconv>
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
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    if (server_fd_ == INVALID_SOCKET) {
#else
    if (server_fd_ < 0) {
#endif
        std::cerr << "Error creating socket" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options" << std::endl;
        CLOSE_SOCKET(server_fd_);
        std::exit(EXIT_FAILURE);
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Error binding socket" << std::endl;
        CLOSE_SOCKET(server_fd_);
        std::exit(EXIT_FAILURE);
    }

    if (listen(server_fd_, SOMAXCONN) < 0) {
        std::cerr << "Error listening on socket" << std::endl;
        CLOSE_SOCKET(server_fd_);
        std::exit(EXIT_FAILURE);
    }
}

void HttpServer::start() {
    setup_socket();
    is_running_ = true;
    std::cout << "[*] nexus_server online at http://localhost:" << port_ << std::endl;

    while (is_running_) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
        SOCKET client_fd = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd == INVALID_SOCKET) {
#else
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
#endif
            if (!is_running_) break;
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

void send_error_and_close(socket_t client_fd, int status, const char* reason, const char* message) {
    HttpResponse res(status, reason);
    res.set_header("Content-Type", "application/json");
    res.set_header("Connection", "close");
    res.set_body(std::string("{\"status\":\"error\",\"message\":\"") + message + "\"}");
    const std::string out = res.to_string();
    send(client_fd, out.data(), static_cast<int>(out.size()), 0);
    CLOSE_SOCKET(client_fd);
}

} // namespace

void HttpServer::handle_client(socket_t client_fd) {
    std::string raw_request;
    std::array<char, 4096> buffer{};
    size_t header_end_pos = std::string::npos;
    size_t content_length = 0;
    bool headers_parsed = false;

    while (true) {
        const int bytes_read = recv(client_fd, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (bytes_read <= 0) break;
        raw_request.append(buffer.data(), static_cast<size_t>(bytes_read));

        if (!headers_parsed && raw_request.size() > MAX_HEADER_SIZE + MAX_BODY_SIZE) {
            send_error_and_close(client_fd, 413, "Payload Too Large", "Request exceeds the maximum allowed size.");
            return;
        }

        if (!headers_parsed) {
            std::string_view req_view(raw_request);
            header_end_pos = req_view.find("\r\n\r\n");
            if (header_end_pos != std::string_view::npos) {
                headers_parsed = true;
                std::string_view headers_view = req_view.substr(0, header_end_pos);

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
                    send_error_and_close(client_fd, 431, "Request Header Fields Too Large", "Request headers are too large.");
                    return;
                }
                if (content_length > MAX_BODY_SIZE) {
                    send_error_and_close(client_fd, 413, "Payload Too Large", "Request body exceeds the 16 MiB limit.");
                    return;
                }
            }
        }

        if (headers_parsed) {
            const size_t body_received = raw_request.size() - (header_end_pos + 4);
            if (body_received > MAX_BODY_SIZE) {
                send_error_and_close(client_fd, 413, "Payload Too Large", "Request body exceeds the 16 MiB limit.");
                return;
            }
            if (body_received >= content_length) {
                break;
            }
        }
    }

    if (!raw_request.empty()) {
        const HttpRequest req = HttpRequestParser::parse(raw_request);
        const Router router(engine_);
        const HttpResponse res = router.route(req);

        const std::string response_str = res.to_string();
        send(client_fd, response_str.data(), static_cast<int>(response_str.size()), 0);
    }

    CLOSE_SOCKET(client_fd);
}

void HttpServer::stop() {
    is_running_ = false;
#ifdef _WIN32
    if (server_fd_ != INVALID_SOCKET) {
#else
    if (server_fd_ >= 0) {
#endif
        CLOSE_SOCKET(server_fd_);
        server_fd_ = static_cast<socket_t>(-1);
    }
}
