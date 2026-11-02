#include "HttpServer.hpp"
#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "Router.hpp"

#include <iostream>
#include <cstring>
#include <string>

#ifdef _WIN32
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define CLOSE_SOCKET(s) close(s)
#endif

HttpServer::HttpServer(uint16_t port, std::shared_ptr<engine::IndexBase> index, size_t num_threads)
    : port_(port), index_(std::move(index)), server_fd_(static_cast<socket_t>(-1)), is_running_(false), thread_pool_(num_threads) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options" << std::endl;
        CLOSE_SOCKET(server_fd_);
        exit(EXIT_FAILURE);
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Error binding socket" << std::endl;
        CLOSE_SOCKET(server_fd_);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd_, SOMAXCONN) < 0) {
        std::cerr << "Error listening on socket" << std::endl;
        CLOSE_SOCKET(server_fd_);
        exit(EXIT_FAILURE);
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

void HttpServer::handle_client(socket_t client_fd) {
    std::string raw_request;
    char buffer[4096];
    size_t header_end_pos = std::string::npos;
    size_t content_length = 0;
    bool headers_parsed = false;

    while (true) {
        int bytes_read = recv(client_fd, buffer, sizeof(buffer), 0);
        if (bytes_read <= 0) break;
        raw_request.append(buffer, bytes_read);

        if (!headers_parsed) {
            header_end_pos = raw_request.find("\r\n\r\n");
            if (header_end_pos != std::string::npos) {
                headers_parsed = true;
                
                std::string cl_key = "Content-Length:";
                auto cl_pos = raw_request.find(cl_key);
                if (cl_pos == std::string::npos) {
                    cl_key = "content-length:";
                    cl_pos = raw_request.find(cl_key);
                }

                if (cl_pos != std::string::npos && cl_pos < header_end_pos) {
                    size_t val_start = raw_request.find_first_not_of(" \t", cl_pos + cl_key.length());
                    size_t val_end = raw_request.find("\r\n", val_start);
                    if (val_start != std::string::npos && val_end != std::string::npos) {
                        try {
                            content_length = std::stoul(raw_request.substr(val_start, val_end - val_start));
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }
        }

        if (headers_parsed) {
            size_t body_received = raw_request.size() - (header_end_pos + 4);
            if (body_received >= content_length) {
                break;
            }
        }
    }

    if (!raw_request.empty()) {
        HttpRequest req = HttpRequestParser::parse(raw_request);
        Router router(index_);
        HttpResponse res = router.route(req);

        std::string response_str = res.to_string();
        send(client_fd, response_str.c_str(), static_cast<int>(response_str.size()), 0);
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
