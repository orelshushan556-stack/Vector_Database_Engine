// End-to-end tests of HttpServer over real loopback sockets. They also give ThreadSanitizer
// something to observe: start() runs on one thread while stop() is called from another.

#include <gtest/gtest.h>

#include "HttpServer.hpp"
#include "engine/core/vector_engine.hpp"
#include "engine/index/flat_index.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #define TEST_CLOSE_SOCKET(s) closesocket(s)
    using test_sock_t = SOCKET;
    constexpr test_sock_t kInvalidSock = INVALID_SOCKET;
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
    #define TEST_CLOSE_SOCKET(s) close(s)
    using test_sock_t = int;
    constexpr test_sock_t kInvalidSock = -1;
#endif

namespace {

using namespace std::chrono_literals;
constexpr uint16_t kPort = 38417;

// Connects (retrying while the server is still starting), sends raw bytes, reads until close.
std::string roundtrip(const std::string& request) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        test_sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd == kInvalidSock) return {};

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(kPort);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            TEST_CLOSE_SOCKET(fd);
            std::this_thread::sleep_for(20ms);
            continue;
        }

        send(fd, request.data(), static_cast<int>(request.size()), 0);
        std::string response;
        char buf[4096];
        while (true) {
            const int n = static_cast<int>(recv(fd, buf, sizeof(buf), 0));
            if (n <= 0) break;
            response.append(buf, static_cast<size_t>(n));
        }
        TEST_CLOSE_SOCKET(fd);
        return response;
    }
    return {};
}

class HttpServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        wal_path_ = "http_server_test.wal";
        std::filesystem::remove(wal_path_);
        engine_ = std::make_shared<engine::VectorEngine>(
            std::make_unique<engine::FlatIndex>(4, engine::MetricType::L2), wal_path_);
        server_ = std::make_unique<HttpServer>(kPort, engine_, 2);
        thread_ = std::thread([this] {
            server_->start();
            finished_.store(true);
        });
    }

    void TearDown() override {
        server_->stop();
        for (int i = 0; i < 250 && !finished_.load(); ++i) std::this_thread::sleep_for(20ms);
        if (finished_.load()) {
            thread_.join();
        } else {
            thread_.detach();  // would otherwise hang the suite
            ADD_FAILURE() << "start() did not return after stop()";
        }
        server_.reset();
        engine_.reset();
        std::filesystem::remove(wal_path_);
    }

    std::string wal_path_;
    std::shared_ptr<engine::VectorEngine> engine_;
    std::unique_ptr<HttpServer> server_;
    std::thread thread_;
    std::atomic<bool> finished_{false};
};

TEST_F(HttpServerTest, ServesStatusAndStopsWhenStopIsCalledFromAnotherThread) {
    const std::string response = roundtrip("GET /api/status HTTP/1.1\r\nHost: x\r\n\r\n");
    EXPECT_NE(response.find("200 OK"), std::string::npos) << response;
    EXPECT_NE(response.find("\"engine_ready\":true"), std::string::npos);
}

TEST_F(HttpServerTest, RejectsOversizedBodyWith413WithoutReadingIt) {
    const std::string response = roundtrip(
        "POST /api/insert HTTP/1.1\r\nContent-Length: 20000000\r\nContent-Type: application/json\r\n\r\n{}");
    EXPECT_NE(response.find("413"), std::string::npos) << response;
}

TEST_F(HttpServerTest, SurvivesMalformedHeadersAndKeepsServing) {
    // "X:" used to make the header parser throw, which leaked the client socket.
    const std::string bad = roundtrip("GET /api/status HTTP/1.1\r\nX:\r\nY:\r\n\r\n");
    EXPECT_NE(bad.find("200 OK"), std::string::npos) << bad;

    const std::string good = roundtrip("GET /api/status HTTP/1.1\r\n\r\n");
    EXPECT_NE(good.find("200 OK"), std::string::npos);
}

TEST_F(HttpServerTest, MapsBadClientInputToClientErrors) {
    auto post = [](const std::string& path, const std::string& body) {
        return roundtrip("POST " + path + " HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body);
    };

    EXPECT_NE(post("/api/insert", "{\"id\":1,\"vector\":[1,nan,3,4]}").find(" 400 "), std::string::npos);
    EXPECT_NE(post("/api/insert", "{\"id\":1,\"vector\":[1,2]}").find(" 422 "), std::string::npos);
    EXPECT_NE(post("/api/search", "{\"vector\":[1,2,3,4],\"k\":1000000}").find(" 400 "), std::string::npos);
    EXPECT_NE(post("/api/insert", "{\"id\":7,\"vector\":[1,2,3,4]}").find(" 200 "), std::string::npos);
    EXPECT_EQ(engine_->size(), 1u);
}

}  // namespace
