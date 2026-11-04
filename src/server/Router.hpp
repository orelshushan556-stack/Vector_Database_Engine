#pragma once

#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "StaticFileHandler.hpp"
#include "engine/core/vector_engine.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

enum class HttpMethod : uint8_t {
    Get,
    Post,
    Options,
    Unknown
};

enum class RouteEndpoint : uint8_t {
    ApiStatus,
    ApiSearch,
    ApiInsert,
    ApiCheckpoint,
    StaticAsset
};

class Router {
public:
    explicit Router(std::shared_ptr<engine::VectorEngine> engine = nullptr);
    [[nodiscard]] HttpResponse route(const HttpRequest& req) const;

private:
    std::shared_ptr<engine::VectorEngine> engine_;
    StaticFileHandler file_handler_;

    [[nodiscard]] static constexpr HttpMethod resolve_method(std::string_view method) noexcept;
    [[nodiscard]] static constexpr RouteEndpoint resolve_endpoint(std::string_view path) noexcept;

    [[nodiscard]] HttpResponse handle_options() const;
    [[nodiscard]] HttpResponse handle_api_status() const;
    [[nodiscard]] HttpResponse handle_api_search(const HttpRequest& req) const;
    [[nodiscard]] HttpResponse handle_api_insert(const HttpRequest& req) const;
    [[nodiscard]] HttpResponse handle_api_checkpoint() const;
    [[nodiscard]] HttpResponse handle_static(std::string_view path) const;
    [[nodiscard]] HttpResponse handle_not_found() const;
    [[nodiscard]] HttpResponse handle_method_not_allowed() const;

    static void apply_cors_headers(HttpResponse& res);
};
