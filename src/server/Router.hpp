#pragma once

#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "StaticFileHandler.hpp"
#include "engine/core/vector_engine.hpp"

#include <memory>

class Router {
public:
    explicit Router(std::shared_ptr<engine::VectorEngine> engine = nullptr);
    HttpResponse route(const HttpRequest& req) const;

private:
    std::shared_ptr<engine::VectorEngine> engine_;
    StaticFileHandler file_handler_;

    HttpResponse route_impl(const HttpRequest& req) const;
    HttpResponse handle_options() const;
    HttpResponse handle_api_status() const;
    HttpResponse handle_api_concepts() const;
    HttpResponse handle_api_search(const HttpRequest& req) const;
    HttpResponse handle_api_insert(const HttpRequest& req) const;
    HttpResponse handle_api_checkpoint(const HttpRequest& req) const;
    HttpResponse handle_not_found() const;

    static void apply_cors_headers(HttpResponse& res);
};
