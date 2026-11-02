#pragma once

#include "HttpRequestParser.hpp"
#include "HttpResponse.hpp"
#include "StaticFileHandler.hpp"
#include "engine/index/index_base.hpp"
#include <memory>
#include <vector>

class Router {
public:
    explicit Router(std::shared_ptr<engine::IndexBase> index = nullptr);
    HttpResponse route(const HttpRequest& req) const;

private:
    std::shared_ptr<engine::IndexBase> index_;
    StaticFileHandler file_handler_;

    HttpResponse handle_options() const;
    HttpResponse handle_api_status() const;
    HttpResponse handle_api_search(const HttpRequest& req) const;
    HttpResponse handle_not_found() const;

    static void apply_cors_headers(HttpResponse& res);
    static bool parse_search_json(const std::string& body, std::vector<float>& out_vec, size_t& out_k);
};
