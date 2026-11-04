#include "Router.hpp"
#include "JsonUtils.hpp"

#include <chrono>
#include <format>
#include <string>
#include <vector>

Router::Router(std::shared_ptr<engine::VectorEngine> engine)
    : engine_(std::move(engine)), file_handler_("public") {
}

constexpr HttpMethod Router::resolve_method(std::string_view method) noexcept {
    if (method == "GET")     return HttpMethod::Get;
    if (method == "POST")    return HttpMethod::Post;
    if (method == "OPTIONS") return HttpMethod::Options;
    return HttpMethod::Unknown;
}

constexpr RouteEndpoint Router::resolve_endpoint(std::string_view path) noexcept {
    if (path == "/api/status")     return RouteEndpoint::ApiStatus;
    if (path == "/api/search")     return RouteEndpoint::ApiSearch;
    if (path == "/api/insert")     return RouteEndpoint::ApiInsert;
    if (path == "/api/checkpoint") return RouteEndpoint::ApiCheckpoint;
    return RouteEndpoint::StaticAsset;
}

void Router::apply_cors_headers(HttpResponse& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

HttpResponse Router::handle_options() const {
    HttpResponse res(204, "No Content");
    apply_cors_headers(res);
    return res;
}

HttpResponse Router::handle_api_status() const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);

    if (engine_) {
        res.set_body(std::format(
            "{{\"status\":\"ok\",\"server\":\"Nexus Vector Database Engine\",\"engine_ready\":true,\"dimension\":{},\"total_vectors\":{}}}",
            engine_->dimension(),
            engine_->size()
        ));
    } else {
        res.set_body("{\"status\":\"ok\",\"server\":\"Nexus Vector Database Engine\",\"engine_ready\":false}");
    }

    return res;
}

HttpResponse Router::handle_api_search(const HttpRequest& req) const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);

    if (!engine_) {
        res.set_status(503, "Service Unavailable");
        res.set_body("{\"status\":\"error\",\"message\":\"Vector engine is not initialized.\"}");
        return res;
    }

    std::vector<float> query_vec;
    size_t k = 10;
    if (!server::json::parse_search_request(req.body, query_vec, k)) {
        res.set_status(400, "Bad Request");
        res.set_body("{\"status\":\"error\",\"message\":\"Invalid JSON. Expected 'vector': [floats] and optional 'k'.\"}");
        return res;
    }

    if (query_vec.size() != engine_->dimension()) {
        res.set_status(422, "Unprocessable Entity");
        res.set_body(std::format(
            "{{\"status\":\"error\",\"message\":\"Dimension mismatch. Expected {}, received {}.\"}}\"",
            engine_->dimension(),
            query_vec.size()
        ));
        return res;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto results = engine_->search(query_vec, k);
    const auto t1 = std::chrono::steady_clock::now();
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    res.set_body(server::json::build_search_response(results, k, duration_us));
    return res;
}

HttpResponse Router::handle_api_insert(const HttpRequest& req) const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);

    if (!engine_) {
        res.set_status(503, "Service Unavailable");
        res.set_body("{\"status\":\"error\",\"message\":\"Vector engine is not initialized.\"}");
        return res;
    }

    uint64_t id = 0;
    std::vector<float> vec;
    if (!server::json::parse_insert_request(req.body, id, vec)) {
        res.set_status(400, "Bad Request");
        res.set_body("{\"status\":\"error\",\"message\":\"Invalid JSON. Expected 'id': uint and 'vector': [floats].\"}");
        return res;
    }

    if (vec.size() != engine_->dimension()) {
        res.set_status(422, "Unprocessable Entity");
        res.set_body(std::format(
            "{{\"status\":\"error\",\"message\":\"Dimension mismatch. Expected {}, received {}.\"}}\"",
            engine_->dimension(),
            vec.size()
        ));
        return res;
    }

    const auto t0 = std::chrono::steady_clock::now();
    engine_->insert(id, vec);
    const auto t1 = std::chrono::steady_clock::now();
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    res.set_body(std::format(
        "{{\"status\":\"success\",\"message\":\"Vector inserted and persisted to WAL\",\"id\":{},\"took_us\":{}}}",
        id,
        duration_us
    ));
    return res;
}

HttpResponse Router::handle_api_checkpoint() const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);

    if (!engine_) {
        res.set_status(503, "Service Unavailable");
        res.set_body("{\"status\":\"error\",\"message\":\"Vector engine is not initialized.\"}");
        return res;
    }

    try {
        engine_->create_checkpoint("index.bin");
        res.set_body("{\"status\":\"success\",\"message\":\"Checkpoint created successfully and WAL truncated.\"}");
    } catch (const std::exception& e) {
        res.set_status(500, "Internal Server Error");
        res.set_body(std::format("{{\"status\":\"error\",\"message\":\"Checkpoint failed: {}\"}}", e.what()));
    }

    return res;
}

HttpResponse Router::handle_static(std::string_view path) const {
    const std::string resolved_path = (path == "/") ? "/index.html" : std::string(path);
    auto static_res = file_handler_.serve(resolved_path);
    if (static_res.has_value()) {
        apply_cors_headers(*static_res);
        return std::move(*static_res);
    }
    return handle_not_found();
}

HttpResponse Router::handle_not_found() const {
    HttpResponse res;
    res.set_status(404, "Not Found");
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);
    res.set_body("{\"status\":\"error\",\"message\":\"404 Endpoint Not Found\"}");
    return res;
}

HttpResponse Router::handle_method_not_allowed() const {
    HttpResponse res;
    res.set_status(405, "Method Not Allowed");
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);
    res.set_body("{\"status\":\"error\",\"message\":\"405 HTTP Method Not Allowed\"}");
    return res;
}

HttpResponse Router::route(const HttpRequest& req) const {
    const HttpMethod method = resolve_method(req.method);
    if (method == HttpMethod::Options) {
        return handle_options();
    }

    switch (resolve_endpoint(req.path)) {
        case RouteEndpoint::ApiStatus:
            return (method == HttpMethod::Get) ? handle_api_status() : handle_method_not_allowed();

        case RouteEndpoint::ApiSearch:
            return (method == HttpMethod::Post) ? handle_api_search(req) : handle_method_not_allowed();

        case RouteEndpoint::ApiInsert:
            return (method == HttpMethod::Post) ? handle_api_insert(req) : handle_method_not_allowed();

        case RouteEndpoint::ApiCheckpoint:
            return (method == HttpMethod::Post) ? handle_api_checkpoint() : handle_method_not_allowed();

        case RouteEndpoint::StaticAsset:
            return (method == HttpMethod::Get) ? handle_static(req.path) : handle_method_not_allowed();
    }

    return handle_not_found();
}
