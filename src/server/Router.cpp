#include "Router.hpp"
#include "JsonUtils.hpp"
#include "ConceptRegistry.hpp"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string error_body(const char* msg) {
    std::string out = "{\"status\":\"error\",\"message\":\"";
    for (const char* c = msg; *c; ++c) {
        if (*c == '"' || *c == '\\') out += '\\';
        if (static_cast<unsigned char>(*c) >= 0x20) out += *c;
    }
    out += "\"}";
    return out;
}
}

Router::Router(std::shared_ptr<engine::VectorEngine> engine)
    : engine_(std::move(engine)), file_handler_("public") {
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

    std::ostringstream ss;
    ss << "{\"status\":\"ok\",\"server\":\"Nexus Vector Database Engine\"";
    if (engine_) {
        ss << ",\"engine_ready\":true"
           << ",\"dimension\":" << engine_->dimension()
           << ",\"total_vectors\":" << engine_->size();
    } else {
        ss << ",\"engine_ready\":false";
    }
    ss << "}";

    res.set_body(ss.str());
    return res;
}

HttpResponse Router::handle_api_concepts() const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);
    res.set_body(ConceptRegistry::instance().get_precompiled_json());
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
        std::ostringstream ss;
        ss << "{\"status\":\"error\",\"message\":\"Dimension mismatch. Expected " 
           << engine_->dimension() << ", received " << query_vec.size() << ".\"}";
        res.set_body(ss.str());
        return res;
    }

    try {
        auto t0 = std::chrono::steady_clock::now();
        auto results = engine_->search(query_vec, k);
        auto t1 = std::chrono::steady_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        res.set_body(server::json::build_search_response(results, k, duration_us));
    } catch (const std::invalid_argument& e) {
        res.set_status(400, "Bad Request");
        res.set_body(error_body(e.what()));
    } catch (const std::exception&) {
        res.set_status(500, "Internal Server Error");
        res.set_body("{\"status\":\"error\",\"message\":\"Internal engine error during search.\"}");
    }
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
        std::ostringstream ss;
        ss << "{\"status\":\"error\",\"message\":\"Dimension mismatch. Expected " 
           << engine_->dimension() << ", received " << vec.size() << ".\"}";
        res.set_body(ss.str());
        return res;
    }

    try {
        auto t0 = std::chrono::steady_clock::now();
        engine_->insert(id, vec);
        auto t1 = std::chrono::steady_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::ostringstream ss;
        ss << "{\"status\":\"success\",\"message\":\"Vector inserted and persisted to WAL\",\"id\":"
           << id << ",\"took_us\":" << duration_us << "}";
        res.set_body(ss.str());
    } catch (const std::invalid_argument& e) {
        res.set_status(400, "Bad Request");
        res.set_body(error_body(e.what()));
    } catch (const std::exception&) {
        res.set_status(500, "Internal Server Error");
        res.set_body("{\"status\":\"error\",\"message\":\"Internal engine error during insert.\"}");
    }
    return res;
}

HttpResponse Router::handle_api_checkpoint(const HttpRequest& req) const {
    (void)req;
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
        std::ostringstream ss;
        ss << "{\"status\":\"error\",\"message\":\"Checkpoint failed: " << e.what() << "\"}";
        res.set_body(ss.str());
    }

    return res;
}

HttpResponse Router::handle_not_found() const {
    HttpResponse res;
    res.set_status(404, "Not Found");
    res.set_header("Content-Type", "text/html");
    apply_cors_headers(res);
    res.set_body("<h1>404 Not Found - Nexus Vector Server</h1>");
    return res;
}

HttpResponse Router::route(const HttpRequest& req) const {
    try {
        return route_impl(req);
    } catch (const std::exception&) {
        HttpResponse res(500, "Internal Server Error");
        res.set_header("Content-Type", "application/json");
        apply_cors_headers(res);
        res.set_body("{\"status\":\"error\",\"message\":\"Internal server error.\"}");
        return res;
    }
}

HttpResponse Router::route_impl(const HttpRequest& req) const {
    if (req.method == "OPTIONS") {
        return handle_options();
    }

    if (req.path == "/api/status" && req.method == "GET") {
        return handle_api_status();
    }

    if (req.path == "/api/concepts" && req.method == "GET") {
        return handle_api_concepts();
    }

    if (req.path == "/api/search" && req.method == "POST") {
        return handle_api_search(req);
    }

    if (req.path == "/api/insert" && req.method == "POST") {
        return handle_api_insert(req);
    }

    if (req.path == "/api/checkpoint" && req.method == "POST") {
        return handle_api_checkpoint(req);
    }

    std::string path_to_serve = (req.path == "/") ? "/index.html" : req.path;
    auto static_res = file_handler_.serve(path_to_serve);
    if (static_res.has_value()) {
        apply_cors_headers(*static_res);
        return std::move(*static_res);
    }

    return handle_not_found();
}
