#include "Router.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

Router::Router(std::shared_ptr<engine::IndexBase> index)
    : index_(std::move(index)), file_handler_("public") {
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
    if (index_) {
        ss << ",\"index_loaded\":true"
           << ",\"dimension\":" << index_->dimension()
           << ",\"total_vectors\":" << index_->size();
    } else {
        ss << ",\"index_loaded\":false";
    }
    ss << "}";

    res.set_body(ss.str());
    return res;
}

bool Router::parse_search_json(const std::string& body, std::vector<float>& out_vec, size_t& out_k) {
    out_k = 10;
    out_vec.clear();

    auto k_pos = body.find("\"k\"");
    if (k_pos != std::string::npos) {
        auto colon = body.find(':', k_pos);
        if (colon != std::string::npos) {
            out_k = std::strtoul(body.c_str() + colon + 1, nullptr, 10);
            if (out_k == 0) out_k = 10;
        }
    }

    auto vec_pos = body.find("\"vector\"");
    if (vec_pos == std::string::npos) return false;

    auto open_bracket = body.find('[', vec_pos);
    auto close_bracket = body.find(']', open_bracket);
    if (open_bracket == std::string::npos || close_bracket == std::string::npos) return false;

    const char* p = body.c_str() + open_bracket + 1;
    const char* end = body.c_str() + close_bracket;

    while (p < end) {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
        if (p >= end) break;
        char* next_p = nullptr;
        float val = std::strtof(p, &next_p);
        if (next_p == p) break;
        out_vec.push_back(val);
        p = next_p;
    }

    return !out_vec.empty();
}

HttpResponse Router::handle_api_search(const HttpRequest& req) const {
    HttpResponse res;
    res.set_header("Content-Type", "application/json");
    apply_cors_headers(res);

    if (!index_) {
        res.set_status(503, "Service Unavailable");
        res.set_body("{\"status\":\"error\",\"message\":\"Vector index is not initialized.\"}");
        return res;
    }

    std::vector<float> query_vec;
    size_t k = 10;
    if (!parse_search_json(req.body, query_vec, k)) {
        res.set_status(400, "Bad Request");
        res.set_body("{\"status\":\"error\",\"message\":\"Invalid JSON. Expected 'vector': [floats] and optional 'k'.\"}");
        return res;
    }

    if (query_vec.size() != index_->dimension()) {
        res.set_status(422, "Unprocessable Entity");
        std::ostringstream ss;
        ss << "{\"status\":\"error\",\"message\":\"Dimension mismatch. Expected " 
           << index_->dimension() << ", received " << query_vec.size() << ".\"}";
        res.set_body(ss.str());
        return res;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto results = index_->search(query_vec, k);
    auto t1 = std::chrono::steady_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5);
    ss << "{\"status\":\"success\",\"took_us\":" << duration_us
       << ",\"k\":" << k
       << ",\"results\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        ss << "{\"id\":" << results[i].id << ",\"distance\":" << results[i].distance << "}";
        if (i + 1 < results.size()) ss << ",";
    }
    ss << "]}";

    res.set_body(ss.str());
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
    if (req.method == "OPTIONS") {
        return handle_options();
    }

    if (req.path == "/api/status" && req.method == "GET") {
        return handle_api_status();
    }

    if (req.path == "/api/search" && req.method == "POST") {
        return handle_api_search(req);
    }

    std::string path_to_serve = (req.path == "/") ? "/index.html" : req.path;
    auto static_res = file_handler_.serve(path_to_serve);
    if (static_res.has_value()) {
        apply_cors_headers(*static_res);
        return std::move(*static_res);
    }

    return handle_not_found();
}
