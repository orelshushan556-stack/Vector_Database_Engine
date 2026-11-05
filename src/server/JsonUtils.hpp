#pragma once

#include "engine/index/index_base.hpp"
#include "ConceptRegistry.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace server::json {

// Upper bound on the client-supplied result count; larger values would size internal
// candidate buffers from untrusted input.
inline constexpr size_t MAX_SEARCH_K = 10000;

inline bool parse_vector_payload(const std::string& body, std::vector<float>& out_vec) {
    out_vec.clear();
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
        // strtof accepts "nan", "inf" and overflowing literals; a non-finite component
        // would corrupt distance ordering and be persisted to the WAL.
        if (!std::isfinite(val)) {
            out_vec.clear();
            return false;
        }
        out_vec.push_back(val);
        p = next_p;
    }

    return !out_vec.empty();
}

inline bool parse_search_request(const std::string& body, std::vector<float>& out_vec, size_t& out_k) {
    out_k = 10;
    auto k_pos = body.find("\"k\"");
    if (k_pos != std::string::npos) {
        auto colon = body.find(':', k_pos);
        if (colon != std::string::npos) {
            out_k = std::strtoul(body.c_str() + colon + 1, nullptr, 10);
            if (out_k == 0) out_k = 10;
        }
    }
    return parse_vector_payload(body, out_vec);
}

inline bool parse_insert_request(const std::string& body, uint64_t& out_id, std::vector<float>& out_vec) {
    auto id_pos = body.find("\"id\"");
    if (id_pos == std::string::npos) return false;

    auto colon = body.find(':', id_pos);
    if (colon == std::string::npos) return false;

    char* next_p = nullptr;
    out_id = std::strtoull(body.c_str() + colon + 1, &next_p, 10);
    if (next_p == body.c_str() + colon + 1) return false;

    return parse_vector_payload(body, out_vec);
}

inline std::string build_search_response(const std::vector<engine::SearchResult>& results, size_t k, int64_t duration_us) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5);
    ss << "{\"status\":\"success\",\"took_us\":" << duration_us
       << ",\"k\":" << k
       << ",\"results\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        ss << "{\"id\":" << results[i].id << ",\"distance\":" << results[i].distance;
        // Ids above 32 bits must not wrap into the concept id range.
        const auto* meta = results[i].id <= std::numeric_limits<uint32_t>::max()
            ? ConceptRegistry::instance().get_concept(static_cast<uint32_t>(results[i].id))
            : nullptr;
        if (meta) {
            ss << ",\"label\":\"" << meta->label << "\""
               << ",\"branch\":" << meta->branch
               << ",\"category\":\"" << meta->category << "\"";
        }
        ss << "}";
        if (i + 1 < results.size()) ss << ",";
    }
    ss << "]}";
    return ss.str();
}

} // namespace server::json
