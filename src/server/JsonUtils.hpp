#pragma once

#include "engine/index/index_base.hpp"
#include <charconv>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace server::json {

constexpr std::string_view trim_whitespace(std::string_view sv) noexcept {
    constexpr std::string_view whitespace = " \t\r\n,";
    const auto first = sv.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    return sv.substr(first);
}

inline std::optional<std::string_view> find_key_value(std::string_view body, std::string_view key) noexcept {
    const auto key_pos = body.find(key);
    if (key_pos == std::string_view::npos) {
        return std::nullopt;
    }
    const auto colon_pos = body.find(':', key_pos + key.size());
    if (colon_pos == std::string_view::npos) {
        return std::nullopt;
    }
    return trim_whitespace(body.substr(colon_pos + 1));
}

inline bool parse_vector_payload(std::string_view body, std::vector<float>& out_vec) {
    out_vec.clear();
    auto val_view = find_key_value(body, "\"vector\"");
    if (!val_view) {
        return false;
    }

    const auto open_bracket = val_view->find('[');
    const auto close_bracket = val_view->find(']', open_bracket);
    if (open_bracket == std::string_view::npos || close_bracket == std::string_view::npos) {
        return false;
    }

    std::string_view array_content = val_view->substr(open_bracket + 1, close_bracket - open_bracket - 1);

    while (!array_content.empty()) {
        array_content = trim_whitespace(array_content);
        if (array_content.empty()) {
            break;
        }

        float value = 0.0f;
        const auto* begin = array_content.data();
        const auto* end = begin + array_content.size();
        auto [ptr, ec] = std::from_chars(begin, end, value);

        if (ec != std::errc{}) {
            return false;
        }

        out_vec.push_back(value);
        array_content.remove_prefix(static_cast<size_t>(ptr - begin));
    }

    return !out_vec.empty();
}

inline bool parse_search_request(std::string_view body, std::vector<float>& out_vec, size_t& out_k) {
    out_k = 10;
    if (auto k_view = find_key_value(body, "\"k\"")) {
        size_t parsed_k = 0;
        const auto* begin = k_view->data();
        const auto* end = begin + k_view->size();
        auto [ptr, ec] = std::from_chars(begin, end, parsed_k);
        if (ec == std::errc{} && parsed_k > 0) {
            out_k = parsed_k;
        }
    }
    return parse_vector_payload(body, out_vec);
}

inline bool parse_insert_request(std::string_view body, uint64_t& out_id, std::vector<float>& out_vec) {
    auto id_view = find_key_value(body, "\"id\"");
    if (!id_view) {
        return false;
    }

    const auto* begin = id_view->data();
    const auto* end = begin + id_view->size();
    auto [ptr, ec] = std::from_chars(begin, end, out_id);
    if (ec != std::errc{}) {
        return false;
    }

    return parse_vector_payload(body, out_vec);
}

inline std::string build_search_response(std::span<const engine::SearchResult> results, size_t k, int64_t duration_us) {
    std::string out = std::format("{{\"status\":\"success\",\"took_us\":{},\"k\":{},\"results\":[", duration_us, k);

    for (size_t i = 0; i < results.size(); ++i) {
        out += std::format("{{\"id\":{},\"distance\":{:.5f}}}", results[i].id, results[i].distance);
        if (i + 1 < results.size()) {
            out += ',';
        }
    }

    out += "]}";
    return out;
}

} // namespace server::json
