#pragma once
#include <fstream>

#include <string_view>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include "HttpResponse.hpp"

class StaticFileHandler {
public:
    explicit StaticFileHandler(const std::filesystem::path& public_root = "public");

    [[nodiscard]] std::optional<HttpResponse> serve(std::string_view raw_path) const;

private:
    static std::string_view get_mime_type(const std::filesystem::path& file_path);
    [[nodiscard]] bool is_safe_path(const std::filesystem::path& resolved_path) const;

    std::filesystem::path public_root_;
};