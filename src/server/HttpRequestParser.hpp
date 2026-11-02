#pragma once

#include <string>
#include <unordered_map>

struct HttpRequest {
    std::string method;
    std::string path;
    std::string version;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
};

class HttpRequestParser {
public:
    static HttpRequest parse(const std::string& raw_request);
};