#include "HttpRequestParser.hpp"
#include <sstream>

HttpRequest HttpRequestParser::parse(const std::string& raw_request) {
    HttpRequest req;
    std::istringstream stream(raw_request);
    std::string line;

    if (std::getline(stream, line)) {
        std::istringstream line_stream(line);
        line_stream >> req.method >> req.path >> req.version;
    }

    while (std::getline(stream, line)) {
        size_t pos = line.find(':');

        if (pos == std::string::npos) {
            break;
        }

        std::string key = line.substr(0, pos);
        // Header values may be empty ("X:") or lack the conventional space after the colon.
        std::string value = line.substr(pos + 1);

        while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        const size_t first = value.find_first_not_of(" \t");
        value.erase(0, first == std::string::npos ? value.size() : first);

        req.headers[key] = value;
    }

    std::string body_line;
    while (std::getline(stream, body_line)) {
        req.body += body_line + "\n";
    }

    return req;
}