#pragma once

#include <string>
#include <unordered_map>
#include <utility>

class HttpResponse {
private:
    int status_code_;
    std::string status_message_;
    std::unordered_map<std::string, std::string> headers_;
    std::string body_;

public:
    HttpResponse();
    HttpResponse(int status_code, const std::string& status_message);

    HttpResponse(const HttpResponse&) = default;
    HttpResponse& operator=(const HttpResponse&) = default;
    HttpResponse(HttpResponse&&) noexcept = default;
    HttpResponse& operator=(HttpResponse&&) noexcept = default;
    ~HttpResponse() = default;

    void set_status(int status_code, const std::string& status_message);
    void set_header(const std::string& key, const std::string& value);

    void set_body(const std::string& body);
    void set_body(std::string&& body);

    int get_status_code() const;
    const std::string& get_status_message() const;
    const std::unordered_map<std::string, std::string>& get_headers() const;
    const std::string& get_body() const;

    std::string to_string() const;
};