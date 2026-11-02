#include "HttpResponse.hpp"

HttpResponse::HttpResponse() : status_code_(200), status_message_("OK") {
}

HttpResponse::HttpResponse(int status_code, const std::string& status_message)
    : status_code_(status_code), status_message_(status_message) {
}

void HttpResponse::set_status(int status_code, const std::string& status_message) {
    status_code_ = status_code;
    status_message_ = status_message;
}

void HttpResponse::set_header(const std::string& key, const std::string& value) {
    headers_[key] = value;
}

void HttpResponse::set_body(const std::string& body) {
    body_ = body;
    set_header("Content-Length", std::to_string(body_.length()));
}

void HttpResponse::set_body(std::string&& body) {
    body_ = std::move(body);
    set_header("Content-Length", std::to_string(body_.length()));
}

int HttpResponse::get_status_code() const {
    return status_code_;
}

const std::string& HttpResponse::get_status_message() const {
    return status_message_;
}

const std::unordered_map<std::string, std::string>& HttpResponse::get_headers() const {
    return headers_;
}

const std::string& HttpResponse::get_body() const {
    return body_;
}

std::string HttpResponse::to_string() const {
    std::string response = "HTTP/1.1 " + std::to_string(status_code_) + " " + status_message_ + "\r\n";

    for (const auto& pair : headers_) {
        response += pair.first + ": " + pair.second + "\r\n";
    }

    response += "\r\n";
    response += body_;

    return response;
}
