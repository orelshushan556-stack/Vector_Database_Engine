#pragma once

// Shared helpers for the benchmark executables: dataset I/O, recall, timing statistics,
// process memory, thread pinning and a minimal JSON writer. Depends only on the standard
// library and the OS API, so it also compiles against older engine revisions
// (benchmarks/fix_compare).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace bench {

// ---------------------------------------------------------------------------------------
// Dataset I/O (.fvecs / .ivecs, little-endian, each row prefixed by its int32 dimension)
// ---------------------------------------------------------------------------------------
struct Matrix {
    std::vector<float> data;
    std::size_t rows{0};
    std::size_t dim{0};
    const float* row(std::size_t i) const { return data.data() + i * dim; }
};

inline Matrix load_fvecs(const std::string& path, std::size_t max_rows = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), 4);
    if (!f || d <= 0) throw std::runtime_error("bad fvecs header: " + path);
    f.seekg(0, std::ios::end);
    const std::size_t bytes = static_cast<std::size_t>(f.tellg());
    const std::size_t row_bytes = 4 + static_cast<std::size_t>(d) * 4;
    Matrix m;
    m.dim = static_cast<std::size_t>(d);
    m.rows = bytes / row_bytes;
    if (max_rows && max_rows < m.rows) m.rows = max_rows;
    m.data.resize(m.rows * m.dim);
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(row_bytes * 4096);
    std::size_t done = 0;
    while (done < m.rows) {
        const std::size_t n = std::min<std::size_t>(4096, m.rows - done);
        f.read(buf.data(), static_cast<std::streamsize>(n * row_bytes));
        for (std::size_t i = 0; i < n; ++i) {
            std::copy_n(reinterpret_cast<const float*>(buf.data() + i * row_bytes + 4), m.dim,
                        m.data.data() + (done + i) * m.dim);
        }
        done += n;
    }
    return m;
}

inline std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, std::size_t max_rows = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<uint32_t>> out;
    int32_t k = 0;
    while (f.read(reinterpret_cast<char*>(&k), 4)) {
        std::vector<uint32_t> row(static_cast<std::size_t>(k));
        f.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(k) * 4);
        out.push_back(std::move(row));
        if (max_rows && out.size() >= max_rows) break;
    }
    return out;
}

// Recall@k: |top-k returned ∩ true top-k| / k.
template <class IdRange>
double recall_at_k(const std::vector<uint32_t>& gt, const IdRange& ids, std::size_t k) {
    std::unordered_set<uint32_t> truth(gt.begin(), gt.begin() + static_cast<std::ptrdiff_t>(std::min(k, gt.size())));
    std::size_t hit = 0, n = 0;
    for (auto id : ids) {
        if (n++ >= k) break;
        if (truth.count(static_cast<uint32_t>(id))) ++hit;
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

// ---------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct LatencyStats {
    double mean_us{0}, p50_us{0}, p95_us{0}, p99_us{0}, min_us{0}, max_us{0};
    std::size_t samples{0};
};

inline LatencyStats latency_stats(std::vector<double> us) {
    LatencyStats s;
    if (us.empty()) return s;
    std::sort(us.begin(), us.end());
    double sum = 0;
    for (double v : us) sum += v;
    auto pct = [&](double p) {
        const std::size_t idx = std::min(us.size() - 1, static_cast<std::size_t>(p * static_cast<double>(us.size())));
        return us[idx];
    };
    s.samples = us.size();
    s.mean_us = sum / static_cast<double>(us.size());
    s.p50_us = pct(0.50);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    s.min_us = us.front();
    s.max_us = us.back();
    return s;
}

// ---------------------------------------------------------------------------------------
// Process memory (bytes)
// ---------------------------------------------------------------------------------------
struct MemInfo {
    uint64_t rss{0};
    uint64_t peak_rss{0};
};

inline MemInfo mem_info() {
    MemInfo m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        m.rss = pmc.WorkingSetSize;
        m.peak_rss = pmc.PeakWorkingSetSize;
    }
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        auto val = [&] { return std::stoull(line.substr(line.find(':') + 1)) * 1024ULL; };
        if (line.rfind("VmRSS:", 0) == 0) m.rss = val();
        if (line.rfind("VmHWM:", 0) == 0) m.peak_rss = val();
    }
#endif
    return m;
}

// ---------------------------------------------------------------------------------------
// Thread pinning: pin the calling thread to one logical CPU. Returns false if unsupported.
// ---------------------------------------------------------------------------------------
inline bool pin_current_thread(int cpu) {
    if (cpu < 0) return false;
#if defined(_WIN32)
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) != 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

inline std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

inline std::string build_flags() {
    std::string s;
#if defined(__AVX2__)
    s += "avx2 ";
#endif
#if defined(__FMA__)
    s += "fma ";
#endif
#if defined(__AVX512F__)
    s += "avx512f ";
#endif
#if defined(NDEBUG)
    s += "NDEBUG ";
#endif
#if defined(NEXUS_COUNT_DISTANCES)
    s += "NEXUS_COUNT_DISTANCES ";
#endif
#if defined(NEXUS_DISABLE_PREFETCH)
    s += "NEXUS_DISABLE_PREFETCH ";
#endif
#if defined(NEXUS_WAL_NO_FSYNC)
    s += "NEXUS_WAL_NO_FSYNC ";
#endif
    if (!s.empty()) s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------------------
// Minimal JSON writer (objects, arrays, numbers, strings, bools)
// ---------------------------------------------------------------------------------------
class Json {
public:
    Json& begin_object() { sep(); os_ << '{'; first_ = true; return *this; }
    Json& end_object() { os_ << '}'; first_ = false; return *this; }
    Json& begin_array() { sep(); os_ << '['; first_ = true; return *this; }
    Json& end_array() { os_ << ']'; first_ = false; return *this; }
    Json& key(const std::string& k) { sep(); str(k); os_ << ':'; first_ = true; return *this; }
    Json& value(const std::string& v) { sep(); str(v); return *this; }
    Json& value(const char* v) { return value(std::string(v)); }
    Json& value(bool v) { sep(); os_ << (v ? "true" : "false"); return *this; }
    Json& value(double v) {
        sep();
        if (!std::isfinite(v)) { os_ << "null"; return *this; }
        std::ostringstream t; t << std::setprecision(10) << v; os_ << t.str();
        return *this;
    }
    Json& value(int v) { sep(); os_ << v; return *this; }
    Json& value(long v) { sep(); os_ << v; return *this; }
    Json& value(long long v) { sep(); os_ << v; return *this; }
    Json& value(unsigned v) { sep(); os_ << v; return *this; }
    Json& value(unsigned long v) { sep(); os_ << v; return *this; }
    Json& value(unsigned long long v) { sep(); os_ << v; return *this; }
    template <class T> Json& kv(const std::string& k, const T& v) { key(k); return value(v); }
    Json& kv_stats(const std::string& k, const LatencyStats& s) {
        key(k).begin_object()
            .kv("mean_us", s.mean_us).kv("p50_us", s.p50_us).kv("p95_us", s.p95_us)
            .kv("p99_us", s.p99_us).kv("min_us", s.min_us).kv("max_us", s.max_us)
            .kv("samples", static_cast<unsigned long long>(s.samples))
            .end_object();
        return *this;
    }
    std::string str() const { return os_.str(); }

private:
    void sep() { if (!first_) os_ << ','; first_ = false; }
    void str(const std::string& s) {
        os_ << '"';
        for (char c : s) {
            switch (c) {
                case '"': os_ << "\\\""; break;
                case '\\': os_ << "\\\\"; break;
                case '\n': os_ << "\\n"; break;
                case '\r': os_ << "\\r"; break;
                case '\t': os_ << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) { os_ << ' '; } else { os_ << c; }
            }
        }
        os_ << '"';
    }
    std::ostringstream os_;
    bool first_{true};
};

inline void write_text(const std::string& path, const std::string& text) {
    if (path.empty() || path == "-") {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fputc('\n', stdout);
        return;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << text << '\n';
}

// ---------------------------------------------------------------------------------------
// Command-line parsing: --key value | --flag
// ---------------------------------------------------------------------------------------
class Args {
public:
    Args(int argc, char** argv, int start) {
        for (int i = start; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + a);
            a = a.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                kv_[a] = argv[++i];
            } else {
                kv_[a] = "1";
            }
        }
    }
    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    std::string req(const std::string& k) const {
        if (!has(k)) throw std::runtime_error("missing --" + k);
        return get(k);
    }
    long long num(const std::string& k, long long def) const { return has(k) ? std::stoll(get(k)) : def; }
    double real(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
    template <class T>
    std::vector<T> list(const std::string& k, const std::vector<T>& def) const {
        if (!has(k)) return def;
        std::vector<T> out;
        std::stringstream ss(get(k));
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) out.push_back(static_cast<T>(std::stod(item)));
        }
        return out;
    }

private:
    std::map<std::string, std::string> kv_;
};

}  // namespace bench
