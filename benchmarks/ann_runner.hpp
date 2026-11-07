#pragma once

// Measurement loop shared by every ANN harness (nexus_bench and hnswlib_bench), so both
// engines are timed by identical code.
//
// One measurement point:
//   1. recall pass: every query once; Recall@k against the ground truth and (if a counter
//      is supplied) the mean number of distance evaluations per query;
//   2. `repeats` timed passes over all queries, each query timed individually with
//      steady_clock. QPS = 1e6 / mean latency (single thread, queries issued back to back).

#include "bench_common.hpp"

#include <functional>
#include <utility>

namespace bench {

struct DistanceCounts {
    double exact{0};
    double quantized{0};
};

struct PointResult {
    double recall{0};
    double recall_min_query{1};
    LatencyStats latency;
    double qps{0};
    bool counted{false};
    DistanceCounts dist;
};

// search(q_index) -> vector of result ids (any integer type), best first.
// counter_reset() / counter_read() may be empty functions when counting is not compiled in.
template <class SearchFn>
PointResult measure_point(const Matrix& queries,
                          const std::vector<std::vector<uint32_t>>& gt,
                          std::size_t k,
                          std::size_t repeats,
                          SearchFn&& search,
                          const std::function<void()>& counter_reset = {},
                          const std::function<DistanceCounts()>& counter_read = {}) {
    PointResult r;
    const std::size_t nq = queries.rows;
    double recall_sum = 0;
    DistanceCounts dsum;
    for (std::size_t q = 0; q < nq; ++q) {
        if (counter_reset) counter_reset();
        const auto ids = search(q);
        if (counter_read) {
            const DistanceCounts c = counter_read();
            dsum.exact += c.exact;
            dsum.quantized += c.quantized;
        }
        const double rc = recall_at_k(gt[q], ids, k);
        recall_sum += rc;
        r.recall_min_query = std::min(r.recall_min_query, rc);
    }
    r.recall = recall_sum / static_cast<double>(nq);
    if (counter_read) {
        r.counted = true;
        r.dist.exact = dsum.exact / static_cast<double>(nq);
        r.dist.quantized = dsum.quantized / static_cast<double>(nq);
    }

    std::vector<double> samples;
    samples.reserve(nq * repeats);
    std::size_t sink = 0;
    for (std::size_t rep = 0; rep < repeats; ++rep) {
        for (std::size_t q = 0; q < nq; ++q) {
            const auto t0 = Clock::now();
            const auto ids = search(q);
            const auto t1 = Clock::now();
            sink += ids.size();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    }
    if (sink == 0 && nq > 0 && repeats > 0) throw std::runtime_error("searches returned no results");
    r.latency = latency_stats(std::move(samples));
    r.qps = r.latency.mean_us > 0 ? 1e6 / r.latency.mean_us : 0;
    return r;
}

inline void write_point(Json& j, const PointResult& p) {
    j.kv("recall", p.recall).kv("recall_min_query", p.recall_min_query).kv("qps", p.qps);
    j.kv_stats("latency", p.latency);
    if (p.counted) {
        j.key("distances_per_query").begin_object()
            .kv("exact", p.dist.exact).kv("quantized", p.dist.quantized)
            .kv("total", p.dist.exact + p.dist.quantized)
            .end_object();
    }
}

inline void write_meta(Json& j, const Args& a, const std::string& engine, int pinned_cpu, bool pin_ok) {
    j.key("meta").begin_object()
        .kv("engine", engine)
        .kv("compiler", compiler_string())
        .kv("build_flags", build_flags())
        .kv("pinned_cpu", pinned_cpu)
        .kv("pin_ok", pin_ok)
        .kv("hardware_threads", static_cast<unsigned>(std::thread::hardware_concurrency()));
    j.key("args").begin_object();
    for (const char* k : {"base", "query", "gt", "metric", "index", "M", "efc", "ef", "rerank", "k",
                          "builds", "repeats", "max-base", "max-queries", "save", "load", "seed"}) {
        if (a.has(k)) j.kv(k, a.get(k));
    }
    j.end_object();
    j.end_object();
}

}  // namespace bench
