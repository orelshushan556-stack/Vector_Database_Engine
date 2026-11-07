// hnswlib_bench: the same `ann` measurement as nexus_bench, for hnswlib (header-only,
// fetched by CMake when NEXUS_BENCH_HNSWLIB=ON). Built with the same compiler flags.
//
// Distance counting: when compiled with NEXUS_COUNT_DISTANCES the space's distance function
// is wrapped by a counting trampoline, so every distance evaluation hnswlib performs is
// counted exactly (hnswlib's own metric_distance_computations counts neighbor-list sizes
// rather than evaluations, and only when metrics collection is enabled).

#include "ann_runner.hpp"
#include "bench_common.hpp"

#include <hnswlib/hnswlib.h>

#include <filesystem>
#include <iostream>
#include <memory>

using namespace bench;

namespace {

thread_local uint64_t g_dist_count = 0;
hnswlib::DISTFUNC<float> g_inner = nullptr;

float counting_dist(const void* a, const void* b, const void* param) {
    ++g_dist_count;
    return g_inner(a, b, param);
}

class CountingSpace : public hnswlib::SpaceInterface<float> {
public:
    explicit CountingSpace(hnswlib::SpaceInterface<float>* inner) : inner_(inner) {
        g_inner = inner->get_dist_func();
    }
    size_t get_data_size() override { return inner_->get_data_size(); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return &counting_dist; }
    void* get_dist_func_param() override { return inner_->get_dist_func_param(); }

private:
    hnswlib::SpaceInterface<float>* inner_;
};

int cmd_ann(const Args& a) {
    const int pin = static_cast<int>(a.num("pin", -1));
    const bool pin_ok = pin_current_thread(pin);
    const auto mem_start = mem_info();
    const Matrix base = load_fvecs(a.req("base"), static_cast<std::size_t>(a.num("max-base", 0)));
    const Matrix queries = load_fvecs(a.req("query"), static_cast<std::size_t>(a.num("max-queries", 0)));
    const auto gt = load_ivecs(a.req("gt"), queries.rows);
    const auto mem_loaded = mem_info();

    const std::string metric = a.get("metric", "l2");
    const std::size_t M = static_cast<std::size_t>(a.num("M", 16));
    const std::size_t efc = static_cast<std::size_t>(a.num("efc", 200));
    const std::size_t k = static_cast<std::size_t>(a.num("k", 10));
    const std::size_t builds = static_cast<std::size_t>(a.num("builds", 1));
    const std::size_t repeats = static_cast<std::size_t>(a.num("repeats", 3));
    const std::size_t seed0 = static_cast<std::size_t>(a.num("seed", 100));
    const auto efs = a.list<std::size_t>("ef", {10, 20, 40, 80, 120, 200, 400, 800});

    std::unique_ptr<hnswlib::SpaceInterface<float>> raw_space;
    if (metric == "l2") raw_space = std::make_unique<hnswlib::L2Space>(base.dim);
    else if (metric == "ip") raw_space = std::make_unique<hnswlib::InnerProductSpace>(base.dim);
    else throw std::runtime_error("hnswlib_bench supports metric l2 or ip");
#if defined(NEXUS_COUNT_DISTANCES)
    CountingSpace counting(raw_space.get());
    hnswlib::SpaceInterface<float>* space = &counting;
    std::function<void()> reset = [] { g_dist_count = 0; };
    std::function<DistanceCounts()> read = [] { return DistanceCounts{static_cast<double>(g_dist_count), 0}; };
#else
    hnswlib::SpaceInterface<float>* space = raw_space.get();
    std::function<void()> reset;
    std::function<DistanceCounts()> read;
#endif

    Json j;
    j.begin_object();
    write_meta(j, a, "hnswlib", pin, pin_ok);
    j.key("dataset").begin_object()
        .kv("base_rows", static_cast<unsigned long long>(base.rows))
        .kv("dim", static_cast<unsigned long long>(base.dim))
        .kv("queries", static_cast<unsigned long long>(queries.rows))
        .end_object();
    j.kv("rss_start_bytes", static_cast<unsigned long long>(mem_start.rss));
    j.kv("rss_after_load_bytes", static_cast<unsigned long long>(mem_loaded.rss));
    j.key("builds").begin_array();

    for (std::size_t b = 0; b < builds; ++b) {
        std::cerr << "[hnswlib] build " << (b + 1) << "/" << builds << "\n";
        const std::size_t seed = seed0 + b;
        const auto rss0 = mem_info();
        double build_s = -1, load_s = -1, save_s = -1;
        const auto t0 = Clock::now();
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
        if (a.has("load")) {
            alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, a.get("load"), false, base.rows);
            load_s = seconds_since(t0);
        } else {
            alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, base.rows, M, efc, seed);
            for (std::size_t i = 0; i < base.rows; ++i) alg->addPoint(base.row(i), i);
            build_s = seconds_since(t0);
        }
        const auto rss1 = mem_info();
        unsigned long long snap_bytes = 0;
        if (a.has("save") && b == 0) {
            const auto ts = Clock::now();
            alg->saveIndex(a.get("save"));
            save_s = seconds_since(ts);
            snap_bytes = std::filesystem::file_size(a.get("save"));
        }
        j.begin_object().kv("build", static_cast<unsigned long long>(b)).kv("seed", static_cast<unsigned long long>(seed))
            .kv("build_seconds", build_s).kv("train_seconds", 0.0).kv("load_seconds", load_s).kv("save_seconds", save_s)
            .kv("snapshot_bytes", snap_bytes)
            .kv("rss_before_build_bytes", static_cast<unsigned long long>(rss0.rss))
            .kv("rss_after_build_bytes", static_cast<unsigned long long>(rss1.rss))
            .kv("index_rss_delta_bytes", static_cast<long long>(rss1.rss) - static_cast<long long>(rss0.rss))
            .kv("peak_rss_bytes", static_cast<unsigned long long>(rss1.peak_rss));
        j.key("points").begin_array();
        for (std::size_t ef : efs) {
            alg->setEf(ef);
            auto search = [&](std::size_t q) {
                auto pq = alg->searchKnn(queries.row(q), k);
                std::vector<std::size_t> ids(pq.size());
                for (std::size_t i = pq.size(); i-- > 0;) {
                    ids[i] = pq.top().second;
                    pq.pop();
                }
                return ids;
            };
            const PointResult p = measure_point(queries, gt, k, repeats, search, reset, read);
            j.begin_object().kv("ef", static_cast<unsigned long long>(ef)).kv("rerank", 0.0);
            write_point(j, p);
            j.end_object();
            std::cerr << "  ef=" << ef << " recall=" << p.recall << " qps=" << p.qps;
            if (p.counted) std::cerr << " dist=" << p.dist.exact;
            std::cerr << "\n";
        }
        j.end_array().end_object();
    }
    j.end_array().end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "ann") {
        std::cerr << "usage: hnswlib_bench ann [--opts]\n";
        return 2;
    }
    try {
        return cmd_ann(Args(argc, argv, 2));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
