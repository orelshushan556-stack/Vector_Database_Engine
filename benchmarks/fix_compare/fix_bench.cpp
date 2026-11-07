// fix_bench: N independent HNSW builds on one dataset, efSearch sweep per build.
// Uses only the public HNSWIndex / HNSWIndexSQ8 API that is unchanged between the compared
// revisions, so the same file compiles against the pre-fix and post-fix engine trees.

#include "ann_runner.hpp"
#include "bench_common.hpp"

#include "engine/index/hnsw_index.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"

#include <iostream>
#include <memory>

using namespace bench;
using namespace engine;

int main(int argc, char** argv) {
    try {
        const Args a(argc, argv, 1);
        const bool pin_ok = pin_current_thread(static_cast<int>(a.num("pin", -1)));
        const Matrix base = load_fvecs(a.req("base"));
        const Matrix queries = load_fvecs(a.req("query"));
        const auto gt = load_ivecs(a.req("gt"), queries.rows);
        const std::string kind = a.get("index", "hnsw");
        const std::size_t builds = static_cast<std::size_t>(a.num("builds", 10));
        const std::size_t repeats = static_cast<std::size_t>(a.num("repeats", 5));
        const auto efs = a.list<std::size_t>("ef", {10, 16, 32, 64, 100});

        Json j;
        j.begin_object();
        write_meta(j, a, "nexus-revision", static_cast<int>(a.num("pin", -1)), pin_ok);
        j.key("builds").begin_array();
        for (std::size_t b = 0; b < builds; ++b) {
            const auto t0 = Clock::now();
            std::unique_ptr<HNSWIndex> h;
            std::unique_ptr<HNSWIndexSQ8> s;
            if (kind == "sq8") {
                ScalarQuantizer sq(base.dim);
                sq.train(base.data);
                s = std::make_unique<HNSWIndexSQ8>(base.dim, MetricType::L2, sq, 16, 200, 50, 2.0f);
                for (std::size_t i = 0; i < base.rows; ++i) s->add(i, VectorView(base.row(i), base.dim));
            } else {
                h = std::make_unique<HNSWIndex>(base.dim, MetricType::L2, 16, 200, 50);
                for (std::size_t i = 0; i < base.rows; ++i) h->add(i, VectorView(base.row(i), base.dim));
            }
            const double build_s = seconds_since(t0);
            j.begin_object().kv("build", static_cast<unsigned long long>(b)).kv("build_seconds", build_s);
            j.key("points").begin_array();
            for (std::size_t ef : efs) {
                auto ids = [&](std::size_t q) {
                    const VectorView qv(queries.row(q), queries.dim);
                    const auto r = s ? s->search(qv, 10) : h->search(qv, 10);
                    std::vector<VectorId> out;
                    for (const auto& x : r) out.push_back(x.id);
                    return out;
                };
                if (s) s->set_ef_search(ef); else h->set_ef_search(ef);
                const PointResult p = measure_point(queries, gt, 10, repeats, ids);
                j.begin_object().kv("ef", static_cast<unsigned long long>(ef)).kv("rerank", s ? 2.0 : 0.0);
                write_point(j, p);
                j.end_object();
            }
            j.end_array().end_object();
            std::cerr << "[fix_bench] " << kind << " build " << b << " done in " << build_s << "s\n";
        }
        j.end_array().end_object();
        write_text(a.get("out", "-"), j.str());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
