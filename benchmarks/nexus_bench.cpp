// nexus_bench: benchmark driver for the Nexus engine. One executable, several subcommands;
// every subcommand writes one JSON document (--out file, or stdout).
//
//   ann            build (or load) an index, sweep efSearch (and re_rank_factor), report
//                  Recall@k, latency percentiles, QPS, build time, memory, distance counts
//   snapshot-load  time Serializer::load of a saved index
//   crash-writer   child process for the crash-recovery test: insert forever, ack each insert
//   crash-verify   recover a WAL after a kill and verify it against the acknowledged count
//   wal-replay     WAL replay / recovery time versus record count
//   write-path     insert throughput with and without the WAL, checkpoint and snapshot costs
//   concurrency    VectorEngine search throughput versus threads, optional read/write mix
//   simd           AVX2 versus scalar distance kernels per dimension
//
// Instrumented variants are separate builds (see engine/diagnostics.hpp):
//   NEXUS_COUNT_DISTANCES, NEXUS_DISABLE_PREFETCH, NEXUS_WAL_NO_FSYNC.

#include "ann_runner.hpp"
#include "bench_common.hpp"

#include "engine/core/vector_engine.hpp"
#include "engine/diagnostics.hpp"
#include "engine/distance/metrics.hpp"
#include "engine/distance/simd_metrics.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/index/hnsw_index.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/index/quantized_flat_index.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"
#include "engine/storage/wal_manager.hpp"

#include <atomic>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

using namespace engine;
using namespace bench;
namespace fs = std::filesystem;

namespace {

MetricType parse_metric(const std::string& m) {
    if (m == "l2") return MetricType::L2;
    if (m == "ip") return MetricType::DotProduct;
    if (m == "cos") return MetricType::Cosine;
    throw std::runtime_error("unknown metric " + m);
}

std::function<void()> counter_reset_fn() {
#if defined(NEXUS_COUNT_DISTANCES)
    return [] { diag::reset_distance_counters(); };
#else
    return {};
#endif
}

std::function<DistanceCounts()> counter_read_fn() {
#if defined(NEXUS_COUNT_DISTANCES)
    return [] {
        return DistanceCounts{static_cast<double>(diag::exact_distance_count),
                              static_cast<double>(diag::quantized_distance_count)};
    };
#else
    return {};
#endif
}

std::vector<VectorId> ids_of(const std::vector<SearchResult>& r) {
    std::vector<VectorId> out;
    out.reserve(r.size());
    for (const auto& x : r) out.push_back(x.id);
    return out;
}

// ---------------------------------------------------------------------------------------
// ann
// ---------------------------------------------------------------------------------------
int cmd_ann(const Args& a) {
    const int pin = static_cast<int>(a.num("pin", -1));
    const bool pin_ok = pin_current_thread(pin);

    const auto mem_start = mem_info();
    const Matrix base = load_fvecs(a.req("base"), static_cast<std::size_t>(a.num("max-base", 0)));
    const Matrix queries = load_fvecs(a.req("query"), static_cast<std::size_t>(a.num("max-queries", 0)));
    const auto gt = load_ivecs(a.req("gt"), queries.rows);
    if (gt.size() < queries.rows) throw std::runtime_error("ground truth shorter than query set");
    // Note: --max-base truncates the base set; the ground truth must then match the truncated set.
    const auto mem_loaded = mem_info();

    const std::string kind = a.get("index", "hnsw");
    const MetricType metric = parse_metric(a.get("metric", "l2"));
    const std::size_t M = static_cast<std::size_t>(a.num("M", 16));
    const std::size_t efc = static_cast<std::size_t>(a.num("efc", 200));
    const std::size_t k = static_cast<std::size_t>(a.num("k", 10));
    const std::size_t builds = static_cast<std::size_t>(a.num("builds", 1));
    const std::size_t repeats = static_cast<std::size_t>(a.num("repeats", 3));
    const auto efs = a.list<std::size_t>("ef", {10, 20, 40, 80, 120, 200, 400, 800});
    const auto reranks = a.list<float>("rerank", {2.0f});

    Json j;
    j.begin_object();
    write_meta(j, a, "nexus", pin, pin_ok);
    j.key("dataset").begin_object()
        .kv("base_rows", static_cast<unsigned long long>(base.rows))
        .kv("dim", static_cast<unsigned long long>(base.dim))
        .kv("queries", static_cast<unsigned long long>(queries.rows))
        .end_object();
    j.kv("rss_start_bytes", static_cast<unsigned long long>(mem_start.rss));
    j.kv("rss_after_load_bytes", static_cast<unsigned long long>(mem_loaded.rss));
    j.key("builds").begin_array();

    for (std::size_t b = 0; b < builds; ++b) {
        std::cerr << "[ann] " << kind << " build " << (b + 1) << "/" << builds << "\n";
        j.begin_object().kv("build", static_cast<unsigned long long>(b));
        const auto rss0 = mem_info();
        double train_s = 0, build_s = 0, load_s = -1, save_s = -1;
        unsigned long long snapshot_bytes = 0;

        std::unique_ptr<IndexBase> index;
        std::unique_ptr<QuantizedFlatIndex> qflat;

        const auto t0 = Clock::now();
        if (a.has("load")) {
            index = Serializer::load(a.get("load"));
            load_s = seconds_since(t0);
        } else if (kind == "hnsw") {
            auto h = std::make_unique<HNSWIndex>(base.dim, metric, M, efc, 50);
            h->reserve(base.rows);
            for (std::size_t i = 0; i < base.rows; ++i) h->add(i, VectorView(base.row(i), base.dim));
            index = std::move(h);
        } else if (kind == "sq8") {
            ScalarQuantizer sq(base.dim);
            sq.train(base.data);
            train_s = seconds_since(t0);
            auto h = std::make_unique<HNSWIndexSQ8>(base.dim, metric, sq, M, efc, 50, reranks.front());
            h->reserve(base.rows);
            for (std::size_t i = 0; i < base.rows; ++i) h->add(i, VectorView(base.row(i), base.dim));
            index = std::move(h);
        } else if (kind == "flat") {
            auto f = std::make_unique<FlatIndex>(base.dim, metric);
            f->reserve(base.rows);
            for (std::size_t i = 0; i < base.rows; ++i) f->add(i, VectorView(base.row(i), base.dim));
            index = std::move(f);
        } else if (kind == "qflat") {
            qflat = std::make_unique<QuantizedFlatIndex>(base.dim);
            qflat->train(base.data);
            train_s = seconds_since(t0);
            qflat->add_batch(base.data);
        } else {
            throw std::runtime_error("unknown index " + kind);
        }
        build_s = seconds_since(t0);
        const auto rss1 = mem_info();

        if (a.has("save") && b == 0 && index) {
            const auto ts = Clock::now();
            Serializer::save(*index, a.get("save"));
            save_s = seconds_since(ts);
            snapshot_bytes = fs::file_size(a.get("save"));
        }

        j.kv("build_seconds", load_s >= 0 ? -1.0 : build_s)
            .kv("train_seconds", train_s)
            .kv("load_seconds", load_s)
            .kv("save_seconds", save_s)
            .kv("snapshot_bytes", snapshot_bytes)
            .kv("rss_before_build_bytes", static_cast<unsigned long long>(rss0.rss))
            .kv("rss_after_build_bytes", static_cast<unsigned long long>(rss1.rss))
            .kv("index_rss_delta_bytes", static_cast<long long>(rss1.rss) - static_cast<long long>(rss0.rss))
            .kv("peak_rss_bytes", static_cast<unsigned long long>(rss1.peak_rss));

        j.key("points").begin_array();
        auto run = [&](auto&& search, std::size_t ef, double rerank) {
            const PointResult p = measure_point(queries, gt, k, repeats, search, counter_reset_fn(), counter_read_fn());
            j.begin_object().kv("ef", static_cast<unsigned long long>(ef)).kv("rerank", rerank);
            write_point(j, p);
            j.end_object();
            std::cerr << "  ef=" << ef << " rr=" << rerank << " recall=" << p.recall << " qps=" << p.qps;
            if (p.counted) std::cerr << " dist=" << (p.dist.exact + p.dist.quantized);
            std::cerr << "\n";
        };

        if (qflat) {
            run([&](std::size_t q) { return ids_of(qflat->search(std::span<const float>(queries.row(q), queries.dim), k)); }, 0, 0);
        } else if (auto* f = dynamic_cast<FlatIndex*>(index.get())) {
            run([&](std::size_t q) { return ids_of(f->search(VectorView(queries.row(q), queries.dim), k)); }, 0, 0);
        } else if (auto* h = dynamic_cast<HNSWIndex*>(index.get())) {
            for (std::size_t ef : efs) {
                h->set_ef_search(ef);
                run([&](std::size_t q) { return ids_of(h->search(VectorView(queries.row(q), queries.dim), k)); }, ef, 0);
            }
        } else if (auto* s = dynamic_cast<HNSWIndexSQ8*>(index.get())) {
            for (float rr : reranks) {
                s->set_re_rank_factor(rr);
                for (std::size_t ef : efs) {
                    s->set_ef_search(ef);
                    run([&](std::size_t q) { return ids_of(s->search(VectorView(queries.row(q), queries.dim), k)); }, ef, rr);
                }
            }
        }
        j.end_array();
        j.end_object();
    }
    j.end_array();
    j.end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

// ---------------------------------------------------------------------------------------
// snapshot-load
// ---------------------------------------------------------------------------------------
int cmd_snapshot_load(const Args& a) {
    const std::string path = a.req("snapshot");
    const int reps = static_cast<int>(a.num("repeats", 3));
    Json j;
    j.begin_object().kv("compiler", compiler_string()).kv("build_flags", build_flags())
        .kv("snapshot_bytes", static_cast<unsigned long long>(fs::file_size(path)));
    j.key("load_seconds").begin_array();
    std::size_t n = 0;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        auto idx = Serializer::load(path);
        j.value(seconds_since(t0));
        n = idx->size();
    }
    j.end_array().kv("vectors", static_cast<unsigned long long>(n)).end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

// ---------------------------------------------------------------------------------------
// Deterministic synthetic vectors for the WAL tests: vector(id) is a pure function of id.
// ---------------------------------------------------------------------------------------
uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

void synth_vector(uint64_t id, std::size_t dim, std::vector<float>& out) {
    out.resize(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        const uint64_t h = splitmix(id * 1315423911ULL + j);
        out[j] = static_cast<float>(static_cast<double>(h >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
    }
}

// ---------------------------------------------------------------------------------------
// crash-writer: open the engine (this replays the WAL), print "READY <size>", then insert
// ids size, size+1, ... and print "A <id>" after each insert returns (= acknowledged).
// ---------------------------------------------------------------------------------------
int cmd_crash_writer(const Args& a) {
    const std::size_t dim = static_cast<std::size_t>(a.num("dim", 128));
    VectorEngine eng(std::make_unique<FlatIndex>(dim, MetricType::L2), a.req("wal"));
    uint64_t id = eng.size();
    std::printf("READY %llu\n", static_cast<unsigned long long>(id));
    std::fflush(stdout);
    std::vector<float> v;
    const uint64_t limit = static_cast<uint64_t>(a.num("limit", 0));
    while (limit == 0 || id < limit) {
        synth_vector(id, dim, v);
        eng.insert(id, v);
        std::printf("A %llu\n", static_cast<unsigned long long>(id));
        std::fflush(stdout);
        ++id;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------
// crash-verify
// ---------------------------------------------------------------------------------------
int cmd_crash_verify(const Args& a) {
    const std::size_t dim = static_cast<std::size_t>(a.num("dim", 128));
    const std::string wal = a.req("wal");
    const uint64_t acked = static_cast<uint64_t>(a.num("acked", 0));
    const long long inject = a.num("inject-torn-bytes", 0);

    uint64_t size_before = fs::exists(wal) ? fs::file_size(wal) : 0;
    uint64_t record_bytes = 0;
    {
        // A record frame produced by the real WalManager (written to a scratch file), so the
        // test does not re-implement the on-disk format.
        const std::string scratch = wal + ".frame";
        fs::remove(scratch);
        {
            storage::WalManager w(scratch);
            std::vector<float> v;
            synth_vector(0xFFFFFFFFULL, dim, v);
            w.append_insert(0xFFFFFFFFULL, v);
        }
        record_bytes = fs::file_size(scratch);
        if (inject > 0) {
            std::ifstream in(scratch, std::ios::binary);
            std::vector<char> frame(static_cast<std::size_t>(record_bytes));
            in.read(frame.data(), static_cast<std::streamsize>(frame.size()));
            const std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(inject), frame.size() - 1);
            std::ofstream out(wal, std::ios::binary | std::ios::app);
            out.write(frame.data(), static_cast<std::streamsize>(n));
        }
        fs::remove(scratch);
    }
    const uint64_t size_injected = fs::exists(wal) ? fs::file_size(wal) : 0;
    const bool torn_before = (size_injected % record_bytes) != 0;

    std::size_t recovered = 0;
    const auto t0 = Clock::now();
    {
        VectorEngine eng(std::make_unique<FlatIndex>(dim, MetricType::L2), wal);
        recovered = eng.size();
    }
    const double recover_s = seconds_since(t0);
    const uint64_t size_after = fs::file_size(wal);

    // Independent second pass: ids must be 0..n-1 in order with bit-exact vectors.
    uint64_t expect = 0, mismatches = 0;
    std::vector<float> v;
    storage::WalManager reader(wal);
    reader.replay([&](uint64_t id, std::span<const float> vec) {
        synth_vector(expect, dim, v);
        if (id != expect || vec.size() != dim || !std::equal(vec.begin(), vec.end(), v.begin())) ++mismatches;
        ++expect;
    });

    const uint64_t lost = acked > recovered ? acked - recovered : 0;
    const bool ok = lost == 0 && mismatches == 0 && expect == recovered && (size_after % record_bytes) == 0 &&
                    size_after == recovered * record_bytes && recovered <= acked + 1;

    Json j;
    j.begin_object()
        .kv("acked", static_cast<unsigned long long>(acked))
        .kv("recovered", static_cast<unsigned long long>(recovered))
        .kv("lost_acked", static_cast<unsigned long long>(lost))
        .kv("unacked_but_durable", static_cast<unsigned long long>(recovered > acked ? recovered - acked : 0))
        .kv("mismatches", static_cast<unsigned long long>(mismatches))
        .kv("record_bytes", static_cast<unsigned long long>(record_bytes))
        .kv("wal_bytes_at_kill", static_cast<unsigned long long>(size_before))
        .kv("injected_torn_bytes", inject)
        .kv("torn_tail_before_recovery", torn_before)
        .kv("wal_bytes_after_recovery", static_cast<unsigned long long>(size_after))
        .kv("tail_truncated", torn_before && (size_after % record_bytes) == 0)
        .kv("recover_seconds", recover_s)
        .kv("ok", ok)
        .end_object();
    write_text(a.get("out", "-"), j.str());
    return ok ? 0 : 3;
}

// ---------------------------------------------------------------------------------------
// wal-replay: write N synthetic records, then time (a) raw WalManager::replay (decode+CRC),
// (b) full VectorEngine recovery into a FlatIndex, (c) recovery into HNSWIndex (optional).
// ---------------------------------------------------------------------------------------
int cmd_wal_replay(const Args& a) {
    const std::size_t dim = static_cast<std::size_t>(a.num("dim", 128));
    const auto counts = a.list<std::size_t>("records", {10000, 100000, 1000000});
    const std::size_t hnsw_max = static_cast<std::size_t>(a.num("hnsw-max", 100000));
    const int reps = static_cast<int>(a.num("repeats", 3));
    const std::string dir = a.req("dir");
    fs::create_directories(dir);

    Json j;
    j.begin_object().kv("compiler", compiler_string()).kv("build_flags", build_flags())
        .kv("dim", static_cast<unsigned long long>(dim)).kv("page_cache", "warm (file just written; caches not dropped)");
    j.key("results").begin_array();
    for (std::size_t n : counts) {
        const std::string wal = (fs::path(dir) / ("replay_" + std::to_string(n) + ".wal")).string();
        fs::remove(wal);
        const auto tw = Clock::now();
        {
            storage::WalManager w(wal);
            std::vector<float> v;
            for (std::size_t i = 0; i < n; ++i) {
                synth_vector(i, dim, v);
                w.append_insert(i, v);
            }
        }
        const double write_s = seconds_since(tw);
        std::cerr << "[wal-replay] n=" << n << " written in " << write_s << "s\n";

        std::vector<double> raw, flat, hnsw;
        for (int r = 0; r < reps; ++r) {
            {
                storage::WalManager w(wal);
                uint64_t cnt = 0;
                const auto t0 = Clock::now();
                w.replay([&](uint64_t, std::span<const float>) { ++cnt; });
                raw.push_back(seconds_since(t0));
                if (cnt != n) throw std::runtime_error("replay count mismatch");
            }
            {
                const auto t0 = Clock::now();
                VectorEngine eng(std::make_unique<FlatIndex>(dim, MetricType::L2), wal);
                flat.push_back(seconds_since(t0));
                if (eng.size() != n) throw std::runtime_error("recovery size mismatch");
            }
            if (n <= hnsw_max && r == 0) {
                const auto t0 = Clock::now();
                VectorEngine eng(std::make_unique<HNSWIndex>(dim, MetricType::L2, 16, 200, 50), wal);
                hnsw.push_back(seconds_since(t0));
            }
        }
        auto arr = [&](const char* key, const std::vector<double>& xs) {
            j.key(key).begin_array();
            for (double x : xs) j.value(x);
            j.end_array();
        };
        j.begin_object()
            .kv("records", static_cast<unsigned long long>(n))
            .kv("wal_bytes", static_cast<unsigned long long>(fs::file_size(wal)))
            .kv("write_seconds", write_s);
        arr("raw_replay_seconds", raw);
        arr("recover_flat_seconds", flat);
        arr("recover_hnsw_seconds", hnsw);
        j.end_object();
        fs::remove(wal);
    }
    j.end_array().end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

// ---------------------------------------------------------------------------------------
// write-path
// ---------------------------------------------------------------------------------------
int cmd_write_path(const Args& a) {
    const int pin = static_cast<int>(a.num("pin", -1));
    const bool pin_ok = pin_current_thread(pin);
    const std::size_t n = static_cast<std::size_t>(a.num("n", 10000));
    const Matrix base = load_fvecs(a.req("base"), n);
    const std::string kind = a.get("index", "hnsw");
    const std::string dir = a.req("dir");
    fs::create_directories(dir);
    const std::string wal = (fs::path(dir) / "write_path.wal").string();
    const std::string snap = (fs::path(dir) / "write_path.snapshot").string();

    auto make_index = [&]() -> std::unique_ptr<IndexBase> {
        if (kind == "flat") return std::make_unique<FlatIndex>(base.dim, MetricType::L2);
        return std::make_unique<HNSWIndex>(base.dim, MetricType::L2, 16, 200, 50);
    };

    Json j;
    j.begin_object().kv("compiler", compiler_string()).kv("build_flags", build_flags())
        .kv("index", kind).kv("n", static_cast<unsigned long long>(base.rows))
        .kv("dim", static_cast<unsigned long long>(base.dim)).kv("pinned_cpu", pin).kv("pin_ok", pin_ok)
        .kv("wal_fsync_per_insert",
#if defined(NEXUS_WAL_NO_FSYNC)
            false
#else
            true
#endif
        );

    // (a) index only, no WAL
    {
        auto idx = make_index();
        const auto t0 = Clock::now();
        for (std::size_t i = 0; i < base.rows; ++i) idx->add(i, VectorView(base.row(i), base.dim));
        const double s = seconds_since(t0);
        j.key("index_only").begin_object().kv("seconds", s).kv("inserts_per_second", base.rows / s).end_object();
    }
    // (b) VectorEngine: WAL append (+ fsync unless NEXUS_WAL_NO_FSYNC) + index add
    {
        fs::remove(wal);
        std::vector<double> lat;
        lat.reserve(base.rows);
        double s = 0;
        {
            VectorEngine eng(make_index(), wal);
            const auto t0 = Clock::now();
            for (std::size_t i = 0; i < base.rows; ++i) {
                const auto ti = Clock::now();
                eng.insert(i, std::span<const float>(base.row(i), base.dim));
                lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - ti).count());
            }
            s = seconds_since(t0);
            const uint64_t wal_bytes = fs::file_size(wal);
            fs::remove(snap);
            const auto tc = Clock::now();
            eng.create_checkpoint(snap);
            const double ckpt_s = seconds_since(tc);
            j.key("engine_with_wal").begin_object().kv("seconds", s).kv("inserts_per_second", base.rows / s)
                .kv("wal_bytes", static_cast<unsigned long long>(wal_bytes));
            j.kv_stats("insert_latency", latency_stats(lat));
            j.end_object();
            j.key("checkpoint").begin_object().kv("seconds", ckpt_s)
                .kv("snapshot_bytes", static_cast<unsigned long long>(fs::file_size(snap)))
                .kv("wal_bytes_after", static_cast<unsigned long long>(fs::file_size(wal)))
                .end_object();
        }
        std::vector<double> loads;
        for (int r = 0; r < 3; ++r) {
            const auto t0 = Clock::now();
            auto idx = Serializer::load(snap);
            loads.push_back(seconds_since(t0));
        }
        j.key("snapshot_load_seconds").begin_array();
        for (double x : loads) j.value(x);
        j.end_array();
    }
    fs::remove(wal);
    fs::remove(snap);
    j.end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

// ---------------------------------------------------------------------------------------
// concurrency
// ---------------------------------------------------------------------------------------
int cmd_concurrency(const Args& a) {
    const Matrix queries = load_fvecs(a.req("query"));
    const Matrix inserts = a.has("insert-base") ? load_fvecs(a.get("insert-base")) : Matrix{};
    const auto threads = a.list<int>("threads", {1, 2, 4, 8, 16});
    const double secs = a.real("seconds", 5.0);
    const double write_frac = a.real("write-frac", 0.0);
    const std::size_t k = static_cast<std::size_t>(a.num("k", 10));
    const std::size_t ef = static_cast<std::size_t>(a.num("ef", 64));
    const std::string wal = a.req("wal");
    fs::remove(wal);

    auto idx = Serializer::load(a.req("snapshot"));
    if (auto* h = dynamic_cast<HNSWIndex*>(idx.get())) h->set_ef_search(ef);
    if (auto* s = dynamic_cast<HNSWIndexSQ8*>(idx.get())) s->set_ef_search(ef);
    const uint64_t base_size = idx->size();
    VectorEngine eng(std::move(idx), wal);
    std::atomic<uint64_t> next_insert{0};

    Json j;
    j.begin_object().kv("compiler", compiler_string()).kv("build_flags", build_flags())
        .kv("ef", static_cast<unsigned long long>(ef)).kv("k", static_cast<unsigned long long>(k))
        .kv("write_frac", write_frac).kv("seconds_per_point", secs)
        .kv("index_size_start", static_cast<unsigned long long>(base_size))
        .kv("hardware_threads", static_cast<unsigned>(std::thread::hardware_concurrency()));
    j.key("points").begin_array();
    for (int t : threads) {
        std::atomic<bool> stop{false};
        std::atomic<bool> go{false};
        std::vector<uint64_t> reads(static_cast<std::size_t>(t), 0), writes(static_cast<std::size_t>(t), 0);
        std::vector<std::thread> pool;
        for (int w = 0; w < t; ++w) {
            pool.emplace_back([&, w] {
                std::mt19937_64 rng(static_cast<uint64_t>(w) * 7919 + 17);
                std::uniform_real_distribution<double> u(0, 1);
                std::size_t q = static_cast<std::size_t>(w) * 97 % queries.rows;
                while (!go.load(std::memory_order_acquire)) {}
                while (!stop.load(std::memory_order_relaxed)) {
                    if (write_frac > 0 && inserts.rows > 0 && u(rng) < write_frac) {
                        const uint64_t i = next_insert.fetch_add(1);
                        eng.insert(base_size + i, std::span<const float>(inserts.row(i % inserts.rows), inserts.dim));
                        ++writes[static_cast<std::size_t>(w)];
                    } else {
                        auto r = eng.search(std::span<const float>(queries.row(q), queries.dim), k);
                        if (r.empty()) std::abort();
                        ++reads[static_cast<std::size_t>(w)];
                        q = (q + 1) % queries.rows;
                    }
                }
            });
        }
        const auto t0 = Clock::now();
        go.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        stop.store(true);
        for (auto& th : pool) th.join();
        const double el = seconds_since(t0);
        uint64_t R = 0, W = 0;
        for (auto x : reads) R += x;
        for (auto x : writes) W += x;
        j.begin_object().kv("threads", t).kv("seconds", el)
            .kv("reads", static_cast<unsigned long long>(R)).kv("writes", static_cast<unsigned long long>(W))
            .kv("read_qps", R / el).kv("write_ops", W / el).kv("total_ops", (R + W) / el).end_object();
        std::cerr << "[concurrency] t=" << t << " read_qps=" << R / el << " writes/s=" << W / el << "\n";
    }
    j.end_array().kv("index_size_end", static_cast<unsigned long long>(eng.size())).end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

// ---------------------------------------------------------------------------------------
// simd: ns per distance call, AVX2 kernel vs the engine's scalar reference implementation
// ---------------------------------------------------------------------------------------
int cmd_simd(const Args& a) {
    const int pin = static_cast<int>(a.num("pin", -1));
    const bool pin_ok = pin_current_thread(pin);
    const auto dims = a.list<std::size_t>("dims", {32, 64, 96, 128, 256, 384, 512, 768, 1024});
    const int trials = static_cast<int>(a.num("trials", 7));
    const double min_s = a.real("min-seconds", 0.2);

    Json j;
    j.begin_object().kv("compiler", compiler_string()).kv("build_flags", build_flags())
        .kv("pinned_cpu", pin).kv("pin_ok", pin_ok).kv("trials", trials)
        .kv("note", "working set of 256 vectors per dim (<= 1 MiB, cache resident); median of trials");
    j.key("results").begin_array();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1, 1);
    for (std::size_t d : dims) {
        const std::size_t nv = 256;
        AlignedVector<float> vecs(nv * d), q(d);
        for (auto& x : vecs) x = u(rng);
        for (auto& x : q) x = u(rng);
        const VectorView qv(q.data(), d);

        auto bench_fn = [&](auto&& fn) {
            // calibrate
            std::size_t iters = 1;
            volatile float sink = 0;
            while (true) {
                const auto t0 = Clock::now();
                for (std::size_t it = 0; it < iters; ++it)
                    for (std::size_t i = 0; i < nv; ++i) sink = sink + fn(qv, VectorView(vecs.data() + i * d, d));
                if (seconds_since(t0) > min_s / 4 || iters > (1u << 24)) break;
                iters *= 2;
            }
            std::vector<double> ns;
            for (int t = 0; t < trials; ++t) {
                const auto t0 = Clock::now();
                for (std::size_t it = 0; it < iters; ++it)
                    for (std::size_t i = 0; i < nv; ++i) sink = sink + fn(qv, VectorView(vecs.data() + i * d, d));
                ns.push_back(seconds_since(t0) * 1e9 / static_cast<double>(iters * nv));
            }
            std::sort(ns.begin(), ns.end());
            return std::pair<double, double>(ns[ns.size() / 2], ns.front());
        };
        auto l2s = bench_fn([](VectorView x, VectorView y) { return l2_distance_squared(x, y); });
        auto l2v = bench_fn([](VectorView x, VectorView y) { return l2_distance_squared_avx2(x, y); });
        auto dps = bench_fn([](VectorView x, VectorView y) { return dot_product(x, y); });
        auto dpv = bench_fn([](VectorView x, VectorView y) { return dot_product_avx2(x, y); });
        auto cos_s = bench_fn([](VectorView x, VectorView y) { return cosine_distance(x, y); });
        auto cos_v = bench_fn([](VectorView x, VectorView y) { return cosine_distance_avx2(x, y); });
        j.begin_object().kv("dim", static_cast<unsigned long long>(d))
            .kv("l2_scalar_ns", l2s.first).kv("l2_avx2_ns", l2v.first)
            .kv("dot_scalar_ns", dps.first).kv("dot_avx2_ns", dpv.first)
            .kv("cos_scalar_ns", cos_s.first).kv("cos_avx2_ns", cos_v.first)
            .end_object();
        std::cerr << "[simd] d=" << d << " l2 " << l2s.first << " -> " << l2v.first << " ns\n";
    }
    j.end_array().end_object();
    write_text(a.get("out", "-"), j.str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: nexus_bench <ann|snapshot-load|crash-writer|crash-verify|wal-replay|write-path|concurrency|simd> [--opts]\n";
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        const Args a(argc, argv, 2);
        if (cmd == "ann") return cmd_ann(a);
        if (cmd == "snapshot-load") return cmd_snapshot_load(a);
        if (cmd == "crash-writer") return cmd_crash_writer(a);
        if (cmd == "crash-verify") return cmd_crash_verify(a);
        if (cmd == "wal-replay") return cmd_wal_replay(a);
        if (cmd == "write-path") return cmd_write_path(a);
        if (cmd == "concurrency") return cmd_concurrency(a);
        if (cmd == "simd") return cmd_simd(a);
        std::cerr << "unknown command " << cmd << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
