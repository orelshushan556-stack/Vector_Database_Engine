#!/usr/bin/env python3
"""Build docs/report.html (self-contained, images inlined as base64, light/dark) and
docs/carousel.pdf (7 slides, 1080x1350) from results/ and the charts in docs/figures/.

  python scripts/plot.py && python scripts/make_report.py

All numbers in the prose are computed here from the raw JSON (see headlines()).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402

FIG = bd.REPO / "docs" / "figures"
REPO_URL = "https://github.com/orelshushan556-stack/Vector_Database_Engine"


# ----------------------------------------------------------------------------- headline numbers
def headlines(m: bd.Machine) -> dict:
    h: dict = {"machine_id": m.machine_id, "cpu": m.cpu, "runtime": m.runtime, "commit": m.commit, "date": m.date}
    for ds in ("sift", "glove100"):
        nx, hl = bd.all_grid(m, ds, "nexus"), bd.all_grid(m, ds, "hnswlib")
        for t in bd.RECALL_TARGETS:
            a, b = bd.best_at(nx, t), bd.best_at(hl, t)
            key = f"{ds}_r{int(t * 100)}"
            h[key] = {"nexus_qps": a.qps if a else None, "nexus_cfg": f"{a.label} ef{a.ef}" if a else None,
                      "nexus_recall": a.recall if a else None,
                      "hnswlib_qps": b.qps if b else None, "hnswlib_cfg": f"{b.label} ef{b.ef}" if b else None,
                      "hnswlib_recall": b.recall if b else None,
                      "ratio": (a.qps / b.qps) if a and b else None}
        dc_res = m.result("distance_computations", f"{ds}_nexus_hnsw")
        dn = bd.ann_points(dc_res)
        dh = bd.ann_points(m.result("distance_computations", f"{ds}_hnswlib_hnsw"))
        if dn:
            n = dc_res["dataset"]["base_rows"]
            a, b = bd.min_dist_at(dn, 0.95), bd.min_dist_at(dh, 0.95)
            h[f"{ds}_dist95"] = {"nexus": a.dist if a else None, "hnswlib": b.dist if b else None, "n": n,
                                 "reduction": n / a.dist if a else None,
                                 "nexus_vs_hnswlib": (a.dist / b.dist) if a and b else None}
        fl = bd.ann_points(m.result("recall_qps", f"{ds}_flat"))
        if fl:
            h[f"{ds}_flat_qps"] = fl[0].qps
        bn = bd.build_stats(m.result("recall_qps", f"{ds}_hnsw"))
        bh = bd.build_stats(m.result("recall_qps", f"{ds}_hnswlib"))
        if bn and bh:
            h[f"{ds}_build"] = {"nexus_s": bn["build_seconds"], "hnswlib_s": bh["build_seconds"],
                                "ratio": bn["build_seconds"] / bh["build_seconds"],
                                "nexus_bpv": bn["bytes_per_vector"], "hnswlib_bpv": bh["bytes_per_vector"],
                                "mem_ratio": bn["bytes_per_vector"] / bh["bytes_per_vector"]}
    for k in ("flat", "qflat", "hnsw", "sq8", "hnswlib"):
        b = bd.build_stats(m.result("recall_qps", f"sift_{k}"))
        if b:
            h.setdefault("sift_memory", {})[k] = b
    # recall stability on SIFT1M (min..max over builds at a mid ef)
    pts = bd.ann_points(m.result("recall_qps", "sift_hnsw"))
    if pts:
        p = next((x for x in pts if x.ef == 64), pts[len(pts) // 2])
        h["sift_stability"] = {"ef": p.ef, "min": p.recall_min, "max": p.recall_max, "builds": p.n_builds}
    fx = {}
    for v in ("before", "after"):
        d = m.result("before_after_fix", f"{v}_hnsw")
        if d:
            per = {}
            for b in d["builds"]:
                for p in b["points"]:
                    per.setdefault(p["ef"], []).append(p["recall"])
            fx[v] = per
    if len(fx) == 2:
        ef = 64 if 64 in fx["after"] else sorted(fx["after"])[0]
        h["fix"] = {"ef": ef, "builds": len(fx["after"][ef]),
                    "before_min": min(fx["before"][ef]), "before_median": st.median(fx["before"][ef]),
                    "after_min": min(fx["after"][ef]), "after_median": st.median(fx["after"][ef]),
                    "before_min_all": min(min(v) for v in fx["before"].values()),
                    "after_min_all": min(min(v) for v in fx["after"].values())}
    kc = m.result("crash_recovery", "kill_cycles")
    if kc:
        h["crash"] = kc["summary"]
    rp = m.result("crash_recovery", "replay_time")
    if rp:
        h["replay"] = {str(x["records"]): {"raw_s": st.median(x["raw_replay_seconds"]),
                                           "flat_s": st.median(x["recover_flat_seconds"]),
                                           "hnsw_s": st.median(x["recover_hnsw_seconds"]) if x["recover_hnsw_seconds"] else None,
                                           "mib": x["wal_bytes"] / 2 ** 20}
                       for x in rp["results"]}
    ro = m.result("concurrency", "read_only")
    if ro:
        runs = [r for r in (ro, m.result("concurrency", "read_only_repeat")) if r]
        pts = bd.mean_scaling(runs)
        q1 = next(p["read_qps"] for p in pts if p["threads"] == 1)
        best = max(pts, key=lambda p: p["read_qps"])
        phys = m.host_physical_cores
        ap = next((p for p in pts if p["threads"] == phys), best)
        h["scaling"] = {"q1": q1, "best_threads": best["threads"], "best_qps": best["read_qps"],
                        "best_speedup": best["read_qps"] / q1, "phys": phys, "at_phys_threads": ap["threads"],
                        "at_phys_speedup": ap["read_qps"] / q1,
                        "at_phys_efficiency": ap["read_qps"] / (q1 * ap["threads"]), "ef": ro["ef"],
                        "sweeps": len(runs), "q16": next((p["read_qps"] for p in pts if p["threads"] == 16), None)}
        mix = m.result("concurrency", "mix_90_10")
        if mix:
            lm = max(mix["points"], key=lambda p: p["total_ops"])
            h["scaling"]["mix_best"] = {"threads": lm["threads"], "reads": lm["read_qps"], "writes": lm["write_ops"]}
    sm = m.result("ablation", "simd_kernels")
    if sm:
        sp = [x["l2_scalar_ns"] / x["l2_avx2_ns"] for x in sm["results"]]
        h["simd"] = {"l2_min": min(sp), "l2_max": max(sp), "dims": [x["dim"] for x in sm["results"]]}
    pairs = [(bd.ann_points(m.result("ablation", f"prefetch_default_{i}")),
              bd.ann_points(m.result("ablation", f"prefetch_noprefetch_{i}"))) for i in range(10)]
    pairs = [(a_, b_) for a_, b_ in pairs if a_ and b_]
    on, off = [a_ for a_, _ in pairs], [b_ for _, b_ in pairs]
    if pairs:
        rs = [st.mean(r[i].qps for r in on) / st.mean(r[i].qps for r in off) for i in range(len(on[0]))]
        per = [a_[i].qps / b_[i].qps for a_, b_ in pairs for i in range(len(a_))]
        h["prefetch"] = {"min": min(rs), "max": max(rs), "pairs": len(pairs), "wins": sum(v > 1 for v in per), "n": len(per)}
    wp = {}
    for v in ("default", "nofsync"):
        for idx in ("flat", "hnsw"):
            r = m.result("write_path", f"{v}_{idx}")
            if r:
                wp[f"{v}_{idx}"] = {"index_only": r["index_only"]["inserts_per_second"],
                                    "engine": r["engine_with_wal"]["inserts_per_second"],
                                    "p50_us": r["engine_with_wal"]["insert_latency"]["p50_us"],
                                    "p99_us": r["engine_with_wal"]["insert_latency"]["p99_us"],
                                    "checkpoint_s": r["checkpoint"]["seconds"],
                                    "snapshot_bytes": r["checkpoint"]["snapshot_bytes"],
                                    "snapshot_load_s": st.median(r["snapshot_load_seconds"])}
    if wp:
        h["write_path"] = wp
    probes = {}
    for tag in ("start", "end"):
        qs = []
        for i in range(3):
            r = m.result(f"sanity_{tag}", f"probe_{i}")
            if r:
                qs += [p["qps"] for b in r["builds"] for p in b["points"] if p["ef"] == 64]
        if qs:
            probes[tag] = st.median(qs)
    if len(probes) == 2:
        h["drift"] = {"start_qps": probes["start"], "end_qps": probes["end"], "change": probes["end"] / probes["start"] - 1}
    # main-pass files that were set aside because of a slow window at the start (DECISIONS.md, item 16)
    sup = m.root / "superseded_slow_start"
    if sup.exists():
        def ef64(path):
            if not path.exists():
                return []
            r = json.loads(path.read_text())["result"]
            return [p["qps"] for b in r["builds"] for p in b["points"] if p["ef"] == 64]
        s0 = [q for i in range(3) for q in ef64(sup / "sanity_start" / f"probe_{i}.json")]
        s1 = [q for i in range(3) for q in ef64(sup / "sanity_end" / f"probe_{i}.json")]
        info = {}
        if s0 and s1:
            info.update(start_qps=st.median(s0), end_qps=st.median(s1))
        bf = sup / "before_after_fix" / "before_hnsw.json"
        if bf.exists() and h.get("fix"):
            r = json.loads(bf.read_text())["result"]
            vals = [p["recall"] for b in r["builds"] for p in b["points"] if p["ef"] == h["fix"]["ef"]]
            if vals:
                info.update(before_min=min(vals), builds=len(vals))
        if info:
            h["superseded"] = info
    sq = bd.best_at(bd.ann_points(m.result("recall_qps", "sift_sq8")), 0.95)
    fp = bd.best_at(bd.ann_points(m.result("recall_qps", "sift_hnsw")), 0.95)
    if sq and fp:
        h["sift_sq8_r95"] = {"sq8_qps": sq.qps, "sq8_recall": sq.recall, "sq8_ef": sq.ef, "sq8_rerank": sq.rerank,
                             "hnsw_qps": fp.qps, "hnsw_recall": fp.recall, "hnsw_ef": fp.ef}
    h["commits"] = m.commits
    h["gaps"] = m.gaps()
    return h


def x(v, nd=1):
    return "n/a" if v is None else f"{v:.{nd}f}x"


def n0(v):
    return "n/a" if v is None else f"{v:,.0f}"


# ----------------------------------------------------------------------------- report sections
def sections(m: bd.Machine, h: dict) -> list[dict]:
    S = []
    r95 = h.get("sift_r95", {})
    g95 = h.get("glove100_r95", {})

    def cmp_sentence(d, name):
        if not d or d.get("ratio") is None:
            return f"No comparison point at 95% recall on {name}."
        r = d["ratio"]
        who = (f"Nexus is {r:.2f}x the throughput of hnswlib" if r >= 1.05 else
               f"hnswlib is {1 / r:.2f}x faster than Nexus" if r <= 1 / 1.05 else "the two engines are within 5%")
        return (f"On {name} at recall >= 0.95, {who}: {n0(d['nexus_qps'])} vs {n0(d['hnswlib_qps'])} queries per second "
                f"on one core (Nexus best config {d['nexus_cfg']}, hnswlib {d['hnswlib_cfg']}).")

    def targets_table(ds):
        rows = []
        for t in bd.RECALL_TARGETS:
            d = h.get(f"{ds}_r{int(t * 100)}", {})
            rows.append(f"<tr><td>&ge; {t:.2f}</td><td>{n0(d.get('nexus_qps'))}</td><td>{html.escape(str(d.get('nexus_cfg')))}</td>"
                        f"<td>{n0(d.get('hnswlib_qps'))}</td><td>{html.escape(str(d.get('hnswlib_cfg')))}</td>"
                        f"<td>{x(d.get('ratio'), 2)}</td></tr>")
        return ("<table><thead><tr><th>Recall target</th><th>Nexus QPS</th><th>Nexus config</th><th>hnswlib QPS</th>"
                "<th>hnswlib config</th><th>Nexus / hnswlib</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    S.append(dict(id="architecture", title="The engine in one picture", fig="architecture",
                  measured="Nothing; this is an explainer. Nexus is an embedded C++20 library: an HNSW graph index (FP32 or "
                           "8-bit scalar-quantized with exact re-ranking), AVX2 distance kernels, and a write-ahead log with "
                           "CRC32-C checksums and checkpoints.",
                  read="The top row is the write path: each insert is appended to the log and flushed to disk before it "
                       "enters the in-memory index. The middle row is recovery after a crash. The bottom row is a search: "
                       "start at the sparse top layer, descend greedily, then search the dense bottom layer.",
                  shows="The pieces the benchmarks below exercise: graph search (recall, throughput, distance counts), the "
                        "SQ8 path (memory, re-rank), and the log (crash recovery, write cost).",
                  caveats="Simplified drawing; see README sections 2 and 3 for the exact data structures."))
    S.append(dict(id="pareto-sift", title="Recall vs throughput: SIFT1M", fig="pareto_sift",
                  measured="For both engines, every (M, efConstruction) pair in the same grid was built, and each graph "
                           "was queried with efSearch from 10 to 768. Each point is Recall@10 over the 10,000 standard "
                           "queries against single-thread queries per second (one query at a time).",
                  read="Each line is one engine's best trade-off (its Pareto frontier). Further right is more accurate; "
                       "higher is faster; up-and-right is better. The dashed vertical line marks 95% recall. The dashed "
                       "horizontal line is an exact brute-force scan.",
                  shows=cmp_sentence(r95, "SIFT1M") + " " + targets_table("sift"),
                  caveats="Absolute QPS depends on this CPU and on WSL2; compare the ratio, not the numbers, across "
                          "machines. Both engines were compiled with the same flags (-O3 -mavx2 -mfma) and run on the "
                          "same pinned vCPU. hnswlib v0.8.0. FAISS, if shown, is called through Python one query at a "
                          "time, which adds overhead."))
    S.append(dict(id="pareto-glove", title="Recall vs throughput: GloVe-100 (angular)", fig="pareto_glove100",
                  measured="Same protocol on 1.18M 100-dimensional GloVe word vectors (cosine similarity, computed as "
                           "inner product on normalized vectors in both engines).",
                  read="As above. GloVe is a harder dataset for graph indexes: neighborhoods are less clustered, so "
                       "high recall needs larger efSearch.",
                  shows=cmp_sentence(g95, "GloVe-100") + " " + targets_table("glove100"),
                  caveats="Fewer grid points than SIFT1M, because GloVe builds take longer. The GloVe ground truth comes "
                          "from the ann-benchmarks file."))
    d95 = h.get("sift_dist95", {})
    S.append(dict(id="distances", title="Algorithmic efficiency: distance computations", fig="distances_sift",
                  measured="The number of vector-distance evaluations each query performs, counted inside the engines "
                           "(a compile-time counter in Nexus, a wrapped distance function in hnswlib) on the same saved "
                           "graphs that were timed.",
                  read="Left is cheaper, up is more accurate. The dotted vertical line is an exact scan, which computes "
                       "one distance per stored vector.",
                  shows=(f"At recall >= 0.95, Nexus HNSW evaluates {n0(d95.get('nexus'))} distances per query instead of "
                         f"{n0(d95.get('n'))}: {n0(d95.get('reduction'))}x fewer. hnswlib needs {n0(d95.get('hnswlib'))} "
                         f"(Nexus / hnswlib = {x(d95.get('nexus_vs_hnswlib'), 2)}). This measure does not depend on the "
                         f"CPU, so it carries over to other machines.") if d95 else "Not available.",
                  caveats="Counts distance calls, not their cost: an SQ8 code distance is cheaper than an FP32 one. "
                          "Equal counts do not mean equal speed, because memory layout and prefetching also matter."))
    fx = h.get("fix", {})
    S.append(dict(id="fix", title="Before and after the neighbor-selection fix", fig="fix_before_after",
                  measured=f"The same harness, compiled against the commit before the fix and against the current code, "
                           f"builds {fx.get('builds', 'n')} independent graphs each on SIFT-10K and sweeps efSearch.",
                  read="Each dot is one build. Gray: before the fix. Blue: after. Red rings mark before-fix builds that "
                       "are worse than the worst after-fix build.",
                  shows=(f"At efSearch {fx.get('ef')}, before-fix recall ranged from {fx.get('before_min', 0):.3f} "
                         f"(median {fx.get('before_median', 0):.3f}); after the fix, the worst build reached "
                         f"{fx.get('after_min', 0):.3f} (median {fx.get('after_median', 0):.3f}). The old link-maintenance "
                         f"rule (evict the farthest neighbor) disconnected parts of the graph in some builds, and no "
                         f"efSearch value could reach them. The heuristic keeps diverse links, so every build is stable."
                         + (f" An earlier pass of the same test (set aside for timing reasons; recall is unaffected) drew "
                            f"{h['superseded']['builds']} other before-fix builds, and its worst was "
                            f"{h['superseded']['before_min']:.3f}: how bad the old rule gets depends on the random graph."
                            if h.get("superseded", {}).get("before_min") is not None else ""))
                  if fx else "Not available.",
                  caveats="SIFT-10K only (10,000 vectors). The fix commit also changed SQ8 re-ranking; this chart uses the "
                          "FP32 index, which only the graph change affects."))
    c = h.get("crash", {})
    rp = h.get("replay", {})
    big = rp.get(max(rp, key=int)) if rp else None
    S.append(dict(id="crash", title="Crash recovery", fig="crash_recovery",
                  measured=f"A writer process inserts vectors through the engine (log append + fsync per insert, then the "
                           f"index) and is killed with SIGKILL at a random moment, {c.get('cycles', 'N')} times. After each "
                           f"kill the log is replayed and checked against what the writer had acknowledged. Every 4th "
                           f"cycle also appends a partial record to simulate a write torn mid-way.",
                  read="Each square is one kill and restart: green means every acknowledged insert came back with "
                       "bit-identical data and the log ended on a record boundary. The right panel shows how replay "
                       "time grows with log size.",
                  shows=(f"{c.get('passed')} of {c.get('cycles')} cycles verified, {c.get('lost_acked_total')} acknowledged "
                         f"inserts lost, {c.get('torn_tails_truncated')} torn tails detected and truncated "
                         f"({c.get('injected_torn_tails')} injected, {c.get('natural_torn_tails')} natural). In "
                         f"{c.get('unacked_but_durable_cycles')} cycles the insert in flight had already reached the disk "
                         f"but was not yet acknowledged; it came back after restart, which is correct behavior. "
                         + (f"Replaying {int(max(rp, key=int)):,} records ({big['mib']:,.0f} MiB) takes {big['raw_s']:.2f} s "
                            f"to read and verify, {big['flat_s']:.2f} s to load into a flat index." if big else ""))
                  if c else "Not available.",
                  caveats="A process kill does not lose the operating system's page cache, so this tests the log "
                          "format and the recovery logic, not power loss. The fsync behavior is that of a WSL2 virtual "
                          "disk."))
    sc = h.get("scaling", {})
    S.append(dict(id="scaling", title="Multi-thread throughput", fig="thread_scaling",
                  measured="Concurrent searches through VectorEngine (the thread-safe wrapper) on the SIFT1M graph: one sweep "
                           "at 5 seconds and a repeat at 10 seconds per thread count (the line is their mean). A third run mixes 10% inserts (each with a log fsync) into the "
                           "searches.",
                  read="The dotted line is perfect linear scaling from one thread. The vertical lines mark the host's "
                       "physical cores and logical CPUs.",
                  shows=(f"Throughput grows {sc.get('at_phys_speedup', 0):.1f}x at {sc.get('at_phys_threads')} threads "
                         f"({sc.get('at_phys_efficiency', 0) * 100:.0f}% of linear) and peaks at {n0(sc.get('best_qps'))} QPS "
                         f"with {sc.get('best_threads')} threads"
                         + (f"; with all 16 logical CPUs busy it falls to {n0(sc['q16'])} QPS." if sc.get("q16") else ".")
                         + (f" With 10% inserts mixed in, the best point is {n0(sc['mix_best']['reads'])} reads/s plus "
                            f"{n0(sc['mix_best']['writes'])} writes/s ({sc['mix_best']['threads']} threads). Each thread "
                            f"issues its own inserts one at a time, and inserts are serialized by the log mutex, each "
                            f"waiting for its fsync, so the whole closed loop runs at the fsync rate: about nine reads per "
                            f"write. Searches are not blocked by an fsync; the log is written before the index lock is "
                            f"taken." if sc.get("mix_best") else "")) if sc else "Not available.",
                  caveats="Threads are not pinned." + (f" Host CPU{bd.known_topology(m.cpu)}: a WSL2 guest "
                          "cannot tell the core types apart, so scaling past the performance-core count mixes core types."
                          if bd.known_topology(m.cpu) else "") + " The mixed run is a closed loop (each thread waits for its "
                          "own insert), so it measures how fsync-bound writes pace a thread, not how much a separate "
                          "writer slows independent readers; that was not measured."))
    sm = h.get("simd", {})
    S.append(dict(id="simd", title="SIMD kernels", fig="simd_speedup",
                  measured="Nanoseconds per distance call for the AVX2+FMA kernels against the engine's scalar reference "
                           "functions, across dimensions 32 to 1024, on cache-resident data.",
                  read="Each bar is how many times faster the AVX2 kernel is at that dimension.",
                  shows=f"The AVX2 L2 kernel is {sm.get('l2_min', 0):.1f}x to {sm.get('l2_max', 0):.1f}x faster than the "
                        f"scalar loop." if sm else "Not available.",
                  caveats="The scalar loop is compiled without -ffast-math, so the compiler does not vectorize the "
                          "floating-point reduction; this is the speedup over straightforward C++, not over the best "
                          "possible scalar code."))
    mem = h.get("sift_memory", {})
    S.append(dict(id="memory", title="Memory and build time", fig="memory_build",
                  measured="Process memory growth while building each index on SIFT1M (one process per index, so "
                           "nothing else is counted), divided by 1M vectors, and single-thread build time.",
                  read="Left: bytes per stored vector. The dotted line marks the raw FP32 vector (128 x 4 = 512 bytes). "
                       "Right: seconds to build.",
                  shows=(f"The pure SQ8 flat index stores {mem['qflat']['bytes_per_vector']:,.0f} B per vector against "
                         f"{mem['flat']['bytes_per_vector']:,.0f} B for FP32. HNSW-SQ8 uses {mem['sq8']['bytes_per_vector']:,.0f} B, "
                         f"compared with {mem['hnsw']['bytes_per_vector']:,.0f} B for FP32 HNSW, because it keeps the FP32 vectors "
                         f"for exact re-ranking: SQ8 here buys cheaper traversal, not less memory. hnswlib uses "
                         f"{mem['hnswlib']['bytes_per_vector']:,.0f} B.")
                  if all(k in mem for k in ("qflat", "flat", "sq8", "hnsw", "hnswlib")) else "Partially available.",
                  caveats="Resident-memory growth includes allocator slack and vector-growth headroom, so it is an upper "
                          "bound on the data-structure size."))
    S.append(dict(id="latency", title="Latency at fixed recall", fig="latency_at_recall",
                  measured="For each recall target, the configuration with the lowest mean latency among all measured "
                           "points at or above that target.",
                  read="The bar is the median (P50) query latency; the tick above it is the 99th percentile (P99).",
                  shows="The gap between P50 and P99 shows tail behavior: single-thread queries on an otherwise idle "
                        "machine, so the tail comes from the query itself (hard queries explore more of the graph) and "
                        "from system noise.",
                  caveats="Microsecond latencies on a virtual machine include timer and scheduler noise."))
    pf = h.get("prefetch", {})
    S.append(dict(id="prefetch", title="Ablation: software prefetch", fig="prefetch_ablation",
                  measured="The same saved SIFT1M graph searched by two builds that differ only in the prefetch toggle, "
                           f"run alternately (on, off, on, off, ...) for {pf.get('pairs', '?')} pairs.",
                  read="Bars above 1 mean prefetching helps at that efSearch.",
                  shows=(f"Prefetch changes throughput by {(pf['min'] - 1) * 100:+.1f}% to {(pf['max'] - 1) * 100:+.1f}% "
                         f"(ratio of means per efSearch); it was faster in {pf.get('wins')} of {pf.get('n')} individual run pairs. "
                         f"README section 2.2 previously said its effect was unmeasured; this is the first measurement.")
                  if pf else "Not available.",
                  caveats="Prefetch benefit depends on the memory system; it may differ on other CPUs."))
    S.append(dict(id="rerank", title="Ablation: SQ8 re-rank factor", fig="rerank_sweep",
                  measured="One saved HNSW-SQ8 graph searched with re-rank factors from 1 to 8.",
                  read="Darker lines re-rank more candidates with exact distances.",
                  shows="A larger factor re-scores more candidates exactly: recall goes up and throughput goes down. "
                        "The title names the factor that is fastest at 95% recall.",
                  caveats="Specific to SIFT1M and to the 8-bit min/max quantizer."))
    w = h.get("write_path", {})
    S.append(dict(id="write", title="Write path: the cost of durability", fig="write_path",
                  measured="10,000 inserts in three modes: straight into the index, through the engine with the log but "
                           "no fsync (a test-only build), and through the default engine (log + fsync per insert). Also "
                           "checkpoint time and snapshot load time.",
                  read="Longer bars are faster; the axis is logarithmic.",
                  shows=(f"With an fsync per insert, the engine sustains {n0(w['default_hnsw']['engine'])} inserts/s into "
                         f"HNSW (P50 {w['default_hnsw']['p50_us'] / 1000:.1f} ms each); without fsync, "
                         f"{n0(w['nofsync_hnsw']['engine'])}/s. The disk flush, not the index, sets the write rate. A "
                         f"checkpoint of 10,000 vectors takes {w['default_hnsw']['checkpoint_s'] * 1000:.0f} ms.")
                  if "default_hnsw" in w and "nofsync_hnsw" in w else "Not available.",
                  caveats="Measured on WSL2's virtual disk; fsync latency on bare-metal Linux or Windows will differ, "
                          "possibly by an order of magnitude. Group commit (batching fsyncs) is not implemented."))
    return S


# ----------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme=light])){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}}
:root[data-theme=dark]{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h1{font-size:clamp(28px,4.5vw,44px);line-height:1.15;margin:8px 0 6px}h2{font-size:clamp(22px,3vw,30px);margin:0 0 10px;line-height:1.2}
h3{font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:22px 0 4px}
p{margin:6px 0 10px;color:var(--ink2)}.lead{font-size:19px;color:var(--ink2);max-width:780px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px;margin:28px 0}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}nav.toc a{color:var(--ink2);text-decoration:none;border:1px solid var(--line);border-radius:99px;padding:4px 12px;font-size:14px}
nav.toc a:hover{border-color:var(--accent);color:var(--ink)}
figure{margin:16px 0}figure img{width:100%;height:auto;border-radius:10px;border:1px solid var(--line);display:block}
img.dark{display:none}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme=light])) img.dark{display:block}:root:where(:not([data-theme=light])) img.light{display:none}}
:root[data-theme=dark] img.dark{display:block}:root[data-theme=dark] img.light{display:none}
:root[data-theme=light] img.light{display:block}:root[data-theme=light] img.dark{display:none}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:15px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:22px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.tile b{display:block;font-size:34px;line-height:1.1;color:var(--ink)}.tile span{color:var(--ink2);font-size:15px}
button.theme{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:99px;padding:8px 14px;font:inherit;font-size:14px;cursor:pointer}
code{font-size:.92em;background:var(--bg);padding:1px 5px;border-radius:5px}
.caveat{border-left:3px solid var(--muted);padding-left:12px}
footer{color:var(--muted);font-size:14px;margin-top:40px}
a{color:var(--accent)}
"""


def img_tag(name: str) -> str:
    out = []
    for theme in ("light", "dark"):
        p = FIG / f"{name}_{theme}.png"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            out.append(f'<img class="{theme}" alt="{html.escape(name.replace("_", " "))} chart ({theme} theme)" '
                       f'loading="lazy" src="data:image/png;base64,{b64}">')
    return "".join(out) if out else "<p><em>Chart not generated (data missing).</em></p>"


def build_html(m: bd.Machine, h: dict) -> str:
    env = m.env
    host = env.get("host", {})
    c = h.get("crash", {})
    d95 = h.get("sift_dist95", {})
    fx = h.get("fix", {})
    r95 = h.get("sift_r95", {})
    tiles = []
    if d95:
        tiles.append((f"{n0(d95.get('reduction'))}x", "fewer distance computations than a full scan at 95% recall (SIFT1M)"))
    if r95.get("nexus_qps"):
        tiles.append((n0(r95["nexus_qps"]), f"queries/s on one core at recall {r95['nexus_recall']:.3f} (SIFT1M)"))
    if fx:
        tiles.append((f"{fx['after_min']:.3f}", f"worst-build Recall@10 after the graph fix (was {fx['before_min']:.2f})"))
    if c:
        tiles.append((f"{c['passed']}/{c['cycles']}", f"SIGKILL crash cycles recovered, {c['lost_acked_total']} acknowledged writes lost"))
    tiles_html = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    secs = sections(m, h)
    toc = "".join(f'<a href="#{s["id"]}">{html.escape(s["title"])}</a>' for s in secs)
    body = []
    for s in secs:
        shows = s["shows"]
        if "<table>" not in shows:
            shows = html.escape(shows)
        else:
            i = shows.index("<table>")
            shows = html.escape(shows[:i]) + "</p>" + shows[i:] + "<p>"
        body.append(f"""<section id="{s['id']}"><h2>{html.escape(s['title'])}</h2>
<figure>{img_tag(s['fig'])}</figure>
<h3>What we measured</h3><p>{html.escape(s['measured'])}</p>
<h3>How to read this chart</h3><p>{html.escape(s['read'])}</p>
<h3>What it shows</h3><p>{shows}</p>
<h3>Caveats</h3><p class="caveat">{html.escape(s['caveats'])}</p></section>""")
    drift = h.get("drift")
    drift_txt = (f"A fixed SIFT-10K graph was timed at the start and at the end of the run: {n0(drift['start_qps'])} vs "
                 f"{n0(drift['end_qps'])} QPS ({drift['change'] * 100:+.1f}%).") if drift else "Drift probe not available."
    sp = h.get("superseded", {})
    if sp.get("start_qps"):
        drift_txt += (f" In the main pass the same probe ran at {n0(sp['start_qps'])} QPS at the start and "
                      f"{n0(sp['end_qps'])} QPS at the end: a slow window in the first minutes of that pass. Its "
                      f"suites from that window (before/after fix, crash recovery, write path) were re-run in a follow-up "
                      f"pass, and the originals are kept under superseded_slow_start/. Suites from the rest of the main "
                      f"pass ran at end-of-run speed.")
    gaps = h.get("gaps", [])
    gaps_html = ("<ul>" + "".join(f"<li><code>{html.escape(g['suite'])}/{html.escape(g['task'])}</code>: {html.escape(g['reason'][:200])}</li>"
                                  for g in gaps) + "</ul>") if gaps else "<p>None: every planned task produced data.</p>"
    losses = limitations(h)
    meth = f"""<section id="method"><h2>Methodology and hardware</h2>
<table><tbody>
<tr><th>Machine</th><td>{html.escape(m.machine_id)}: {html.escape(m.cpu)}; host {html.escape(str(host.get('physical_cores', '?')))} cores / {html.escape(str(host.get('logical_processors', '?')))} threads{html.escape(bd.known_topology(m.cpu))}, RAM {int(host.get('ram_bytes') or env.get('ram_bytes') or 0) / 2**30:.0f} GiB</td></tr>
<tr><th>Runtime</th><td>{html.escape(env.get('runtime', '?'))}, {html.escape(env.get('os_release', ''))}, kernel {html.escape(env.get('kernel', ''))}; host {html.escape(clean_text(host.get('os', '')))}</td></tr>
<tr><th>Compiler</th><td>{html.escape(m.build.get('compiler_version', ''))}; {html.escape(m.build.get('cxx_flags', ''))}</td></tr>
<tr><th>Pinning</th><td>{html.escape(env.get('pinning', {}).get('note', ''))}</td></tr>
<tr><th>Disk / WAL</th><td>{html.escape(env.get('work_dir_filesystem', ''))}; host disks: {html.escape(host.get('disks', '').replace(chr(10), '; '))}</td></tr>
<tr><th>Power plan</th><td>{html.escape(power_plan(host.get('power_scheme', env.get('power_scheme', ''))))}</td></tr>
<tr><th>Commit</th><td><code>{html.escape(' + '.join(h.get('commits', [])))}</code> (engine, C++ harness and CMake files are identical in both; the second only adds three prefetch run pairs and a second concurrency sweep to run_all.py, plus docs and chart scripts)</td></tr>
<tr><th>Datasets</th><td>SIFT-10K and SIFT1M (INRIA Texmex, L2), GloVe-100 angular (ann-benchmarks); SHA-256 of every file in the JSON</td></tr>
</tbody></table>
<p>Protocol: every query is timed individually with a monotonic clock; QPS = 1 / mean latency on one thread. Recall@10 is computed against the published ground truth. Graph indexes were built at least 5 times (SIFT-10K) and 3 times (SIFT1M); medians are shown with min/max where builds vary. Engine comparisons are best-vs-best: both engines get the same (M, efConstruction) grid, and for each recall target the fastest measured point at or above the target is reported, with no interpolation. Instrumented builds (distance counters, prefetch off, fsync off) are separate binaries; the default engine is unmodified. Full decisions log: <code>docs/DECISIONS.md</code>; reproduce: <code>benchmarks/README.md</code>.</p>
<p>Thermal / drift check: {html.escape(drift_txt)}</p></section>"""
    lim = f"""<section id="limits"><h2>Limitations</h2><ul>{''.join(f'<li>{html.escape(l)}</li>' for l in losses)}</ul>
<h3>Gaps in this run</h3>{gaps_html}</section>"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexus vector engine: benchmark report ({html.escape(m.machine_id)})</title><style>{CSS}</style></head>
<body><main>
<header class="top"><div><p style="margin:0;color:var(--muted)">Benchmark report, {html.escape(m.date)}, {html.escape(m.machine_id)}</p>
<h1>Nexus vector search engine, measured</h1>
<p class="lead">What an embedded C++20 HNSW engine with SQ8 quantization, AVX2 kernels and a write-ahead log actually does on one desktop CPU, compared fairly with hnswlib. Every number on this page was measured; raw JSON is in <code>results/</code>.</p></div>
<button class="theme" onclick="var r=document.documentElement;var d=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');r.setAttribute('data-theme',d==='dark'?'light':'dark')">Toggle light / dark</button></header>
<div class="tiles">{tiles_html}</div>
<nav class="toc">{toc}<a href="#method">Methodology</a><a href="#limits">Limitations</a></nav>
{''.join(body)}{meth}{lim}
<footer>Generated by <code>scripts/make_report.py</code> from <code>results/{html.escape(m.machine_id)}/{html.escape(m.date)}/</code>. Source: <a href="{REPO_URL}">{REPO_URL}</a></footer>
</main></body></html>"""


def losses(h: dict) -> list[tuple[str, str, float, float, float]]:
    """Every recall target where hnswlib's best measured point beat Nexus's: (dataset, target, hnswlib, nexus, x)."""
    out = []
    for ds, name in (("sift", "SIFT1M"), ("glove100", "GloVe-100")):
        for t in bd.RECALL_TARGETS:
            d = h.get(f"{ds}_r{int(t * 100)}", {})
            if d.get("ratio") and d["ratio"] < 1:
                out.append((name, f"{t:.2f}", d["hnswlib_qps"], d["nexus_qps"], 1 / d["ratio"]))
    return out


def clean_text(t: str) -> str:
    """Drop characters mangled by a console code page (Windows tools under a non-English locale)."""
    return "".join(c for c in t if c.isascii() and (c.isprintable() or c == " ")).strip(" ?")


# Windows' built-in power scheme GUIDs (identical on every Windows install)
POWER_SCHEMES = {"8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
                 "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
                 "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
                 "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate Performance"}


def power_plan(t: str) -> str:
    for guid, name in POWER_SCHEMES.items():
        if guid in t:
            return f"{name} ({guid})"
    return clean_text(t)


def limitations(h: dict, short: bool = False) -> list[str]:
    L = []
    ls = losses(h)
    if ls:
        cases = "; ".join((f"{n} >= {t}: {r:.2f}x" if short else f"{n} at recall >= {t}: {hq:,.0f} vs {nq:,.0f} QPS ({r:.2f}x)")
                          for n, t, hq, nq, r in ls)
        L.append(f"hnswlib is faster in {len(ls)} of 6 (dataset, recall-target) comparisons: {cases}."
                 + ("" if short else " A likely reason (not separately measured): hnswlib stores each vector next to "
                    "its layer-0 links in one memory block, while Nexus keeps vectors and links in separate arrays, so "
                    "each graph hop touches more cache lines."))
    for ds, name in (("sift", "SIFT1M"), ("glove100", "GloVe-100")):
        b = h.get(f"{ds}_build")
        if b and b["ratio"] > 1:
            L.append(f"Graph construction on {name} is {b['ratio']:.2f}x slower than hnswlib's, single-threaded "
                     f"({b['nexus_s']:.0f} s vs {b['hnswlib_s']:.0f} s); Nexus has no parallel build.")
    mem = h.get("sift_memory", {})
    if "sq8" in mem and "hnsw" in mem and mem["sq8"]["bytes_per_vector"] > mem["hnsw"]["bytes_per_vector"]:
        sq = h.get("sift_sq8_r95")
        speed = (f" It is faster at equal recall ({sq['sq8_qps']:,.0f} vs {sq['hnsw_qps']:,.0f} QPS at recall >= 0.95 "
                 f"on SIFT1M), so it trades memory for speed, not the other way round." if sq else "")
        L.append(f"HNSWIndexSQ8 uses more memory than FP32 HNSW ({mem['sq8']['bytes_per_vector']:.0f} vs "
                 f"{mem['hnsw']['bytes_per_vector']:.0f} bytes/vector), because it keeps FP32 copies for re-ranking."
                 + ("" if short else speed))
    w = h.get("write_path", {})
    if "default_hnsw" in w:
        L.append(f"Durable writes are bounded by one fsync per insert ({w['default_hnsw']['engine']:,.0f} inserts/s here); "
                 f"there is no group commit.")
    sc = h.get("scaling", {})
    if sc.get("q16") and sc["q16"] < sc["best_qps"] * 0.9 and not short:
        L.append(f"Read throughput drops from {sc['best_qps']:,.0f} QPS at {sc['best_threads']} threads to "
                 f"{sc['q16']:,.0f} QPS at 16 threads (all logical CPUs, both sweeps).")
    if short:
        L.append("One machine (Intel i5-13400F) under WSL2; absolute numbers will differ elsewhere.")
        return L
    L += ["One machine (Intel i5-13400F) under WSL2. Absolute numbers will differ elsewhere; ratios and distance counts "
          "should carry over better, but that is not shown until another machine is measured.",
          "WSL2 hides the hybrid P/E core layout; single-thread runs were pinned to a virtual CPU, which Windows schedules "
          "on a host core of its choosing.",
          "Crash tests kill the process; they do not simulate power loss or disk firmware behavior.",
          "There is no delete operation, and ids are not deduplicated (README section 2.2)."]
    return L


# ----------------------------------------------------------------------------- carousel
def build_carousel(m: bd.Machine, h: dict, out: Path, theme: str = "dark") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import plot as P

    T = P.THEMES[theme]
    W, H = 1080 / 72, 1350 / 72  # inches, so the PDF page is 1080 x 1350 pt

    def page_from_png(pdf, name, n):
        p = FIG / f"{name}_{theme}_1080x1350.png"
        fig = plt.figure(figsize=(W, H), facecolor=T["surface"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        if p.exists():
            ax.imshow(mpimg.imread(p), interpolation="lanczos")
        else:
            ax.text(0.5, 0.5, f"{name}: data missing", ha="center", color=T["ink"], fontsize=30, transform=ax.transAxes)
        fig.text(0.955, 0.012, f"{n}/7", ha="right", va="bottom", color=T["muted"], fontsize=20)
        pdf.savefig(fig, facecolor=T["surface"])
        plt.close(fig)

    def text_page(pdf, n, kicker, title, lines, link=None):
        fig = plt.figure(figsize=(W, H), facecolor=T["surface"])
        fig.text(0.075, 0.92, kicker, color=T["nexus"], fontsize=30, fontweight="bold", va="top")
        y = 0.86
        import textwrap
        for ln in textwrap.wrap(title, 26):
            fig.text(0.075, y, ln, color=T["ink"], fontsize=60, fontweight="bold", va="top")
            y -= 0.058
        y -= 0.03
        # shrink the body until it fits above the link line (keeps every item; never truncates)
        for fs_, width, lh in ((32, 46, 0.032), (28, 53, 0.028), (25, 60, 0.025), (22, 68, 0.022)):
            need = sum(len(textwrap.wrap(it, width)) * lh + 0.02 for it in lines)
            if y - need > 0.13:
                break
        for item in lines:
            wrapped = textwrap.wrap(item, width)
            fig.text(0.075, y, "•", color=T["nexus"], fontsize=fs_, va="top")
            for ln in wrapped:
                fig.text(0.11, y, ln, color=T["ink2"], fontsize=fs_, va="top")
                y -= lh
            y -= 0.02
        if link:
            fig.text(0.075, 0.06, link, color=T["ink"], fontsize=26, va="bottom")
        fig.text(0.955, 0.012, f"{n}/7", ha="right", va="bottom", color=T["muted"], fontsize=20)
        pdf.savefig(fig, facecolor=T["surface"])
        plt.close(fig)

    with PdfPages(out) as pdf:
        page_from_png(pdf, "hero", 1)
        page_from_png(pdf, "architecture", 2)
        page_from_png(pdf, "pareto_sift", 3)
        page_from_png(pdf, "distances_sift", 4)
        page_from_png(pdf, "fix_before_after", 5)
        page_from_png(pdf, "crash_recovery", 6)
        text_page(pdf, 7, "HONEST LIMITS", "What it does not do (yet)", limitations(h, short=True),
                  link=f"Code, raw data and how to reproduce:\n{REPO_URL.replace('https://', '')}")
        d = pdf.infodict()
        d["Title"] = "Nexus vector engine: measured"
        d["Author"] = "orel shushan"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(bd.REPO / "docs" / "report.html"))
    ap.add_argument("--pdf", default=str(bd.REPO / "docs" / "carousel.pdf"))
    ap.add_argument("--headlines", default=None, help="write headline numbers JSON (default results/<id>/headlines.json)")
    a = ap.parse_args()
    ms = bd.load_machines()
    if not ms:
        print("no results")
        return 1
    m = ms[0]
    h = headlines(m)
    hp = Path(a.headlines) if a.headlines else (bd.RESULTS / m.machine_id / "headlines.json")
    hp.write_text(json.dumps(h, indent=1, default=str))
    Path(a.html).write_text(build_html(m, h), encoding="utf-8")
    print(f"wrote {a.html} ({Path(a.html).stat().st_size / 2**20:.1f} MiB)")
    build_carousel(m, h, Path(a.pdf))
    print(f"wrote {a.pdf}")
    print(f"wrote {hp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
