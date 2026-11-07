#!/usr/bin/env python3
"""Optional FAISS IndexHNSWFlat reference run (pip install faiss-cpu numpy).

Same protocol and JSON schema as nexus_bench/hnswlib_bench `ann`: one build, efSearch sweep,
recall pass + timed pass with one query per call (single thread, omp_set_num_threads(1)).
Per-query calls go through the Python binding, which adds a few microseconds of overhead per
query, so QPS here is a lower bound for FAISS itself; the JSON records this caveat.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np


def read_vecs(path: str, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].view(dtype).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("base", "query", "gt", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", default="10,20,40,80,160")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--pin", type=int, default=-1)
    a = ap.parse_args()

    import faiss
    faiss.omp_set_num_threads(1)
    pinned = False
    if a.pin >= 0 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {a.pin})
        pinned = True

    xb = read_vecs(a.base, np.float32)
    xq = read_vecs(a.query, np.float32)
    gt = read_vecs(a.gt, np.int32)
    d = xb.shape[1]
    metric = faiss.METRIC_L2 if a.metric == "l2" else faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexHNSWFlat(d, a.M, metric)
    index.hnsw.efConstruction = a.efc
    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0

    points = []
    for ef in [int(x) for x in a.ef.split(",")]:
        index.hnsw.efSearch = ef
        hits = 0
        for i in range(len(xq)):
            _, ids = index.search(xq[i:i + 1], a.k)
            hits += len(set(ids[0].tolist()) & set(gt[i, :a.k].tolist()))
        recall = hits / (len(xq) * a.k)
        lat = []
        for _ in range(a.repeats):
            for i in range(len(xq)):
                q = xq[i:i + 1]
                s = time.perf_counter()
                index.search(q, a.k)
                lat.append((time.perf_counter() - s) * 1e6)
        lat.sort()
        mean = sum(lat) / len(lat)
        pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
        points.append({"ef": ef, "rerank": 0, "recall": recall, "qps": 1e6 / mean,
                       "latency": {"mean_us": mean, "p50_us": pct(.5), "p95_us": pct(.95), "p99_us": pct(.99),
                                   "min_us": lat[0], "max_us": lat[-1], "samples": len(lat)}})
        print(f"  ef={ef} recall={recall:.4f} qps={1e6 / mean:.0f}", flush=True)

    doc = {"meta": {"engine": "faiss", "faiss_version": faiss.__version__, "python": platform.python_version(),
                    "omp_threads": 1, "pinned_cpu": a.pin, "pin_ok": pinned,
                    "caveat": "per-query calls through the Python binding; QPS includes binding overhead",
                    "args": vars(a)},
           "dataset": {"base_rows": int(xb.shape[0]), "dim": int(d), "queries": int(xq.shape[0])},
           "builds": [{"build": 0, "build_seconds": build_s, "points": points}]}
    with open(a.out, "w") as f:
        json.dump(doc, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
