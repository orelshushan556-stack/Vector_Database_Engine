# Benchmark summary: intel-i5-13400f (2026-11-07)

**Machine.** Intel Core i5-13400F (6 P-cores with Hyper-Threading + 4 E-cores, 16 threads), 16 GB RAM,
Windows 11 Pro host with the power plan set to High performance. The benchmarks ran inside WSL2 (Ubuntu 26.04,
kernel 6.18, GCC 15.2, `-O3 -mavx2 -mfma`, no AVX-512). The guest has 16 vCPUs and 7.6 GB RAM, and its ext4 virtual
disk sits on an NVMe SSD. Single-thread runs were pinned to vCPU 2.

**Commits.** `6b47c62` (main pass) and `b48c33b` (follow-up pass). The engine, the C++ harnesses and the CMake files
are identical in both. `b48c33b` adds three prefetch run pairs and a second concurrency sweep to `scripts/run_all.py`,
plus docs and chart scripts. Every result JSON records the commit that produced it.

**Wall time (measured, from the run logs in `2026-11-07/logs/`).**
- Main pass: 5 h 45 min (14:39:18 to 20:24:45 local), including compiling the four build variants.
- Follow-up pass: 9.5 min (20:26:02 to 20:35:35).

Data download and conversion took place earlier and are not included.

All numbers below are copied from `headlines.json`, which `scripts/make_report.py` computes from the raw JSON.

## Headlines

| What | Result |
|------|--------|
| SIFT1M, QPS at recall >= 0.95, best of the same grid for both engines | Nexus 11,337 vs hnswlib 10,619 (1.07x) |
| SIFT1M, QPS at recall >= 0.99 | Nexus 5,607 vs hnswlib 4,916 (1.14x) |
| GloVe-100, QPS at recall >= 0.95 | Nexus 1,356 vs hnswlib 1,271 (1.07x) |
| Distance computations per query at recall >= 0.95, SIFT1M | 1,109, which is 902x fewer than a 1,000,000-vector full scan (hnswlib: 1,250) |
| Distance computations per query at recall >= 0.95, GloVe-100 | 16,820, which is 70x fewer than a full scan (hnswlib: 16,375) |
| Neighbor-selection fix, SIFT-10K, efSearch 64, 10 builds each | worst-build recall 0.641 before, 0.998 after. An earlier pass's worst before-fix build was 0.394 |
| Crash recovery | 500 / 500 SIGKILL cycles verified, 0 acknowledged inserts lost, 125 / 125 injected torn tails truncated |
| WAL replay | 1,000,000 records (505 MiB) read and CRC-checked in 0.12 s, loaded into a FlatIndex in 0.40 s |
| Recovery into HNSW (rebuilds the graph) | 1.4 s for 10,000 records, 28.6 s for 100,000 |
| Durable inserts (fsync per insert) | 212 / s into HNSW, P50 4.5 ms. Without fsync: 11,567 / s |
| HNSW-SQ8 at recall >= 0.95, SIFT1M | 14,571 QPS vs 11,337 for FP32 HNSW |
| Memory, SIFT1M | FP32 HNSW 680 B/vector, HNSW-SQ8 808, hnswlib 718, QuantizedFlat 128, Flat 520 |
| Multi-thread search, SIFT1M ef 64 | 8,521 QPS on 1 thread, 5.4x at 10 threads (54% of linear), peak 49,737 at 12 |
| AVX2 vs scalar L2 kernel | 2.8x to 11.8x (d = 32 to 1024); dot 2.7x to 11.9x; cosine 4.9x to 19.2x |
| Software prefetch, SIFT1M | +2.3% to +10.0% QPS by efSearch; faster in 24 of 25 alternating run pairs |
| Recall stability, SIFT1M (3 builds, ef 64) | 0.9676 to 0.9681 |

## Losses (stated, not hidden)

- **hnswlib is faster in 3 of 6 recall comparisons:**
  - SIFT1M at >= 0.90: 17,253 vs 16,032 QPS (1.08x)
  - GloVe-100 at >= 0.90: 3,134 vs 2,984 (1.05x)
  - GloVe-100 at >= 0.99: 461 vs 438 (1.05x)

  A likely cause, which was not separately measured: hnswlib stores each vector next to its links, and Nexus keeps them in separate arrays.
- On SIFT-10K with a single configuration, hnswlib is also faster at recall >= 0.90 (200,251 vs 164,979 QPS).
- **Graph build on SIFT1M is 1.10x slower** than hnswlib's (297 s vs 269 s, single thread; Nexus has no parallel
  build). On GloVe-100 Nexus was slightly faster (353 s vs 361 s).
- **HNSW-SQ8 uses more memory than FP32 HNSW** (808 vs 680 B/vector) because it keeps FP32 copies for re-ranking.
  It trades memory for speed.
- **Durable write rate is set by one fsync per insert** (212 / s; no group commit). In the closed-loop 90/10 mix,
  each thread waits for its own fsync'd insert, so the best point is 1,815 reads/s + 195 writes/s.
- **Read throughput falls at 16 threads** (30,616 QPS vs 49,737 at 12) in both sweeps.

## What was blocked or changed during the run

- **Native Windows benchmarks were blocked.** Smart App Control (enforce mode) blocked some freshly compiled, unsigned
  test executables. Everything therefore ran in WSL2. Turning Smart App Control off cannot be undone without
  reinstalling Windows, so that was left to the machine owner (`docs/DECISIONS.md`, item 1).
- **WSL2 hides the P/E core layout.** Pinning targets a vCPU, not a specific P-core, and fsync goes through Hyper-V to
  the host NVMe.
- **A slow window at the start of the main pass** (about 2.2x slower drift probes at 12:39 UTC) was detected by the
  drift probes. The four suites from that window were re-run, and the originals are kept in
  `2026-11-07/superseded_slow_start/` (`docs/DECISIONS.md`, item 16).
  - Drift in the follow-up pass: start 33,437 QPS, end 34,920 QPS (+4.4%), same graph and recall.
  - Main pass: start 15,548 QPS, end 34,408 QPS.
- **Recovery into HNSW at 1,000,000 records was not run**, by design (`--hnsw-max 100000`). It rebuilds the graph,
  and building a 1M graph alone measured 297 s in `build_memory`. 10k and 100k were measured.
- **Docker** images do not contain `.git`, so `before_after_fix` reports a gap there.
- `gaps.json`: none. Every planned task produced data.
- FAISS (`IndexHNSWFlat`, faiss-cpu 1.15.1) ran through Python one query at a time, so its QPS includes binding
  overhead. It is a reference curve, not part of the comparison.

## Run it on another machine

```bash
git clone https://github.com/orelshushan556-stack/Vector_Database_Engine && cd Vector_Database_Engine
git checkout benchmarks
python3 -m venv .venv && . .venv/bin/activate
pip install numpy h5py matplotlib faiss-cpu ninja
python3 scripts/download_data.py --data-dir ~/nexus_data
python3 scripts/run_all.py --machine-id <slug> --data-dir ~/nexus_data --build-root ~/nexus_bench_build
python3 scripts/plot.py && python3 scripts/make_report.py   # adds the new machine to every chart as an overlay
```

`<slug>` is the CPU model in lower case with dashes, for example `amd-ryzen-ai-7-350`. If omitted, it is derived
automatically. Add `--resume` to continue after an interruption. Native Windows: see `benchmarks/README.md`.
