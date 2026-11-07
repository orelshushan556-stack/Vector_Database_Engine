# Benchmark suite

Every number published from this suite comes from an actual run. The raw JSON is committed
under `results/<machine_id>/<date>/`, and each file records the CPU, OS/runtime, compiler and
flags, git commit, dataset SHA-256, parameters, repeats, and seeds. Charts and the report are
regenerated from that JSON. Their titles and text are computed from the data, not written by hand.

- Visual report: [`docs/report.html`](../docs/report.html) (self-contained; open it in a browser)
- Per-machine summary: `results/<machine_id>/SUMMARY.md`
- Decisions and their reasons: [`docs/DECISIONS.md`](../docs/DECISIONS.md)

## Reproduce

Requirements: an x86-64 CPU with AVX2, FMA and SSE4.2; CMake >= 3.25; a C++20 compiler (GCC 13+,
Clang 17+, or MSVC); Ninja (optional); Python 3.9+. Optional: `numpy` and `h5py` (to convert
GloVe), `faiss-cpu` (the FAISS reference run), and `matplotlib` (charts).

### Linux / WSL2

```bash
git clone https://github.com/orelshushan556-stack/Vector_Database_Engine && cd Vector_Database_Engine
git checkout benchmarks
python3 -m venv .venv && . .venv/bin/activate
pip install numpy h5py matplotlib faiss-cpu ninja
python3 scripts/download_data.py --data-dir data               # SIFT-10K, SIFT1M, GloVe-100 (~1.1 GB)
python3 scripts/run_all.py --machine-id <cpu-slug> --data-dir data
python3 scripts/plot.py && python3 scripts/make_report.py
```

On WSL2, keep `data/` and the build/work directory on the Linux filesystem (for example `~/`),
not under `/mnt/c`. `/mnt/c` is a 9p network mount; it is slow and its fsync is not
representative:

```bash
python3 scripts/run_all.py --machine-id <cpu-slug> --data-dir ~/nexus_data --build-root ~/nexus_bench_build
```

### Windows (native)

```powershell
winget install Kitware.CMake Ninja-build.Ninja BrechtSanders.WinLibs.POSIX.UCRT   # or VS Build Tools
py -m pip install numpy h5py matplotlib
py scripts\download_data.py --data-dir data
py scripts\run_all.py --machine-id <cpu-slug> --data-dir data
```

Put Kitware's CMake before WinLibs on `PATH`. WinLibs ships a CMake without TLS trust anchors,
and FetchContent downloads then fail. If Windows Smart App Control is in enforce mode, it may
block freshly compiled benchmark executables; use WSL2 in that case (see `docs/DECISIONS.md`).

### Docker

```bash
docker build --target bench -t vector-engine:bench .
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/src/results" vector-engine:bench \
    --machine-id <cpu-slug> --data-dir /data
```

Docker Desktop runs containers inside a VM. The git history is not in the image, so
`before_after_fix` is reported as a gap there.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--suites a b ...` | run a subset (`sanity_start before_after_fix crash_recovery write_path recall_qps distance_computations build_memory concurrency ablation vs_hnswlib sanity_end`) |
| `--resume` | skip tasks whose JSON already exists (restart after a crash or reboot) |
| `--quick` | CI smoke run: SIFT-10K recall only |
| `--crash-cycles N` | kill/recover cycles (default 200) |
| `--work-dir DIR` | snapshots and WAL files; must be a local disk |

The run on this repository's first machine used
`--crash-cycles 500 --data-dir ~/nexus_data --build-root ~/nexus_bench_build`; its measured wall
time is recorded in `results/<machine_id>/SUMMARY.md`.

## What is measured

| Suite | Question | Headline metric (hardware-robust) |
|-------|----------|-----------------------------------|
| `recall_qps` | Recall@10 vs single-thread QPS for HNSWIndex, HNSWIndexSQ8 (re-rank 1/2/5), FlatIndex, QuantizedFlatIndex, hnswlib; SIFT-10K (5 builds), SIFT1M (3 builds), GloVe-100 | recall stability across builds |
| `distance_computations` | distance evaluations per query vs recall, counted in both engines | reduction vs exact scan; Nexus/hnswlib ratio |
| `vs_hnswlib` | the same (M, efConstruction) grid for both engines, efSearch sweep; FAISS HNSWFlat via Python if installed | QPS, build-time and memory ratios at equal recall |
| `before_after_fix` | the commit before the neighbor-selection fix vs HEAD, the same harness source, 10 builds each | worst-build recall |
| `crash_recovery` | SIGKILL / TerminateProcess at random moments, restart, verify; replay time vs record count | pass rate, acknowledged inserts lost |
| `write_path` | inserts/s: index only, WAL without fsync, WAL + fsync; checkpoint; snapshot load | fsync cost ratio |
| `build_memory` | build time, RSS growth, bytes/vector at 1M; snapshot load time | bytes per vector |
| `concurrency` | VectorEngine search QPS at 1..16 threads; 90/10 read/write mix | scaling efficiency |
| `ablation` | AVX2 vs scalar kernels (d = 32..1024); prefetch on/off on the same graph; SQ8 re-rank factor | speedup ratios |
| `sanity_start` / `sanity_end` | the same saved graph timed at the start and end of the run | thermal / background drift |

### Protocol

- **Timing.** Single thread, pinned (`sched_setaffinity` / `SetThreadAffinityMask`) to one
  logical CPU chosen by `run_all.py`: a P-core on visible Intel hybrid topologies, otherwise the
  first SMT sibling of a physical core. One recall pass, then `repeats` timed passes. Each query
  is timed individually with `steady_clock`. QPS = 1e6 / mean latency. P50/P95/P99 are reported.
- **Fair comparison.** Both engines use the same compiler, the same flags (`-O3 -mavx2 -mfma`,
  no AVX-512), the same pinned CPU, `k = 10`, and the same queries. Both get the same
  (M, efConstruction) grid (`benchmarks/config.json`). For each recall target (0.90 / 0.95 /
  0.99) the reported number is the **highest measured QPS among points with measured recall at
  or above the target**. There is no interpolation, and points are never moved.
- **Distance counts.** These come from separate instrumented binaries that load the *same saved
  graphs* the timing runs used. Nexus uses `NEXUS_COUNT_DISTANCES`; hnswlib uses a counting
  wrapper around its distance function. Timing binaries are never instrumented.
- **Independent builds.** Nexus seeds its level generator from `std::random_device` (not
  controllable without changing the engine), so builds are independent but not bit-reproducible.
  hnswlib builds use seeds 100, 101, ...
- **Datasets.** SIFT (INRIA Texmex, L2). GloVe-100 angular from ann-benchmarks, L2-normalized
  and searched by inner product in both engines (identical ranking to cosine).

### Engine instrumentation (compile-time, default OFF)

`include/engine/diagnostics.hpp`: `NEXUS_COUNT_DISTANCES`, `NEXUS_DISABLE_PREFETCH`,
`NEXUS_WAL_NO_FSYNC`. With all three OFF the macros expand to nothing and the engine is
unchanged. The full test suite passes with the toggles OFF and with the counting and
prefetch toggles ON.

## Files

```
benchmarks/
  bench_common.hpp     dataset I/O, recall, timing, RSS, pinning, JSON writer
  ann_runner.hpp       the measurement loop shared by every ANN harness
  nexus_bench.cpp      Nexus driver: ann, snapshot-load, crash-writer/verify, wal-replay,
                       write-path, concurrency, simd
  hnswlib_bench.cpp    hnswlib driver (CMake option NEXUS_BENCH_HNSWLIB fetches hnswlib v0.8.0)
  fix_compare/         builds one harness against any engine revision (before/after fix)
  config.json          (M, efConstruction) grid and main configuration
  bench_sift.cpp       original single-file SIFT-10K harness (unchanged)
scripts/
  download_data.py     fetch + convert datasets, SHA-256 manifests
  run_all.py           orchestration: detect hardware, build variants, run suites, write JSON
  bench_faiss.py       optional FAISS IndexHNSWFlat reference run
  bench_data.py        loading + derived numbers shared by the chart/report scripts
  plot.py              charts (light/dark, 2x PNG, SVG, 1200x675, 1080x1350)
  make_report.py       docs/report.html + results/<id>/headlines.json
```

## Limitations

- The first published run is from **one machine** (Intel i5-13400F) **under WSL2**. The guest
  cannot see the hybrid P/E layout, and pinning targets a vCPU, not a physical core. Absolute
  QPS and latency belong to that machine; ratios and distance counts are the numbers expected to
  carry over, and that has not been checked until a second machine is run.
- fsync cost is platform-specific. Under WSL2 it passes through Hyper-V to the host NVMe drive.
- Crash tests kill the process; they do not model power loss or disk caches.
- FAISS is called through its Python binding one query at a time, so its QPS includes binding
  overhead.
- Exact-scan baselines on 1M vectors use the first 1,000 queries.
