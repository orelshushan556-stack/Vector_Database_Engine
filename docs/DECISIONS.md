# Benchmark decisions log

Decisions taken while building and running the benchmark suite (branch `benchmarks`), with the
reason for each. Newest last.

## Environment

1. **Benchmarks run inside WSL2 (Ubuntu 26.04, GCC 15.2), not native Windows.** The host has
   Windows Smart App Control in enforce mode. It blocked 3 of 12 freshly compiled, unsigned test
   executables ("An Application Control policy has blocked this file"), and which binaries get
   blocked is not predictable. Turning Smart App Control off can't be undone without reinstalling
   Windows, so that was left to the owner. WSL2 binaries are not subject to it. The native Windows
   build (MinGW-w64 GCC 16.2 UCRT) compiles cleanly. The 8 test binaries that were allowed to run
   (test_hnsw ran with no output under Git Bash) all passed; the full suite (82/82) passed in WSL2.
2. **Consequences of WSL2, stated in every chart footer and the docs:**
   - The guest sees 16 vCPUs presented as 8 cores x 2 threads. The real host is an
     i5-13400F with 6 P-cores (with Hyper-Threading) and 4 E-cores (10 cores, 16 threads). The
     hybrid P/E layout is not visible inside the VM.
   - Single-thread runs are pinned with `sched_setaffinity` to **vCPU 2**. Hyper-V decides which
     host logical processor runs that vCPU, so pinning to a specific P-core is not possible from
     inside the guest. Windows normally schedules a busy vCPU on a P-core, but this is not
     guaranteed.
   - WAL benchmarks write to the guest's ext4 filesystem (a VHDX on the host NVMe SSD). An
     fsync inside the guest goes through Hyper-V to the host's disk flush, so fsync latency here
     is *platform-specific* and not representative of bare-metal Linux.
3. **Power plan:** the host's power plan was switched from Balanced to **High performance**
   (`powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`). To revert:
   `powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e`.
4. **Toolchain installed with winget:** Kitware CMake 4.4, Ninja 1.13, GitHub CLI, WinLibs
   GCC 16.2 (UCRT). WinLibs ships its own CMake without TLS trust anchors (FetchContent then fails
   with "SSL peer certificate"), so Kitware's CMake must come first on PATH. In WSL a Python venv
   (`~/nexus_venv`) provides numpy, h5py, ninja and faiss-cpu (no sudo was available).

## Engine changes (compile-time only, default off)

5. **Three toggles in `include/engine/diagnostics.hpp`**, all OFF by default. With them OFF the
   macros expand to `((void)0)` and the engine compiles exactly as before.
   - `NEXUS_COUNT_DISTANCES`: thread-local counters, incremented in `HNSWIndex::distance_to` and
     in the SQ8 index's quantized and exact distance functions.
   - `NEXUS_DISABLE_PREFETCH`: turns `prefetch_bytes` into a no-op.
   - `NEXUS_WAL_NO_FSYNC`: skips the per-record `fdatasync`/`FlushFileBuffers`. It is used only by
     the write-path comparison, and to generate large WAL files quickly for the replay-time test
     (replay cost does not depend on how the file was written).

   The full test suite passes with every toggle OFF, with `NEXUS_COUNT_DISTANCES` ON, and with
   `NEXUS_DISABLE_PREFETCH` ON (82/82 each). No other engine code was touched.
6. **The HNSW random seed is not controllable.** The engine seeds `std::mt19937` from
   `std::random_device`, and adding a seed parameter would change engine behavior. Our
   "independent builds" are therefore independent by construction, but not reproducible
   bit-for-bit. hnswlib builds use seeds 100, 101, ... (recorded per build).

## Methodology

7. **Distance counting uses the same saved graph.** Build 0 of each SIFT1M/GloVe index is
   saved. The counting build (`NEXUS_COUNT_DISTANCES`) loads that same snapshot, so counts
   describe exactly the graph whose QPS was measured, and the instrumented build never affects
   any timing.
8. **hnswlib distance counts come from wrapping its distance function** (`CountingSpace` in
   `benchmarks/hnswlib_bench.cpp`). Every call is counted exactly. hnswlib's built-in
   `metric_distance_computations` adds up neighbor-list sizes, including already-visited
   neighbors, and is collected only when metrics collection is enabled, so it is not comparable.
9. **GloVe-100 angular:** vectors are L2-normalized at conversion time, and both engines use
   inner product (Nexus `MetricType::DotProduct`, hnswlib `InnerProductSpace`). On unit vectors
   this ranks exactly like cosine. Nexus's `Cosine` metric was not used because it recomputes both
   norms on every call, and because it throws on zero vectors, which GloVe contains. Ground truth
   is the `neighbors` table from the ann-benchmarks HDF5 file.
10. **Best-vs-best comparison.** Both engines are built on the same (M, efConstruction) grid
    (`benchmarks/config.json`), with an efSearch sweep on each build. For each recall target
    (0.90 / 0.95 / 0.99) we report the **highest measured QPS among points whose measured
    recall is at or above the target**. There is no interpolation between points, so the reported
    QPS is what a user would actually get at a recall at least that high.
11. **QPS is single-thread, one query at a time:** 1e6 / mean per-query latency, with each query
    timed individually. Batch APIs are not used for either engine.
12. **The exact-scan baselines on 1M vectors use the first 1,000 queries** to keep the run
    length reasonable. Recall is 1.0 by construction. The latency is per query, so the query
    count only affects precision.
13. **Crash recovery uses SIGKILL**, which a process cannot catch, and kills at a uniformly
    random time between 20 ms and 600 ms after the writer is ready. An insert counts as
    "acknowledged" when the writer printed `A <id>` after `insert()` returned; the parent reads
    the pipe to the end after the kill. An insert that reached the disk but was not acknowledged
    before the kill legitimately reappears after recovery, so it is counted separately
    (`unacked_but_durable`), not as an error. Process kills rarely tear a write (the bytes are
    already in the page cache), so every 4th cycle **injects** a torn tail: a random-length
    prefix of a real WAL frame, produced by `WalManager` itself, so the test does not
    re-implement the format. A process kill does not test power loss; that is out of scope.
14. **Concurrency threads are not pinned.** Pinning N threads inside the VM would pin to vCPUs,
    not physical cores, so it would add no control. Physical-core boundaries are marked from the
    host's topology.
15. **Before/after the neighbor-selection fix:** "before" is `2bf962f^` (= `414cadf`), checked out
    with `git worktree`, and "after" is the benchmark-branch HEAD. Both are built by the same
    harness source (`benchmarks/fix_compare`). Commit `2bf962f` also changed SQ8 re-ranking, so
    the SQ8 comparison mixes both changes; the headline uses the FP32 `HNSWIndex`.
