# Nexus Vector Database Engine

Nexus is an embedded C++20 vector search engine. It provides approximate nearest neighbor (ANN) search over dense FP32 vectors using a hierarchical navigable small world (HNSW) graph, AVX2/FMA distance kernels, 8-bit scalar quantization (SQ8) with exact re-ranking, and a CRC32-C-protected write-ahead log (WAL). The engine is built as a static library, `engine_core`, and is also packaged as a standalone HTTP service, `nexus_server`, which serves a React and Three.js dashboard.

Author: orel shushan

## Contents

1. System overview
2. Systems pillars
3. Architecture
4. Performance and telemetry
5. HTTP API reference
6. Deployment and native build
7. Repository layout

## 1. System Overview

| Component            | Implementation                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| Language / standard  | C++20                                                                          |
| Index types          | `FlatIndex`, `QuantizedFlatIndex`, `HNSWIndex`, `HNSWIndexSQ8`                  |
| Metrics              | `L2` (squared Euclidean), `DotProduct` (negated for ranking), `Cosine` (1 - cos) |
| SIMD requirement     | AVX2 and FMA (index code paths call the AVX2 kernels unconditionally)           |
| Durability           | Append-only WAL, fsync per record, CRC32-C via SSE4.2                           |
| Snapshot             | Binary index serializer with per-index-type magic numbers                       |
| Service              | Blocking-socket HTTP/1.1 server, fixed worker pool, JSON over REST              |

The service binary builds a 128-dimensional `HNSWIndexSQ8` with the L2 metric over a registry of 100 built-in concept vectors, then opens the WAL and replays it into the index.

## 2. Systems Pillars

### 2.1 SIMD vector kernels

Source: `src/distance/simd_metrics.cpp`, `include/engine/distance/simd_metrics.hpp`.

Kernels operate on `__m256` (8 x FP32) registers using unaligned loads (`_mm256_loadu_ps`) and fused multiply-add (`_mm256_fmadd_ps`).

| Kernel                          | Main loop                                  | Notes                                                                 |
|---------------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| `l2_distance_squared_avx2`      | 32 floats per iteration, 4 accumulators    | `_mm256_sub_ps` followed by FMA on the difference                      |
| `dot_product_avx2`              | 32 floats per iteration, 4 accumulators    | FMA on operand pairs                                                   |
| `cosine_distance_avx2`          | 16 floats per iteration, 2 accumulator sets | Dot product and both squared norms in one pass; throws on zero norm   |
| `quantized_l2_distance_sq_avx2` | 32 bytes per iteration, 4 accumulators     | `_mm256_cvtepu8_epi32` + `_mm256_cvtepi32_ps`, dequantizes with `fmadd(code, scale, min)` against an FP32 query |

Each kernel reduces its accumulators (`reduce_accumulators`), then processes remaining 8-wide blocks, then a scalar tail, so any dimension is supported. Horizontal reduction uses 128-bit extract/add/`_mm_movehl_ps`/`_mm_movehdup_ps`.

Memory layout: vector storage in the indexes uses `AlignedVector<float>`, backed by `AlignedAllocator` with 32-byte alignment (AVX2 register width). The kernels themselves use unaligned loads and do not require alignment.

Scalar reference implementations (`l2_distance_squared`, `dot_product`, `cosine_distance`, `src/distance/metrics.cpp`) exist as portable references. There is no runtime CPU dispatch: the HNSW, flat, and SQ8 index code call the `_avx2` kernels directly, so the library requires an AVX2/FMA-capable CPU and is compiled with `-mavx2 -mfma` (or `/arch:AVX2` on MSVC).

### 2.2 HNSW index topology

Source: `src/index/hnsw_index.cpp`, `src/index/hnsw_index_sq8.cpp`.

Parameters and defaults (both `HNSWIndex` and `HNSWIndexSQ8`):

| Parameter        | Default | Meaning                                                              |
|------------------|---------|----------------------------------------------------------------------|
| `M`              | 16      | Maximum neighbors per node on layers >= 1                            |
| `M0`             | 2 x M   | Maximum neighbors per node on layer 0 (32 by default)                |
| `efConstruction` | 200     | Candidate list width during insertion                                |
| `efSearch`       | 50      | Candidate list width at query time; runtime-adjustable via `set_ef_search` |
| `re_rank_factor` | 2.0     | `HNSWIndexSQ8` only; see section 2.3                                 |

Structure:

- Node level is drawn as `floor(-ln(U) * (1 / ln(M)))` with `U` uniform in (0, 1]. The generator is a `std::mt19937` seeded from `std::random_device`, so graph structure is not deterministic across runs.
- Layer 0 adjacency is a flat `uint32_t` array with a fixed stride of `M0 + 1` per node (degree followed by neighbor ids). Upper layers are stored per node as `level * (M + 1)` entries.
- Internal ids are dense `uint32_t` indexes; user-visible `VectorId` (`uint64_t`) is mapped through an `ids_` table.
- Insertion: greedy descent with ef = 1 from the entry point down to the node's level, then at each level from `min(max_level, node_level)` to 0 a beam search with width `efConstruction` selects up to `M` (or `M0`) nearest candidates, and bidirectional links are added.
- Link pruning: when a node is at capacity, the new link replaces the current farthest neighbor only if it is closer. This is a nearest-neighbor replacement rule; the diversity-based neighbor-selection heuristic from the HNSW paper is not implemented.
- Query: greedy descent through upper layers (hill climbing with a single entry point), then a best-first beam search on layer 0 with width `max(efSearch, k)`.
- Visited-set tracking uses a thread-local tag array, avoiding per-query allocation and clearing.
- There is no delete operation. `add` with an existing id appends a second node; ids are not deduplicated.

`HNSWIndex` is not internally synchronized; thread safety is provided by `VectorEngine` (section 2.5).

### 2.3 SQ8 quantization pipeline

Source: `src/quantization/scalar_quantizer.cpp`, `src/index/hnsw_index_sq8.cpp`.

Training (`ScalarQuantizer::train`): a single pass computes the per-dimension minimum and maximum over the training set (AVX2 `_mm256_min_ps` / `_mm256_max_ps`). Per dimension:

```
scale[d] = (max[d] - min[d]) / 255        (scale = 1.0 when range < 1e-7)
code[d]  = clamp((x[d] - min[d]) / scale[d] + 0.5, 0, 255)    truncated to uint8
decode   = min[d] + code[d] * scale[d]
```

`encode` and `decode` throw if the quantizer is untrained or sizes mismatch. `HNSWIndexSQ8` requires a trained quantizer at construction. Values outside the training range saturate at 0 or 255.

Search flow in `HNSWIndexSQ8`:

1. Graph traversal, construction, and link pruning use the quantized distance for the L2 metric (FP32 query against dequantized codes). For Cosine and DotProduct the traversal uses the stored FP32 vectors.
2. Layer-0 search collects `max(efSearch, k) * re_rank_factor` candidates (100 by default).
3. Candidates are re-ranked by exact FP32 distance and the top `k` are returned, so returned distances are exact.

Memory footprint: `HNSWIndexSQ8` stores both the uint8 codes and the original FP32 vectors (`quantized_storage_` and `raw_storage_`), which is 5 bytes per component, not 1. The FP32 copy is required for re-ranking and for the Cosine/DotProduct paths. The 4x reduction (1 byte versus 4 bytes per component) applies to `QuantizedFlatIndex`, which stores only codes.

### 2.4 Write-ahead logging and crash recovery

Source: `src/storage/wal_manager.cpp`, `include/engine/storage/crc32.hpp`.

Record layout (little-endian on x86-64, native byte order via `std::bit_cast`):

```
offset  size  field
0       1     op type            0x01 = Insert (the only defined operation)
1       4     CRC32-C            over bytes [5, 5 + payload_len)
5       8     vector id          uint64
13      4     dimension          uint32
17      4*d   vector payload     d FP32 values
17+4*d  1     end marker         0xAA
```

Record size is `18 + 4*d` bytes. The log has no file header and no magic number.

Checksum: CRC32-C (Castagnoli) computed with the SSE4.2 instructions `_mm_crc32_u64` and `_mm_crc32_u8`, initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`. The CRC covers the id, dimension, and vector payload; it does not cover the op byte or the end marker.

Write path: `append_insert` serializes the record into a reusable buffer, issues one `fwrite`, flushes the C buffer, and then calls `fdatasync` (POSIX) or `FlushFileBuffers` (Windows). Every insert is therefore synced before `insert` returns. `VectorEngine::insert` appends to the WAL first and then applies the vector to the in-memory index.

Replay (`WalManager::replay`, invoked from the `VectorEngine` constructor) reads records sequentially and applies each through the index's `add`:

| Condition                                              | Behavior                                          |
|--------------------------------------------------------|---------------------------------------------------|
| Clean end of file                                      | Replay ends normally                              |
| Truncated header (fewer than 17 bytes)                 | Replay stops silently; earlier records are kept   |
| Truncated vector payload                               | Replay stops silently; earlier records are kept   |
| Unknown op type byte                                   | Throws `std::runtime_error`                       |
| Missing end marker or marker not equal to `0xAA`       | Throws `std::runtime_error`                       |
| CRC32-C mismatch                                       | Throws `std::runtime_error`                       |

A torn write at the tail of the log that cuts a record short is tolerated and discarded. Any record that is complete but fails validation aborts replay with an exception; the server does not catch it during startup, so the process terminates and the log must be repaired or removed by an operator. Replay does not truncate the file.

Checkpointing (`VectorEngine::create_checkpoint`) serializes the index with `Serializer::save` and then truncates the WAL (`clear`, which reopens the file with `"wb+"`). Both steps run while holding the WAL mutex and a shared index lock. Snapshot files begin with a 32-bit magic (`VECI` flat, `HNSW`, `SQ8H`) and a version field (currently 1). On startup, `nexus_server` loads `index.bin` if it exists in the working directory, otherwise it builds the default concept index.

### 2.5 Thread pool and concurrency model

Source: `src/server/WorkerThreadPool.*`, `src/server/HttpServer.cpp`, `src/core/vector_engine.cpp`.

- `HttpServer` runs a blocking accept loop. Each accepted connection is enqueued as a task. `main` creates the server with 4 worker threads.
- `WorkerThreadPool` is a fixed set of `std::thread` workers consuming a single `std::queue<std::function<void()>>` guarded by one `std::mutex` and a `std::condition_variable`. It is a plain FIFO pool; there is no work stealing and no per-thread queues. Destruction sets a stop flag, wakes all workers, and joins them after the queue drains.
- Each connection is handled as one request/response exchange (read until `Content-Length` bytes are received, route, send, close). There is no keep-alive. Request bytes are read in 4096-byte chunks; request size is not capped.
- `VectorEngine` uses two locks. `index_rw_lock_` (`std::shared_mutex`) allows concurrent `search` calls and gives `insert` exclusive access while it mutates the index. `wal_mutex_` serializes WAL appends. Insert acquires the WAL mutex, appends and syncs, releases it, then takes the exclusive index lock. Because the fsync happens outside the index lock, concurrent searches are not blocked by disk I/O.
- Because WAL append and index update are separate critical sections, concurrent inserts can reach the log and the index in different orders. Ids are explicit, so replay reconstructs the same set of vectors, though not necessarily the same graph.

## 3. Architecture

```
  HTTP client / dashboard
          |
          v
  +-----------------------+      +-----------------------------------+
  | HttpServer            |      | WorkerThreadPool (4 threads)      |
  | blocking accept loop  +----->| single FIFO queue, mutex + condvar|
  +-----------------------+      +----------------+------------------+
                                                  |  one task per connection
                                                  v
                          +--------------------------------------------+
                          | HttpRequestParser -> Router                |
                          | JsonUtils (hand-rolled JSON in/out)        |
                          | StaticFileHandler (public/)                |
                          +-----+-------------------+------------------+
                       POST /api/search     POST /api/insert
                                |                   |
                                |                   v
                                |        VectorEngine::insert
                                |          1. wal_mutex_  -> WalManager::append_insert
                                |             [op|crc32c|id|dim|floats|0xAA] + fdatasync
                                |          2. unique_lock(index_rw_lock_)
                                |                   |
                                v                   v
                          +--------------------------------------------+
                          | HNSWIndexSQ8  (L2, dim 128 in nexus_server)|
                          |  insert: ScalarQuantizer::encode (uint8)   |
                          |          layered graph link (M=16, M0=32)  |
                          |  search: shared_lock(index_rw_lock_)       |
                          |    descent L_top..L1 (greedy, ef=1)        |
                          |    layer 0 beam search over SQ8 codes      |
                          |      -> quantized_l2_distance_sq_avx2      |
                          |    exact FP32 re-rank (l2_distance_squared |
                          |      _avx2) -> top-k                       |
                          +--------------------------------------------+

  Startup:   index.bin (if present) or built-in concept index
             -> WalManager::replay (verify CRC32-C) -> index.add per record
  Checkpoint: Serializer::save(index.bin) -> WAL truncate
```

## 4. Performance and Telemetry

### 4.1 Latency instrumentation

- `POST /api/search` and `POST /api/insert` time the engine call with `std::chrono::steady_clock` and return the duration in microseconds as `took_us`. This measures the engine call only, excluding socket I/O, parsing, and JSON serialization.
- `GET /api/status` reports the engine dimension and live vector count.

### 4.2 Benchmark harness

`benchmarks/bench_sift.cpp` (target `bench_sift`) evaluates the SIFT-small dataset against ground truth and reports mean per-query latency in microseconds and recall@10 for `FlatIndex` (exact), `QuantizedFlat` (SQ8), `HNSW`, and `HNSW-SQ8` across a sweep of `efSearch` values. It reports means, not percentiles. This repository does not commit measured results; run the harness on the target hardware to obtain figures. It reads `data/siftsmall/siftsmall_base.fvecs`, `siftsmall_query.fvecs` and the ground-truth ivecs file relative to the working directory; `data/` is git-ignored.

### 4.3 Memory efficiency

| Structure             | Bytes per vector component | Notes                                                    |
|-----------------------|----------------------------|----------------------------------------------------------|
| `HNSWIndex`           | 4                          | FP32 only                                                |
| `QuantizedFlatIndex`  | 1                          | uint8 codes; 4x smaller than FP32                         |
| `HNSWIndexSQ8`        | 5                          | uint8 codes plus FP32 copy for re-ranking                |

Graph overhead is `(2M + 1) * 4` bytes per node on layer 0 (132 bytes at M = 16) plus `level * (M + 1) * 4` bytes for upper layers. Per-dimension quantizer parameters add `2 * d * 4` bytes in total.

## 5. HTTP API Reference

Base URL: `http://localhost:8080`. All handlers set `Access-Control-Allow-Origin: *`; `OPTIONS` requests return `204`. There are no WebSocket endpoints. Request bodies are parsed by a minimal hand-written JSON scanner (`src/server/JsonUtils.hpp`), not a general JSON parser.

| Method | Path              | Request body                                | Success response                                                                 |
|--------|-------------------|---------------------------------------------|----------------------------------------------------------------------------------|
| GET    | `/api/status`     | none                                        | `{"status":"ok","server":"...","engine_ready":true,"dimension":128,"total_vectors":N}` |
| GET    | `/api/concepts`   | none                                        | Precompiled JSON catalog `{"concepts":[...]}` of the 100 built-in concepts        |
| POST   | `/api/search`     | `{"vector":[floats], "k":10}` (`k` optional, default 10) | `{"status":"success","took_us":T,"k":K,"results":[{"id","distance"[,"label","branch","category"]}]}` |
| POST   | `/api/insert`     | `{"id":uint, "vector":[floats]}`            | `{"status":"success","message":"...","id":ID,"took_us":T}`                         |
| POST   | `/api/checkpoint` | none                                        | `{"status":"success","message":"Checkpoint created successfully and WAL truncated."}` |

Result entries include `label`, `branch`, and `category` only when the id matches a built-in concept (ids 1 to 100). Distances for the default L2 index are squared Euclidean distances.

Error responses:

| Status | Condition                                                                |
|--------|--------------------------------------------------------------------------|
| 400    | Body is not parseable as the expected search or insert JSON               |
| 422    | Vector length differs from the engine dimension (message includes both)   |
| 500    | `/api/checkpoint` failed (message includes the exception text)            |
| 503    | Engine not initialized                                                    |
| 404    | Unknown path and no matching file under `public/`                         |

Any other path is served as a static file from `public/`; `/` maps to `/index.html`.

Example:

```bash
curl -s http://localhost:8080/api/status
curl -s -X POST http://localhost:8080/api/search \
     -H 'Content-Type: application/json' \
     -d '{"k":5,"vector":[ ...128 floats... ]}'
```

## 6. Deployment and Native Build

### 6.1 Docker Compose

```bash
git clone <repo-url>
cd Vector_Database_Engine
docker compose up --build -d
```

The image is built in three stages: a Node 20 stage builds the frontend into `public/`, an Ubuntu 24.04 stage compiles `nexus_server` with `-DCMAKE_BUILD_TYPE=Release -mavx2 -mfma -O3`, and a runtime stage carries the binary, `public/`, and `libstdc++`. The service listens on port 8080.

Persistence: `./data` on the host is mounted at `/app/data`, and `/app/nexus.wal` is a symlink to `/app/data/nexus.wal`, so the WAL survives container replacement. The checkpoint file `index.bin` is written to the container working directory (`/app/index.bin`), which is not on the volume, so a checkpoint is lost when the container is recreated; because checkpointing also truncates the WAL, do not rely on `/api/checkpoint` for durability in the default compose configuration.

Operations: `docker compose logs --tail=30`, `docker compose ps`, `docker compose down`.

### 6.2 Native build

Prerequisites:

- CMake 3.25 or newer
- A C++20 compiler: GCC 13+, Clang 17+, or MSVC (the build adds `/arch:AVX2`)
- Ninja (optional; any CMake generator works)
- An x86-64 CPU with AVX2, FMA, and SSE4.2
- Node.js 20+ for the dashboard

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
./build/nexus_server          # run from a directory containing public/
```

Targets: `engine_core` (library), `nexus_server`, `vector_engine_app` (example), `bench_sift`. Unit tests live in `tests/` (GoogleTest, fetched at configure time by `FetchContent`) and are registered through CTest.

The server resolves `public/`, `nexus.wal`, and `index.bin` relative to its working directory.

Dashboard:

```bash
cd web
npm ci
npm run dev       # Vite development server
npm run build     # type-check and emit the bundle to ../public
```

### 6.3 Dashboard

The dashboard (`web/`, React 19, TypeScript, Vite, Three.js, GSAP) visualizes the 100 built-in concepts across four branches. Implemented capabilities:

- 3D scene (`web/src/scene/HoloTopology.ts`) with orbit camera controls, instanced meshes for concept nodes, raycast picking against an instanced pick mesh, and custom shader materials for conduits, packet animation, and result trails.
- Selecting a concept sends a `POST /api/search` with a query vector derived from that concept's branch and displays the returned neighbors and the engine-reported `took_us` latency.
- Status and vector count are read from `GET /api/status`; the concept catalog from `GET /api/concepts`.
- When the backend is unreachable, the UI falls back to a local simulation and labels itself as a standalone preview. Simulated results are not engine output.

## 7. Repository Layout

```
include/engine/       public headers: core, distance, index, quantization, storage
src/                  engine implementation
src/server/           HTTP server, router, worker pool, concept registry
web/                  React + Three.js dashboard (builds into public/)
benchmarks/           SIFT benchmark
examples/             embedded usage example
tests/                GoogleTest unit tests
Dockerfile            multi-stage image build
docker-compose.yml    service definition and volume mapping
```
