# Nexus Vector Database Engine

Nexus is a low-latency, embedded C++20 vector search engine. It targets sub-100 microsecond approximate nearest neighbor (ANN) retrieval using AVX2/FMA SIMD distance kernels, a hierarchical navigable small world (HNSW) graph index, SQ8 scalar quantization, and an append-only write-ahead log (WAL) for durability. The engine is built as a static library (`engine_core`) and is also shipped as a standalone HTTP service (`nexus_server`) with a bundled visualization dashboard.

Author: orel shushan

## Contents

1. Core systems pillars
2. Architecture
3. Performance
4. Dashboard and telemetry
5. Quick start (Docker)
6. REST interface
7. Local build toolchain
8. Repository layout

## 1. Core Systems Pillars

### Hardware SIMD kernels

Distance computation is the dominant cost in graph traversal, so it is implemented directly against 256-bit AVX2 registers.

- Kernels operate on `__m256` lanes (8 x FP32) and accumulate with fused multiply-add (`_mm256_fmadd_ps`), which removes an intermediate rounding step and halves the instruction count of the multiply-accumulate loop.
- Multiple independent accumulators are kept live per loop iteration to hide FMA latency and keep both vector ports busy.
- Hot buffers are aligned to cache-line boundaries (`alignas(64)`) to avoid split loads.
- A portable scalar path (`src/distance/metrics.cpp`) is retained as a reference implementation and fallback for targets without AVX2.

### HNSW graph topology

The primary index is a multi-layer HNSW graph (`HNSWIndex`, `HNSWIndexSQ8`).

- Each node is assigned a maximum layer from an exponentially decaying distribution. Upper layers are sparse skip-list-like routing structures; layer 0 contains every node and is stored in a contiguous adjacency array for traversal locality.
- Queries descend from the entry point through the upper layers with greedy routing (ef = 1), then run a best-first beam search on layer 0 with width `efSearch`.
- Graph connectivity is controlled by `M` (maximum out-degree per node, default 16) and `efConstruction` (candidate list width during insertion, default 200). `efSearch` (default 50) is adjustable at runtime and trades recall for latency.
- Exact `FlatIndex` and `QuantizedFlatIndex` implementations are provided as brute-force baselines for recall measurement.

### SQ8 scalar quantization

`ScalarQuantizer` compresses FP32 vectors to 8-bit codes.

- Per-dimension minimum and scale parameters are stored, so each dimension is mapped independently onto the 256 available bins. This preserves resolution on dimensions with small dynamic range.
- Storage drops from 4 bytes to 1 byte per component, a 75% reduction (4x density), which also reduces memory bandwidth per distance evaluation.
- `HNSWIndexSQ8` runs graph traversal over quantized codes.

### Durability and WAL

`WalManager` implements an append-only binary commit log.

- Every insert is appended as a framed record containing the vector id, dimension, payload, and a CRC32-C checksum.
- Checksums use the hardware CRC32 instruction (`_mm_crc32_u64` / `_mm_crc32_u8`, SSE4.2).
- On startup the log is replayed through the engine. Each record's CRC32-C is verified before it is applied; a mismatch or invalid framing raises an error rather than silently loading corrupted vectors.
- `POST /api/checkpoint` snapshots engine state via the serializer.
- In the container deployment the WAL is written to `/app/data`, which is bind-mounted to `./data` on the host so it survives container replacement.

### Concurrency

- `VectorEngine` guards the index with a `std::shared_mutex`: concurrent readers (queries) proceed in parallel, while inserts take an exclusive lock.
- `nexus_server` dispatches accepted connections to `WorkerThreadPool`, a fixed-size pool of worker threads draining a single mutex- and condition-variable-protected FIFO task queue.

## 2. Architecture

```
                         +---------------------------+
   Client / Dashboard -->|  HttpServer (accept loop) |
        REST/JSON        +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         |  WorkerThreadPool         |
                         |  (FIFO task queue)        |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         |  Router / JSON parsing    |
                         +------+------------+-------+
                           query|            |insert
                                |            v
                                |   +----------------------+
                                |   |  SQ8 ScalarQuantizer |
                                |   |  FP32 -> uint8 codes |
                                |   +----------+-----------+
                                v              v
                         +---------------------------+
                         |  VectorEngine             |
                         |  (shared_mutex: R/W)      |
                         |                           |
                         |  HNSW multi-layer         |
                         |  traversal (L_top .. L0)  |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         |  AVX2/FMA distance kernel |
                         |  (scalar fallback)        |
                         +---------------------------+

   insert path only:
   VectorEngine --> WalManager: append record + CRC32-C --> data/nexus.wal
   startup:         WalManager::replay (verify CRC32-C) --> VectorEngine
```

## 3. Performance

Reference figures below are reported for single-query latency on a 128-dimensional SIFT-style workload (`benchmarks/bench_sift.cpp`) on an AVX2-capable x86-64 host. Results depend on CPU, dataset, and `M` / `efSearch`; reproduce them on your own hardware with the benchmark binary before relying on them.

| Metric                        | Value                                             |
|-------------------------------|---------------------------------------------------|
| Query latency, P50            | 72 us                                             |
| Query latency, P99            | < 120 us                                          |
| Memory per component (FP32)   | 4 bytes                                           |
| Memory per component (SQ8)    | 1 byte (4x density, 75% reduction)                |

Dimension scaling: distance cost is linear in dimensionality. Each AVX2 iteration consumes 8 components, so supported dimensions (128 through 1536) should be multiples of 8 for a full-width inner loop; other sizes incur a scalar tail. Per-vector SQ8 storage equals the dimension in bytes (128 B at d=128, 1536 B at d=1536, versus 512 B and 6144 B in FP32).

## 4. Dashboard and Telemetry

The `web/` directory contains a React and TypeScript (Vite) single-page application with a Three.js scene, the "Obsidian Chamber", that renders the indexed concept set in 3D. Search requests issued from the UI are instrumented with high-resolution timers, and server-side engine latency is surfaced in microseconds alongside end-to-end round-trip time. The production bundle is built in the first Docker stage and served as static files by `nexus_server`.

## 5. Quick Start (Docker)

```bash
git clone <repo-url>
cd Vector_Database_Engine
docker compose up --build -d
```

- UI: `http://localhost:8080`
- WAL persistence: `./data/nexus.wal` (bind-mounted to `/app/data`)
- Logs: `docker compose logs --tail=30`
- Stop: `docker compose down`

The image is a multi-stage build: a Node stage builds the frontend, a GCC/CMake/Ninja stage compiles `nexus_server` in Release mode with `-mavx2 -mfma -O3`, and a slim runtime stage carries the binary and static assets.

## 6. REST Interface

| Method | Path              | Description                                  |
|--------|-------------------|----------------------------------------------|
| GET    | `/api/status`     | Engine status and live vector count          |
| GET    | `/api/concepts`   | Indexed concept registry                     |
| POST   | `/api/search`     | k-nearest-neighbor query                     |
| POST   | `/api/insert`     | Insert a vector (WAL-appended)               |
| POST   | `/api/checkpoint` | Snapshot engine state                        |

All other paths are served as static files; `/` resolves to `index.html`. Consult `src/server/Router.cpp` for the exact request and response schemas.

## 7. Local Build Toolchain

Prerequisites:

- CMake 3.25 or newer
- GCC 13+ or Clang 17+ with C++20 support (MSVC with `/arch:AVX2` is also supported by the build files)
- Ninja
- A CPU with AVX2, FMA, and SSE4.2
- Node.js 20+ (frontend only)

Engine and server:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
./build/nexus_server
```

Targets: `engine_core` (library), `nexus_server` (HTTP service), `vector_engine_app` (example), `bench_sift` (benchmark).

Frontend:

```bash
cd web
npm ci
npm run dev      # development server with HMR
npm run build    # production bundle
```

## 8. Repository Layout

```
include/engine/   public headers (core, distance, index, quantization, storage)
src/              engine implementation and HTTP server (src/server)
web/              React + Three.js dashboard
benchmarks/       SIFT benchmark harness
examples/         embedded-library usage example
tests/            unit tests
Dockerfile        multi-stage build
docker-compose.yml  service orchestration and volume mapping
```
