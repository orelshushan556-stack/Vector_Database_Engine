#include "ConceptRegistry.hpp"
#include <random>
#include <cmath>

ConceptRegistry::ConceptRegistry() {
    init_concepts();
    generate_semantic_vectors();
    precompile_json();
}

void ConceptRegistry::populate_index(engine::IndexBase& index) const {
    for (size_t i = 0; i < TOTAL_CONCEPTS; ++i) {
        index.add(concepts_[i].id, vectors_[i]);
    }
}

void ConceptRegistry::generate_semantic_vectors() {
    vectors_.resize(TOTAL_CONCEPTS, std::vector<float>(DIMENSION, 0.0f));

    // 4 וקטורי מרכז (Centroids) אורתוגונליים במרחב 128 ממדים
    std::vector<std::vector<float>> centroids(4, std::vector<float>(DIMENSION, 0.0f));
    for (size_t b = 0; b < 4; ++b) {
        for (size_t d = b * 32; d < (b + 1) * 32; ++d) {
            centroids[b][d] = 1.0f;
        }
    }

    std::mt19937 rng(1337);
    std::normal_distribution<float> noise(0.0f, 0.08f);

    for (size_t i = 0; i < TOTAL_CONCEPTS; ++i) {
        uint32_t b = concepts_[i].branch;
        float norm_sq = 0.0f;

        for (size_t d = 0; d < DIMENSION; ++d) {
            float val = centroids[b][d] + noise(rng);
            vectors_[i][d] = val;
            norm_sq += val * val;
        }

        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (size_t d = 0; d < DIMENSION; ++d) {
            vectors_[i][d] *= inv_norm;
        }
    }
}

void ConceptRegistry::precompile_json() {
    precompiled_json_.reserve(32768);
    precompiled_json_ = "{\"status\":\"success\",\"total\":100,\"concepts\":[";

    for (size_t i = 0; i < TOTAL_CONCEPTS; ++i) {
        const auto& c = concepts_[i];
        precompiled_json_ += "{\"id\":";
        precompiled_json_ += std::to_string(c.id);
        precompiled_json_ += ",\"branch\":";
        precompiled_json_ += std::to_string(c.branch);
        precompiled_json_ += ",\"label\":\"";
        precompiled_json_ += c.label;
        precompiled_json_ += "\",\"category\":\"";
        precompiled_json_ += c.category;
        precompiled_json_ += "\",\"description\":\"";
        precompiled_json_ += c.description;
        precompiled_json_ += "\",\"code_ref\":\"";
        precompiled_json_ += c.code_ref;
        precompiled_json_ += "\"}";
        if (i + 1 < TOTAL_CONCEPTS) precompiled_json_ += ",";
    }
    precompiled_json_ += "]}";
}

void ConceptRegistry::init_concepts() {
    // -------------------------------------------------------------
    // BRANCH 0: SIMD & Hardware Optimization (1-25)
    // -------------------------------------------------------------
    concepts_[0]  = { 1,  0, "AVX2 256-bit Registers", "SIMD", "Utilizing 8-wide float vector registers (_mm256) for parallel arithmetic.", "engine/distance/simd_metrics.cpp" };
    concepts_[1]  = { 2,  0, "Fused Multiply-Add (FMA)", "SIMD", "Single-cycle (A * B) + C execution via _mm256_fmadd_ps.", "engine/distance/simd_metrics.cpp" };
    concepts_[2]  = { 3,  0, "L2 Distance Squared Kernel", "SIMD", "AVX2-accelerated Euclidean distance calculation.", "engine/distance/simd_metrics.cpp" };
    concepts_[3]  = { 4,  0, "Cosine Distance Kernel", "SIMD", "Normalized inner product distance with SIMD dot products.", "engine/distance/simd_metrics.cpp" };
    concepts_[4]  = { 5,  0, "Dot Product AVX2", "SIMD", "High-throughput vector projection using horizontal sum.", "engine/distance/simd_metrics.cpp" };
    concepts_[5]  = { 6,  0, "Loop Unrolling 4x", "Optimization", "Unrolling vector loops to maximize instruction pipeline saturation.", "engine/distance/simd_metrics.cpp" };
    concepts_[6]  = { 7,  0, "Tail Remainder Handling", "SIMD", "Scalar processing for dimensions not divisible by 8.", "engine/distance/simd_metrics.cpp" };
    concepts_[7]  = { 8,  0, "Horizontal Vector Sum", "SIMD", "Reduction of 256-bit register to single float scalar.", "engine/distance/simd_metrics.cpp" };
    concepts_[8]  = { 9, 0, "32-Byte Aligned Storage", "Memory", "AlignedAllocator places vector storage on 32-byte boundaries, matching the AVX2 register width.", "include/engine/types.hpp" };
    concepts_[9]  = { 10, 0, "AlignedVector Container", "Memory", "std::vector with a custom aligned allocator built on aligned operator new.", "include/engine/types.hpp" };
    concepts_[10] = { 11, 0, "VectorView Zero-Copy", "Memory", "Non-owning std::span over contiguous float memory.", "include/engine/types.hpp" };
    concepts_[11] = { 12, 0, "Software Prefetching", "CPU Cache", "_mm_prefetch (T0) of unvisited neighbor vectors before distance evaluation in search_layer.", "src/index/hnsw_common.hpp" };
    concepts_[12] = { 13, 0, "Unaligned Loads", "Memory", "Kernels use _mm256_loadu_ps, so input pointers need no particular alignment.", "src/distance/simd_metrics.cpp" };
    concepts_[13] = { 14, 0, "Metric Dispatch Per Call", "Architecture", "A switch on MetricType selects the AVX2 kernel; the branch is predictable but is evaluated on every distance call.", "src/index/hnsw_index.cpp" };
    concepts_[14] = { 15, 0, "Scalar Reference Kernels", "Testing", "Portable scalar L2, dot product and cosine used as reference implementations.", "src/distance/metrics.cpp" };
    concepts_[15] = { 16, 0, "SSE4.2 CRC32-C Intrinsics", "Hardware", "_mm_crc32_u64 hardware checksum for WAL record validation.", "include/engine/storage/crc32.hpp" };
    concepts_[16] = { 17, 0, "Compile-Time AVX2 Requirement", "Hardware", "Built with -mavx2 -mfma; there is no runtime CPU dispatch, so an AVX2/FMA CPU is required.", "CMakeLists.txt" };
    concepts_[17] = { 18, 0, "Multi-Accumulator Kernels", "Optimization", "Independent FMA accumulators hide instruction latency in the L2 and dot product kernels.", "src/distance/simd_metrics.cpp" };
    concepts_[18] = { 19, 0, "Per-Query Heap Allocation", "Memory", "search_layer reserves its candidate heaps per query; the visited set is reused thread-locally.", "src/index/hnsw_common.hpp" };
    concepts_[19] = { 20, 0, "Scalar Tail Handling", "SIMD", "Dimensions that are not multiples of the vector width finish in a scalar loop.", "src/distance/simd_metrics.cpp" };
    concepts_[20] = { 21, 0, "Interleaved FMA Accumulators", "CPU", "Four independent accumulators per loop iteration for instruction-level parallelism.", "src/distance/simd_metrics.cpp" };
    concepts_[21] = { 22, 0, "Instruction Parallelism", "CPU", "Interleaving independent FMA accumulators for ILP.", "engine/distance/simd_metrics.cpp" };
    concepts_[22] = { 23, 0, "SIFT-10K Dataset", "Benchmark", "Benchmarks use the 128-dimensional SIFT-10K (siftsmall) dataset.", "benchmarks/bench_sift.cpp" };
    concepts_[23] = { 24, 0, "Switch-Based Metric Selection", "Architecture", "MetricType enum selects among L2, cosine and dot product kernels at call time.", "src/index/hnsw_index.cpp" };
    concepts_[24] = { 25, 0, "Contiguous Neighbor Layout", "Memory", "Flattening graph neighbor lists into contiguous buffers.", "engine/index/hnsw_index.hpp" };

    // -------------------------------------------------------------
    // BRANCH 1: Graph Topology & HNSW (26-50)
    // -------------------------------------------------------------
    concepts_[25] = { 26, 1, "HNSW Multi-Layer Skip Graph", "Graph", "Multi-layer skip-list graph for logarithmic nearest-neighbor search.", "engine/index/hnsw_index.cpp" };
    concepts_[26] = { 27, 1, "Exponential Level Generator", "Algorithm", "get_random_level() using geometric decay: -ln(unif) * mult.", "engine/index/hnsw_index.cpp" };
    concepts_[27] = { 28, 1, "Entry Point at Top Layer", "Graph", "Search starts from the entry node on the highest populated layer.", "include/engine/index/hnsw_index.hpp" };
    concepts_[28] = { 29, 1, "Greedy Layer Descent", "Search", "1-greedy hopping through upper layers to locate local neighborhood.", "engine/index/hnsw_index.cpp" };
    concepts_[29] = { 30, 1, "search_layer Priority Queues", "Search", "Dual-heap (candidates & top_candidates) bounded traversal.", "engine/index/hnsw_index.cpp" };
    concepts_[30] = { 31, 1, "Layer 0 Base Connectivity", "Graph", "max_m0_ (2*M) degree allowance for dense base clustering.", "engine/index/hnsw_index.hpp" };
    concepts_[31] = { 32, 1, "ef_search Beam Width", "Tuning", "Runtime balance between query recall and search latency.", "engine/index/hnsw_index.hpp" };
    concepts_[32] = { 33, 1, "ef_construction Quality", "Indexing", "Candidate beam size during index insertion graph building.", "engine/index/hnsw_index.hpp" };
    concepts_[33] = { 34, 1, "Diversity Neighbor Selection", "Graph", "Neighbors are chosen with the HNSW paper heuristic (Algorithm 4, pruned connections kept), which preserves bridges between clusters.", "src/index/hnsw_common.hpp" };
    concepts_[34] = { 35, 1, "Bounded Neighbor Lists", "Graph", "Neighbor lists are capped at M (upper layers) and 2*M (layer 0).", "src/index/hnsw_common.hpp" };
    concepts_[35] = { 36, 1, "Scalar Quantizer (SQ8)", "Quantization", "Per-dimension 8-bit quantization of FP32 values; HNSWIndexSQ8 also keeps the raw floats for re-ranking.", "src/quantization/scalar_quantizer.cpp" };
    concepts_[36] = { 37, 1, "Asymmetric Distance Calc (ADC)", "Search", "Query float vector against 8-bit quantized database vectors.", "engine/index/hnsw_index_sq8.cpp" };
    concepts_[37] = { 38, 1, "Re-ranking Pipeline", "Algorithm", "Top candidate re-evaluation using exact full-precision vectors.", "engine/index/hnsw_index_sq8.cpp" };
    concepts_[38] = { 39, 1, "Recall@10 SIFT Benchmark", "Benchmark", "Recall@10, P50, P99 and QPS measured on SIFT-10K over 10 independent builds (see README, section 4.2).", "benchmarks/bench_sift.cpp" };
    concepts_[39] = { 40, 1, "FlatIndex Ground Truth", "Index", "Exact brute-force index for baseline accuracy validation.", "engine/index/flat_index.cpp" };
    concepts_[40] = { 41, 1, "QuantizedFlatIndex Scan", "Index", "Linear scan over compact SQ8 vector payloads.", "engine/index/quantized_flat_index.cpp" };
    concepts_[41] = { 42, 1, "Flat Link Storage Buffer", "Memory", "Contiguous link buffer avoiding pointer indirection overhead.", "engine/index/hnsw_index.hpp" };
    concepts_[42] = { 43, 1, "C++20 std::span Link Access", "C++20", "std::span<const uint32_t> for zero-copy graph inspection.", "engine/index/hnsw_index.hpp" };
    concepts_[43] = { 44, 1, "Epoch-Tagged Visited Set", "Optimization", "Integer epoch tags avoid clearing the visited set between queries.", "src/index/hnsw_common.hpp" };
    concepts_[44] = { 45, 1, "reserve() Capacity Hint", "Memory", "reserve() pre-allocates vector, id and adjacency storage when the final size is known.", "src/index/hnsw_index.cpp" };
    concepts_[45] = { 46, 1, "Compact Upper Graph Storage", "Graph", "Compact nested link storage for layers > 0.", "engine/index/hnsw_index.hpp" };
    concepts_[46] = { 47, 1, "KNN Bounded Result Heap", "Search", "std::priority_queue retaining only top-K nearest matches.", "engine/index/hnsw_index.cpp" };
    concepts_[47] = { 48, 1, "MetricType Enum", "Architecture", "Single enum covering L2, Cosine and DotProduct.", "include/engine/types.hpp" };
    concepts_[48] = { 49, 1, "Dimension Validation", "Validation", "Mismatched dimensions are rejected at the engine, the router and the index.", "src/core/vector_engine.cpp" };
    concepts_[49] = { 50, 1, "Recall Against Exact Baseline", "Testing", "HighRecallAgainstFlatIndex compares HNSW results with FlatIndex ground truth.", "tests/test_hnsw.cpp" };

    // -------------------------------------------------------------
    // BRANCH 2: Storage & WAL Persistence (51-75)
    // -------------------------------------------------------------
    concepts_[50] = { 51, 2, "Write-Ahead Log (WAL)", "Persistence", "Each insert is appended and synchronized to the log before it is applied to the in-memory index.", "src/storage/wal_manager.cpp" };
    concepts_[51] = { 52, 2, "Hardware CRC32-C Integrity", "Reliability", "CRC32-C is computed per record and verified before a record is replayed.", "src/storage/wal_manager.cpp" };
    concepts_[52] = { 53, 2, "Append-Only Log Strategy", "I/O", "Records are only appended; the log is truncated after a checkpoint.", "src/storage/wal_manager.cpp" };
    concepts_[53] = { 54, 2, "Crash Recovery Replay", "Persistence", "Replay rebuilds the in-memory index from the log at startup.", "src/storage/wal_manager.cpp" };
    concepts_[54] = { 55, 2, "Unified Serializer Interface", "Serialization", "Serializer::save and Serializer::load dispatch on the concrete index type.", "src/storage/serializer.cpp" };
    concepts_[55] = { 56, 2, "Magic Number 0x56454349 (VECI)", "Binary Format", "0x56454349 file header identifying FlatIndex archives.", "src/storage/serializer.cpp" };
    concepts_[56] = { 57, 2, "Magic Number 0x484E5357 (HNSW)", "Binary Format", "0x484E5357 file header identifying HNSWIndex archives.", "src/storage/serializer.cpp" };
    concepts_[57] = { 58, 2, "Magic Number 0x53513848 (SQ8H)", "Binary Format", "0x53513848 file header identifying HNSWIndexSQ8 archives.", "src/storage/serializer.cpp" };
    concepts_[58] = { 59, 2, "Factory-Style Index Loader", "Design Pattern", "Serializer::load instantiates the matching index type from the magic number.", "src/storage/serializer.cpp" };
    concepts_[59] = { 60, 2, "Torn-Tail Truncation", "Recovery", "An incomplete record at the end of the log is discarded and the file is truncated to the last valid record.", "src/storage/wal_manager.cpp" };
    concepts_[60] = { 61, 2, "Replay Dimension Bound", "Reliability", "A record dimension of 0 or above 4096 is treated as corruption before any buffer is sized.", "src/storage/wal_manager.cpp" };
    concepts_[61] = { 62, 2, "Checked Sync Return Codes", "Reliability", "fdatasync and FlushFileBuffers results are checked; failures are logged and raised as std::system_error.", "src/storage/wal_manager.cpp" };
    concepts_[62] = { 63, 2, "WAL Truncation at Checkpoint", "Maintenance", "The log is emptied once the snapshot has been written.", "src/core/vector_engine.cpp" };
    concepts_[63] = { 64, 2, "Atomic Snapshot Write", "Persistence", "Snapshot goes to a temporary file, is fsynced, then renamed over the target.", "src/storage/serializer.cpp" };
    concepts_[64] = { 65, 2, "Native Byte Order Encoding", "Format", "Fixed-width fields are written in native byte order (little-endian on x86-64); not portable across endianness.", "src/storage/serializer.cpp" };
    concepts_[65] = { 66, 2, "End-Marker Framing", "Protocol", "Each WAL record ends with a 0xAA marker that is validated on replay.", "src/storage/wal_manager.cpp" };
    concepts_[66] = { 67, 2, "Directory Entry Durability", "Reliability", "POSIX: fsync of the parent directory after rename. Windows: MoveFileExW with write-through.", "src/storage/serializer.cpp" };
    concepts_[67] = { 68, 2, "Corrupted Record Detection", "Recovery", "A complete record with a bad CRC, op type or end marker aborts replay.", "src/storage/wal_manager.cpp" };
    concepts_[68] = { 69, 2, "Bounds-Checked Replay", "Reliability", "Payload length is validated before reading, so corrupt headers cannot force large allocations.", "src/storage/wal_manager.cpp" };
    concepts_[69] = { 70, 2, "Fixed 17-Byte Record Header", "Format", "Op type, CRC32-C, vector id and dimension precede the float payload.", "src/storage/wal_manager.cpp" };
    concepts_[70] = { 71, 2, "Checkpoint Under WAL Mutex", "Concurrency", "Checkpoint and insert share the WAL mutex, so no acknowledged insert is lost between snapshot and truncation.", "src/core/vector_engine.cpp" };
    concepts_[71] = { 72, 2, "Snapshot Format Version", "Format", "Snapshot headers carry a magic number and a version field.", "src/storage/serializer.cpp" };
    concepts_[72] = { 73, 2, "Snapshot Load at Startup", "Persistence", "nexus_server loads index.bin when present, then replays the WAL.", "src/server/main.cpp" };
    concepts_[73] = { 74, 2, "Sync Before Acknowledge", "Reliability", "append_insert returns only after the record has been flushed and synchronized.", "src/storage/wal_manager.cpp" };
    concepts_[74] = { 75, 2, "Replay Callback Interface", "Design", "Replay delivers each verified record to a caller-supplied callback.", "src/storage/wal_manager.cpp" };

    // -------------------------------------------------------------
    // BRANCH 3: Concurrency & OS Runtime (76-100)
    // -------------------------------------------------------------
    concepts_[75] = { 76, 3, "thread_local Visited Set", "Concurrency", "Per-thread visited buffer; searches need no lock for it.", "src/index/hnsw_common.hpp" };
    concepts_[76] = { 77, 3, "Concurrent Read Path", "Concurrency", "Concurrent read path with std::shared_mutex; searches share the lock and inserts take it exclusively.", "src/core/vector_engine.cpp" };
    concepts_[77] = { 78, 3, "Worker Thread Pool", "Threading", "Fixed-size pool of std::thread workers; nexus_server starts four.", "src/server/WorkerThreadPool.cpp" };
    concepts_[78] = { 79, 3, "FIFO Task Queue", "Scheduling", "Single std::queue guarded by a mutex and condition variable; no work stealing.", "src/server/WorkerThreadPool.cpp" };
    concepts_[79] = { 80, 3, "std::shared_mutex R/W Locks", "Synchronization", "Shared-read, exclusive-write locking around the index.", "include/engine/core/vector_engine.hpp" };
    concepts_[80] = { 81, 3, "Task Exception Isolation", "Reliability", "worker_loop catches exceptions from tasks and logs them instead of terminating the worker.", "src/server/WorkerThreadPool.cpp" };
    concepts_[81] = { 82, 3, "Request Size Limits", "Security", "Bodies above 16 MiB are rejected with HTTP 413; oversized headers with HTTP 431.", "src/server/HttpServer.cpp" };
    concepts_[82] = { 83, 3, "Winsock2 / POSIX Abstraction", "Cross-Platform", "Socket wrapper that compiles on Windows and Linux.", "src/server/HttpServer.cpp" };
    concepts_[83] = { 84, 3, "Custom HTTP Engine (nexus_server)", "Network", "Small HTTP/1.1 server without external dependencies.", "src/server/HttpServer.cpp" };
    concepts_[84] = { 85, 3, "Streaming Content-Length Reader", "Network", "recv loop that accumulates segmented requests until Content-Length bytes arrive.", "src/server/HttpServer.cpp" };
    concepts_[85] = { 86, 3, "REST API /api/search Route", "API", "JSON vector search endpoint.", "src/server/Router.cpp" };
    concepts_[86] = { 87, 3, "REST API /api/status Telemetry", "API", "Reports engine readiness, dimension and vector count.", "src/server/Router.cpp" };
    concepts_[87] = { 88, 3, "REST API /api/concepts Catalog", "API", "Serves a JSON catalog that is built once at startup.", "src/server/Router.cpp" };
    concepts_[88] = { 89, 3, "Microsecond Latency Telemetry", "Diagnostics", "std::chrono::steady_clock timing of the engine call, returned as took_us.", "src/server/Router.cpp" };
    concepts_[89] = { 90, 3, "CORS Headers", "Network", "OPTIONS preflight handling and permissive cross-origin headers (no authentication).", "src/server/Router.cpp" };
    concepts_[90] = { 91, 3, "Blocking Socket I/O", "Network", "Blocking accept and recv on sockets; each connection is handled by one pool worker.", "src/server/HttpServer.cpp" };
    concepts_[91] = { 92, 3, "HTTP Error Mapping", "API", "Invalid input maps to HTTP 400, engine errors to HTTP 500.", "src/server/Router.cpp" };
    concepts_[92] = { 93, 3, "Single-Request Connections", "Network", "One request and response per connection; there is no keep-alive.", "src/server/HttpServer.cpp" };
    concepts_[93] = { 94, 3, "Lock Ordering", "Concurrency", "wal_mutex_ is always acquired before index_rw_lock_.", "src/core/vector_engine.cpp" };
    concepts_[94] = { 95, 3, "Shared Graph Core", "Architecture", "HNSWIndex and HNSWIndexSQ8 share one templated traversal and linking implementation.", "src/index/hnsw_common.hpp" };
    concepts_[95] = { 96, 3, "Concurrency Stress Tests", "Testing", "Tests run concurrent searches and inserts against the engine.", "tests/test_vector_engine.cpp" };
    concepts_[96] = { 97, 3, "Server Shutdown", "Lifecycle", "stop() closes the listening socket; the pool destructor drains queued tasks and joins workers.", "src/server/HttpServer.cpp" };
    concepts_[97] = { 98, 3, "Static File Serving", "Network", "StaticFileHandler serves the dashboard from public/.", "src/server/StaticFileHandler.cpp" };
    concepts_[98] = { 99, 3, "Mutex-Guarded Task Dispatch", "Concurrency", "Connections are handed to workers through a mutex-protected queue.", "src/server/WorkerThreadPool.cpp" };
    concepts_[99] = { 100, 3, "Monotonic Clock Timing", "Telemetry", "std::chrono::steady_clock, backed by the operating system monotonic clock.", "src/server/Router.cpp" };
}
