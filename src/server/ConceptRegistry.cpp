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
    concepts_[8]  = { 9,  0, "64-Byte Cache Alignment", "Memory", "Aligning vector storage to L1 cache lines to avoid split loads.", "engine/core/aligned_allocator.hpp" };
    concepts_[9]  = { 10, 0, "AlignedVector Container", "Memory", "Custom STL-compatible container with _aligned_malloc.", "engine/core/aligned_allocator.hpp" };
    concepts_[10] = { 11, 0, "VectorView Zero-Copy", "Memory", "Lightweight non-owning span over contiguous memory.", "engine/core/vector_view.hpp" };
    concepts_[11] = { 12, 0, "Software Prefetching", "CPU Cache", "Hardware prefetching of candidate nodes before distance calculation.", "engine/index/hnsw_index.cpp" };
    concepts_[12] = { 13, 0, "Stream Non-Temporal Stores", "Memory", "Bypassing cache hierarchies during bulk ingestion.", "engine/core/aligned_allocator.hpp" };
    concepts_[13] = { 14, 0, "Branch Elimination", "Optimization", "Moving metric switches outside tight loops for zero branch penalty.", "engine/index/flat_index.cpp" };
    concepts_[14] = { 15, 0, "Fast Reciprocal SQRT", "SIMD", "_mm256_rsqrt_ps for instant vector normalization.", "engine/distance/simd_metrics.cpp" };
    concepts_[15] = { 16, 0, "SSE4.2 CRC32-C Intrinsics", "Hardware", "_mm_crc32_u64 hardware checksum for WAL validation.", "engine/storage/wal.cpp" };
    concepts_[16] = { 17, 0, "Runtime CPUID Detection", "Hardware", "Runtime CPUID inspection to detect AVX2, FMA, and SSE support.", "engine/distance/simd_metrics.cpp" };
    concepts_[17] = { 18, 0, "Register Pressure Balancing", "Optimization", "Balancing register usage to prevent stack spills on x86_64.", "engine/distance/simd_metrics.cpp" };
    concepts_[18] = { 19, 0, "Zero Allocation Search Path", "Memory", "Eliminating dynamic heap allocation in the hot query path.", "engine/index/hnsw_index.cpp" };
    concepts_[19] = { 20, 0, "Auto-Vectorization Primitives", "Optimization", "Compiler-friendly loop structures promoting auto-vectorization.", "engine/distance/simd_metrics.cpp" };
    concepts_[20] = { 21, 0, "Unaligned Load Safety", "SIMD", "_mm256_loadu_ps handling arbitrary user vector pointers safely.", "engine/distance/simd_metrics.cpp" };
    concepts_[21] = { 22, 0, "Instruction Parallelism", "CPU", "Interleaving independent FMA accumulators for ILP.", "engine/distance/simd_metrics.cpp" };
    concepts_[22] = { 23, 0, "SIFT-10K 128D Tuning", "Benchmark", "128-dimensional optimizations matching standard vector benchmarks.", "benchmarks/bench_sift.cpp" };
    concepts_[23] = { 24, 0, "Inline Metric Functors", "Architecture", "Static dispatch of distance functions avoiding vtable overhead.", "engine/distance/simd_metrics.hpp" };
    concepts_[24] = { 25, 0, "Contiguous Neighbor Layout", "Memory", "Flattening graph neighbor lists into contiguous buffers.", "engine/index/hnsw_index.hpp" };

    // -------------------------------------------------------------
    // BRANCH 1: Graph Topology & HNSW (26-50)
    // -------------------------------------------------------------
    concepts_[25] = { 26, 1, "HNSW Multi-Layer Skip Graph", "Graph", "Multi-layer skip-list graph for logarithmic nearest-neighbor search.", "engine/index/hnsw_index.cpp" };
    concepts_[26] = { 27, 1, "Exponential Level Generator", "Algorithm", "get_random_level() using geometric decay: -ln(unif) * mult.", "engine/index/hnsw_index.cpp" };
    concepts_[27] = { 28, 1, "Layer 2 Apex Entry Point", "Graph", "Global entry node at top sparse level for coarse navigation.", "engine/index/hnsw_index.hpp" };
    concepts_[28] = { 29, 1, "Greedy Layer Descent", "Search", "1-greedy hopping through upper layers to locate local neighborhood.", "engine/index/hnsw_index.cpp" };
    concepts_[29] = { 30, 1, "search_layer Priority Queues", "Search", "Dual-heap (candidates & top_candidates) bounded traversal.", "engine/index/hnsw_index.cpp" };
    concepts_[30] = { 31, 1, "Layer 0 Base Connectivity", "Graph", "max_m0_ (2*M) degree allowance for dense base clustering.", "engine/index/hnsw_index.hpp" };
    concepts_[31] = { 32, 1, "ef_search Beam Width", "Tuning", "Runtime balance between query recall and search latency.", "engine/index/hnsw_index.hpp" };
    concepts_[32] = { 33, 1, "ef_construction Quality", "Indexing", "Candidate beam size during index insertion graph building.", "engine/index/hnsw_index.hpp" };
    concepts_[33] = { 34, 1, "Heuristic Neighbor Selection", "Graph", "Pruning neighbors to retain diverse directions and avoid clustering.", "engine/index/hnsw_index.cpp" };
    concepts_[34] = { 35, 1, "Bi-directional Edge Shrink", "Graph", "Keeping graph in-degree and out-degree within bounded M limits.", "engine/index/hnsw_index.cpp" };
    concepts_[35] = { 36, 1, "Scalar Quantizer (SQ8)", "Quantization", "Quantizing 32-bit floats to 8-bit unsigned integers (75% RAM save).", "engine/quantization/scalar_quantizer.cpp" };
    concepts_[36] = { 37, 1, "Asymmetric Distance Calc (ADC)", "Search", "Query float vector against 8-bit quantized database vectors.", "engine/index/hnsw_index_sq8.cpp" };
    concepts_[37] = { 38, 1, "Re-ranking Pipeline", "Algorithm", "Top candidate re-evaluation using exact full-precision vectors.", "engine/index/hnsw_index_sq8.cpp" };
    concepts_[38] = { 39, 1, "Recall@10 SIFT Benchmark", "Benchmark", "Achieved 99.7%–100% ground-truth accuracy on SIFT-10K.", "benchmarks/bench_sift.cpp" };
    concepts_[39] = { 40, 1, "FlatIndex Ground Truth", "Index", "Exact brute-force index for baseline accuracy validation.", "engine/index/flat_index.cpp" };
    concepts_[40] = { 41, 1, "QuantizedFlatIndex Scan", "Index", "Linear scan over compact SQ8 vector payloads.", "engine/index/quantized_flat_index.cpp" };
    concepts_[41] = { 42, 1, "Flat Link Storage Buffer", "Memory", "Contiguous link buffer avoiding pointer indirection overhead.", "engine/index/hnsw_index.hpp" };
    concepts_[42] = { 43, 1, "C++20 std::span Link Access", "C++20", "std::span<const uint32_t> for zero-copy graph inspection.", "engine/index/hnsw_index.hpp" };
    concepts_[43] = { 44, 1, "Magic Versioning Tags", "Optimization", "Integer version tags eliminating memset operations on visited sets.", "engine/index/hnsw_index.cpp" };
    concepts_[44] = { 45, 1, "Pre-allocated Adjacency Reserve", "Memory", "Pre-allocating vector and adjacency storage to prevent reallocations.", "engine/index/hnsw_index.cpp" };
    concepts_[45] = { 46, 1, "Compact Upper Graph Storage", "Graph", "Compact nested link storage for layers > 0.", "engine/index/hnsw_index.hpp" };
    concepts_[46] = { 47, 1, "KNN Bounded Result Heap", "Search", "std::priority_queue retaining only top-K nearest matches.", "engine/index/hnsw_index.cpp" };
    concepts_[47] = { 48, 1, "MetricType Abstraction Enum", "Architecture", "Unified enum supporting L2, Cosine, and Inner Product metrics.", "engine/core/types.hpp" };
    concepts_[48] = { 49, 1, "Dimension Validation", "Validation", "Enforcing vector dimensional consistency at insert and query time.", "engine/index/hnsw_index.cpp" };
    concepts_[49] = { 50, 1, "Graph Connectivity Traversal", "Diagnostics", "Validating that all nodes remain reachable from entry point.", "tests/test_hnsw.cpp" };

    // -------------------------------------------------------------
    // BRANCH 2: Storage & WAL Persistence (51-75)
    // -------------------------------------------------------------
    concepts_[50] = { 51, 2, "Write-Ahead Log (WAL)", "Persistence", "Sequential logging of mutations before in-memory index application.", "engine/storage/wal.cpp" };
    concepts_[51] = { 52, 2, "Hardware CRC32-C Integrity", "Reliability", "Hardware-accelerated checksum verification for every log record.", "engine/storage/wal.cpp" };
    concepts_[52] = { 53, 2, "Append-Only Log Strategy", "I/O", "Eliminating random disk writes for optimal NVMe throughput.", "engine/storage/wal.cpp" };
    concepts_[53] = { 54, 2, "Crash Recovery Replay", "Persistence", "Rebuilding state from WAL upon ungraceful shutdown.", "engine/storage/wal.cpp" };
    concepts_[54] = { 55, 2, "Unified Serializer Interface", "Serialization", "Unified serializer separating file persistence from core index logic.", "engine/storage/serializer.cpp" };
    concepts_[55] = { 56, 2, "Magic Number 0x56454349 (VECI)", "Binary Format", "0x56454349 file header identifying FlatIndex archives.", "engine/storage/serializer.hpp" };
    concepts_[56] = { 57, 2, "Magic Number 0x484E5357 (HNSW)", "Binary Format", "0x484E5357 file header identifying HNSWIndex archives.", "engine/storage/serializer.hpp" };
    concepts_[57] = { 58, 2, "Magic Number 0x53513848 (SQ8H)", "Binary Format", "0x53513848 file header identifying HNSWIndexSQ8 archives.", "engine/storage/serializer.hpp" };
    concepts_[58] = { 59, 2, "Factory Pattern Index Loader", "Design Pattern", "Automatic polymorphic index instantiation from magic number.", "engine/storage/serializer.cpp" };
    concepts_[59] = { 60, 2, "Slotted Page Layout", "Storage", "Fixed-size page headers with variable offset indirection.", "engine/storage/page_manager.hpp" };
    concepts_[60] = { 61, 2, "Memory-Mapped Files (mmap)", "OS Memory", "Virtual memory paging bypassing user-space buffer copies.", "engine/storage/mmap_storage.cpp" };
    concepts_[61] = { 62, 2, "Hardware Fsync / Flush Guarantee", "Reliability", "Forcing disk controller caches to persistent non-volatile media.", "engine/storage/wal.cpp" };
    concepts_[62] = { 63, 2, "WAL Log Compaction", "Maintenance", "Purging obsolete mutation records after full checkpoint.", "engine/storage/wal.cpp" };
    concepts_[63] = { 64, 2, "Snapshot Checkpointing", "Persistence", "Atomic point-in-time index serialization to secondary storage.", "engine/storage/serializer.cpp" };
    concepts_[64] = { 65, 2, "Endian-Safe Binary Encoding", "Portability", "Explicit little-endian serialisation across CPU architectures.", "engine/storage/serializer.cpp" };
    concepts_[65] = { 66, 2, "Monotonic Sequence Numbers", "Protocol", "Strict ordering of operations across WAL and memory graph.", "engine/storage/wal.hpp" };
    concepts_[66] = { 67, 2, "Atomic Log File Renaming", "Reliability", "POSIX rename / MoveFileEx guaranteeing atomic file swaps.", "engine/storage/wal.cpp" };
    concepts_[67] = { 68, 2, "Corrupted Frame Discard", "Recovery", "Safely truncating damaged WAL tails without index corruption.", "engine/storage/wal.cpp" };
    concepts_[68] = { 69, 2, "Zero-Copy Record Parsing", "Performance", "Direct struct casting over raw mmap record buffers.", "engine/storage/wal.cpp" };
    concepts_[69] = { 70, 2, "Variable-Length Record Header", "Format", "Compact 16-byte metadata preamble per WAL entry.", "engine/storage/wal.hpp" };
    concepts_[70] = { 71, 2, "Buffer Pool Manager", "Memory", "Frame caching coordinating dirty page flushes.", "engine/storage/buffer_pool.hpp" };
    concepts_[71] = { 72, 2, "Page Eviction Policy", "Algorithm", "CLOCK / LRU replacement avoiding buffer thrashing.", "engine/storage/buffer_pool.cpp" };
    concepts_[72] = { 73, 2, "Persistent Segment Allocation", "Storage", "Pre-allocating 64MB extents to minimize fragmentation.", "engine/storage/wal.cpp" };
    concepts_[73] = { 74, 2, "Direct I/O (O_DIRECT)", "I/O", "Bypassing OS page cache for deterministic disk latency.", "engine/storage/wal.cpp" };
    concepts_[74] = { 75, 2, "Write Stall Throttling", "Flow Control", "Backpressure protection during high-volume persistence peaks.", "engine/storage/wal.cpp" };

    // -------------------------------------------------------------
    // BRANCH 3: Concurrency & OS Runtime (76-100)
    // -------------------------------------------------------------
    concepts_[75] = { 76, 3, "thread_local VisitedBuffer", "Lock-Free", "Zero-contention visited buffer per worker thread.", "engine/index/hnsw_index.cpp" };
    concepts_[76] = { 77, 3, "Zero-Contention Search Traversal", "Concurrency", "Permitting 100% lock-free concurrent read operations.", "engine/index/hnsw_index.cpp" };
    concepts_[77] = { 78, 3, "Worker Thread Pool", "Threading", "Fixed-size thread pool matching hardware thread concurrency.", "src/server/HttpServer.cpp" };
    concepts_[78] = { 79, 3, "Work-Stealing Task Queue", "Scheduling", "Load balancing requests across CPU worker cores.", "src/server/ThreadPool.hpp" };
    concepts_[79] = { 80, 3, "std::shared_mutex R/W Locks", "Synchronization", "Granular shared-read exclusive-write concurrency model.", "engine/core/vector_engine.hpp" };
    concepts_[80] = { 81, 3, "Atomic Sequence Generator", "Atomics", "Lock-free std::atomic<uint64_t> for global query IDs.", "src/server/Router.cpp" };
    concepts_[81] = { 82, 3, "Epoll / Completion Ports", "Network OS", "Edge-triggered socket event polling on Linux/Windows.", "src/server/HttpServer.cpp" };
    concepts_[82] = { 83, 3, "Winsock2 / POSIX Abstraction", "Cross-Platform", "Unified socket wrapper compiling on Windows and Linux.", "src/server/HttpServer.cpp" };
    concepts_[83] = { 84, 3, "Custom HTTP Engine (nexus_server)", "Network", "Lightweight embedded HTTP/1.1 engine without external dependencies.", "src/server/HttpServer.cpp" };
    concepts_[84] = { 85, 3, "Streaming Content-Length Reader", "Network", "Resilient recv loop handling segmented TCP packet arrivals.", "src/server/HttpServer.cpp" };
    concepts_[85] = { 86, 3, "REST API /api/search Route", "API", "High-speed JSON vector search endpoint.", "src/server/Router.cpp" };
    concepts_[86] = { 87, 3, "REST API /api/status Telemetry", "API", "Telemetry endpoint exposing engine health and vector count.", "src/server/Router.cpp" };
    concepts_[87] = { 88, 3, "REST API /api/concepts Catalog", "API", "Fast O(1) serving of technical concepts and architecture mapping.", "src/server/Router.cpp" };
    concepts_[88] = { 89, 3, "Microsecond Latency Telemetry", "Diagnostics", "Precision std::chrono::steady_clock benchmark reporting.", "src/server/Router.cpp" };
    concepts_[89] = { 90, 3, "CORS Policy Enforcement", "Security", "Preflight OPTIONS handling and permissive cross-origin headers.", "src/server/Router.cpp" };
    concepts_[90] = { 91, 3, "Non-blocking Socket Handshake", "Network", "fcntl / ioctlsocket for non-blocking client connections.", "src/server/HttpServer.cpp" };
    concepts_[91] = { 92, 3, "Thread Affinity (CPU Pinning)", "OS", "Binding worker threads to dedicated CPU cores for L1 cache residency.", "src/server/ThreadPool.cpp" };
    concepts_[92] = { 93, 3, "Memory Fence & Barrier Tuning", "Hardware", "std::memory_order_acquire/release synchronization.", "engine/core/vector_engine.cpp" };
    concepts_[93] = { 94, 3, "False Sharing Prevention", "Cache", "Cache-line spacing preventing L1/L2 bounce between cores.", "src/server/ThreadPool.hpp" };
    concepts_[94] = { 95, 3, "alignas(64) Worker Padding", "Memory", "64-byte structural alignment on thread-local states.", "src/server/ThreadPool.hpp" };
    concepts_[95] = { 96, 3, "Concurrent Batch Ingestion", "Parallelism", "Partitioned parallel vector ingestion with SIMD metrics.", "engine/core/vector_engine.cpp" };
    concepts_[96] = { 97, 3, "Graceful Server Shutdown", "Lifecycle", "Atomic cancellation token ensuring inflight queries finish.", "src/server/HttpServer.cpp" };
    concepts_[97] = { 98, 3, "Connection Keep-Alive Engine", "Network", "Reusing TCP sockets to minimize TCP 3-way handshake overhead.", "src/server/HttpServer.cpp" };
    concepts_[98] = { 99, 3, "Lock-Free Single-Producer Queue", "Lock-Free", "Atomic ring buffer for inter-thread task dispatching.", "src/server/ThreadPool.hpp" };
    concepts_[99] = { 100, 3, "High-Resolution Steady Clock", "Telemetry", "Sub-microsecond timing using OS monotonic TSC registers.", "src/server/Router.cpp" };
}
