// Generated from src/server/ConceptRegistry.cpp: offline fallback mirroring the 100 indexed concepts.
import type { Concept } from '../services/api';

export const FALLBACK_CONCEPTS: Concept[] = [
  {
    "id": 1,
    "branch": 0,
    "label": "AVX2 256-bit Registers",
    "category": "SIMD",
    "description": "Utilizing 8-wide float vector registers (_mm256) for parallel arithmetic.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 2,
    "branch": 0,
    "label": "Fused Multiply-Add (FMA)",
    "category": "SIMD",
    "description": "Single-cycle (A * B) + C execution via _mm256_fmadd_ps.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 3,
    "branch": 0,
    "label": "L2 Distance Squared Kernel",
    "category": "SIMD",
    "description": "AVX2-accelerated Euclidean distance calculation.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 4,
    "branch": 0,
    "label": "Cosine Distance Kernel",
    "category": "SIMD",
    "description": "Normalized inner product distance with SIMD dot products.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 5,
    "branch": 0,
    "label": "Dot Product AVX2",
    "category": "SIMD",
    "description": "High-throughput vector projection using horizontal sum.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 6,
    "branch": 0,
    "label": "Loop Unrolling 4x",
    "category": "Optimization",
    "description": "Unrolling vector loops to maximize instruction pipeline saturation.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 7,
    "branch": 0,
    "label": "Tail Remainder Handling",
    "category": "SIMD",
    "description": "Scalar processing for dimensions not divisible by 8.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 8,
    "branch": 0,
    "label": "Horizontal Vector Sum",
    "category": "SIMD",
    "description": "Reduction of 256-bit register to single float scalar.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 9,
    "branch": 0,
    "label": "64-Byte Cache Alignment",
    "category": "Memory",
    "description": "Aligning vector storage to L1 cache lines to avoid split loads.",
    "code_ref": "engine/core/aligned_allocator.hpp"
  },
  {
    "id": 10,
    "branch": 0,
    "label": "AlignedVector Container",
    "category": "Memory",
    "description": "Custom STL-compatible container with _aligned_malloc.",
    "code_ref": "engine/core/aligned_allocator.hpp"
  },
  {
    "id": 11,
    "branch": 0,
    "label": "VectorView Zero-Copy",
    "category": "Memory",
    "description": "Lightweight non-owning span over contiguous memory.",
    "code_ref": "engine/core/vector_view.hpp"
  },
  {
    "id": 12,
    "branch": 0,
    "label": "Software Prefetching",
    "category": "CPU Cache",
    "description": "Hardware prefetching of candidate nodes before distance calculation.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 13,
    "branch": 0,
    "label": "Stream Non-Temporal Stores",
    "category": "Memory",
    "description": "Bypassing cache hierarchies during bulk ingestion.",
    "code_ref": "engine/core/aligned_allocator.hpp"
  },
  {
    "id": 14,
    "branch": 0,
    "label": "Branch Elimination",
    "category": "Optimization",
    "description": "Moving metric switches outside tight loops for zero branch penalty.",
    "code_ref": "engine/index/flat_index.cpp"
  },
  {
    "id": 15,
    "branch": 0,
    "label": "Fast Reciprocal SQRT",
    "category": "SIMD",
    "description": "_mm256_rsqrt_ps for instant vector normalization.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 16,
    "branch": 0,
    "label": "SSE4.2 CRC32-C Intrinsics",
    "category": "Hardware",
    "description": "_mm_crc32_u64 hardware checksum for WAL validation.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 17,
    "branch": 0,
    "label": "Runtime CPUID Detection",
    "category": "Hardware",
    "description": "Runtime CPUID inspection to detect AVX2, FMA, and SSE support.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 18,
    "branch": 0,
    "label": "Register Pressure Balancing",
    "category": "Optimization",
    "description": "Balancing register usage to prevent stack spills on x86_64.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 19,
    "branch": 0,
    "label": "Zero Allocation Search Path",
    "category": "Memory",
    "description": "Eliminating dynamic heap allocation in the hot query path.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 20,
    "branch": 0,
    "label": "Auto-Vectorization Primitives",
    "category": "Optimization",
    "description": "Compiler-friendly loop structures promoting auto-vectorization.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 21,
    "branch": 0,
    "label": "Unaligned Load Safety",
    "category": "SIMD",
    "description": "_mm256_loadu_ps handling arbitrary user vector pointers safely.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 22,
    "branch": 0,
    "label": "Instruction Parallelism",
    "category": "CPU",
    "description": "Interleaving independent FMA accumulators for ILP.",
    "code_ref": "engine/distance/simd_metrics.cpp"
  },
  {
    "id": 23,
    "branch": 0,
    "label": "SIFT-10K 128D Tuning",
    "category": "Benchmark",
    "description": "128-dimensional optimizations matching standard vector benchmarks.",
    "code_ref": "benchmarks/bench_sift.cpp"
  },
  {
    "id": 24,
    "branch": 0,
    "label": "Inline Metric Functors",
    "category": "Architecture",
    "description": "Static dispatch of distance functions avoiding vtable overhead.",
    "code_ref": "engine/distance/simd_metrics.hpp"
  },
  {
    "id": 25,
    "branch": 0,
    "label": "Contiguous Neighbor Layout",
    "category": "Memory",
    "description": "Flattening graph neighbor lists into contiguous buffers.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 26,
    "branch": 1,
    "label": "HNSW Multi-Layer Skip Graph",
    "category": "Graph",
    "description": "Multi-layer skip-list graph for logarithmic nearest-neighbor search.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 27,
    "branch": 1,
    "label": "Exponential Level Generator",
    "category": "Algorithm",
    "description": "get_random_level() using geometric decay: -ln(unif) * mult.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 28,
    "branch": 1,
    "label": "Layer 2 Apex Entry Point",
    "category": "Graph",
    "description": "Global entry node at top sparse level for coarse navigation.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 29,
    "branch": 1,
    "label": "Greedy Layer Descent",
    "category": "Search",
    "description": "1-greedy hopping through upper layers to locate local neighborhood.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 30,
    "branch": 1,
    "label": "search_layer Priority Queues",
    "category": "Search",
    "description": "Dual-heap (candidates & top_candidates) bounded traversal.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 31,
    "branch": 1,
    "label": "Layer 0 Base Connectivity",
    "category": "Graph",
    "description": "max_m0_ (2*M) degree allowance for dense base clustering.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 32,
    "branch": 1,
    "label": "ef_search Beam Width",
    "category": "Tuning",
    "description": "Runtime balance between query recall and search latency.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 33,
    "branch": 1,
    "label": "ef_construction Quality",
    "category": "Indexing",
    "description": "Candidate beam size during index insertion graph building.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 34,
    "branch": 1,
    "label": "Heuristic Neighbor Selection",
    "category": "Graph",
    "description": "Pruning neighbors to retain diverse directions and avoid clustering.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 35,
    "branch": 1,
    "label": "Bi-directional Edge Shrink",
    "category": "Graph",
    "description": "Keeping graph in-degree and out-degree within bounded M limits.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 36,
    "branch": 1,
    "label": "Scalar Quantizer (SQ8)",
    "category": "Quantization",
    "description": "Quantizing 32-bit floats to 8-bit unsigned integers (75% RAM save).",
    "code_ref": "engine/quantization/scalar_quantizer.cpp"
  },
  {
    "id": 37,
    "branch": 1,
    "label": "Asymmetric Distance Calc (ADC)",
    "category": "Search",
    "description": "Query float vector against 8-bit quantized database vectors.",
    "code_ref": "engine/index/hnsw_index_sq8.cpp"
  },
  {
    "id": 38,
    "branch": 1,
    "label": "Re-ranking Pipeline",
    "category": "Algorithm",
    "description": "Top candidate re-evaluation using exact full-precision vectors.",
    "code_ref": "engine/index/hnsw_index_sq8.cpp"
  },
  {
    "id": 39,
    "branch": 1,
    "label": "Recall@10 SIFT Benchmark",
    "category": "Benchmark",
    "description": "Achieved 99.7%–100% ground-truth accuracy on SIFT-10K.",
    "code_ref": "benchmarks/bench_sift.cpp"
  },
  {
    "id": 40,
    "branch": 1,
    "label": "FlatIndex Ground Truth",
    "category": "Index",
    "description": "Exact brute-force index for baseline accuracy validation.",
    "code_ref": "engine/index/flat_index.cpp"
  },
  {
    "id": 41,
    "branch": 1,
    "label": "QuantizedFlatIndex Scan",
    "category": "Index",
    "description": "Linear scan over compact SQ8 vector payloads.",
    "code_ref": "engine/index/quantized_flat_index.cpp"
  },
  {
    "id": 42,
    "branch": 1,
    "label": "Flat Link Storage Buffer",
    "category": "Memory",
    "description": "Contiguous link buffer avoiding pointer indirection overhead.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 43,
    "branch": 1,
    "label": "C++20 std::span Link Access",
    "category": "C++20",
    "description": "std::span<const uint32_t> for zero-copy graph inspection.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 44,
    "branch": 1,
    "label": "Magic Versioning Tags",
    "category": "Optimization",
    "description": "Integer version tags eliminating memset operations on visited sets.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 45,
    "branch": 1,
    "label": "Pre-allocated Adjacency Reserve",
    "category": "Memory",
    "description": "Pre-allocating vector and adjacency storage to prevent reallocations.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 46,
    "branch": 1,
    "label": "Compact Upper Graph Storage",
    "category": "Graph",
    "description": "Compact nested link storage for layers > 0.",
    "code_ref": "engine/index/hnsw_index.hpp"
  },
  {
    "id": 47,
    "branch": 1,
    "label": "KNN Bounded Result Heap",
    "category": "Search",
    "description": "std::priority_queue retaining only top-K nearest matches.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 48,
    "branch": 1,
    "label": "MetricType Abstraction Enum",
    "category": "Architecture",
    "description": "Unified enum supporting L2, Cosine, and Inner Product metrics.",
    "code_ref": "engine/core/types.hpp"
  },
  {
    "id": 49,
    "branch": 1,
    "label": "Dimension Validation",
    "category": "Validation",
    "description": "Enforcing vector dimensional consistency at insert and query time.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 50,
    "branch": 1,
    "label": "Graph Connectivity Traversal",
    "category": "Diagnostics",
    "description": "Validating that all nodes remain reachable from entry point.",
    "code_ref": "tests/test_hnsw.cpp"
  },
  {
    "id": 51,
    "branch": 2,
    "label": "Write-Ahead Log (WAL)",
    "category": "Persistence",
    "description": "Sequential logging of mutations before in-memory index application.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 52,
    "branch": 2,
    "label": "Hardware CRC32-C Integrity",
    "category": "Reliability",
    "description": "Hardware-accelerated checksum verification for every log record.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 53,
    "branch": 2,
    "label": "Append-Only Log Strategy",
    "category": "I/O",
    "description": "Eliminating random disk writes for optimal NVMe throughput.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 54,
    "branch": 2,
    "label": "Crash Recovery Replay",
    "category": "Persistence",
    "description": "Rebuilding state from WAL upon ungraceful shutdown.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 55,
    "branch": 2,
    "label": "Unified Serializer Interface",
    "category": "Serialization",
    "description": "Unified serializer separating file persistence from core index logic.",
    "code_ref": "engine/storage/serializer.cpp"
  },
  {
    "id": 56,
    "branch": 2,
    "label": "Magic Number 0x56454349 (VECI)",
    "category": "Binary Format",
    "description": "0x56454349 file header identifying FlatIndex archives.",
    "code_ref": "engine/storage/serializer.hpp"
  },
  {
    "id": 57,
    "branch": 2,
    "label": "Magic Number 0x484E5357 (HNSW)",
    "category": "Binary Format",
    "description": "0x484E5357 file header identifying HNSWIndex archives.",
    "code_ref": "engine/storage/serializer.hpp"
  },
  {
    "id": 58,
    "branch": 2,
    "label": "Magic Number 0x53513848 (SQ8H)",
    "category": "Binary Format",
    "description": "0x53513848 file header identifying HNSWIndexSQ8 archives.",
    "code_ref": "engine/storage/serializer.hpp"
  },
  {
    "id": 59,
    "branch": 2,
    "label": "Factory Pattern Index Loader",
    "category": "Design Pattern",
    "description": "Automatic polymorphic index instantiation from magic number.",
    "code_ref": "engine/storage/serializer.cpp"
  },
  {
    "id": 60,
    "branch": 2,
    "label": "Slotted Page Layout",
    "category": "Storage",
    "description": "Fixed-size page headers with variable offset indirection.",
    "code_ref": "engine/storage/page_manager.hpp"
  },
  {
    "id": 61,
    "branch": 2,
    "label": "Memory-Mapped Files (mmap)",
    "category": "OS Memory",
    "description": "Virtual memory paging bypassing user-space buffer copies.",
    "code_ref": "engine/storage/mmap_storage.cpp"
  },
  {
    "id": 62,
    "branch": 2,
    "label": "Hardware Fsync / Flush Guarantee",
    "category": "Reliability",
    "description": "Forcing disk controller caches to persistent non-volatile media.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 63,
    "branch": 2,
    "label": "WAL Log Compaction",
    "category": "Maintenance",
    "description": "Purging obsolete mutation records after full checkpoint.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 64,
    "branch": 2,
    "label": "Snapshot Checkpointing",
    "category": "Persistence",
    "description": "Atomic point-in-time index serialization to secondary storage.",
    "code_ref": "engine/storage/serializer.cpp"
  },
  {
    "id": 65,
    "branch": 2,
    "label": "Endian-Safe Binary Encoding",
    "category": "Portability",
    "description": "Explicit little-endian serialisation across CPU architectures.",
    "code_ref": "engine/storage/serializer.cpp"
  },
  {
    "id": 66,
    "branch": 2,
    "label": "Monotonic Sequence Numbers",
    "category": "Protocol",
    "description": "Strict ordering of operations across WAL and memory graph.",
    "code_ref": "engine/storage/wal.hpp"
  },
  {
    "id": 67,
    "branch": 2,
    "label": "Atomic Log File Renaming",
    "category": "Reliability",
    "description": "POSIX rename / MoveFileEx guaranteeing atomic file swaps.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 68,
    "branch": 2,
    "label": "Corrupted Frame Discard",
    "category": "Recovery",
    "description": "Safely truncating damaged WAL tails without index corruption.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 69,
    "branch": 2,
    "label": "Zero-Copy Record Parsing",
    "category": "Performance",
    "description": "Direct struct casting over raw mmap record buffers.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 70,
    "branch": 2,
    "label": "Variable-Length Record Header",
    "category": "Format",
    "description": "Compact 16-byte metadata preamble per WAL entry.",
    "code_ref": "engine/storage/wal.hpp"
  },
  {
    "id": 71,
    "branch": 2,
    "label": "Buffer Pool Manager",
    "category": "Memory",
    "description": "Frame caching coordinating dirty page flushes.",
    "code_ref": "engine/storage/buffer_pool.hpp"
  },
  {
    "id": 72,
    "branch": 2,
    "label": "Page Eviction Policy",
    "category": "Algorithm",
    "description": "CLOCK / LRU replacement avoiding buffer thrashing.",
    "code_ref": "engine/storage/buffer_pool.cpp"
  },
  {
    "id": 73,
    "branch": 2,
    "label": "Persistent Segment Allocation",
    "category": "Storage",
    "description": "Pre-allocating 64MB extents to minimize fragmentation.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 74,
    "branch": 2,
    "label": "Direct I/O (O_DIRECT)",
    "category": "I/O",
    "description": "Bypassing OS page cache for deterministic disk latency.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 75,
    "branch": 2,
    "label": "Write Stall Throttling",
    "category": "Flow Control",
    "description": "Backpressure protection during high-volume persistence peaks.",
    "code_ref": "engine/storage/wal.cpp"
  },
  {
    "id": 76,
    "branch": 3,
    "label": "thread_local VisitedBuffer",
    "category": "Lock-Free",
    "description": "Zero-contention visited buffer per worker thread.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 77,
    "branch": 3,
    "label": "Zero-Contention Search Traversal",
    "category": "Concurrency",
    "description": "Permitting 100% lock-free concurrent read operations.",
    "code_ref": "engine/index/hnsw_index.cpp"
  },
  {
    "id": 78,
    "branch": 3,
    "label": "Worker Thread Pool",
    "category": "Threading",
    "description": "Fixed-size thread pool matching hardware thread concurrency.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 79,
    "branch": 3,
    "label": "Work-Stealing Task Queue",
    "category": "Scheduling",
    "description": "Load balancing requests across CPU worker cores.",
    "code_ref": "src/server/ThreadPool.hpp"
  },
  {
    "id": 80,
    "branch": 3,
    "label": "std::shared_mutex R/W Locks",
    "category": "Synchronization",
    "description": "Granular shared-read exclusive-write concurrency model.",
    "code_ref": "engine/core/vector_engine.hpp"
  },
  {
    "id": 81,
    "branch": 3,
    "label": "Atomic Sequence Generator",
    "category": "Atomics",
    "description": "Lock-free std::atomic<uint64_t> for global query IDs.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 82,
    "branch": 3,
    "label": "Epoll / Completion Ports",
    "category": "Network OS",
    "description": "Edge-triggered socket event polling on Linux/Windows.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 83,
    "branch": 3,
    "label": "Winsock2 / POSIX Abstraction",
    "category": "Cross-Platform",
    "description": "Unified socket wrapper compiling on Windows and Linux.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 84,
    "branch": 3,
    "label": "Custom HTTP Engine (nexus_server)",
    "category": "Network",
    "description": "Lightweight embedded HTTP/1.1 engine without external dependencies.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 85,
    "branch": 3,
    "label": "Streaming Content-Length Reader",
    "category": "Network",
    "description": "Resilient recv loop handling segmented TCP packet arrivals.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 86,
    "branch": 3,
    "label": "REST API /api/search Route",
    "category": "API",
    "description": "High-speed JSON vector search endpoint.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 87,
    "branch": 3,
    "label": "REST API /api/status Telemetry",
    "category": "API",
    "description": "Telemetry endpoint exposing engine health and vector count.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 88,
    "branch": 3,
    "label": "REST API /api/concepts Catalog",
    "category": "API",
    "description": "Fast O(1) serving of technical concepts and architecture mapping.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 89,
    "branch": 3,
    "label": "Microsecond Latency Telemetry",
    "category": "Diagnostics",
    "description": "Precision std::chrono::steady_clock benchmark reporting.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 90,
    "branch": 3,
    "label": "CORS Policy Enforcement",
    "category": "Security",
    "description": "Preflight OPTIONS handling and permissive cross-origin headers.",
    "code_ref": "src/server/Router.cpp"
  },
  {
    "id": 91,
    "branch": 3,
    "label": "Non-blocking Socket Handshake",
    "category": "Network",
    "description": "fcntl / ioctlsocket for non-blocking client connections.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 92,
    "branch": 3,
    "label": "Thread Affinity (CPU Pinning)",
    "category": "OS",
    "description": "Binding worker threads to dedicated CPU cores for L1 cache residency.",
    "code_ref": "src/server/ThreadPool.cpp"
  },
  {
    "id": 93,
    "branch": 3,
    "label": "Memory Fence & Barrier Tuning",
    "category": "Hardware",
    "description": "std::memory_order_acquire/release synchronization.",
    "code_ref": "engine/core/vector_engine.cpp"
  },
  {
    "id": 94,
    "branch": 3,
    "label": "False Sharing Prevention",
    "category": "Cache",
    "description": "Cache-line spacing preventing L1/L2 bounce between cores.",
    "code_ref": "src/server/ThreadPool.hpp"
  },
  {
    "id": 95,
    "branch": 3,
    "label": "alignas(64) Worker Padding",
    "category": "Memory",
    "description": "64-byte structural alignment on thread-local states.",
    "code_ref": "src/server/ThreadPool.hpp"
  },
  {
    "id": 96,
    "branch": 3,
    "label": "Concurrent Batch Ingestion",
    "category": "Parallelism",
    "description": "Partitioned parallel vector ingestion with SIMD metrics.",
    "code_ref": "engine/core/vector_engine.cpp"
  },
  {
    "id": 97,
    "branch": 3,
    "label": "Graceful Server Shutdown",
    "category": "Lifecycle",
    "description": "Atomic cancellation token ensuring inflight queries finish.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 98,
    "branch": 3,
    "label": "Connection Keep-Alive Engine",
    "category": "Network",
    "description": "Reusing TCP sockets to minimize TCP 3-way handshake overhead.",
    "code_ref": "src/server/HttpServer.cpp"
  },
  {
    "id": 99,
    "branch": 3,
    "label": "Lock-Free Single-Producer Queue",
    "category": "Lock-Free",
    "description": "Atomic ring buffer for inter-thread task dispatching.",
    "code_ref": "src/server/ThreadPool.hpp"
  },
  {
    "id": 100,
    "branch": 3,
    "label": "High-Resolution Steady Clock",
    "category": "Telemetry",
    "description": "Sub-microsecond timing using OS monotonic TSC registers.",
    "code_ref": "src/server/Router.cpp"
  }
];
