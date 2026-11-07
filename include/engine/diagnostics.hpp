#pragma once

// Compile-time benchmark instrumentation. Every toggle defaults to off; with all of them off
// this header defines only no-op macros and the engine compiles exactly as before.
//
//   NEXUS_COUNT_DISTANCES  Count distance evaluations made by the HNSW indexes on the calling
//                          thread (thread-local counters, read and reset by the benchmark).
//   NEXUS_DISABLE_PREFETCH Make hnsw_detail::prefetch_bytes a no-op (prefetch ablation).
//   NEXUS_WAL_NO_FSYNC     Skip the per-record fdatasync/FlushFileBuffers in the WAL (write-path
//                          benchmark only; records are still fflush'ed to the OS). Never use in
//                          production: acknowledged inserts are no longer durable on power loss.

#include <cstdint>

#if defined(NEXUS_COUNT_DISTANCES)
namespace engine::diag {
// Full-precision (FP32) distance evaluations.
inline thread_local uint64_t exact_distance_count = 0;
// SQ8 code distance evaluations (HNSWIndexSQ8 traversal with the L2 metric).
inline thread_local uint64_t quantized_distance_count = 0;

inline void reset_distance_counters() noexcept {
    exact_distance_count = 0;
    quantized_distance_count = 0;
}
} // namespace engine::diag
#define NEXUS_COUNT_EXACT_DISTANCE() (++::engine::diag::exact_distance_count)
#define NEXUS_COUNT_QUANTIZED_DISTANCE() (++::engine::diag::quantized_distance_count)
#else
#define NEXUS_COUNT_EXACT_DISTANCE() ((void)0)
#define NEXUS_COUNT_QUANTIZED_DISTANCE() ((void)0)
#endif
