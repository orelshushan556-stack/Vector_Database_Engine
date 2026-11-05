// Architectural roadmap shown in the cinematic tour.
//
// NOTE: the benchmark figures below are reference targets for a 100k x 128-d
// float32 corpus. Replace them with the numbers printed by
// `benchmarks/bench_sift.cpp` on your hardware.

export type TourFocus = 'simd' | 'hnsw' | 'sq8' | 'wal' | 'network';

export interface RoadmapStep {
  id: TourFocus;
  index: string;
  title: string;
  subtitle: string;
  bottleneck: string;
  solution: string;
  codeRef: string;
  benchmarks: { label: string; value: string; detail: string }[];
}

export const ROADMAP: RoadmapStep[] = [
  {
    id: 'simd',
    index: '01',
    title: 'SIMD Kernels',
    subtitle: 'Distance computation at register width',
    bottleneck:
      'Every hop scores candidates by L2 distance. Scalar loops handle one float per cycle, so the CPU sits idle while the search waits on arithmetic.',
    solution:
      'AVX2 256-bit kernels (l2_distance_squared_avx2, dot_product_avx2) process 8 lanes per instruction across four independent FMA accumulators, then combine them with a horizontal add.',
    codeRef: 'include/engine/distance/simd_metrics.hpp',
    benchmarks: [
      { label: 'L2 / 128-d', value: '~14 ns', detail: 'vs ~95 ns scalar' },
      { label: 'Throughput', value: '~6.8x', detail: '4 accumulators' },
      { label: 'Footprint', value: '512 B', detail: 'per float32 vector' },
    ],
  },
  {
    id: 'hnsw',
    index: '02',
    title: 'HNSW Hierarchy',
    subtitle: 'Logarithmic descent through skip-list layers',
    bottleneck:
      'Brute-force k-NN is O(N·d). At 100k vectors that means 12.8M multiply-adds per query, which rules out sub-millisecond latency.',
    solution:
      'Multi-layer navigable small-world graph (M=16, efConstruction=200, efSearch=50). Greedy search starts at a sparse apex layer and refines layer by layer down to a dense base, where a beam search runs.',
    codeRef: 'include/engine/index/hnsw_index.hpp',
    benchmarks: [
      { label: 'Query p50', value: '~120 µs', detail: 'vs ~9.8 ms flat' },
      { label: 'Recall@10', value: '~0.97', detail: 'efSearch = 50' },
      { label: 'Graph', value: '128 B', detail: '2M links / node at L0' },
    ],
  },
  {
    id: 'sq8',
    index: '03',
    title: 'SQ8 Quantization',
    subtitle: 'Compressed memory density',
    bottleneck:
      'float32 vectors are memory-bandwidth bound. The dense base layer evicts cache lines faster than the SIMD units can use them.',
    solution:
      'ScalarQuantizer encodes each dimension to uint8 using per-dimension scales. HNSWIndexSQ8 scores directly on the quantized bytes with quantized_l2_distance_sq_avx2.',
    codeRef: 'include/engine/quantization/scalar_quantizer.hpp',
    benchmarks: [
      { label: 'Compression', value: '4x', detail: '512 B to 128 B' },
      { label: '100k corpus', value: '12.8 MB', detail: 'vs 51.2 MB float32' },
      { label: 'Latency', value: '-30%', detail: 'fewer cache misses' },
    ],
  },
  {
    id: 'wal',
    index: '04',
    title: 'WAL Durability',
    subtitle: 'Crash-safe persistence with CRC32-C',
    bottleneck:
      'An in-memory index loses every insert on a crash, and rebuilding the graph from raw data takes minutes.',
    solution:
      'Append-only write-ahead log (WalManager) with a CRC32-C checksum on every record. The checksum uses the SSE4.2 _mm_crc32_u64 instruction, 8 bytes per step. Binary checkpoints from the Serializer bound replay time.',
    codeRef: 'include/engine/storage/wal_manager.hpp',
    benchmarks: [
      { label: 'CRC32-C', value: '>10 GB/s', detail: 'hardware SSE4.2' },
      { label: 'Append', value: '~3 µs', detail: 'per buffered record' },
      { label: 'Replay', value: '~80 ms', detail: '100k records' },
    ],
  },
  {
    id: 'network',
    index: '05',
    title: 'Network Engine',
    subtitle: 'Native HTTP/REST surface',
    bottleneck:
      'Spawning a thread per request and serialising JSON with heap-heavy libraries adds more latency than the search itself.',
    solution:
      'Native HttpServer with a WorkerThreadPool, a hand-rolled request parser and a Router for /api/status, /api/search, /api/insert and /api/checkpoint. The /api/concepts catalog is precompiled once and served without rebuilding it.',
    codeRef: 'src/server/Router.cpp',
    benchmarks: [
      { label: 'took_us', value: 'live', detail: 'steady_clock, per query' },
      { label: 'Catalog', value: 'O(1)', detail: 'precompiled JSON' },
      { label: 'Workers', value: 'pooled', detail: 'no thread-per-request' },
    ],
  },
];
