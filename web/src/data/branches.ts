export interface Branch {
  id: number;
  name: string;
  tab: string;
  color: string;
  /** Tailwind-free token used for CSS custom properties. */
  token: 'sky' | 'indigo' | 'amber' | 'emerald';
  /** Why concepts in this branch cluster together in vector space. */
  cluster: string;
}

export const BRANCHES: Branch[] = [
  {
    id: 0,
    name: 'SIMD & Hardware',
    tab: 'SIMD & Hardware',
    color: '#38bdf8',
    token: 'sky',
    cluster: 'dimensions 0-31',
  },
  {
    id: 1,
    name: 'Graph & HNSW',
    tab: 'HNSW Topology',
    color: '#818cf8',
    token: 'indigo',
    cluster: 'dimensions 32-63',
  },
  {
    id: 2,
    name: 'Storage & WAL',
    tab: 'WAL Durability',
    color: '#f59e0b',
    token: 'amber',
    cluster: 'dimensions 64-95',
  },
  {
    id: 3,
    name: 'Concurrency & Network',
    tab: 'Concurrency',
    color: '#10b981',
    token: 'emerald',
    cluster: 'dimensions 96-127',
  },
];

/** Flagship concepts surfaced as quick-select chips (ids from ConceptRegistry). */
export const FLAGSHIPS: { id: number; label: string }[] = [
  { id: 1, label: 'AVX2 Registers' },
  { id: 2, label: 'FMA Dot Product' },
  { id: 26, label: 'Skip-List Layering' },
  { id: 64, label: 'WAL Checkpoint' },
  { id: 78, label: 'Worker Thread Pool' },
];
