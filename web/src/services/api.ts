// Thin client for the Nexus C++ REST engine.
// Every call degrades gracefully: when the backend is unreachable the UI keeps
// working against a deterministic local simulation of the same 100 concepts.

import { FALLBACK_CONCEPTS } from '../data/concepts';

export interface Concept {
  id: number;
  branch: number;
  label: string;
  category: string;
  description: string;
  code_ref: string;
}

export interface SearchHit {
  id: number;
  distance: number;
}

export interface SearchResponse {
  took_us: number;
  k: number;
  results: SearchHit[];
  /** True when the response was synthesised locally (backend offline). */
  simulated: boolean;
}

export interface EngineStatus {
  online: boolean;
  server: string;
  engineReady: boolean;
  dimension: number;
  totalVectors: number;
}

export const FALLBACK_TOTAL_VECTORS = 100_000;
export const DIMENSION = 128;
/** Each branch owns one orthogonal 32-dimension block of the 128-d space. */
export const BLOCK = 32;

const OFFLINE_STATUS: EngineStatus = {
  online: false,
  server: 'Nexus Vector Engine (standalone preview)',
  engineReady: false,
  dimension: DIMENSION,
  totalVectors: FALLBACK_TOTAL_VECTORS,
};

async function fetchWithTimeout(input: string, init: RequestInit = {}, timeoutMs = 2500): Promise<Response> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: ctrl.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

export async function fetchEngineStatus(): Promise<EngineStatus> {
  try {
    const res = await fetchWithTimeout('/api/status');
    if (!res.ok) return OFFLINE_STATUS;
    const data = (await res.json()) as {
      status?: string;
      server?: string;
      engine_ready?: boolean;
      dimension?: number;
      total_vectors?: number;
    };
    if (data.status !== 'ok') return OFFLINE_STATUS;
    return {
      online: true,
      server: data.server ?? 'Nexus Vector Database Engine',
      engineReady: Boolean(data.engine_ready),
      dimension: data.dimension && data.dimension > 0 ? data.dimension : DIMENSION,
      totalVectors:
        typeof data.total_vectors === 'number' && data.total_vectors > 0
          ? data.total_vectors
          : FALLBACK_TOTAL_VECTORS,
    };
  } catch {
    return OFFLINE_STATUS;
  }
}

function isConcept(x: unknown): x is Concept {
  const c = x as Partial<Concept> | null;
  return (
    !!c &&
    typeof c.id === 'number' &&
    typeof c.branch === 'number' &&
    typeof c.label === 'string' &&
    typeof c.category === 'string' &&
    typeof c.description === 'string' &&
    typeof c.code_ref === 'string'
  );
}

/** Loads the indexed concepts from GET /api/concepts, or the local mirror. */
export async function fetchConcepts(): Promise<{ concepts: Concept[]; fromEngine: boolean }> {
  try {
    const res = await fetchWithTimeout('/api/concepts');
    if (res.ok) {
      const data = (await res.json()) as { concepts?: unknown[] };
      if (Array.isArray(data.concepts) && data.concepts.length > 0 && data.concepts.every(isConcept)) {
        return { concepts: data.concepts as Concept[], fromEngine: true };
      }
    }
  } catch {
    // fall through to the local mirror
  }
  return { concepts: FALLBACK_CONCEPTS, fromEngine: false };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Query vector for a concept.
 *
 * The server builds every concept vector as its branch centroid (ones on the
 * branch's 32-dim block) plus N(0, 0.08) noise, L2-normalised, but it does not
 * expose the vectors. We rebuild the same construction with a deterministic
 * per-concept noise seed, so the query lands inside the right branch and
 * neighbours are ranked by the engine's real distance computation.
 */
export function conceptQueryVector(concept: Concept): number[] {
  const rand = mulberry32(concept.id * 7919 + 1337);
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand());
  const v = new Array<number>(DIMENSION);
  let norm = 0;
  for (let d = 0; d < DIMENSION; d++) {
    const base = d >= concept.branch * BLOCK && d < (concept.branch + 1) * BLOCK ? 1 : 0;
    v[d] = base + gauss() * 0.08;
    norm += v[d] * v[d];
  }
  const inv = 1 / Math.sqrt(norm);
  return v.map((x) => Number((x * inv).toFixed(6)));
}

/** Offline model: same-branch concepts are near, other branches are ~orthogonal. */
function simulateSearch(concept: Concept, concepts: Concept[], k: number): SearchResponse {
  const scored = concepts
    .filter((c) => c.id !== concept.id)
    .map((c) => {
      const r = mulberry32(concept.id * 131 + c.id * 31)();
      const distance = c.branch === concept.branch ? 0.012 + r * 0.05 : 1.9 + r * 0.2;
      return { id: c.id, distance: Number(distance.toFixed(5)) };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
  return { took_us: Math.round(58 + Math.log2(k + 1) * 14 + Math.random() * 40), k, results: scored, simulated: true };
}

export async function searchConcept(concept: Concept, concepts: Concept[], k: number): Promise<SearchResponse> {
  // Ask for one extra hit: the query concept itself may be returned.
  const ask = k + 1;
  try {
    const res = await fetchWithTimeout(
      '/api/search',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: ask, vector: conceptQueryVector(concept) }),
      },
      4000,
    );
    if (res.ok) {
      const data = (await res.json()) as { status?: string; took_us?: number; results?: SearchHit[] };
      if (data.status === 'success' && typeof data.took_us === 'number' && Array.isArray(data.results)) {
        const results = data.results
          .filter((h) => h.id !== concept.id)
          .slice(0, k)
          .map((h) => ({ id: h.id, distance: h.distance }));
        return { took_us: data.took_us, k, results, simulated: false };
      }
    }
  } catch {
    // fall through to the local simulation
  }
  await new Promise((r) => window.setTimeout(r, 90 + Math.random() * 110));
  return simulateSearch(concept, concepts, k);
}
