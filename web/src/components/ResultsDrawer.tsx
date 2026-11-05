import { ArrowRight, CircleCheck, Clock, FileCode, LoaderCircle, Sliders, X } from 'lucide-react';
import { BRANCHES } from '../data/branches';
import type { Concept, SearchResponse } from '../services/api';

export interface PathStep {
  key: number;
  layer: 'L2' | 'L1' | 'L0';
  text: string;
}

function whyCluster(query: Concept, hit: Concept): string {
  const qb = BRANCHES[query.branch];
  const hb = BRANCHES[hit.branch];
  if (!qb || !hb) return '';
  if (qb.id === hb.id) {
    const shared = query.category === hit.category ? ` Both are filed under ${hit.category}.` : '';
    return `Same ${qb.name} subspace (${qb.cluster}), so the vectors point the same way and squared L2 distance stays small.${shared}`;
  }
  return `Crosses branches: ${hb.name} lives in ${hb.cluster}, an orthogonal block to ${qb.name} (${qb.cluster}), so the distance is large.`;
}

function BranchBadge({ branch }: { branch: number }) {
  const b = BRANCHES[branch];
  if (!b) return null;
  return (
    <span className="badge" style={{ ['--badge' as string]: b.color }}>
      <span className="dot" style={{ background: b.color }} />
      {b.name}
    </span>
  );
}

interface Props {
  concept: Concept;
  conceptsById: Map<number, Concept>;
  k: number;
  onK: (k: number) => void;
  onClose: () => void;
  searching: boolean;
  latencyUs: number | null;
  simulated: boolean;
  path: PathStep[];
  results: SearchResponse | null;
}

export function ResultsDrawer(props: Props) {
  const { concept, conceptsById, k, onK, onClose, searching, latencyUs, simulated, path, results } = props;
  return (
    <aside className="drawer glass" aria-label="Semantic search results">
      <header className="drawer-head">
        <div className="drawer-badges">
          <BranchBadge branch={concept.branch} />
          {latencyUs !== null && (
            <span className="badge badge-latency">
              <Clock size={12} />
              Latency: {latencyUs.toLocaleString('en-US')} {'µ'}s{simulated ? ' (simulated)' : ''}
            </span>
          )}
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close results">
          <X size={16} />
        </button>
      </header>

      <h2 className="drawer-title">{concept.label}</h2>
      <p className="drawer-desc">{concept.description}</p>
      <span className="ref-badge">
        <FileCode size={12} />
        {concept.code_ref}
      </span>

      <div className="k-row">
        <label htmlFor="k-slider">
          <Sliders size={12} />
          Nearest neighbors
        </label>
        <span className="k-val">k = {k}</span>
      </div>
      <input
        id="k-slider"
        className="slider"
        type="range"
        min={1}
        max={15}
        value={k}
        onChange={(e) => onK(Number(e.target.value))}
        style={{ ['--fill' as string]: `${((k - 1) / 14) * 100}%` }}
      />

      <section className="path">
        <h3>Index traversal</h3>
        <ol>
          {path.map((s) => (
            <li key={s.key} className="path-step">
              <CircleCheck size={14} className={`path-icon layer-${s.layer}`} />
              <span className="path-layer">{s.layer}</span>
              {s.text}
            </li>
          ))}
          {searching && (
            <li className="path-step is-pending">
              <LoaderCircle size={14} className="spin" />
              Descending the hierarchy
            </li>
          )}
        </ol>
      </section>

      {results && (
        <section className="neighbors">
          <h3>
            Top {results.results.length} semantic neighbors
            <ArrowRight size={12} />
          </h3>
          <ol>
            {results.results.map((hit, i) => {
              const c = conceptsById.get(hit.id);
              if (!c) return null;
              return (
                <li key={hit.id} className="nn-card" style={{ animationDelay: `${Math.min(i, 8) * 45}ms` }}>
                  <div className="nn-top">
                    <span className="nn-rank">{i + 1}</span>
                    <span className="nn-title">{c.label}</span>
                    <span className="nn-dist">d = {hit.distance.toFixed(5)}</span>
                  </div>
                  <div className="nn-branch">
                    <BranchBadge branch={c.branch} />
                  </div>
                  <p className="nn-desc">{c.description}</p>
                  <p className="nn-why">{whyCluster(concept, c)}</p>
                  <span className="ref-badge">
                    <FileCode size={12} />
                    {c.code_ref}
                  </span>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </aside>
  );
}
