import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MousePointerClick } from 'lucide-react';
import { HoloTopology, type TraversalEvent } from './scene/HoloTopology';
import { ArchitectureButton, Brand, SearchPanel, Telemetry } from './components/TopBar';
import { ArchitectureDrawer } from './components/ArchitectureDrawer';
import { ResultsDrawer, type PathStep } from './components/ResultsDrawer';
import { AnchorDot, RoadmapDock, SpecCard } from './components/Roadmap';
import { BRANCHES } from './data/branches';
import { FALLBACK_CONCEPTS } from './data/concepts';
import { ROADMAP } from './data/roadmap';
import {
  fetchConcepts,
  fetchEngineStatus,
  searchConcept,
  type Concept,
  type EngineStatus,
  type SearchResponse,
} from './services/api';

const STATUS_POLL_MS = 5000;
const DEFAULT_K = 5;
/** Which branch each tour step spotlights (SQ8 lives inside the HNSW index). */
const TOUR_BRANCH = [0, 1, 1, 2, 3] as const;

function detectWebGLError(): string | null {
  try {
    return document.createElement('canvas').getContext('webgl2') ? null : 'WebGL2 is not available in this browser';
  } catch {
    return 'WebGL2 is not available in this browser';
  }
}

export default function App() {
  const stageRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<HoloTopology | null>(null);
  const pathKey = useRef(0);
  const searchToken = useRef(0);
  const kTimer = useRef<number | undefined>(undefined);

  const [concepts, setConcepts] = useState<Concept[]>(FALLBACK_CONCEPTS);
  const [fromEngine, setFromEngine] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [selected, setSelected] = useState<Concept | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [k, setK] = useState(DEFAULT_K);
  const [searching, setSearching] = useState(false);
  const [latencyUs, setLatencyUs] = useState<number | null>(null);
  const [simulated, setSimulated] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [path, setPath] = useState<PathStep[]>([]);
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [tabBranch, setTabBranch] = useState(-1);
  const [archOpen, setArchOpen] = useState(false);
  const [webglError] = useState<string | null>(detectWebGLError);

  const conceptsById = useMemo(() => new Map(concepts.map((c) => [c.id, c])), [concepts]);

  // Latest-value refs so the long-lived scene handlers never go stale.
  const conceptsRef = useRef(concepts);
  const selectedRef = useRef<Concept | null>(null);
  const kRef = useRef(DEFAULT_K);
  const runSearchRef = useRef<(c: Concept, k: number) => void>(() => {});
  useEffect(() => {
    conceptsRef.current = concepts;
    selectedRef.current = selected;
    kRef.current = k;
  }, [concepts, selected, k]);

  // Concept catalog -------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    fetchConcepts().then((r) => {
      if (!alive) return;
      setConcepts(r.concepts);
      setFromEngine(r.fromEngine);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Engine status polling -------------------------------------------------------
  useEffect(() => {
    const run = { alive: true };
    const poll = async () => {
      const s = await fetchEngineStatus();
      if (run.alive) setStatus(s);
    };
    poll();
    const id = window.setInterval(poll, STATUS_POLL_MS);
    return () => {
      run.alive = false;
      window.clearInterval(id);
    };
  }, []);

  // Scene lifecycle -------------------------------------------------------------
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || webglError) return;
    let scene: HoloTopology;
    try {
      scene = new HoloTopology(stage, {
        onHover: (id) => setHoverId(id),
        onPick: (id) => {
          const c = conceptsRef.current.find((x) => x.id === id);
          if (c) runSearchRef.current(c, kRef.current);
        },
      });
    } catch (err) {
      console.error('[nexus] failed to start WebGL scene', err);
      return;
    }
    sceneRef.current = scene;
    scene.setOverlayElements({ card: cardRef.current, dot: dotRef.current, selected: pinRef.current, hover: hoverRef.current });
    scene.intro();
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, [webglError]);

  // The chamber is its own box now, so the spec card only needs a small margin.
  useEffect(() => {
    sceneRef.current?.setSafeArea({ left: 16, right: 16, top: 16, bottom: 16 });
  }, [webglError]);

  // Focus isolation: concept > branch tab > tour step.
  const focusBranch = selected ? selected.branch : tabBranch !== -1 ? tabBranch : activeStep !== null ? TOUR_BRANCH[activeStep] : null;
  useEffect(() => {
    sceneRef.current?.setBranchFocus(focusBranch);
  }, [focusBranch, webglError]);

  // Search ----------------------------------------------------------------------
  const runSearch = useCallback(
    async (concept: Concept, kk: number) => {
      const token = ++searchToken.current;
      const scene = sceneRef.current;
      const stale = () => token !== searchToken.current;

      setSelected(concept);
      setSearching(true);
      setResults(null);
      setPath([]);
      setActiveStep(null);
      scene?.setSelected(concept.id);
      scene?.focusConcept(concept.id);

      const push = (layer: PathStep['layer'], text: string) =>
        setPath((p) => (stale() ? p : [...p, { key: pathKey.current++, layer, text }]));

      const res = await searchConcept(concept, conceptsRef.current, kk);
      if (stale()) return;
      setLatencyUs(res.took_us);
      setSimulated(res.simulated);

      if (!scene) {
        setResults(res);
        setSearching(false);
        return;
      }

      const onEvent = (e: TraversalEvent) => {
        if (stale()) return;
        switch (e.kind) {
          case 'entry':
            push('L2', `Entry point, apex node #${e.node}`);
            break;
          case 'hop':
            push(e.layer === 2 ? 'L2' : 'L1', `Greedy hop to ${e.layer === 2 ? 'apex node' : 'hub'} #${e.node}`);
            break;
          case 'descend':
            push(e.to === 1 ? 'L1' : 'L0', e.to === 1 ? `Descend to routing hub #${e.node}` : `Enter ${BRANCHES[e.node]?.name ?? 'branch'} cluster`);
            break;
          case 'locate':
            push('L0', `Located ${concept.label}`);
            break;
          case 'converge':
            push('L0', `Ranked ${res.results.length} neighbors by L2 distance`);
            setResults(res);
            break;
          case 'done':
            break;
        }
      };
      await scene.runTraversal(concept.id, res.results.map((h) => h.id), onEvent);
      if (!stale()) setSearching(false);
    },
    [],
  );
  useEffect(() => {
    runSearchRef.current = runSearch;
  }, [runSearch]);

  const clearSearch = useCallback(() => {
    searchToken.current++;
    window.clearTimeout(kTimer.current);
    setSelected(null);
    setResults(null);
    setPath([]);
    setSearching(false);
    sceneRef.current?.clearSearch();
    sceneRef.current?.focus('overview', 1.4);
  }, []);

  const onK = useCallback(
    (v: number) => {
      setK(v);
      window.clearTimeout(kTimer.current);
      if (selectedRef.current) {
        const c = selectedRef.current;
        kTimer.current = window.setTimeout(() => runSearch(c, v), 350);
      }
    },
    [runSearch],
  );

  // Roadmap tour ----------------------------------------------------------------
  const selectStep = useCallback((i: number | null) => {
    setActiveStep(i);
    sceneRef.current?.focus(i === null ? 'overview' : ROADMAP[i].id, i === null ? 1.8 : 1.6);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') selectStep(activeStep === null ? 0 : Math.min(ROADMAP.length - 1, activeStep + 1));
      else if (e.key === 'ArrowLeft') selectStep(activeStep === null ? ROADMAP.length - 1 : Math.max(0, activeStep - 1));
      else if (e.key === 'Escape') selectStep(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeStep, selectStep]);

  const hoverConcept = hoverId === null ? null : conceptsById.get(hoverId);
  const selectedColor = selected ? BRANCHES[selected.branch]?.color : undefined;

  return (
    <div className="app">
      <header className="hud-top">
        <Brand status={status} fromEngine={fromEngine} />
        <SearchPanel
          concepts={concepts}
          selectedId={selected?.id ?? null}
          onSelect={(c) => runSearch(c, k)}
          onClear={clearSearch}
          onBranchChange={setTabBranch}
        />
        <div className="hud-top-end">
          <ArchitectureButton open={archOpen} onClick={() => setArchOpen(true)} />
        </div>
      </header>

      <main className="workspace">
        <Telemetry status={status} latencyUs={latencyUs} />

        <section className="chamber" aria-label="3D vector index chamber">
          <div ref={stageRef} className="stage" />

          {webglError && <div className="webgl-error">3D view unavailable: {webglError}. Search and telemetry still work.</div>}

          <div ref={pinRef} className="pin pin-selected" style={{ ['--pin' as string]: selectedColor }}>
            <span className="pin-dot" />
            {selected?.label}
          </div>
          <div ref={hoverRef} className="pin pin-hover">
            {hoverConcept?.label}
          </div>

          <div className={`hint ${activeStep !== null || selected ? 'is-hidden' : ''}`}>
            <MousePointerClick size={14} />
            Click any node in Layer 0 to run a semantic search. Drag to orbit, scroll to zoom.
          </div>

          <AnchorDot ref={dotRef} />
          <SpecCard ref={cardRef} active={activeStep} onClose={() => selectStep(null)} />
        </section>

        {selected && (
          <ResultsDrawer
            concept={selected}
            conceptsById={conceptsById}
            k={k}
            onK={onK}
            onClose={clearSearch}
            searching={searching}
            latencyUs={latencyUs}
            simulated={simulated}
            path={path}
            results={results}
          />
        )}
      </main>

      <RoadmapDock active={activeStep} onSelect={selectStep} />
      <ArchitectureDrawer open={archOpen} onClose={() => setArchOpen(false)} />
    </div>
  );
}
