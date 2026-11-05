import { useMemo, useRef, useState } from 'react';
import { Activity, Cpu, Database, Layers, Search, ShieldCheck, X } from 'lucide-react';
import { BRANCHES, FLAGSHIPS } from '../data/branches';
import type { Concept, EngineStatus } from '../services/api';

export function Brand({ status, fromEngine }: { status: EngineStatus | null; fromEngine: boolean }) {
  const state = status === null ? 'checking' : status.online ? 'online' : 'offline';
  return (
    <div className="brand">
      <div className="brand-name">Nexus</div>
      <div className="brand-sub">Vector Database Engine</div>
      <div className={`status status-${state}`} role="status">
        <span className="status-dot" />
        {state === 'checking' ? 'Connecting' : state === 'online' ? 'Engine online' : 'Standalone preview'}
        {state === 'online' && !fromEngine ? ' (local catalog)' : ''}
      </div>
    </div>
  );
}

export function Telemetry({ status, latencyUs }: { status: EngineStatus | null; latencyUs: number | null }) {
  const state = status === null ? 'checking' : status.online ? 'online' : 'offline';
  const rows = [
    { icon: <Activity size={14} />, label: 'Status', value: state === 'online' ? 'Online' : state === 'offline' ? 'Standalone preview' : 'Connecting', state },
    { icon: <Database size={14} />, label: 'Total vectors', value: (status?.totalVectors ?? 100_000).toLocaleString('en-US') },
    { icon: <Cpu size={14} />, label: 'Hardware accel', value: 'AVX2 256-bit SIMD' },
    { icon: <ShieldCheck size={14} />, label: 'WAL durability', value: 'Checkpointed (CRC32-C)' },
  ];
  return (
    <aside className="telemetry glass" aria-label="Engine telemetry">
      <h2 className="tele-title">Engine telemetry</h2>
      <dl>
        {rows.map((r) => (
          <div key={r.label} className="tele-row">
            <dt>
              {r.icon}
              {r.label}
            </dt>
            <dd>
              {r.state && <span className={`status-dot status-dot-${r.state}`} />}
              {r.value}
            </dd>
          </div>
        ))}
      </dl>
      <div className={`latency-badge ${latencyUs === null ? 'is-idle' : ''}`} aria-live="polite">
        <span className="latency-label">Query latency</span>
        <span className="latency-value">
          {latencyUs === null ? 'No query yet' : latencyUs.toLocaleString('en-US')}
          {latencyUs !== null && <small> {'µ'}s</small>}
        </span>
      </div>
    </aside>
  );
}

export function ArchitectureButton({ onClick, open }: { onClick: () => void; open: boolean }) {
  return (
    <button className="arch-btn" onClick={onClick} aria-haspopup="dialog" aria-expanded={open}>
      <Layers size={15} />
      Architecture Specs
    </button>
  );
}

interface SearchPanelProps {
  concepts: Concept[];
  selectedId: number | null;
  onSelect: (concept: Concept) => void;
  onClear: () => void;
  onBranchChange: (branch: number) => void;
}

export function SearchPanel({ concepts, selectedId, onSelect, onClear, onBranchChange }: SearchPanelProps) {
  const [query, setQuery] = useState('');
  const [branch, setBranch] = useState(-1);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return concepts
      .filter((c) => branch === -1 || c.branch === branch)
      .filter((c) => !q || c.label.toLowerCase().includes(q) || c.category.toLowerCase().includes(q) || c.description.toLowerCase().includes(q))
      .slice(0, 8);
  }, [concepts, query, branch]);

  const chips = useMemo(
    () =>
      FLAGSHIPS.map((f) => ({ ...f, concept: concepts.find((c) => c.id === f.id) })).filter(
        (f) => f.concept && (branch === -1 || f.concept.branch === branch),
      ),
    [concepts, branch],
  );

  const showList = open && (query.trim() !== '' || branch !== -1);

  const choose = (c: Concept) => {
    onSelect(c);
    setQuery('');
    setOpen(false);
    inputRef.current?.blur();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, matches.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === 'Enter' && matches[active]) {
      choose(matches[active]);
    } else if (e.key === 'Escape') {
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  return (
    <div className="search-panel">
      <div className="search-box glass">
        <Search size={16} className="search-icon" />
        <input
          ref={inputRef}
          value={query}
          placeholder="Search 100 architecture concepts"
          aria-label="Search architecture concepts"
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          onKeyDown={onKeyDown}
        />
        {(query || selectedId !== null) && (
          <button
            className="icon-btn"
            aria-label="Clear search"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              setQuery('');
              onClear();
            }}
          >
            <X size={14} />
          </button>
        )}
        {showList && (
          <ul className="search-list glass" role="listbox">
            {matches.length === 0 && <li className="search-empty">No matching concepts</li>}
            {matches.map((c, i) => (
              <li
                key={c.id}
                role="option"
                aria-selected={i === active}
                className={i === active ? 'is-active' : ''}
                onMouseDown={(e) => {
                  e.preventDefault();
                  choose(c);
                }}
                onMouseEnter={() => setActive(i)}
              >
                <span className="dot" style={{ background: BRANCHES[c.branch]?.color }} />
                <span className="li-label">{c.label}</span>
                <span className="li-meta">{c.category}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="segmented glass" role="tablist" aria-label="Filter by architecture branch">
        {[{ id: -1, tab: 'All' }, ...BRANCHES.map((b) => ({ id: b.id, tab: b.tab }))].map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={branch === t.id}
            className={branch === t.id ? 'is-active' : ''}
            onClick={() => {
              setBranch(t.id);
              onBranchChange(t.id);
              setActive(0);
              inputRef.current?.focus();
            }}
          >
            {t.tab}
          </button>
        ))}
      </div>

      <div className="chips" aria-label="Flagship concepts">
        {chips.map((f) => (
          <button
            key={f.id}
            className={`chip ${selectedId === f.id ? 'is-active' : ''}`}
            onClick={() => f.concept && choose(f.concept)}
          >
            <span className="dot" style={{ background: BRANCHES[f.concept!.branch]?.color }} />
            {f.label}
          </button>
        ))}
      </div>
    </div>
  );
}
