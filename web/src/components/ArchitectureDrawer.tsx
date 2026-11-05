import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Cpu, FileCode, HardDrive, Layers, Network, X } from 'lucide-react';
import { BRANCHES } from '../data/branches';

interface Pillar {
  index: string;
  title: string;
  icon: ReactNode;
  branch: number;
  bullets: ReactNode[];
  codeRef: string;
}

const PILLARS: Pillar[] = [
  {
    index: '01',
    title: 'Hardware & Compute Acceleration',
    icon: <Cpu size={16} />,
    branch: 0,
    bullets: [
      <>
        C++20 compiled with <code>-mavx2 -mfma -O3</code>.
      </>,
      '256-bit SIMD registers computing 8 parallel float32 products per clock cycle.',
      '64-byte aligned buffers matching CPU cache lines.',
    ],
    codeRef: 'include/engine/distance/simd_metrics.hpp',
  },
  {
    index: '02',
    title: 'Vector Graph Index (HNSW + SQ8)',
    icon: <Layers size={16} />,
    branch: 1,
    bullets: [
      <>
        Multi-layer graph topology with hierarchical skip-list traversal, <code>O(log N)</code> complexity.
      </>,
      'SQ8 scalar quantization compressing 32-bit floats to 8-bit integers (75% memory footprint reduction).',
    ],
    codeRef: 'include/engine/index/hnsw_index.hpp',
  },
  {
    index: '03',
    title: 'Durability & Storage (WAL Engine)',
    icon: <HardDrive size={16} />,
    branch: 2,
    bullets: [
      'Append-only write-ahead log with hardware-accelerated CRC32-C checksumming.',
      'Zero-data-loss crash recovery via sequential binary replay.',
    ],
    codeRef: 'include/engine/storage/wal_manager.hpp',
  },
  {
    index: '04',
    title: 'Network & Concurrency',
    icon: <Network size={16} />,
    branch: 3,
    bullets: [
      'High-throughput Worker Thread Pool with non-blocking task queue.',
      'Zero-copy HTTP protocol parser delivering sub-100 microsecond response latency.',
    ],
    codeRef: 'src/server/Router.cpp',
  },
];

export function ArchitectureDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <>
      <div className={`arch-scrim ${open ? 'is-open' : ''}`} onClick={onClose} aria-hidden />
      <aside
        className={`arch-panel ${open ? 'is-open' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label="Architecture and tech stack"
        aria-hidden={!open}
        inert={!open}
      >
        <header className="arch-head">
          <div>
            <p className="arch-eyebrow">Nexus Vector Database Engine</p>
            <h2>Architecture &amp; Tech Stack</h2>
            <p className="arch-lede">Four systems-engineering pillars, from registers to sockets.</p>
          </div>
          <button ref={closeRef} className="icon-btn" onClick={onClose} aria-label="Close architecture specs">
            <X size={16} />
          </button>
        </header>

        <div className="arch-body">
          {PILLARS.map((p) => {
            const b = BRANCHES[p.branch];
            return (
              <section key={p.index} className="pillar" style={{ ['--accent' as string]: b.color }}>
                <header className="pillar-head">
                  <span className="pillar-icon">{p.icon}</span>
                  <div>
                    <span className="pillar-index">Pillar {p.index}</span>
                    <h3>{p.title}</h3>
                  </div>
                </header>
                <ul>
                  {p.bullets.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
                <span className="ref-badge">
                  <FileCode size={12} />
                  {p.codeRef}
                </span>
              </section>
            );
          })}
        </div>
      </aside>
    </>
  );
}
