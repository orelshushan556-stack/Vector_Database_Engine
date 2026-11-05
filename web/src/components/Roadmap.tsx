import { forwardRef, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import gsap from 'gsap';
import { Cpu, HardDrive, Layers, Network, Shrink, X } from 'lucide-react';
import { ROADMAP } from '../data/roadmap';

const STEP_ICONS: ReactNode[] = [
  <Cpu key="simd" size={14} />,
  <Layers key="hnsw" size={14} />,
  <Shrink key="sq8" size={14} />,
  <HardDrive key="wal" size={14} />,
  <Network key="net" size={14} />,
];

export function RoadmapDock(props: { active: number | null; onSelect: (i: number | null) => void }) {
  const { active, onSelect } = props;
  return (
    <nav className="dock glass" aria-label="Architecture roadmap tour">
      <ol>
        {ROADMAP.map((s, i) => (
          <li key={s.id}>
            <button
              className={`dock-step ${active === i ? 'is-active' : ''}`}
              onClick={() => onSelect(active === i ? null : i)}
              aria-current={active === i ? 'step' : undefined}
            >
              <span className="dock-icon">{STEP_ICONS[i]}</span>
              <span className="dock-index">{s.index}</span>
              <span className="dock-title">{s.title}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

/**
 * Frosted spec card. The scene positions it every frame (via ref) so it tracks
 * the focused 3D anchor without React re-renders.
 */
export const SpecCard = forwardRef<HTMLDivElement, { active: number | null; onClose: () => void }>(
  function SpecCard({ active, onClose }, ref) {
    const innerRef = useRef<HTMLElement>(null);
    const step = active === null ? null : ROADMAP[active];

    useEffect(() => {
      const el = innerRef.current;
      if (!el || !step) return;
      const items = el.querySelectorAll('[data-reveal]');
      const tl = gsap.timeline();
      tl.fromTo(
        el,
        { autoAlpha: 0, y: 14, scale: 0.98 },
        { autoAlpha: 1, y: 0, scale: 1, duration: 0.6, ease: 'power3.out', delay: 0.3 },
      );
      tl.fromTo(items, { autoAlpha: 0, y: 6 }, { autoAlpha: 1, y: 0, duration: 0.4, stagger: 0.07, ease: 'power2.out' }, '-=0.35');
      return () => {
        tl.kill();
      };
    }, [step]);

    return (
      <div ref={ref} className={`spec-anchor ${step ? 'is-visible' : ''}`}>
        {step && (
          <article ref={innerRef} className="spec-card glass" key={step.id}>
            <header className="spec-head" data-reveal>
              <span className="spec-index">{step.index}</span>
              <div>
                <h2>{step.title}</h2>
                <p>{step.subtitle}</p>
              </div>
              <button className="icon-btn" onClick={onClose} aria-label="Close spec card">
                <X size={14} />
              </button>
            </header>
            <section data-reveal>
              <h3>Engineering bottleneck</h3>
              <p>{step.bottleneck}</p>
            </section>
            <section data-reveal>
              <h3>C++ solution</h3>
              <p>{step.solution}</p>
              <code className="ref-badge">{step.codeRef}</code>
            </section>
            <section data-reveal>
              <h3>Measured performance</h3>
              <div className="bench-grid">
                {step.benchmarks.map((b) => (
                  <div key={b.label} className="bench">
                    <div className="bench-value">{b.value}</div>
                    <div className="bench-label">{b.label}</div>
                    <div className="bench-detail">{b.detail}</div>
                  </div>
                ))}
              </div>
              <div className="bench-note">Reference figures for 100k x 128-d. See benchmarks/bench_sift.cpp.</div>
            </section>
          </article>
        )}
      </div>
    );
  },
);

export const AnchorDot = forwardRef<HTMLDivElement>(function AnchorDot(_props, ref) {
  return (
    <div ref={ref} className="anchor-dot" aria-hidden>
      <span />
    </div>
  );
});
