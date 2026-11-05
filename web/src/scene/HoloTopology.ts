import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import gsap from 'gsap';
import type { TourFocus } from '../data/roadmap';
import { BRANCHES } from '../data/branches';
import * as S from './shaders';

// Layout ---------------------------------------------------------------------
const LAYER_Y = [0, 12, 24] as const; // L0 base, L1 routing hubs, L2 apex
const LAYER_R = [44, 30, 18] as const;
const CHASSIS_Y = -6;
const CHASSIS_R = 47;
const CONCEPTS = 100;
const PER_BRANCH = 25;
const CRC_SEGMENTS = 54;
const RESULT_POOL = 15;

const C = {
  blue: new THREE.Color('#38bdf8'),
  indigo: new THREE.Color('#818cf8'),
  amber: new THREE.Color('#f59e0b'),
  emerald: new THREE.Color('#10b981'),
  white: new THREE.Color('#ffffff'),
  ink: new THREE.Color('#e2e8f0'),
  coolGray: new THREE.Color('#64748b'),
  softBlue: new THREE.Color('#7dd3fc'),
  grid: new THREE.Color('#1e293b'),
  surface: new THREE.Color('#0b1220'),
  crcIdle: new THREE.Color('#1e293b'),
  chamber: new THREE.Color('#070a13'),
};
const BRANCH_COLORS = BRANCHES.map((b) => new THREE.Color(b.color));

export type TraversalEvent =
  | { kind: 'entry'; node: number }
  | { kind: 'hop'; layer: 2 | 1; node: number }
  | { kind: 'descend'; from: 2 | 1; to: 1 | 0; node: number }
  | { kind: 'locate'; concept: number }
  | { kind: 'converge'; concept: number; neighbors: number[] }
  | { kind: 'done' };

export interface SceneHandlers {
  onHover: (conceptId: number | null) => void;
  onPick: (conceptId: number) => void;
}

export interface OverlayElements {
  card: HTMLElement | null;
  dot: HTMLElement | null;
  selected: HTMLElement | null;
  hover: HTMLElement | null;
}

interface Waypoint {
  id: number;
  layer: 0 | 1 | 2;
  pos: THREE.Vector3;
}

interface GraphNode {
  id: number;
  pos: THREE.Vector3;
}

interface Packet {
  curve: THREE.Curve<THREE.Vector3>;
  t: number;
  speed: number;
}

interface Pose {
  pos: THREE.Vector3;
  target: THREE.Vector3;
  anchor: THREE.Vector3 | null;
  boost: number;
  quant: number;
  chassis: number;
  stream: number;
  focusConduit: number;
  autoRotate: boolean;
}

/** Deterministic PRNG so the topology is identical on every load. */
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

function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function xzDist(a: THREE.Vector3, b: THREE.Vector3): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

/** Smooth conduit between two nodes: gentle arc in-plane, S-curve between layers. */
function makeCurve(a: THREE.Vector3, b: THREE.Vector3): THREE.Curve<THREE.Vector3> {
  const dy = a.y - b.y;
  if (Math.abs(dy) < 0.01) {
    const mid = a.clone().lerp(b, 0.5);
    mid.y += 0.5 + a.distanceTo(b) * 0.04;
    return new THREE.QuadraticBezierCurve3(a.clone(), mid, b.clone());
  }
  return new THREE.CubicBezierCurve3(
    a.clone(),
    new THREE.Vector3(a.x, a.y - dy * 0.55, a.z),
    new THREE.Vector3(b.x, b.y + dy * 0.55, b.z),
    b.clone(),
  );
}

function makeGlowTexture(): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.3, 'rgba(255,255,255,0.45)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeLabelSprite(text: string): THREE.Sprite {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 128;
  const g = c.getContext('2d')!;
  g.font = '600 46px -apple-system, "SF Pro Text", "Segoe UI", Inter, sans-serif';
  g.textBaseline = 'middle';
  g.fillStyle = '#7c8ba3';
  (g as CanvasRenderingContext2D & { letterSpacing: string }).letterSpacing = '6px';
  g.fillText(text, 16, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const s = new THREE.Sprite(mat);
  s.scale.set(20, 2.5, 1);
  s.center.set(0, 0.5);
  return s;
}

export class HoloTopology {
  private readonly container: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private readonly pixelRatio: number;
  private readonly reducedMotion: boolean;
  private readonly rand = mulberry32(0x5eed);
  private readonly handlers: SceneHandlers;

  // topology
  private branchCenters: THREE.Vector3[] = [];
  private conceptPos: THREE.Vector3[] = [];
  private hubs: GraphNode[] = [];
  private apex: GraphNode[] = [];
  private adj2 = new Map<number, number[]>();
  private adj1 = new Map<number, number[]>();
  private hubParent = new Map<number, number>();

  // animated materials / objects
  private readonly time = { value: 0 };
  private readonly shared = { uTime: this.time, uPixelRatio: { value: 1 } };
  private readonly focusUniforms = { uFocusBranch: { value: -1 }, uFocusAmt: { value: 0 } };
  private conduitMaterials: THREE.ShaderMaterial[] = [];
  private conduitBoost = { value: 0 };
  private cloudMaterial!: THREE.ShaderMaterial;
  private streamMaterial!: THREE.ShaderMaterial;
  private chassisRim!: THREE.ShaderMaterial;
  private focusConduitMat!: THREE.ShaderMaterial;
  private apexMeshes: THREE.Mesh[] = [];
  private crcSegments!: THREE.InstancedMesh;
  private packets: Packet[] = [];
  private packetGeom!: THREE.BufferGeometry;
  private conduitCurves: THREE.Curve<THREE.Vector3>[] = [];
  private focusCurve!: THREE.Curve<THREE.Vector3>;
  private chassisGlow = { value: 0.3 };
  private crcSweep = { value: 0.4 };
  private currentStream = 0.2;
  private packetMaterial!: THREE.ShaderMaterial;
  private haloMaterial!: THREE.ShaderMaterial;
  private hubMesh!: THREE.InstancedMesh;
  private focusState = { amt: 0, branch: -1 };
  private focusApplied = { amt: -1, branch: -2 };
  private focusTween: gsap.core.Tween | null = null;

  // concept nodes + interaction
  private conceptMesh!: THREE.InstancedMesh;
  private pickMesh!: THREE.InstancedMesh;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private pointerInside = false;
  private pointerDown: { x: number; y: number } | null = null;
  private hoverId: number | null = null;
  private selectedId: number | null = null;

  // traversal
  private beam!: THREE.Mesh<THREE.CylinderGeometry, THREE.ShaderMaterial>;
  private probe!: THREE.Mesh;
  private probeHalo!: THREE.Sprite;
  private shockRing!: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  private flashPool: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>[] = [];
  private flashCursor = 0;
  private selMarker!: THREE.Mesh;
  private selRing!: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  private resultMarkers: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>[] = [];
  private trail: THREE.Mesh<THREE.TubeGeometry, THREE.ShaderMaterial> | null = null;
  private resultTubes: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> | null = null;
  private traversalTl: gsap.core.Timeline | null = null;

  // tour
  private presets!: Record<TourFocus | 'overview', Pose>;
  private cameraTl: gsap.core.Timeline | null = null;
  private effectsTl: gsap.core.Timeline | null = null;
  private anchor: THREE.Vector3 | null = null;
  private overlay: OverlayElements = { card: null, dot: null, selected: null, hover: null };
  private cardPos = { x: -1, y: -1 };
  private safe = { left: 16, right: 16, top: 160, bottom: 110 };

  private disposed = false;
  private readonly tmpV = new THREE.Vector3();
  private readonly tmpC = new THREE.Color();

  constructor(container: HTMLElement, handlers: SceneHandlers) {
    this.container = container;
    this.handlers = handlers;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    this.shared.uPixelRatio.value = this.pixelRatio;

    const { clientWidth: w, clientHeight: h } = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0xffffff, 0);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(46, w / Math.max(h, 1), 0.1, 800);
    this.camera.position.set(120, 95, 140);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 200;
    this.controls.maxPolarAngle = Math.PI * 0.6;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.target.set(0, 8, 0);
    this.controls.addEventListener('start', () => {
      this.controls.autoRotate = false;
    });
    this.controls.update();

    this.buildTopology();
    this.buildPlanes();
    this.buildChassis();
    this.buildPointCloud();
    this.buildNodes();
    this.buildConduits();
    this.buildPackets();
    this.buildStreams();
    this.buildTraversalRig();
    this.buildPresets();
    this.bindPointer();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);

    // Rendering runs on the GSAP ticker so tweens apply *before* each frame
    // renders, eliminating one-frame lag between camera tween and render.
    gsap.ticker.add(this.tick);
  }

  // Public API -----------------------------------------------------------------

  setOverlayElements(els: OverlayElements): void {
    this.overlay = els;
    this.cardPos = { x: -1, y: -1 };
  }

  /** Screen-space insets (px) the floating spec card must stay clear of. */
  setSafeArea(insets: Partial<{ left: number; right: number; top: number; bottom: number }>): void {
    this.safe = { ...this.safe, ...insets };
  }

  intro(): void {
    this.focus('overview', this.reducedMotion ? 0.6 : 2.6);
  }

  /** Glide camera + target to a roadmap focus with zero jitter. */
  focus(step: TourFocus | 'overview', duration = 1.6): void {
    this.glideTo(this.presets[step], duration);
  }

  /** Re-frame the whole 3-layer stack from the side of a concept's branch. */
  focusConcept(conceptId: number, duration = 1.5): void {
    const c = this.branchCenters[this.branchOf(conceptId)];
    const dir = new THREE.Vector3(c.x, 0, c.z).normalize();
    const target = new THREE.Vector3(c.x * 0.3, 15, c.z * 0.3);
    const pos = target.clone().addScaledVector(dir, 80).add(new THREE.Vector3(0, 38, 0));
    this.glideTo({ ...this.presets.overview, pos, target }, duration);
  }

  /**
   * Focus isolation: every node, conduit and cloud point outside `branch` fades to
   * 15%, while the branch itself brightens. Pass null to restore the full topology.
   */
  setBranchFocus(branch: number | null): void {
    const next = branch === null ? this.focusState.branch : branch;
    this.focusState.branch = next;
    this.focusTween?.kill();
    this.focusTween = gsap.to(this.focusState, {
      amt: branch === null ? 0 : 1,
      duration: this.reducedMotion ? 0.05 : 0.7,
      ease: 'power2.inOut',
    });
  }

  /** Select a concept: pins its marker in Layer 0 before/without a search. */
  setSelected(conceptId: number | null): void {
    this.selectedId = conceptId;
    if (conceptId === null) {
      this.selMarker.visible = false;
      this.selRing.visible = false;
      return;
    }
    const p = this.conceptPos[conceptId - 1];
    const color = BRANCH_COLORS[this.branchOf(conceptId)];
    (this.selMarker.material as THREE.MeshBasicMaterial).color.copy(color);
    this.selRing.material.color.copy(color);
    this.selMarker.position.set(p.x, p.y + 0.4, p.z);
    this.selRing.position.set(p.x, 0.12, p.z);
    this.selMarker.visible = true;
    this.selRing.visible = true;
  }

  /** Animate a search: apex beam, hub hops, ripple on the concept, neighbour arcs. */
  runTraversal(conceptId: number, neighborIds: number[], onEvent: (e: TraversalEvent) => void): Promise<void> {
    this.cleanupTraversal();
    this.setSelected(conceptId);
    this.selMarker.scale.setScalar(0.01);
    this.selRing.visible = false;

    const target = this.conceptPos[conceptId - 1];
    const accent = BRANCH_COLORS[this.branchOf(conceptId)];
    const accentHot = accent.clone().lerp(C.white, 0.6);
    const plan = this.planPath(conceptId);
    const path = new THREE.CurvePath<THREE.Vector3>();
    for (let i = 0; i < plan.length - 1; i++) path.add(makeCurve(plan[i].pos, plan[i + 1].pos));
    const total = path.getLength();
    const boundaries = path.getCurveLengths().map((l) => l / total);

    const trailMat = this.makeConduitMaterial(accent, accentHot, 0.95, 1.6);
    trailMat.uniforms.uReveal.value = 0;
    trailMat.uniforms.uBoost.value = 1;
    const trailGeom = new THREE.TubeGeometry(path, Math.max(64, plan.length * 40), 0.17, 10, false);
    this.addSeedAttribute(trailGeom, 0);
    this.trail = new THREE.Mesh(trailGeom, trailMat);
    this.trail.renderOrder = 6;
    this.scene.add(this.trail);

    const entry = plan[0].pos;
    const beamTop = 70;
    const beamState = { bottom: beamTop };
    const progress = { u: 0 };
    let emitted = 0;
    const segTime = this.reducedMotion ? 0.18 : 0.5;

    const beamMat = this.beam.material;
    beamMat.uniforms.uColor.value.copy(accent);
    (this.probe.material as THREE.MeshBasicMaterial).color.copy(accentHot);
    (this.probeHalo.material as THREE.SpriteMaterial).color.copy(accent);
    this.shockRing.material.color.copy(accent);
    this.cloudMaterial.uniforms.uHitColor.value.copy(accent);
    this.beam.visible = true;
    beamMat.uniforms.uOpacity.value = 1;
    this.probe.visible = false;
    this.probeHalo.visible = false;

    const updateBeam = (x: number, z: number, bottom: number) => {
      this.beam.position.set(x, beamTop, z);
      this.beam.scale.set(1, Math.max(beamTop - bottom, 0.01), 1);
    };
    updateBeam(entry.x, entry.z, beamTop);

    return new Promise<void>((resolve) => {
      const tl = gsap.timeline();
      this.traversalTl = tl;

      // 1) Laser pulse drops onto the apex entry point.
      tl.to(beamState, {
        bottom: entry.y,
        duration: this.reducedMotion ? 0.2 : 0.5,
        ease: 'power4.in',
        onUpdate: () => updateBeam(entry.x, entry.z, beamState.bottom),
        onComplete: () => {
          this.flash(entry, accent, 4);
          onEvent({ kind: 'entry', node: plan[0].id });
          this.probe.visible = true;
          this.probeHalo.visible = true;
          this.probe.position.copy(entry);
          this.probeHalo.position.copy(entry);
        },
      });

      // 2) Probe descends the hierarchy; the beam tracks it.
      tl.to(progress, {
        u: 1,
        duration: segTime * (plan.length - 1),
        ease: 'none',
        onUpdate: () => {
          const p = path.getPointAt(Math.min(progress.u, 1), this.tmpV);
          this.probe.position.copy(p);
          this.probeHalo.position.copy(p);
          trailMat.uniforms.uReveal.value = progress.u;
          updateBeam(p.x, p.z, p.y);
          while (emitted < boundaries.length && progress.u >= boundaries[emitted] - 1e-6) {
            const from = plan[emitted];
            const to = plan[emitted + 1];
            this.flash(to.pos, to.layer === 0 ? accentHot : accent, to.layer === 0 ? 3 : 2.6);
            if (to.layer === 0 && emitted + 1 === plan.length - 1) {
              onEvent({ kind: 'locate', concept: conceptId });
            } else if (from.layer === to.layer) {
              onEvent({ kind: 'hop', layer: to.layer as 2 | 1, node: to.id });
            } else {
              onEvent({ kind: 'descend', from: from.layer as 2 | 1, to: to.layer as 1 | 0, node: to.id });
            }
            emitted++;
          }
        },
      });

      // 3) Ripple on the concept + neighbour arcs.
      const shock = this.cloudMaterial.uniforms;
      tl.add(() => {
        shock.uShockCenter.value.copy(target);
        shock.uShockTime.value = 0;
        shock.uHitRadius.value = 5;
        this.shockRing.position.set(target.x, 0.08, target.z);
        this.shockRing.visible = true;
        this.shockRing.scale.setScalar(0.1);
        this.shockRing.material.opacity = 1;
        this.selRing.visible = true;
        this.flash(this.tmpV.copy(target).setY(0.1), accentHot, 16);
        this.buildResultTubes(target, neighborIds, accent);
        const stream = this.streamMaterial.uniforms.uStream;
        gsap.killTweensOf(stream);
        gsap.fromTo(stream, { value: Math.max(stream.value, 0.7) }, { value: this.currentStream, duration: 3, delay: 1, ease: 'power2.inOut' });
      });
      tl.to(shock.uShockTime, { value: 4, duration: 4, ease: 'none' }, '<');
      tl.to(shock.uHit, { value: 1, duration: 0.5, ease: 'power2.out' }, '<');
      tl.to(this.shockRing.scale, { x: 24, y: 24, z: 24, duration: 1.8, ease: 'expo.out' }, '<');
      tl.to(this.shockRing.material, { opacity: 0, duration: 1.8, ease: 'power2.out' }, '<');
      tl.to(beamMat.uniforms.uOpacity, { value: 0, duration: 0.8, ease: 'power2.out' }, '<0.15');
      tl.to(this.probe.scale, { x: 0.01, y: 0.01, z: 0.01, duration: 0.5 }, '<');
      tl.to(this.selMarker.scale, { x: 1, y: 1, z: 1, duration: 0.5, ease: 'back.out(3)' }, '<0.1');
      if (this.resultTubes) {
        tl.to(this.resultTubes.material.uniforms.uReveal, { value: 1, duration: 0.7, ease: 'power2.out' }, '<');
      }
      neighborIds.slice(0, RESULT_POOL).forEach((id, i) => {
        const m = this.resultMarkers[i];
        const p = this.conceptPos[id - 1];
        if (!p) return;
        m.material.color.copy(BRANCH_COLORS[this.branchOf(id)]);
        m.position.set(p.x, p.y + 0.4, p.z);
        m.scale.setScalar(0.01);
        m.visible = true;
        tl.to(m.scale, { x: 1, y: 1, z: 1, duration: 0.45, ease: 'back.out(3)' }, `<${i === 0 ? 0.2 : 0.05}`);
      });
      tl.add(() => onEvent({ kind: 'converge', concept: conceptId, neighbors: neighborIds }));
      tl.add(() => {
        this.probe.visible = false;
        this.probeHalo.visible = false;
        this.probe.scale.setScalar(1);
        this.beam.visible = false;
        onEvent({ kind: 'done' });
        resolve();
      }, '+=0.15');
      // Afterglow fade, not awaited.
      const fade = gsap.timeline();
      tl.add(fade);
      fade.to(trailMat.uniforms.uOpacity, { value: 0.22, duration: 2.5, ease: 'power1.out' });
      fade.to(shock.uHit, { value: 0.35, duration: 3, ease: 'power1.out' }, '<');
    });
  }

  /** Clears every search overlay (markers, arcs, ripple) without touching the camera. */
  clearSearch(): void {
    this.cleanupTraversal();
    this.setSelected(null);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    gsap.ticker.remove(this.tick);
    this.cameraTl?.kill();
    this.effectsTl?.kill();
    this.traversalTl?.kill();
    this.focusTween?.kill();
    this.resizeObserver.disconnect();
    this.controls.dispose();
    const el = this.renderer.domElement;
    el.removeEventListener('pointermove', this.onPointerMove);
    el.removeEventListener('pointerleave', this.onPointerLeave);
    el.removeEventListener('pointerdown', this.onPointerDown);
    el.removeEventListener('pointerup', this.onPointerUp);
    this.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      mesh.geometry?.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    });
    this.renderer.dispose();
    el.remove();
  }

  // Camera ---------------------------------------------------------------------

  private glideTo(preset: Pose, duration: number): void {
    if (this.reducedMotion) duration = Math.min(duration, 0.6);
    this.cameraTl?.kill();
    this.flushControlMomentum();
    this.controls.enabled = false;
    this.controls.autoRotate = false;

    // Animate BOTH camera.position and controls.target together, updating the
    // controls each frame; hand control back only when the glide completes.
    const tl = gsap.timeline({
      defaults: { duration, ease: 'power3.inOut' },
      onUpdate: () => this.controls.update(),
      onComplete: () => {
        this.flushControlMomentum();
        this.controls.enabled = true;
        this.controls.autoRotate = preset.autoRotate;
        this.cameraTl = null;
      },
    });
    tl.to(this.camera.position, { x: preset.pos.x, y: preset.pos.y, z: preset.pos.z }, 0);
    tl.to(this.controls.target, { x: preset.target.x, y: preset.target.y, z: preset.target.z }, 0);
    this.cameraTl = tl;

    this.anchor = preset.anchor;
    this.currentStream = preset.stream;

    this.effectsTl?.kill();
    const fx = gsap.timeline({ defaults: { duration: duration * 0.9, ease: 'power2.inOut' } });
    fx.to(this.conduitBoost, { value: preset.boost }, 0);
    fx.to(this.cloudMaterial.uniforms.uQuant, { value: preset.quant }, 0);
    fx.to(this.chassisGlow, { value: preset.chassis }, 0);
    fx.to(this.crcSweep, { value: preset.chassis > 0.6 ? 1.6 : 0.4 }, 0);
    fx.to(this.streamMaterial.uniforms.uStream, { value: preset.stream }, 0);
    fx.to(this.focusConduitMat.uniforms.uOpacity, { value: preset.focusConduit }, 0);
    this.effectsTl = fx;
  }

  /** Discard residual OrbitControls inertia so it can't fight a tween. */
  private flushControlMomentum(): void {
    const internal = this.controls as unknown as {
      _sphericalDelta?: THREE.Spherical;
      _panOffset?: THREE.Vector3;
      _scale?: number;
    };
    internal._sphericalDelta?.set(0, 0, 0);
    internal._panOffset?.set(0, 0, 0);
    if (typeof internal._scale === 'number') internal._scale = 1;
  }

  // Frame loop -----------------------------------------------------------------

  private readonly tick = (_time: number, deltaMs: number): void => {
    if (this.disposed) return;
    const dt = Math.min(deltaMs / 1000, 0.05);
    this.time.value += dt;
    const t = this.time.value;

    for (const m of this.conduitMaterials) m.uniforms.uBoost.value = this.conduitBoost.value;
    this.applyFocus();
    this.chassisRim.uniforms.uOpacity.value = 0.35 + this.chassisGlow.value * 0.6;

    this.apexMeshes.forEach((m, i) => {
      const base = m.userData.baseY as number;
      m.position.y = base + Math.sin(t * 1.3 + i * 1.7) * 0.18;
      m.scale.setScalar(1 + Math.sin(t * 2.2 + i) * 0.05);
    });

    const pos = this.packetGeom.attributes.position as THREE.BufferAttribute;
    const speedMul = 1 + this.conduitBoost.value * 3.5;
    this.packets.forEach((p, i) => {
      p.t = (p.t + p.speed * dt * speedMul) % 1;
      p.curve.getPoint(p.t, this.tmpV);
      pos.setXYZ(i, this.tmpV.x, this.tmpV.y, this.tmpV.z);
    });
    pos.needsUpdate = true;

    // CRC32-C verification sweep around the chassis.
    const sweep = (t * this.crcSweep.value * 14) % CRC_SEGMENTS;
    for (let i = 0; i < CRC_SEGMENTS; i++) {
      let d = sweep - i;
      if (d < 0) d += CRC_SEGMENTS;
      const verified = d < 12 ? 1 - d / 12 : 0;
      const head = Math.exp(-d * d * 0.6);
      this.tmpC.copy(C.crcIdle).lerp(C.amber, verified * (0.35 + this.chassisGlow.value * 0.65));
      this.tmpC.lerp(C.white, head * (0.3 + this.chassisGlow.value * 0.5));
      this.crcSegments.setColorAt(i, this.tmpC);
    }
    if (this.crcSegments.instanceColor) this.crcSegments.instanceColor.needsUpdate = true;

    this.updateHover();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.updateOverlay();
  };

  /** Push the current focus tween into shader uniforms and instance colours. */
  private applyFocus(): void {
    const { amt, branch } = this.focusState;
    if (amt === this.focusApplied.amt && branch === this.focusApplied.branch) return;
    this.focusApplied = { amt, branch };

    this.focusUniforms.uFocusAmt.value = amt;
    this.focusUniforms.uFocusBranch.value = branch;
    for (const m of this.conduitMaterials) {
      const b = m.userData.branch as number | undefined;
      if (b === undefined) continue;
      m.uniforms.uDim.value = b === branch ? 1 + amt * 0.9 : 1 - amt * 0.85;
    }
    this.packetMaterial.uniforms.uIntensity.value = 0.95 * (1 - amt * 0.75);

    const tint = (i: number, base: THREE.Color, match: boolean, mesh: THREE.InstancedMesh) => {
      this.tmpC.copy(base);
      if (match) this.tmpC.lerp(C.white, amt * 0.22);
      else this.tmpC.lerp(C.chamber, amt * 0.85);
      mesh.setColorAt(i, this.tmpC);
    };
    for (let i = 0; i < CONCEPTS; i++) {
      const b = this.branchOf(i + 1);
      tint(i, BRANCH_COLORS[b], b === branch, this.conceptMesh);
    }
    this.hubs.forEach((_h, i) => tint(i, i < 4 ? BRANCH_COLORS[i] : C.indigo, i === branch, this.hubMesh));
    if (this.conceptMesh.instanceColor) this.conceptMesh.instanceColor.needsUpdate = true;
    if (this.hubMesh.instanceColor) this.hubMesh.instanceColor.needsUpdate = true;
  }

  private updateOverlay(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.positionCard(w, h);
    this.pin(this.overlay.selected, this.selectedId, w, h);
    this.pin(this.overlay.hover, this.hoverId !== this.selectedId ? this.hoverId : null, w, h);
  }

  private pin(el: HTMLElement | null, conceptId: number | null, w: number, h: number): void {
    if (!el) return;
    if (conceptId === null) {
      el.style.opacity = '0';
      return;
    }
    const p = this.tmpV.copy(this.conceptPos[conceptId - 1]).add(new THREE.Vector3(0, 0.9, 0)).project(this.camera);
    const x = (p.x * 0.5 + 0.5) * w;
    const y = (-p.y * 0.5 + 0.5) * h;
    const visible = p.z < 1 && x > 0 && x < w && y > 0 && y < h;
    el.style.opacity = visible ? '1' : '0';
    el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -100%)`;
  }

  private positionCard(w: number, h: number): void {
    const card = this.overlay.card;
    if (!card) return;
    const cw = card.offsetWidth;
    const ch = card.offsetHeight;
    let tx: number;
    let ty: number;
    let dotVisible = false;
    let ax = 0;
    let ay = 0;

    if (this.anchor) {
      const p = this.tmpV.copy(this.anchor).project(this.camera);
      ax = (p.x * 0.5 + 0.5) * w;
      ay = (-p.y * 0.5 + 0.5) * h;
      dotVisible = p.z < 1 && ax > 0 && ax < w && ay > 0 && ay < h;
      const gap = 56;
      tx = ax < w * 0.55 ? ax + gap : ax - cw - gap;
      ty = ay - ch * 0.5;
    } else {
      tx = w - cw - 28;
      ty = h * 0.5 - ch * 0.5;
    }
    const narrow = w < 640;
    const s = this.safe;
    if (narrow) {
      tx = (w - cw) / 2;
      ty = s.top;
    }
    tx = Math.min(Math.max(tx, narrow ? 12 : s.left), Math.max(s.left, w - cw - s.right));
    ty = Math.min(Math.max(ty, s.top), Math.max(s.top, h - ch - s.bottom));

    if (this.cardPos.x < 0) this.cardPos = { x: tx, y: ty };
    else {
      this.cardPos.x += (tx - this.cardPos.x) * 0.12;
      this.cardPos.y += (ty - this.cardPos.y) * 0.12;
    }
    card.style.transform = `translate3d(${this.cardPos.x.toFixed(1)}px, ${this.cardPos.y.toFixed(1)}px, 0)`;

    const dot = this.overlay.dot;
    if (dot) {
      dot.style.opacity = dotVisible && !narrow ? '1' : '0';
      dot.style.transform = `translate3d(${ax.toFixed(1)}px, ${ay.toFixed(1)}px, 0)`;
    }
  }

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  // Picking --------------------------------------------------------------------

  private bindPointer(): void {
    const el = this.renderer.domElement;
    el.addEventListener('pointermove', this.onPointerMove);
    el.addEventListener('pointerleave', this.onPointerLeave);
    el.addEventListener('pointerdown', this.onPointerDown);
    el.addEventListener('pointerup', this.onPointerUp);
  }

  private readonly onPointerMove = (e: PointerEvent): void => {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.pointerInside = true;
  };

  private readonly onPointerLeave = (): void => {
    this.pointerInside = false;
  };

  private readonly onPointerDown = (e: PointerEvent): void => {
    this.pointerDown = { x: e.clientX, y: e.clientY };
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    const d = this.pointerDown;
    this.pointerDown = null;
    if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return; // a drag, not a click
    if (this.hoverId !== null) this.handlers.onPick(this.hoverId);
  };

  private updateHover(): void {
    let next: number | null = null;
    if (this.pointerInside && !this.pointerDown) {
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObject(this.pickMesh, false)[0];
      if (hit && hit.instanceId !== undefined) next = hit.instanceId + 1;
    }
    if (next === this.hoverId) return;
    this.setNodeScale(this.hoverId, 1);
    this.hoverId = next;
    this.setNodeScale(next, 2);
    this.renderer.domElement.style.cursor = next === null ? '' : 'pointer';
    this.handlers.onHover(next);
  }

  private setNodeScale(conceptId: number | null, scale: number): void {
    if (conceptId === null) return;
    const m = new THREE.Matrix4();
    const p = this.conceptPos[conceptId - 1];
    m.compose(new THREE.Vector3(p.x, p.y + 0.35, p.z), new THREE.Quaternion(), new THREE.Vector3(scale, scale, scale));
    this.conceptMesh.setMatrixAt(conceptId - 1, m);
    this.conceptMesh.instanceMatrix.needsUpdate = true;
  }

  private branchOf(conceptId: number): number {
    return Math.min(3, Math.floor((conceptId - 1) / PER_BRANCH));
  }

  // Topology generation --------------------------------------------------------

  private buildTopology(): void {
    const r = this.rand;

    // L0: four branch clusters, 25 concept nodes each.
    for (let b = 0; b < 4; b++) {
      const a = Math.PI / 4 + (b * Math.PI) / 2;
      const center = new THREE.Vector3(Math.cos(a) * 17, LAYER_Y[0], Math.sin(a) * 17);
      this.branchCenters.push(center);
      const local: THREE.Vector3[] = [];
      for (let i = 0; i < PER_BRANCH; i++) {
        let best: THREE.Vector3 | null = null;
        for (let tries = 0; tries < 60; tries++) {
          const p = new THREE.Vector3(center.x + gaussian(r) * 3.6, LAYER_Y[0], center.z + gaussian(r) * 3.6);
          if (xzDist(p, center) > 9) continue;
          best = p;
          if (local.every((o) => xzDist(o, p) > 1.9)) break;
        }
        const p = best ?? center.clone();
        local.push(p);
        this.conceptPos.push(p);
      }
    }

    // L1: one routing hub above each branch plus a few free routers.
    for (let b = 0; b < 4; b++) {
      const c = this.branchCenters[b];
      this.hubs.push({ id: b, pos: new THREE.Vector3(c.x * 0.8, LAYER_Y[1], c.z * 0.8) });
    }
    for (let i = 0; i < 5; i++) {
      const a = r() * Math.PI * 2;
      const rad = 4 + Math.sqrt(r()) * 20;
      this.hubs.push({ id: this.hubs.length, pos: new THREE.Vector3(Math.cos(a) * rad, LAYER_Y[1], Math.sin(a) * rad) });
    }

    // L2: single entry point near the centre plus a sparse ring.
    this.apex.push({ id: 0, pos: new THREE.Vector3(0.8, LAYER_Y[2], -0.6) });
    for (let i = 1; i < 4; i++) {
      const a = (i / 3) * Math.PI * 2 + 0.4;
      this.apex.push({ id: i, pos: new THREE.Vector3(Math.cos(a) * 10, LAYER_Y[2], Math.sin(a) * 10) });
    }

    const link = (adj: Map<number, number[]>, a: number, b: number) => {
      if (!adj.get(a)!.includes(b)) adj.get(a)!.push(b);
      if (!adj.get(b)!.includes(a)) adj.get(b)!.push(a);
    };
    const knn = (nodes: GraphNode[], k: number, adj: Map<number, number[]>) => {
      nodes.forEach((n) => adj.set(n.id, []));
      for (const n of nodes) {
        nodes
          .filter((o) => o.id !== n.id)
          .sort((a, b) => xzDist(a.pos, n.pos) - xzDist(b.pos, n.pos))
          .slice(0, k)
          .forEach((o) => link(adj, n.id, o.id));
      }
    };
    knn(this.apex, 2, this.adj2);
    for (const a of this.apex) if (a.id !== 0) link(this.adj2, 0, a.id); // entry sees every apex node
    knn(this.hubs, 3, this.adj1);

    for (const h of this.hubs) {
      const nearest = this.apex.reduce((a, b) => (xzDist(a.pos, h.pos) < xzDist(b.pos, h.pos) ? a : b));
      this.hubParent.set(h.id, nearest.id);
    }
  }

  /** Greedy descent: apex entry -> apex hop -> hub hops -> branch cluster -> concept. */
  private planPath(conceptId: number): Waypoint[] {
    const b = this.branchOf(conceptId);
    const query = this.conceptPos[conceptId - 1];
    const plan: Waypoint[] = [{ id: 0, layer: 2, pos: this.apex[0].pos }];

    const greedy = (start: GraphNode, nodes: GraphNode[], adj: Map<number, number[]>, layer: 1 | 2) => {
      let cur = start;
      for (let step = 0; step < 6; step++) {
        let best = cur;
        for (const nId of adj.get(cur.id) ?? []) {
          const n = nodes[nId];
          if (xzDist(n.pos, query) < xzDist(best.pos, query)) best = n;
        }
        if (best === cur) break;
        plan.push({ id: best.id, layer, pos: best.pos });
        cur = best;
      }
      return cur;
    };

    const lastApex = greedy(this.apex[0], this.apex, this.adj2, 2);
    const children = this.hubs.filter((h) => this.hubParent.get(h.id) === lastApex.id);
    const pool = children.length ? children : this.hubs;
    const firstHub = pool.reduce((a, c) => (xzDist(a.pos, query) < xzDist(c.pos, query) ? a : c));
    plan.push({ id: firstHub.id, layer: 1, pos: firstHub.pos });
    const lastHub = greedy(firstHub, this.hubs, this.adj1, 1);
    if (lastHub.id !== b) plan.push({ id: b, layer: 1, pos: this.hubs[b].pos });

    const center = this.branchCenters[b];
    plan.push({ id: b, layer: 0, pos: new THREE.Vector3(center.x, 0.3, center.z) });
    plan.push({ id: conceptId, layer: 0, pos: new THREE.Vector3(query.x, 0.35, query.z) });
    return plan;
  }

  // Scene construction ---------------------------------------------------------

  private makeConduitMaterial(a: THREE.Color, b: THREE.Color, opacity: number, freq: number): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      vertexShader: S.conduitVertex,
      fragmentShader: S.conduitFragment,
      uniforms: {
        uTime: this.time,
        uBoost: { value: 0 },
        uOpacity: { value: opacity },
        uReveal: { value: 1 },
        uFreq: { value: freq },
        uDim: { value: 1 },
        uColorA: { value: a.clone() },
        uColorB: { value: b.clone() },
      },
      transparent: true,
      depthWrite: false,
    });
  }

  private makeHaloMaterial(intensity: number, pulse: number, focusable = false): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      vertexShader: S.haloVertex,
      fragmentShader: S.haloFragment,
      uniforms: {
        ...this.shared,
        ...(focusable ? this.focusUniforms : { uFocusBranch: { value: -1 }, uFocusAmt: { value: 0 } }),
        uPulse: { value: pulse },
        uIntensity: { value: intensity },
      },
      transparent: true,
      depthWrite: false,
    });
  }

  private addSeedAttribute(geom: THREE.BufferGeometry, seed: number): void {
    const count = geom.attributes.position.count;
    geom.setAttribute('aSeed', new THREE.Float32BufferAttribute(new Float32Array(count).fill(seed), 1));
  }

  private makePlaneMaterial(o: {
    radius: number;
    cell: number;
    line: THREE.Color;
    fill: THREE.Color;
    rim: THREE.Color;
    lineAlpha: number;
    fillAlpha: number;
    rimAlpha: number;
  }): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      vertexShader: S.planeVertex,
      fragmentShader: S.planeFragment,
      uniforms: {
        uRadius: { value: o.radius },
        uCell: { value: o.cell },
        uLine: { value: o.line.clone() },
        uFill: { value: o.fill.clone() },
        uRim: { value: o.rim.clone() },
        uLineAlpha: { value: o.lineAlpha },
        uFillAlpha: { value: o.fillAlpha },
        uRimAlpha: { value: o.rimAlpha },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  private buildPlanes(): void {
    const cfg = [
      { line: C.coolGray, fill: C.surface, rim: C.coolGray, lineAlpha: 0.28, fillAlpha: 0.7, rimAlpha: 0.5, cell: 3 },
      { line: C.indigo, fill: C.indigo, rim: C.indigo, lineAlpha: 0.16, fillAlpha: 0.05, rimAlpha: 0.55, cell: 3.5 },
      { line: C.blue, fill: C.blue, rim: C.blue, lineAlpha: 0.16, fillAlpha: 0.055, rimAlpha: 0.55, cell: 3 },
    ];
    const labels = ['LAYER 0  BASE VECTORS', 'LAYER 1  ROUTING HUBS', 'LAYER 2  APEX ENTRY'];
    for (let l = 0; l < 3; l++) {
      const R = LAYER_R[l];
      const geom = new THREE.PlaneGeometry(R * 2.1, R * 2.1);
      geom.rotateX(-Math.PI / 2);
      const mesh = new THREE.Mesh(geom, this.makePlaneMaterial({ radius: R, ...cfg[l] }));
      mesh.position.y = LAYER_Y[l] - (l === 0 ? 0.6 : 0.02);
      mesh.renderOrder = l - 3;
      this.scene.add(mesh);

      const label = makeLabelSprite(labels[l]);
      const a = -Math.PI * 0.25;
      label.position.set(Math.cos(a) * R * 0.98, LAYER_Y[l] + 1.4, Math.sin(a) * R * 0.98);
      this.scene.add(label);
    }
  }

  private buildChassis(): void {
    const corners: THREE.Vector3[] = [];
    for (let k = 0; k < 6; k++) {
      const a = (k * Math.PI) / 3;
      corners.push(new THREE.Vector3(Math.sin(a) * CHASSIS_R, 0, Math.cos(a) * CHASSIS_R));
    }
    const topY = CHASSIS_Y + 1.2;

    const wall = new THREE.Mesh(
      new THREE.CylinderGeometry(CHASSIS_R, CHASSIS_R + 2.5, 2.4, 6, 1, true),
      new THREE.MeshBasicMaterial({ color: '#0d1424', side: THREE.DoubleSide }),
    );
    wall.position.y = CHASSIS_Y;
    this.scene.add(wall);

    const shape = new THREE.Shape(corners.map((c) => new THREE.Vector2(c.x, -c.z)));
    const deckGeom = new THREE.ShapeGeometry(shape);
    deckGeom.rotateX(-Math.PI / 2);
    const deck = new THREE.Mesh(
      deckGeom,
      this.makePlaneMaterial({
        radius: 200,
        cell: 4,
        line: C.grid,
        fill: new THREE.Color('#0a0f1c'),
        rim: C.grid,
        lineAlpha: 0.9,
        fillAlpha: 1,
        rimAlpha: 0,
      }),
    );
    deck.position.y = topY;
    deck.renderOrder = -5;
    this.scene.add(deck);

    // Luminous teal rims (tubes, not hairlines).
    this.chassisRim = this.makeConduitMaterial(C.amber, C.white, 0.6, 3);
    this.chassisRim.userData.branch = 2;
    this.conduitMaterials.push(this.chassisRim);
    const rimGeoms: THREE.BufferGeometry[] = [];
    for (const [y, r] of [
      [topY, 1],
      [CHASSIS_Y - 1.2, (CHASSIS_R + 2.5) / CHASSIS_R],
    ] as const) {
      const path = new THREE.CurvePath<THREE.Vector3>();
      for (let k = 0; k < 6; k++) {
        path.add(
          new THREE.LineCurve3(
            corners[k].clone().multiplyScalar(r).setY(y),
            corners[(k + 1) % 6].clone().multiplyScalar(r).setY(y),
          ),
        );
      }
      const g = new THREE.TubeGeometry(path, 240, 0.16, 6, true);
      this.addSeedAttribute(g, this.rand());
      rimGeoms.push(g);
    }
    this.scene.add(new THREE.Mesh(mergeGeometries(rimGeoms), this.chassisRim));

    // CRC32-C segment ring: a verification sweep runs around it.
    this.crcSegments = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1.7, 0.3, 0.55),
      new THREE.MeshBasicMaterial({ color: '#ffffff' }),
      CRC_SEGMENTS,
    );
    const per = CRC_SEGMENTS / 6;
    const dummy = new THREE.Object3D();
    const inset = (CHASSIS_R - 2.4) / CHASSIS_R;
    for (let i = 0; i < CRC_SEGMENTS; i++) {
      const edge = Math.floor(i / per);
      const f = ((i % per) + 0.5) / per;
      const a = corners[edge].clone().multiplyScalar(inset);
      const b = corners[(edge + 1) % 6].clone().multiplyScalar(inset);
      dummy.position.copy(a.lerp(b, f)).setY(topY + 0.2);
      const dir = corners[(edge + 1) % 6].clone().sub(corners[edge]);
      dummy.rotation.set(0, Math.atan2(-dir.z, dir.x), 0);
      dummy.updateMatrix();
      this.crcSegments.setMatrixAt(i, dummy.matrix);
      this.crcSegments.setColorAt(i, C.crcIdle);
    }
    this.scene.add(this.crcSegments);

    const label = makeLabelSprite('WAL CHASSIS  CRC32-C');
    label.position.set(-8, topY + 1.6, CHASSIS_R * 0.9);
    this.scene.add(label);
  }

  private buildPointCloud(): void {
    const n = window.innerWidth < 800 ? 14000 : 30000;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const size = new Float32Array(n);
    const tint = new Float32Array(n);
    const branchAttr = new Float32Array(n);
    const R = LAYER_R[0] * 0.9;
    for (let i = 0; i < n; i++) {
      let x: number;
      let z: number;
      let y: number;
      if (this.rand() < 0.78) {
        const bi = Math.floor(this.rand() * 4);
        const c = this.branchCenters[bi];
        branchAttr[i] = bi;
        x = c.x + gaussian(this.rand) * 5.2;
        z = c.z + gaussian(this.rand) * 5.2;
        y = Math.abs(gaussian(this.rand)) * 0.5;
        tint[i] = 0.45 + this.rand() * 0.55;
      } else {
        const a = this.rand() * Math.PI * 2;
        const rad = Math.sqrt(this.rand()) * R;
        x = Math.cos(a) * rad;
        z = Math.sin(a) * rad;
        y = this.rand() * 0.4;
        tint[i] = this.rand() * 0.3;
        branchAttr[i] = -1;
      }
      pos.set([x, LAYER_Y[0] + y, z], i * 3);
      seed[i] = this.rand();
      size[i] = 0.5 + this.rand() * 0.9;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aTint', new THREE.BufferAttribute(tint, 1));
    g.setAttribute('aBranch', new THREE.BufferAttribute(branchAttr, 1));
    this.cloudMaterial = new THREE.ShaderMaterial({
      vertexShader: S.cloudVertex,
      fragmentShader: S.cloudFragment,
      uniforms: {
        ...this.shared,
        ...this.focusUniforms,
        uQuant: { value: 0 },
        uShockCenter: { value: new THREE.Vector3() },
        uShockTime: { value: -1 },
        uHit: { value: 0 },
        uHitRadius: { value: 5 },
        uColorA: { value: C.coolGray.clone() },
        uColorB: { value: C.softBlue.clone() },
        uHitColor: { value: C.blue.clone() },
      },
      transparent: true,
      depthWrite: false,
    });
    const pts = new THREE.Points(g, this.cloudMaterial);
    pts.frustumCulled = false;
    pts.renderOrder = -2;
    this.scene.add(pts);
  }

  private buildNodes(): void {
    // Apex orbs
    const apexGeom = new THREE.SphereGeometry(0.8, 32, 16);
    this.apex.forEach((a, i) => {
      const m = new THREE.Mesh(apexGeom, new THREE.MeshBasicMaterial({ color: i === 0 ? C.blue : C.softBlue }));
      m.position.copy(a.pos);
      m.userData.baseY = a.pos.y;
      this.apexMeshes.push(m);
      this.scene.add(m);
    });
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.7, 0.05, 8, 64),
      new THREE.MeshBasicMaterial({ color: C.blue, transparent: true, opacity: 0.55 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(this.apex[0].pos);
    this.scene.add(ring);

    // Hubs
    const hubMesh = (this.hubMesh = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.46, 1),
      new THREE.MeshBasicMaterial({ color: '#ffffff' }),
      this.hubs.length,
    ));
    const dummy = new THREE.Object3D();
    this.hubs.forEach((h, i) => {
      dummy.position.copy(h.pos);
      dummy.updateMatrix();
      hubMesh.setMatrixAt(i, dummy.matrix);
      hubMesh.setColorAt(i, i < 4 ? BRANCH_COLORS[i] : C.indigo);
    });
    this.scene.add(hubMesh);

    // Concept nodes: 100 indexed concepts, tinted by architecture branch.
    this.conceptMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.3, 16, 12),
      new THREE.MeshBasicMaterial({ color: '#ffffff' }),
      CONCEPTS,
    );
    this.pickMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.95, 8, 6),
      new THREE.MeshBasicMaterial({ visible: false }),
      CONCEPTS,
    );
    for (let i = 0; i < CONCEPTS; i++) {
      const p = this.conceptPos[i];
      dummy.position.set(p.x, p.y + 0.35, p.z);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      this.conceptMesh.setMatrixAt(i, dummy.matrix);
      this.conceptMesh.setColorAt(i, BRANCH_COLORS[this.branchOf(i + 1)]);
      this.pickMesh.setMatrixAt(i, dummy.matrix);
    }
    this.pickMesh.computeBoundingSphere();
    this.scene.add(this.conceptMesh, this.pickMesh);

    // Soft halos
    const halos: { p: THREE.Vector3; c: THREE.Color; s: number; b: number }[] = [];
    this.apex.forEach((a, i) => halos.push({ p: a.pos, c: C.blue, s: i === 0 ? 7 : 5, b: -1 }));
    this.hubs.forEach((h, i) => halos.push({ p: h.pos, c: i < 4 ? BRANCH_COLORS[i] : C.indigo, s: 3.4, b: i < 4 ? i : -1 }));
    this.branchCenters.forEach((c, i) => halos.push({ p: c.clone().setY(0.3), c: BRANCH_COLORS[i], s: 12, b: i }));
    const pos = new Float32Array(halos.length * 3);
    const col = new Float32Array(halos.length * 3);
    const size = new Float32Array(halos.length);
    const seed = new Float32Array(halos.length);
    const hb = new Float32Array(halos.length);
    halos.forEach((h, i) => {
      hb[i] = h.b;
      pos.set([h.p.x, h.p.y, h.p.z], i * 3);
      col.set([h.c.r, h.c.g, h.c.b], i * 3);
      size[i] = h.s;
      seed[i] = this.rand();
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    g.setAttribute('aBranch', new THREE.BufferAttribute(hb, 1));
    this.haloMaterial = this.makeHaloMaterial(0.34, 1, true);
    const halo = new THREE.Points(g, this.haloMaterial);
    halo.renderOrder = -1;
    this.scene.add(halo);
  }

  private buildConduits(): void {
    // Groups 0-2 are shared upper-layer wiring; 3-6 are L1 -> L0 trunks and 7-10 the
    // per-concept threads, split per branch so each branch can be isolated.
    const groupMat = (a: THREE.Color, b: THREE.Color, opacity: number, freq: number, branch: number) => {
      const m = this.makeConduitMaterial(a, b, opacity, freq);
      m.userData.branch = branch;
      return m;
    };
    const groups: { geoms: THREE.BufferGeometry[]; mat: THREE.ShaderMaterial }[] = [
      { geoms: [], mat: groupMat(C.blue, C.indigo, 0.7, 1.2, -1) }, // L2 intra
      { geoms: [], mat: groupMat(C.indigo, C.blue, 0.65, 1.5, -1) }, // L2 -> L1
      { geoms: [], mat: groupMat(C.indigo, C.softBlue, 0.55, 1.6, -1) }, // L1 intra
      ...BRANCH_COLORS.map((c, b) => ({ geoms: [], mat: groupMat(c, c.clone().lerp(C.white, 0.5), 0.8, 1.8, b) })), // L1 -> L0 trunks
      ...BRANCH_COLORS.map((c, b) => ({ geoms: [], mat: groupMat(c.clone().lerp(C.coolGray, 0.5), c, 0.5, 2.2, b) })), // concept threads
    ];
    const THREADS = 7;
    const add = (group: number, a: THREE.Vector3, b: THREE.Vector3, radius: number, track = true) => {
      const curve = makeCurve(a, b);
      if (track) this.conduitCurves.push(curve);
      const thread = group >= THREADS;
      const g = new THREE.TubeGeometry(curve, thread ? 14 : 40, radius, thread ? 4 : 6, false);
      this.addSeedAttribute(g, this.rand());
      groups[group].geoms.push(g);
      return curve;
    };

    const seen = new Set<string>();
    const once = (adj: Map<number, number[]>, group: number, nodes: GraphNode[], radius: number) => {
      seen.clear();
      for (const [id, ns] of adj) {
        for (const n of ns) {
          const key = id < n ? `${id}-${n}` : `${n}-${id}`;
          if (seen.has(key)) continue;
          seen.add(key);
          add(group, nodes[id].pos, nodes[n].pos, radius);
        }
      }
    };
    once(this.adj2, 0, this.apex, 0.1);
    for (const h of this.hubs) add(1, this.apex[this.hubParent.get(h.id)!].pos, h.pos, 0.075);
    once(this.adj1, 2, this.hubs, 0.06);
    for (let b = 0; b < 4; b++) {
      const curve = add(3 + b, this.hubs[b].pos, this.branchCenters[b].clone().setY(0.3), 0.08);
      if (b === 0) this.focusCurve = curve;
    }
    for (let i = 0; i < CONCEPTS; i++) {
      const b = this.branchOf(i + 1);
      const p = this.conceptPos[i];
      add(THREADS + b, this.branchCenters[b].clone().setY(0.3), new THREE.Vector3(p.x, 0.35, p.z), 0.022, false);
    }

    for (const grp of groups) {
      this.conduitMaterials.push(grp.mat);
      this.scene.add(new THREE.Mesh(mergeGeometries(grp.geoms), grp.mat));
      grp.geoms.forEach((g) => g.dispose());
    }

    // SIMD showcase conduit: brighter and thicker, visible only in step 01.
    this.focusConduitMat = this.makeConduitMaterial(C.blue, C.indigo, 0, 4);
    this.focusConduitMat.uniforms.uBoost.value = 1;
    const fg = new THREE.TubeGeometry(this.focusCurve, 80, 0.13, 10, false);
    this.addSeedAttribute(fg, 0.3);
    this.scene.add(new THREE.Mesh(fg, this.focusConduitMat));
  }

  private buildPackets(): void {
    const n = 120;
    const pos = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const seed = new Float32Array(n);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const curve = i < 16 ? this.focusCurve : this.conduitCurves[Math.floor(this.rand() * this.conduitCurves.length)];
      this.packets.push({ curve, t: this.rand(), speed: 0.1 + this.rand() * 0.22 });
      size[i] = i < 16 ? 1.5 : 0.9 + this.rand() * 0.5;
      seed[i] = this.rand();
      const c = i % 3 === 0 ? C.indigo : C.softBlue;
      col.set([c.r, c.g, c.b], i * 3);
    }
    this.packetGeom = new THREE.BufferGeometry();
    this.packetGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.packetGeom.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    this.packetGeom.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    this.packetGeom.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    this.packetMaterial = this.makeHaloMaterial(0.95, 0.4);
    const pts = new THREE.Points(this.packetGeom, this.packetMaterial);
    pts.frustumCulled = false;
    pts.renderOrder = 3;
    this.scene.add(pts);
  }

  private buildStreams(): void {
    const columns = 18;
    const per = 40;
    const n = columns * per;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const offset = new Float32Array(n);
    for (let c = 0; c < columns; c++) {
      const a = (c / columns) * Math.PI * 2;
      const r = CHASSIS_R + 3;
      for (let j = 0; j < per; j++) {
        const i = c * per + j;
        pos.set([Math.cos(a) * r + gaussian(this.rand) * 0.3, 0, Math.sin(a) * r + gaussian(this.rand) * 0.3], i * 3);
        seed[i] = this.rand();
        offset[i] = this.rand() * 72;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    g.setAttribute('aOffset', new THREE.BufferAttribute(offset, 1));
    this.streamMaterial = new THREE.ShaderMaterial({
      vertexShader: S.streamVertex,
      fragmentShader: S.streamFragment,
      uniforms: { ...this.shared, uStream: { value: 0.2 }, uColor: { value: C.emerald.clone() } },
      transparent: true,
      depthWrite: false,
    });
    const pts = new THREE.Points(g, this.streamMaterial);
    pts.frustumCulled = false;
    this.scene.add(pts);
  }

  private buildTraversalRig(): void {
    const beamGeom = new THREE.CylinderGeometry(0.28, 0.28, 1, 20, 1, true);
    beamGeom.translate(0, -0.5, 0); // top at origin, extends downward
    this.beam = new THREE.Mesh(
      beamGeom,
      new THREE.ShaderMaterial({
        vertexShader: S.beamVertex,
        fragmentShader: S.beamFragment,
        uniforms: { uColor: { value: C.blue.clone() }, uOpacity: { value: 0 } },
        transparent: true,
        depthWrite: false,
      }),
    );
    this.beam.visible = false;
    this.beam.renderOrder = 7;
    this.scene.add(this.beam);

    this.probe = new THREE.Mesh(new THREE.SphereGeometry(0.38, 24, 12), new THREE.MeshBasicMaterial({ color: C.blue }));
    this.probe.visible = false;
    this.probe.renderOrder = 8;
    this.scene.add(this.probe);
    this.probeHalo = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: makeGlowTexture(), color: C.blue, transparent: true, opacity: 0.55, depthWrite: false }),
    );
    this.probeHalo.scale.set(5, 5, 1);
    this.probeHalo.visible = false;
    this.scene.add(this.probeHalo);

    const ringGeom = new THREE.RingGeometry(0.9, 1, 128);
    ringGeom.rotateX(-Math.PI / 2);
    this.shockRing = new THREE.Mesh(
      ringGeom,
      new THREE.MeshBasicMaterial({ color: C.blue, transparent: true, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.shockRing.visible = false;
    this.scene.add(this.shockRing);

    const flashGeom = new THREE.RingGeometry(0.75, 1, 64);
    flashGeom.rotateX(-Math.PI / 2);
    for (let i = 0; i < 8; i++) {
      const f = new THREE.Mesh(
        flashGeom,
        new THREE.MeshBasicMaterial({ color: C.blue, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
      );
      f.visible = false;
      this.flashPool.push(f);
      this.scene.add(f);
    }

    this.selMarker = new THREE.Mesh(new THREE.SphereGeometry(0.62, 28, 16), new THREE.MeshBasicMaterial({ color: C.blue }));
    this.selMarker.visible = false;
    this.selMarker.renderOrder = 9;
    this.scene.add(this.selMarker);
    const selRingGeom = new THREE.RingGeometry(1.35, 1.5, 64);
    selRingGeom.rotateX(-Math.PI / 2);
    this.selRing = new THREE.Mesh(
      selRingGeom,
      new THREE.MeshBasicMaterial({ color: C.blue, transparent: true, opacity: 0.8, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.selRing.visible = false;
    this.scene.add(this.selRing);

    const markerGeom = new THREE.SphereGeometry(0.5, 20, 12);
    const markerRingGeom = new THREE.RingGeometry(0.95, 1.1, 48);
    markerRingGeom.rotateX(-Math.PI / 2);
    for (let i = 0; i < RESULT_POOL; i++) {
      const m = new THREE.Mesh(markerGeom, new THREE.MeshBasicMaterial({ color: C.emerald }));
      const halo = new THREE.Mesh(markerRingGeom, new THREE.MeshBasicMaterial({ color: C.ink, transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }));
      halo.position.y = -0.3;
      m.add(halo);
      m.visible = false;
      m.renderOrder = 9;
      this.resultMarkers.push(m);
      this.scene.add(m);
    }
  }

  private flash(p: THREE.Vector3, color: THREE.Color, size: number): void {
    const f = this.flashPool[this.flashCursor++ % this.flashPool.length];
    gsap.killTweensOf([f.scale, f.material]);
    f.position.copy(p);
    f.material.color.copy(color);
    f.material.opacity = 0.9;
    f.scale.setScalar(0.4);
    f.visible = true;
    gsap.to(f.scale, { x: size, y: size, z: size, duration: 0.7, ease: 'expo.out' });
    gsap.to(f.material, { opacity: 0, duration: 0.7, ease: 'power2.out', onComplete: () => (f.visible = false) });
  }

  private buildResultTubes(from: THREE.Vector3, neighborIds: number[], accent: THREE.Color): void {
    const geoms: THREE.BufferGeometry[] = [];
    const a = from.clone().setY(0.4);
    for (const id of neighborIds.slice(0, RESULT_POOL)) {
      const q = this.conceptPos[id - 1];
      if (!q) continue;
      const b = new THREE.Vector3(q.x, 0.4, q.z);
      const mid = a.clone().lerp(b, 0.5);
      mid.y += 0.8 + a.distanceTo(b) * 0.22;
      const g = new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 24, 0.07, 6, false);
      this.addSeedAttribute(g, this.rand());
      geoms.push(g);
    }
    if (!geoms.length) return;
    const mat = this.makeConduitMaterial(accent, accent.clone().lerp(C.white, 0.5), 1, 2);
    mat.uniforms.uReveal.value = 0;
    mat.uniforms.uBoost.value = 0.6;
    this.resultTubes = new THREE.Mesh(mergeGeometries(geoms), mat);
    geoms.forEach((g) => g.dispose());
    this.resultTubes.renderOrder = 5;
    this.scene.add(this.resultTubes);
  }

  private cleanupTraversal(): void {
    this.traversalTl?.kill();
    this.traversalTl = null;
    for (const obj of [this.trail, this.resultTubes]) {
      if (!obj) continue;
      this.scene.remove(obj);
      obj.geometry.dispose();
      obj.material.dispose();
    }
    this.trail = null;
    this.resultTubes = null;
    this.resultMarkers.forEach((m) => {
      gsap.killTweensOf(m.scale);
      m.visible = false;
    });
    gsap.killTweensOf([this.shockRing.scale, this.shockRing.material, this.probe.scale, this.selMarker.scale]);
    this.shockRing.visible = false;
    this.probe.scale.setScalar(1);
    this.probe.visible = false;
    this.probeHalo.visible = false;
    this.beam.visible = false;
    this.selMarker.scale.setScalar(1);
    const u = this.cloudMaterial.uniforms;
    gsap.killTweensOf([u.uShockTime, u.uHit, this.beam.material.uniforms.uOpacity]);
    u.uShockTime.value = -1;
    u.uHit.value = 0;
  }

  private buildPresets(): void {
    const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    const base = { boost: 0, quant: 0, chassis: 0.3, stream: 0.2, focusConduit: 0, autoRotate: false };

    const conduitMid = this.focusCurve.getPoint(0.5);
    const outward = v(conduitMid.x, 0, conduitMid.z).normalize();
    const side = v(-outward.z, 0, outward.x);
    const simdPos = conduitMid.clone().addScaledVector(outward, 11).addScaledVector(side, 5).add(v(0, 2.5, 0));

    const sc = this.branchCenters[0];
    const scOut = v(sc.x, 0, sc.z).normalize();
    const sqTarget = sc.clone().setY(0.4);
    const sqPos = sqTarget.clone().addScaledVector(scOut, 12).addScaledVector(v(-scOut.z, 0, scOut.x), -3).add(v(0, 8, 0));

    this.presets = {
      overview: { ...base, pos: v(46, 32, 53), target: v(0, 14, 0), anchor: null },
      simd: { ...base, pos: simdPos, target: conduitMid, anchor: conduitMid.clone(), boost: 1, focusConduit: 1 },
      hnsw: { ...base, pos: v(50, 40, 48), target: v(0, 11, 0), anchor: this.apex[0].pos.clone().add(v(0, 1.4, 0)) },
      sq8: { ...base, pos: sqPos, target: sqTarget, anchor: sqTarget.clone().add(v(0, 0.6, 0)), quant: 0.9 },
      wal: {
        ...base,
        pos: v(18, 1.5, 70),
        target: v(0, -4.5, 26),
        anchor: v(0, CHASSIS_Y + 1.4, CHASSIS_R * 0.9),
        chassis: 1,
      },
      network: { ...base, pos: v(0, 74, 104), target: v(0, 8, 0), anchor: null, stream: 1, chassis: 0.6, autoRotate: true },
    };
  }
}
