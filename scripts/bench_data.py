"""Load benchmark results (results/<machine_id>/<date>/...) and derive the numbers the charts,
report and carousel show. Every derived number is computed from the raw JSON; nothing is
hard-coded."""
from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
RECALL_TARGETS = (0.90, 0.95, 0.99)


@dataclass
class Machine:
    machine_id: str
    date: str
    root: Path
    env: dict = field(default_factory=dict)
    build: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ raw access
    def doc(self, suite: str, task: str) -> dict | None:
        p = self.root / suite / f"{task}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def result(self, suite: str, task: str) -> dict | None:
        d = self.doc(suite, task)
        return d["result"] if d else None

    def tasks(self, suite: str) -> list[str]:
        d = self.root / suite
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def gaps(self) -> list[dict]:
        p = self.root / "gaps.json"
        return json.loads(p.read_text()) if p.exists() else []

    # ------------------------------------------------------------------ labels
    @property
    def cpu(self) -> str:
        h = self.env.get("host", {})
        return h.get("cpu_model") or self.env.get("cpu", {}).get("model", "?")

    @property
    def commits(self) -> list[str]:
        """Every commit that produced a published result file (a run can span a resumed pass)."""
        seen: dict[str, str] = {}
        for p in self.root.glob("*/*.json"):
            if p.parent.name.startswith("superseded"):
                continue
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            c = ((d.get("env") or {}).get("git_commit") or "")[:7]
            if c:
                seen.setdefault(c, d.get("timestamp_utc", ""))
        return sorted(seen, key=lambda c: seen[c]) or [(self.env.get("git_commit") or "?")[:7]]

    @property
    def commit(self) -> str:
        return " + ".join(self.commits)

    @property
    def runtime(self) -> str:
        return self.env.get("runtime", "?")

    @property
    def host_physical_cores(self) -> int | None:
        h = self.env.get("host", {})
        try:
            return int(h.get("physical_cores")) if h.get("physical_cores") else self.env["cpu"].get("physical_cores_visible")
        except (TypeError, ValueError):
            return None

    @property
    def logical_cpus(self) -> int | None:
        h = self.env.get("host", {})
        try:
            return int(h.get("logical_processors")) if h.get("logical_processors") else self.env["cpu"].get("logical_cpus")
        except (TypeError, ValueError):
            return None


def load_machines(results: Path = RESULTS) -> list[Machine]:
    out = []
    if not results.exists():
        return out
    for mdir in sorted(p for p in results.iterdir() if p.is_dir()):
        dates = sorted(p for p in mdir.iterdir() if p.is_dir() and (p / "environment.json").exists())
        if not dates:
            continue
        root = dates[-1]  # latest run per machine
        envdoc = json.loads((root / "environment.json").read_text())
        out.append(Machine(mdir.name, root.name, root, envdoc.get("env", {}), envdoc.get("build", {}),
                           envdoc.get("config", {})))
    return out


# ---------------------------------------------------------------------- ANN aggregation
@dataclass
class Point:
    ef: int
    rerank: float
    recall: float
    qps: float
    recall_min: float = 0
    recall_max: float = 0
    qps_min: float = 0
    qps_max: float = 0
    p50: float = 0
    p99: float = 0
    mean_us: float = 0
    dist: float | None = None
    n_builds: int = 1
    label: str = ""


def ann_points(res: dict | None, rerank: float | None = None) -> list[Point]:
    """Median over builds per (ef, rerank), with min/max bands."""
    if not res:
        return []
    groups: dict[tuple, list[dict]] = {}
    for b in res.get("builds", []):
        for p in b.get("points", []):
            if rerank is not None and abs(float(p.get("rerank", 0)) - rerank) > 1e-6:
                continue
            groups.setdefault((int(p["ef"]), float(p.get("rerank", 0))), []).append(p)
    pts = []
    for (ef, rr), ps in sorted(groups.items()):
        rec = [p["recall"] for p in ps]
        qps = [p["qps"] for p in ps]
        dist = [p["distances_per_query"]["total"] for p in ps if "distances_per_query" in p]
        pts.append(Point(ef=ef, rerank=rr, recall=st.median(rec), qps=st.median(qps) if any(qps) else 0,
                         recall_min=min(rec), recall_max=max(rec), qps_min=min(qps), qps_max=max(qps),
                         p50=st.median(p["latency"]["p50_us"] for p in ps),
                         p99=st.median(p["latency"]["p99_us"] for p in ps),
                         mean_us=st.median(p["latency"]["mean_us"] for p in ps),
                         dist=st.median(dist) if dist else None, n_builds=len(ps)))
    return pts


def build_stats(res: dict | None) -> dict | None:
    if not res or not res.get("builds"):
        return None
    bs = [b for b in res["builds"] if b.get("build_seconds", -1) and b.get("build_seconds", -1) > 0]
    if not bs:
        return None
    n = res["dataset"]["base_rows"]
    secs = [b["build_seconds"] for b in bs]
    delta = [b["index_rss_delta_bytes"] for b in bs]
    return {"build_seconds": st.median(secs), "build_seconds_min": min(secs), "build_seconds_max": max(secs),
            "rss_delta_bytes": st.median(delta), "bytes_per_vector": st.median(delta) / n,
            "peak_rss_bytes": max(b["peak_rss_bytes"] for b in bs), "n": n, "dim": res["dataset"]["dim"],
            "builds": len(bs), "snapshot_bytes": max(b.get("snapshot_bytes", 0) for b in bs),
            "save_seconds": max(b.get("save_seconds", -1) for b in bs)}


def pareto(points: list[Point]) -> list[Point]:
    """Points not dominated in (recall, qps); returned in ascending recall."""
    best, out = -1.0, []
    for p in sorted(points, key=lambda p: (-p.recall, -p.qps)):
        if p.qps > best:
            out.append(p)
            best = p.qps
    return sorted(out, key=lambda p: p.recall)


def best_at(points: list[Point], target: float) -> Point | None:
    """Highest measured QPS among points whose measured (median) recall >= target."""
    ok = [p for p in points if p.recall >= target - 1e-12]
    return max(ok, key=lambda p: p.qps) if ok else None


def fastest_latency_at(points: list[Point], target: float) -> Point | None:
    ok = [p for p in points if p.recall >= target - 1e-12]
    return min(ok, key=lambda p: p.mean_us) if ok else None


def min_dist_at(points: list[Point], target: float) -> Point | None:
    ok = [p for p in points if p.recall >= target - 1e-12 and p.dist]
    return min(ok, key=lambda p: p.dist) if ok else None


def grid_points(m: Machine, dataset: str, engine: str) -> dict[str, list[Point]]:
    """All tuning-grid configurations for an engine, including the main recall_qps config."""
    out: dict[str, list[Point]] = {}
    for t in m.tasks("vs_hnswlib"):
        parts = t.split("_")
        if parts[0] == dataset and parts[1] == engine:
            pts = ann_points(m.result("vs_hnswlib", t))
            for p in pts:
                p.label = "_".join(parts[2:])
            out["_".join(parts[2:])] = pts
    main = m.result("recall_qps", f"{dataset}_{'hnsw' if engine == 'nexus' else engine}")
    if main:
        cfg = m.config
        key = f"M{cfg.get('M' if dataset != 'glove100' else 'glove_M', 16)}_efc{cfg.get('efc' if dataset != 'glove100' else 'glove_efc', 200)}"
        pts = ann_points(main)
        for p in pts:
            p.label = key
        out.setdefault(key, pts)
    return out


def all_grid(m: Machine, dataset: str, engine: str) -> list[Point]:
    return [p for pts in grid_points(m, dataset, engine).values() for p in pts]


# Core layouts of hybrid CPUs from the vendor's specification. Used only for explanatory text
# when the topology is hidden from the benchmark (e.g. inside WSL2); never used in a number.
KNOWN_TOPOLOGY = {
    "i5-13400F": " (Intel spec: 6 P-cores with Hyper-Threading + 4 E-cores)",
    "i5-13400": " (Intel spec: 6 P-cores with Hyper-Threading + 4 E-cores)",
}


def known_topology(cpu: str) -> str:
    for k, v in KNOWN_TOPOLOGY.items():
        if k in cpu:
            return v
    return ""


def mean_scaling(runs: list[dict]) -> list[dict]:
    """Average read QPS per thread count over several concurrency sweeps."""
    by: dict[int, list[float]] = {}
    for r in runs:
        for p in r["points"]:
            by.setdefault(p["threads"], []).append(p["read_qps"])
    return [{"threads": t, "read_qps": sum(v) / len(v), "n": len(v)} for t, v in sorted(by.items())]


def fmt_x(r: float) -> str:
    return f"{r:.1f}x" if r < 10 else f"{r:.0f}x"


def fmt_int(v: float) -> str:
    return f"{v:,.0f}"
