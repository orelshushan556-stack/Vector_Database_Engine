#!/usr/bin/env python3
"""Render every benchmark chart from results/ (all machines found there are overlaid).

  python scripts/plot.py [--out docs/figures] [--only pareto_sift ...]

Each chart is exported for a light and a dark theme as:
  <name>_<theme>.png             2x resolution of the 1200x675 layout (2400x1350)
  <name>_<theme>.svg             vector, same layout
  <name>_<theme>_1200x675.png    social card (GitHub)
  <name>_<theme>_1080x1350.png   portrait carousel slide (separate layout)

Titles state the takeaway and are computed from the measured data; nothing is hard-coded.
Colors: one validated colorblind-safe categorical palette (dataviz reference palette); an
entity keeps its color in every chart (Nexus HNSW blue, hnswlib orange, Nexus SQ8 aqua,
FAISS violet, exact scan neutral gray).
"""
from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402

# ----------------------------------------------------------------------------- theme
THEMES = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ring="#fcfcfb",
                  nexus="#2a78d6", hnswlib="#eb6834", sq8="#1baf7a", faiss="#6250d6", exact="#898781",
                  good="#0ca30c", critical="#d03b3b", warning="#fab219",
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", ring="#1a1a19",
                 nexus="#3987e5", hnswlib="#d95926", sq8="#199e70", faiss="#9085e9", exact="#898781",
                 good="#0ca30c", critical="#d03b3b", warning="#fab219",
                 ramp=["#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]),
}
LAYOUTS = {"wide": (12.0, 6.75), "tall": (10.8, 13.5)}
MACHINE_STYLES = ["-", "--", ":", "-."]
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT, "svg.fonttype": "path",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})


class Canvas:
    """Figure with an action title, subtitle, footer and a content area."""

    def __init__(self, theme: str, layout: str, title: str, subtitle: str, footer: str):
        self.T = THEMES[theme]
        self.layout = layout
        w, h = LAYOUTS[layout]
        self.fig = plt.figure(figsize=(w, h), dpi=100, facecolor=self.T["surface"])
        tall = layout == "tall"
        tsize = 25 if tall else 21
        ssize = 14 if tall else 12.5
        wrap_t = 44 if tall else 78
        wrap_s = 96 if tall else 130
        lh = lambda pt: pt * 1.32 / 72 / h  # noqa: E731 line height in figure fraction
        y = 1 - 0.42 / h
        left = 0.6 / w
        for line in textwrap.wrap(title, wrap_t):
            self.fig.text(left, y, line, fontsize=tsize, fontweight="bold", color=self.T["ink"], va="top")
            y -= lh(tsize)
        y -= 0.05 / h
        for line in textwrap.wrap(subtitle, wrap_s):
            self.fig.text(left, y, line, fontsize=ssize, color=self.T["ink2"], va="top")
            y -= lh(ssize)
        fsize = 10 if tall else 8.5
        foot_lines = textwrap.wrap(footer, 150 if tall else 175)
        fy = 0.28 / h
        for line in reversed(foot_lines):
            self.fig.text(left, fy, line, fontsize=fsize, color=self.T["muted"], va="bottom")
            fy += lh(fsize)
        self.top = y - 0.3 / h
        self.bottom = fy + 0.45 / h
        self.left = left
        self.right = 1 - 0.5 / w

    def rect(self, l=0.0, b=0.0, w=1.0, h=1.0, pad_l=0.75, pad_b=0.55, pad_r=0.0, pad_t=0.0):
        """Sub-rectangle of the content area (fractions), with padding in inches for tick labels."""
        W, H = LAYOUTS[self.layout]
        cw, ch = self.right - self.left, self.top - self.bottom
        x0 = self.left + l * cw + pad_l / W
        y0 = self.bottom + b * ch + pad_b / H
        return [x0, y0, w * cw - (pad_l + pad_r) / W, h * ch - (pad_b + pad_t) / H]

    def axes(self, rect, xlabel="", ylabel=""):
        T = self.T
        ax = self.fig.add_axes(rect, facecolor=T["surface"])
        fs = 14 if self.layout == "tall" else 12.5
        ax.set_xlabel(xlabel, fontsize=fs, color=T["ink2"], labelpad=8)
        ax.set_ylabel(ylabel, fontsize=fs, color=T["ink2"], labelpad=8)
        ax.tick_params(colors=T["muted"], labelcolor=T["ink2"], labelsize=fs - 1.5, length=0, pad=6)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(T["axis"])
        ax.grid(True, color=T["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        return ax

    def legend(self, ax, handles=None, loc="lower left", ncol=1, **kw):
        T = self.T
        fs = 12.5 if self.layout == "tall" else 11
        leg = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=True, fontsize=fs, **kw)
        leg.get_frame().set_facecolor(T["surface"])
        leg.get_frame().set_edgecolor(T["grid"])
        for t in leg.get_texts():
            t.set_color(T["ink2"])
        return leg

    def save(self, out: Path, name: str, theme: str) -> list[Path]:
        paths = []
        if self.layout == "wide":
            p = out / f"{name}_{theme}.png"
            self.fig.savefig(p, dpi=200, facecolor=self.T["surface"])
            paths.append(p)
            p = out / f"{name}_{theme}.svg"
            self.fig.savefig(p, facecolor=self.T["surface"])
            paths.append(p)
            p = out / f"{name}_{theme}_1200x675.png"
            self.fig.savefig(p, dpi=100, facecolor=self.T["surface"])
            paths.append(p)
        else:
            p = out / f"{name}_{theme}_1080x1350.png"
            self.fig.savefig(p, dpi=100, facecolor=self.T["surface"])
            paths.append(p)
        plt.close(self.fig)
        return paths


def footer(ms: list[bd.Machine]) -> str:
    parts = [f"{m.machine_id} ({m.cpu}, {m.runtime}, commit {m.commit})" for m in ms]
    return " | ".join(parts) + " | all numbers measured, raw data in results/"


def note(ax, C: Canvas, x, y, text, xy_text, ha="left", fs=None, arrow=True, coords="data"):
    T = C.T
    fs = fs or (12.5 if C.layout == "tall" else 11)
    ax.annotate(text, xy=(x, y), xytext=xy_text, textcoords="offset points", xycoords=coords,
                fontsize=fs, color=T["ink"], ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=T["surface"], ec=T["grid"], lw=0.8),
                arrowprops=dict(arrowstyle="-|>", color=T["muted"], lw=1.1, shrinkA=2, shrinkB=4) if arrow else None,
                zorder=10)


def end_label(ax, C: Canvas, x, y, text, color, dx=8, dy=0):
    """Direct label at the end of a line: a colored dot beside ink text."""
    T = C.T
    fs = 12.5 if C.layout == "tall" else 11
    ax.annotate("● ", xy=(x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=fs - 2,
                va="center", ha="left", zorder=9)
    ax.annotate(text, xy=(x, y), xytext=(dx + fs, dy), textcoords="offset points", color=T["ink"], fontsize=fs,
                va="center", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=T["surface"], ec="none", alpha=0.85))


def qps_fmt(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def better_arrow(ax, C: Canvas, direction="up-right", where=(0.86, 0.12)):
    T = C.T
    x, y = where
    dx, dy = {"up-right": (0.07, 0.10), "up-left": (-0.07, 0.10)}[direction]
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=T["muted"], lw=1.6))
    ax.text(x - 0.01, y - 0.01, "better", transform=ax.transAxes, color=T["muted"],
            fontsize=12 if C.layout == "tall" else 10.5, ha="right", va="top")


# ----------------------------------------------------------------------------- charts
CHARTS = {}


def chart(name):
    def deco(f):
        CHARTS[name] = f
        return f
    return deco


DS_LABEL = {"sift": "SIFT1M (1,000,000 x 128-d, L2)", "glove100": "GloVe-100 (1,183,514 x 100-d, angular)",
            "siftsmall": "SIFT-10K (10,000 x 128-d, L2)"}


def pareto_chart(dataset):
    def make(ms, theme, layout):
        m0 = ms[0]
        nx = bd.pareto(bd.all_grid(m0, dataset, "nexus"))
        hl = bd.pareto(bd.all_grid(m0, dataset, "hnswlib"))
        if not nx or not hl:
            return None
        a, b = bd.best_at(nx, 0.95), bd.best_at(hl, 0.95)
        if a and b:
            r = a.qps / b.qps
            if r >= 1.05:
                title = f"At 95% recall on {'SIFT1M' if dataset == 'sift' else 'GloVe-100'}, Nexus answers {r:.2f}x the queries per second of hnswlib"
            elif r <= 1 / 1.05:
                title = f"At 95% recall on {'SIFT1M' if dataset == 'sift' else 'GloVe-100'}, hnswlib is {1 / r:.2f}x faster than Nexus on one core"
            else:
                title = f"At 95% recall on {'SIFT1M' if dataset == 'sift' else 'GloVe-100'}, Nexus and hnswlib are within 5% of each other"
        else:
            title = f"Recall vs throughput on {DS_LABEL[dataset]}"
        sub = (f"{DS_LABEL[dataset]} | k = 10 | best measured point of each engine across the same (M, efConstruction) grid | "
               f"single thread, one query at a time")
        C = Canvas(theme, layout, title, sub, footer(ms))
        T = C.T
        ax = C.axes(C.rect(pad_r=1.6 if layout == "wide" else 1.4), "Recall@10", "Queries per second (log scale)")
        ax.set_yscale("log")
        series = []
        for mi, m in enumerate(ms):
            ls = MACHINE_STYLES[mi % len(MACHINE_STYLES)]
            s = [("Nexus HNSW (FP32)", "nexus", bd.pareto(bd.all_grid(m, dataset, "nexus")), "o"),
                 ("hnswlib", "hnswlib", bd.pareto(bd.all_grid(m, dataset, "hnswlib")), "s")]
            if dataset == "sift":
                s.append(("Nexus HNSW-SQ8 (re-rank 2)", "sq8", bd.pareto(bd.ann_points(m.result("recall_qps", "sift_sq8"), rerank=2.0)), "D"))
            faiss = [t for t in m.tasks("vs_hnswlib") if t.startswith(f"{dataset}_faiss")]
            if faiss:
                s.append(("FAISS HNSWFlat (Python)", "faiss", bd.pareto(bd.ann_points(m.result("vs_hnswlib", faiss[0]))), "^"))
            for label, key, pts, mk in s:
                pts = [p for p in pts if p.recall >= 0.5]
                if not pts:
                    continue
                lab = label if len(ms) == 1 else f"{label} - {m.machine_id}"
                ax.plot([p.recall for p in pts], [p.qps for p in pts], ls, color=T[key], lw=2.2, marker=mk, ms=6.5,
                        mec=T["ring"], mew=1.5, label=lab, zorder=4)
                if mi == 0:
                    series.append((label, key, pts))
            flat = bd.ann_points(m.result("recall_qps", f"{dataset}_flat"))
            if flat and mi == 0:
                ax.axhline(flat[0].qps, color=T["exact"], lw=2, ls=(0, (2, 3)), zorder=2,
                           label=f"exact scan (FlatIndex), recall {flat[0].recall:.3f}")
                ax.text(0.99, flat[0].qps, f"exact scan: {flat[0].qps:,.0f} QPS at recall {flat[0].recall:.3f}",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=T["ink2"],
                        fontsize=12 if layout == "tall" else 10.5)
        lo = min(p.recall for _, _, pts in series for p in pts)
        hi = max(p.recall for _, _, pts in series for p in pts)
        # logit axis: equal space per "nine" (0.9, 0.99, 0.999), where ANN trade-offs live
        ax.set_xscale("logit", one_half="0.5")
        ax.set_xlim(max(0.5, lo - 0.02), min(0.9995, 1 - (1 - hi) / 1.6))
        ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
                 if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(qps_fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.axvline(0.95, color=T["muted"], lw=1, ls=(0, (4, 4)), zorder=1)
        if a and b:
            fx = ax.transAxes.inverted().transform(ax.transData.transform((a.recall, a.qps)))[0]
            dx = 150 if fx < 0.55 else -170
            # keep the lower callout inside the axes: never push it below ~25% of the axes height
            fy = min(ax.transAxes.inverted().transform(ax.transData.transform((p.recall, p.qps)))[1] for p in (a, b))
            down = -min(130, max(40, (fy - 0.25) * ax.bbox.height * 72 / C.fig.dpi))
            ddx = dx
            if C.layout == "tall":
                # curves fall to the right, so below-left of the lower point is empty space
                w_pt = ax.bbox.width * 72 / C.fig.dpi
                ddx = -max(0.0, min(150.0, fx * w_pt - 125))
            up, dn = ((dx, 70), (ddx, down)) if a.qps >= b.qps else ((ddx, down), (dx, 70))
            note(ax, C, a.recall, a.qps,
                 f"Nexus {a.qps:,.0f} QPS\n(recall {a.recall:.3f}, {a.label.replace('_', ' ')}, ef {a.ef})", up, ha="center")
            note(ax, C, b.recall, b.qps,
                 f"hnswlib {b.qps:,.0f} QPS\n(recall {b.recall:.3f}, {b.label.replace('_', ' ')}, ef {b.ef})", dn, ha="center")
        better_arrow(ax, C, where=(0.88, 0.80))
        C.legend(ax, loc="lower left")
        return C
    return make


chart("pareto_sift")(pareto_chart("sift"))
chart("pareto_glove100")(pareto_chart("glove100"))


def distance_chart(dataset):
    def make(ms, theme, layout):
        m0 = ms[0]
        nx = bd.ann_points(m0.result("distance_computations", f"{dataset}_nexus_hnsw"))
        hl = bd.ann_points(m0.result("distance_computations", f"{dataset}_hnswlib_hnsw"))
        if not nx:
            return None
        n_base = m0.result("distance_computations", f"{dataset}_nexus_hnsw")["dataset"]["base_rows"]
        a = bd.min_dist_at(nx, 0.95)
        title = (f"At 95% recall, HNSW computes {n_base / a.dist:,.0f}x fewer distances per query than a full scan"
                 if a else f"Distance computations per query on {DS_LABEL[dataset]}")
        b = bd.min_dist_at(hl, 0.95) if hl else None
        sub = (f"{DS_LABEL[dataset]} | k = 10 | counted with NEXUS_COUNT_DISTANCES (Nexus) and a counting wrapper around "
               f"hnswlib's distance function, on the same saved graphs as the timing runs | hardware-independent measure")
        C = Canvas(theme, layout, title, sub, footer(ms))
        T = C.T
        ax = C.axes(C.rect(pad_r=1.5 if layout == "wide" else 1.3),
                    "Distance computations per query (log scale)", "Recall@10")
        ax.set_xscale("log")
        hl_label = "hnswlib"
        both = {p.ef: p for p in nx if p.dist}
        rel = [abs(q.dist - both[q.ef].dist) / both[q.ef].dist for q in hl if q.dist and q.ef in both]
        if rel and max(rel) < 0.10:  # say so when the measured curves sit on top of each other
            hl_label = f"hnswlib (under Nexus; within {max(rel) * 100:.0f}% at every ef)"
        series = [("Nexus HNSW (FP32)", "nexus", nx, "o"), (hl_label, "hnswlib", hl, "s")]
        sq = bd.ann_points(m0.result("distance_computations", f"{dataset}_nexus_sq8"))
        if sq:
            series.append(("Nexus HNSW-SQ8 (codes + FP32 re-rank)", "sq8", sq, "D"))
        for mi, m in enumerate(ms):
            ls = MACHINE_STYLES[mi % len(MACHINE_STYLES)]
            for label, key, _, mk in series:
                task = {"nexus": f"{dataset}_nexus_hnsw", "hnswlib": f"{dataset}_hnswlib_hnsw", "sq8": f"{dataset}_nexus_sq8"}[key]
                pts = [p for p in bd.ann_points(m.result("distance_computations", task)) if p.dist]
                if not pts:
                    continue
                ax.plot([p.dist for p in pts], [p.recall for p in pts], ls, color=T[key], lw=2.2, marker=mk, ms=6.5,
                        mec=T["ring"], mew=1.5, label=label if len(ms) == 1 else f"{label} - {m.machine_id}",
                        zorder=6 if key == "nexus" else 4)  # Nexus on top: the curves can coincide
        ax.axvline(n_base, color=T["exact"], lw=2, ls=(0, (2, 3)))
        ax.text(n_base, 0.40, f"exact scan  \n{n_base:,} per query  ", transform=ax.get_xaxis_transform(),
                color=T["ink2"], fontsize=11 if layout == "wide" else 12.5, ha="right", va="bottom")
        ax.axhline(0.95, color=T["muted"], lw=1, ls=(0, (4, 4)))
        xs = [p.dist for _, _, pts, _ in series for p in pts if p.dist]
        ax.set_xlim(min(xs) * 0.7, n_base * 1.6)
        lo = min(p.recall for _, _, pts, _ in series for p in pts)
        ax.set_ylim(max(0.0, lo - 0.03), 1.005)
        ax.xaxis.set_major_formatter(FuncFormatter(qps_fmt))
        if a:
            txt = f"Nexus: {a.dist:,.0f} distances\nat recall {a.recall:.3f}"
            if b:
                txt += f"\nhnswlib: {b.dist:,.0f} at {b.recall:.3f}"
            note(ax, C, a.dist, a.recall, txt, (60, -70))
        better_arrow(ax, C, "up-left", where=(0.72, 0.62))
        C.legend(ax, loc="lower left", bbox_to_anchor=(0.36 if layout == "wide" else 0.2, 0.02))
        return C
    return make


chart("distances_sift")(distance_chart("sift"))
chart("distances_glove100")(distance_chart("glove100"))


def fix_data(m: bd.Machine, index="hnsw"):
    out = {}
    for v in ("before", "after"):
        d = m.doc("before_after_fix", f"{v}_{index}")
        if not d:
            return None
        per_ef = {}
        for b in d["result"]["builds"]:
            for p in b["points"]:
                per_ef.setdefault(p["ef"], []).append(p["recall"])
        out[v] = per_ef
        out[v + "_commit"] = d.get("engine_commit", "")[:7]
    return out


@chart("fix_before_after")
def fix_chart(ms, theme, layout):
    m0 = ms[0]
    d = fix_data(m0)
    if not d:
        return None
    efs = sorted(d["after"])
    key_ef = 64 if 64 in efs else efs[len(efs) // 2]
    bmin, bmax = min(d["before"][key_ef]), max(d["before"][key_ef])
    amin, amax = min(d["after"][key_ef]), max(d["after"][key_ef])
    nb = len(d["before"][key_ef])
    title = (f"Neighbor-selection fix: worst of {nb} builds went from recall {bmin:.2f} to {amin:.3f} "
             f"(efSearch {key_ef})")
    sub = (f"{DS_LABEL['siftsmall']} | Nexus HNSWIndex M=16, efConstruction=200, k=10 | {nb} independent builds per version, "
           f"each dot is one build | before = {d['before_commit']} (farthest-neighbor eviction), "
           f"after = {d['after_commit']} (HNSW Algorithm 4 heuristic)")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(pad_t=0.5), "efSearch", "Recall@10 per build")
    import random
    rnd = random.Random(7)
    xs = list(range(len(efs)))
    flagged = 0
    for i, ef in enumerate(efs):
        a_min = min(d["after"][ef])
        for v, off, col in (("before", -0.18, T["exact"]), ("after", 0.18, T["nexus"])):
            vals = d[v][ef]
            jit = [i + off + rnd.uniform(-0.07, 0.07) for _ in vals]
            ax.scatter(jit, vals, s=70, color=col, edgecolor=T["ring"], lw=1.3, zorder=4,
                       label=("before fix" if v == "before" else "after fix") if i == 0 else None)
            if v == "before":
                low = [(x, y) for x, y in zip(jit, vals) if y < a_min]
                flagged += len(low) if ef == key_ef else 0
                if low:
                    ax.scatter([x for x, _ in low], [y for _, y in low], s=150, facecolor="none",
                               edgecolor=T["critical"], lw=1.8, zorder=5,
                               label="before-fix build below the worst after-fix build" if i == 0 else None)
    ax.set_xticks(xs, [str(e) for e in efs])
    ki = efs.index(key_ef)
    note(ax, C, ki - 0.18, bmin, f"worst before: {bmin:.3f}", (-120, -10), ha="center")
    note(ax, C, ki + 0.18, amin, f"worst after: {amin:.3f}", (95, -40), ha="center")
    allv = [v for ver in ("before", "after") for e in efs for v in d[ver][e]]
    ax.set_ylim(max(0, min(allv) - 0.05), 1.01)
    ax.set_xlim(-0.6, len(efs) - 0.4)
    C.legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.3)
    return C


@chart("crash_recovery")
def crash_chart(ms, theme, layout):
    m0 = ms[0]
    kc = m0.result("crash_recovery", "kill_cycles")
    rp = m0.result("crash_recovery", "replay_time")
    if not kc:
        return None
    s = kc["summary"]
    title = (f"{s['passed']} of {s['cycles']} SIGKILL crashes recovered every acknowledged insert "
             f"({s['lost_acked_total']} lost, {s['torn_tails_truncated']} torn tails truncated)")
    sub = (f"A writer inserts 128-d vectors (WAL append + CRC32-C + fsync per insert) and is killed at a random moment; "
           f"after restart: no acknowledged insert missing, every vector bit-exact, torn tail truncated | "
           f"every 4th cycle injects a partial record | WAL on {m0.env.get('work_dir_filesystem', '?').split(' on ')[0]}")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    if layout == "wide":
        r1, r2 = C.rect(0, 0, 0.5, 1, pad_l=0.1, pad_b=0.55), C.rect(0.53, 0, 0.47, 1, pad_l=0.8)
    else:
        r1, r2 = C.rect(0, 0.45, 1, 0.55, pad_l=0.1, pad_b=0.6), C.rect(0, 0, 1, 0.42, pad_l=0.85)
    ax = C.fig.add_axes(r1, facecolor=T["surface"])
    cyc = kc["cycles"]
    n = len(cyc)
    cols = int(math.ceil(math.sqrt(n * (r1[2] * LAYOUTS[layout][0]) / (r1[3] * LAYOUTS[layout][1]))))
    rows = int(math.ceil(n / cols))
    for i, c in enumerate(cyc):
        x, y = i % cols, rows - 1 - i // cols
        ax.add_patch(patches.FancyBboxPatch((x + 0.08, y + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.18",
                                            fc=T["good"] if c["ok"] else T["critical"], ec="none"))
        if c["torn_tail_before_recovery"]:
            ax.plot([x + 0.3, x + 0.7], [y + 0.3, y + 0.7], color=T["surface"], lw=1.4)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    fs = 12.5 if layout == "tall" else 11
    from matplotlib.lines import Line2D
    hs = [patches.Patch(color=T["good"], label=f"recovery verified ({s['passed']})"),
          Line2D([0], [0], color=T["good"], lw=0, marker="s", ms=12, mfc=T["good"], mec=T["ink2"],
                 label=f"torn tail found and truncated ({s['torn_tails_truncated']})")]
    if s["passed"] < s["cycles"]:
        hs.append(patches.Patch(color=T["critical"], label=f"failed ({s['cycles'] - s['passed']})"))
    C.legend(ax, handles=hs, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, borderaxespad=0)
    ax.text(0, 1.02, f"Each square is one kill + restart cycle ({n})", transform=ax.transAxes, color=T["ink2"],
            fontsize=fs + 0.5, va="bottom")
    ax2 = C.axes(r2, "WAL records replayed (log scale)", "Seconds (log scale)")
    if rp:
        res = rp["results"]
        import statistics as stt
        lines = [("raw replay (read + CRC check)", "raw_replay_seconds", "nexus", "-", "o"),
                 ("recover into FlatIndex", "recover_flat_seconds", "nexus", "--", "s"),
                 ("recover into HNSWIndex (rebuilds graph)", "recover_hnsw_seconds", "sq8", "-", "D")]
        for label, k, col, ls, mk in lines:
            pts = [(x["records"], stt.median(x[k])) for x in res if x[k]]
            if pts:
                ax2.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=T[col] if col != "sq8" else T["exact"],
                         lw=2.2, marker=mk, ms=6.5, mec=T["ring"], mew=1.5, label=label)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ymax = max(stt.median(x[k]) for x in res for _, k, *_ in lines if x[k])
        ymin = min(stt.median(x[k]) for x in res for _, k, *_ in lines if x[k])
        ax2.set_ylim(ymin / 2, ymax * 40)
        big = max(res, key=lambda x: x["records"])
        note(ax2, C, big["records"], stt.median(big["raw_replay_seconds"]),
             f"{big['records']:,} records ({big['wal_bytes'] / 2**20:,.0f} MiB)\nreplayed in {stt.median(big['raw_replay_seconds']):.2f} s",
             (-90, -60), ha="center")
        ax2.xaxis.set_major_formatter(FuncFormatter(qps_fmt))
        C.legend(ax2, loc="upper left")
        small = min(res, key=lambda x: x["records"])
        k = big["records"] / small["records"]
        g = stt.median(big["raw_replay_seconds"]) / stt.median(small["raw_replay_seconds"])
        ax2.set_title(f"Replay time: {k:,.0f}x the records took {g:,.0f}x the time", fontsize=fs + 0.5,
                      color=T["ink2"], loc="left", pad=8)
    return C


@chart("thread_scaling")
def scaling_chart(ms, theme, layout):
    m0 = ms[0]
    ro = m0.result("concurrency", "read_only")
    if not ro:
        return None
    runs = [r for r in (ro, m0.result("concurrency", "read_only_repeat")) if r]
    pts = bd.mean_scaling(runs)
    q1 = next(p["read_qps"] for p in pts if p["threads"] == 1)
    phys = m0.host_physical_cores
    best = max(pts, key=lambda p: p["read_qps"])
    at_phys = next((p for p in pts if p["threads"] == phys), None)
    ref = at_phys or best
    title = (f"Search throughput grows {ref['read_qps'] / q1:.1f}x from 1 to {ref['threads']} threads "
             f"({100 * ref['read_qps'] / (q1 * ref['threads']):.0f}% of linear); peak {best['read_qps']:,.0f} QPS at {best['threads']} threads")
    sub = (f"SIFT1M | Nexus HNSWIndex through VectorEngine (shared lock per search) | efSearch {ro['ef']} | k = 10 | "
           f"{len(runs)} sweep(s) ({', '.join(f'{r['seconds_per_point']:.0f} s' for r in runs)} per point), title uses their mean | "
           f"threads not pinned | squares: 90/10 read/write mix, every write fsyncs")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(pad_r=1.8 if layout == "wide" else 1.2), "Threads", "Search queries per second")
    for mi, m in enumerate(ms):
        r = m.result("concurrency", "read_only")
        if not r:
            continue
        p = r["points"]
        base = next(x["read_qps"] for x in p if x["threads"] == 1)
        ts = [x["threads"] for x in p]
        if mi == 0:
            ax.plot(ts, [base * t for t in ts], color=T["muted"], lw=1.4, ls=(0, (2, 3)), label="ideal linear scaling")
        ax.plot(ts, [x["read_qps"] for x in p], MACHINE_STYLES[mi], color=T["nexus"], lw=2.4, marker="o", ms=7,
                mec=T["ring"], mew=1.5, label="read-only" + ("" if len(ms) == 1 else f" - {m.machine_id}"), zorder=4)
        rep_ = m.result("concurrency", "read_only_repeat")
        if rep_:
            rp = rep_["points"]
            ax.plot([x["threads"] for x in rp], [x["read_qps"] for x in rp], ":", color=T["nexus"], lw=2.0, marker="^",
                    ms=7, mec=T["ring"], mew=1.5, label=f"read-only, repeat ({rep_['seconds_per_point']:.0f} s per point)", zorder=4)
        mix = m.result("concurrency", "mix_90_10")
        if mix:
            mp = mix["points"]
            ax.plot([x["threads"] for x in mp], [x["read_qps"] for x in mp], "--", color=T["nexus"], lw=2.0, marker="s",
                    ms=6, mec=T["ring"], mew=1.5, label="reads during 90/10 mix" + ("" if len(ms) == 1 else f" - {m.machine_id}"))
    top = max(x["read_qps"] for r in runs for x in r["points"])
    ax.set_ylim(0, max(top * 1.25, q1 * 2))
    ax.set_xlim(0, max(x["threads"] for x in pts) + 0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(qps_fmt))
    fs = 12 if layout == "tall" else 10.5
    for v, lab in ((phys, f"{phys} physical cores"), (m0.logical_cpus, f"{m0.logical_cpus} logical CPUs")):
        if v:
            ax.axvline(v, color=T["axis"], lw=1.5)
            # labels sit low, between the mix line and the read-only curves, so they cross no data
            ax.text(v, 0.08, " " + lab, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="bottom",
                    color=T["ink2"], fontsize=fs)
    last = pts[-1]
    end_label(ax, C, last["threads"], last["read_qps"], f"{last['read_qps']:,.0f}", T["nexus"])
    mix = m0.result("concurrency", "mix_90_10")
    if mix:
        lm = mix["points"][-1]
        end_label(ax, C, lm["threads"], lm["read_qps"], f"{lm['read_qps']:,.0f} reads/s\n+ {lm['write_ops']:,.0f} writes/s", T["nexus"])
    C.legend(ax, loc="upper left")
    return C


@chart("simd_speedup")
def simd_chart(ms, theme, layout):
    m0 = ms[0]
    r = m0.result("ablation", "simd_kernels")
    if not r:
        return None
    res = r["results"]
    sp = [x["l2_scalar_ns"] / x["l2_avx2_ns"] for x in res]
    dsp = [x["dot_scalar_ns"] / x["dot_avx2_ns"] for x in res]
    csp = [x["cos_scalar_ns"] / x["cos_avx2_ns"] for x in res]
    title = (f"AVX2 L2 kernel is {min(sp):.1f}x to {max(sp):.1f}x faster than the scalar loop "
             f"(d = {res[0]['dim']} to {res[-1]['dim']})")
    sub = (f"ns per distance call, scalar reference vs AVX2+FMA kernel, same binary (-O3 -mavx2 -mfma, no -ffast-math) | "
           f"256 cache-resident vectors per dimension, median of {r['trials']} trials, pinned to one CPU | "
           f"dot product: {min(dsp):.1f}x to {max(dsp):.1f}x, cosine: {min(csp):.1f}x to {max(csp):.1f}x")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(), "Vector dimension", "Speedup, scalar time / AVX2 time")
    xs = list(range(len(res)))
    bars = ax.bar(xs, sp, width=0.62, color=T["nexus"], zorder=3, label="L2 squared distance")
    for b in bars:
        b.set_linewidth(0)
    ax.set_xticks(xs, [str(x["dim"]) for x in res])
    ax.axhline(1, color=T["axis"], lw=1.2)
    fs = 12.5 if layout == "tall" else 11
    for i in {sp.index(min(sp)), sp.index(max(sp)), len(sp) - 1}:
        x = res[i]
        ax.text(i, sp[i], f"{sp[i]:.1f}x\n{x['l2_scalar_ns']:.0f} -> {x['l2_avx2_ns']:.1f} ns",
                ha="center", va="bottom", color=T["ink"], fontsize=fs - 0.5, linespacing=1.1)
    ax.set_ylim(0, max(sp) * 1.28)
    return C


@chart("memory_build")
def memory_chart(ms, theme, layout):
    m0 = ms[0]
    rows = [("Flat FP32", "exact", "sift_flat"), ("QuantizedFlat SQ8", "exact", "sift_qflat"),
            ("HNSW FP32", "nexus", "sift_hnsw"), ("HNSW-SQ8\n(+FP32 for re-rank)", "sq8", "sift_sq8"),
            ("hnswlib", "hnswlib", "sift_hnswlib")]
    data = [(lab, key, bd.build_stats(m0.result("recall_qps", t))) for lab, key, t in rows]
    data = [x for x in data if x[2]]
    if not data:
        return None
    by = {t: bd.build_stats(m0.result("recall_qps", t)) for _, _, t in rows}
    h, s, f, q = by.get("sift_hnsw"), by.get("sift_sq8"), by.get("sift_flat"), by.get("sift_qflat")
    title = "Memory per vector and build time at 1M vectors"
    if h and s and f and q:
        title = (f"SQ8 codes alone cut vector memory {f['bytes_per_vector'] / q['bytes_per_vector']:.1f}x, but HNSW-SQ8 also keeps FP32 "
                 f"copies for re-ranking: {s['bytes_per_vector']:,.0f} vs {h['bytes_per_vector']:,.0f} bytes per vector")
    sub = ("SIFT1M (1M x 128-d) | bytes per vector = process RSS growth during the build / 1M (includes graph links and allocator "
           "slack) | build time single thread, median of builds, M=16, efConstruction=200")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    if layout == "wide":
        r1, r2 = C.rect(0, 0, 0.5, 1, pad_l=1.55), C.rect(0.5, 0, 0.5, 1, pad_l=1.55)
    else:
        r1, r2 = C.rect(0, 0.5, 1, 0.5, pad_l=1.9), C.rect(0, 0, 1, 0.5, pad_l=1.9)
    ax = C.axes(r1, "Bytes per vector", "")
    labels = [d[0] for d in data]
    ys = list(range(len(data)))[::-1]
    fs = 12.5 if layout == "tall" else 11
    for y, (lab, key, st) in zip(ys, data):
        ax.barh(y, st["bytes_per_vector"], height=0.6, color=T[key], zorder=3)
        ax.text(st["bytes_per_vector"], y, f"  {st['bytes_per_vector']:,.0f} B", va="center", color=T["ink"], fontsize=fs)
    ax.set_yticks(ys, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(d[2]["bytes_per_vector"] for d in data) * 1.3)
    ax.axvline(512, color=T["muted"], lw=1, ls=(0, (3, 3)))
    ax.text(512, 1.01, "raw FP32 vector = 512 B", color=T["ink2"], fontsize=fs - 1, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax2 = C.axes(r2, "Build time, seconds (single thread)", "")
    for y, (lab, key, st) in zip(ys, data):
        ax2.barh(y, st["build_seconds"], height=0.6, color=T[key], zorder=3)
        rng = f" ({st['build_seconds_min']:.0f}-{st['build_seconds_max']:.0f})" if st["builds"] > 1 else ""
        ax2.text(st["build_seconds"], y, f"  {st['build_seconds']:,.1f} s{rng}", va="center", color=T["ink"], fontsize=fs)
    ax2.set_yticks(ys, labels)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, max(d[2]["build_seconds"] for d in data) * 1.35)
    return C


@chart("latency_at_recall")
def latency_chart(ms, theme, layout):
    m0 = ms[0]
    engines = [("Nexus HNSW", "nexus", bd.all_grid(m0, "sift", "nexus")),
               ("hnswlib", "hnswlib", bd.all_grid(m0, "sift", "hnswlib")),
               ("Nexus HNSW-SQ8", "sq8", bd.ann_points(m0.result("recall_qps", "sift_sq8")))]
    engines = [e for e in engines if e[2]]
    if not engines:
        return None
    sel = {(lab, t): bd.fastest_latency_at(pts, t) for lab, _, pts in engines for t in bd.RECALL_TARGETS}
    a, b = sel.get(("Nexus HNSW", 0.95)), sel.get(("hnswlib", 0.95))
    title = "Latency at fixed recall targets"
    if a and b:
        title = (f"At 95% recall: P50 {a.p50:,.0f} us / P99 {a.p99:,.0f} us for Nexus vs "
                 f"{b.p50:,.0f} / {b.p99:,.0f} us for hnswlib")
    sub = ("SIFT1M | k = 10 | for each target, the configuration (M, efConstruction, efSearch) with the lowest mean latency "
           "among measured points at or above the target | bar = P50, tick = P99 | single thread")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(), "Recall@10 target", "Latency per query, microseconds")
    n = len(engines)
    w = 0.8 / n
    fs = 12 if layout == "tall" else 10.5
    for ei, (lab, key, _) in enumerate(engines):
        for ti, t in enumerate(bd.RECALL_TARGETS):
            p = sel[(lab, t)]
            x = ti + (ei - (n - 1) / 2) * w
            if not p:
                ax.text(x, 0, "n/a", ha="center", va="bottom", color=T["muted"], fontsize=fs)
                continue
            ax.bar(x, p.p50, width=w * 0.86, color=T[key], zorder=3, label=lab if ti == 0 else None)
            ax.plot([x - w * 0.35, x + w * 0.35], [p.p99, p.p99], color=T["ink"], lw=2.2, zorder=4)
            ax.plot([x, x], [p.p50, p.p99], color=T["ink2"], lw=1, zorder=4)
            ax.text(x, p.p99, f"{p.p99:,.0f}", ha="center", va="bottom", color=T["ink2"], fontsize=fs - 1)
            ax.text(x, p.p50 / 2, f"{p.p50:,.0f}", ha="center", va="center", color="#ffffff" if key != "sq8" or theme == "dark" else "#0b0b0b",
                    fontsize=fs - 1, fontweight="bold")
    ax.set_xticks(range(len(bd.RECALL_TARGETS)), [f">= {t:.2f}" for t in bd.RECALL_TARGETS])
    ax.grid(axis="x", visible=False)
    C.legend(ax, loc="upper left")
    return C


@chart("prefetch_ablation")
def prefetch_chart(ms, theme, layout):
    m0 = ms[0]
    on = [bd.ann_points(m0.result("ablation", f"prefetch_default_{i}")) for i in range(10)]
    off = [bd.ann_points(m0.result("ablation", f"prefetch_noprefetch_{i}")) for i in range(10)]
    pairs = [(x, y) for x, y in zip(on, off) if x and y]
    if not pairs:
        return None
    on, off = [x for x, _ in pairs], [y for _, y in pairs]
    import statistics as stt
    efs = [p.ef for p in on[0]]
    ratio = []
    for i, ef in enumerate(efs):
        q_on = stt.mean(r[i].qps for r in on)
        q_off = stt.mean(r[i].qps for r in off)
        ratio.append((ef, q_on / q_off, q_on, q_off, [r[i].qps for r in on], [r[i].qps for r in off]))
    lo, hi = min(r[1] for r in ratio), max(r[1] for r in ratio)
    med = sorted(r[1] for r in ratio)[len(ratio) // 2]
    per_pair = [a / b for r in ratio for a, b in zip(r[4], r[5])]
    wins = sum(1 for v in per_pair if v > 1)
    if med > 1.02:
        title = (f"Software prefetch: median {(med - 1) * 100:+.0f}% search QPS (range {(lo - 1) * 100:+.0f}% to "
                 f"{(hi - 1) * 100:+.0f}% by efSearch); faster in {wins} of {len(per_pair)} run pairs")
    elif med < 0.98:
        title = f"Software prefetch slows search down: {(lo - 1) * 100:+.0f}% to {(hi - 1) * 100:+.0f}% QPS"
    else:
        title = f"Software prefetch has no consistent effect: {(lo - 1) * 100:+.0f}% to {(hi - 1) * 100:+.0f}% QPS"
    sub = ("SIFT1M | the same saved HNSW graph loaded by two builds that differ only in NEXUS_DISABLE_PREFETCH | "
           f"{len(on)} interleaved run pairs (on, off, on, off, ...), bar = ratio of means, dots = per-pair ratio | single thread")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(), "efSearch", "QPS with prefetch / QPS without")
    xs = list(range(len(ratio)))
    ax.bar(xs, [r[1] - 1 for r in ratio], bottom=1, width=0.6, color=T["nexus"], zorder=3)
    for x, r in zip(xs, ratio):
        runs = [a / b for a, b in zip(r[4], r[5])]
        ax.scatter([x] * len(runs), runs, s=36, color=T["ink2"], edgecolor=T["ring"], lw=1, zorder=5)
    fs = 12 if layout == "tall" else 11
    top = max(max(per_pair), hi)
    span = max(abs(min(per_pair) - 1), abs(top - 1), 0.02)
    for x, r in zip(xs, ratio):
        # one row of value labels above every bar and dot, so labels never collide with data
        ax.text(x, top + span * 0.12, f"{(r[1] - 1) * 100:+.1f}%\n{r[2]:,.0f} vs\n{r[3]:,.0f} QPS", ha="center",
                va="bottom", color=T["ink"], fontsize=fs - 0.5)
    ax.axhline(1, color=T["axis"], lw=1.4)
    ax.set_xticks(xs, [str(r[0]) for r in ratio])
    ax.set_ylim(1 - span * 1.25, top + span * 0.95)
    return C


@chart("rerank_sweep")
def rerank_chart(ms, theme, layout):
    m0 = ms[0]
    res = m0.result("ablation", "rerank_sweep")
    if not res:
        return None
    rrs = sorted({float(p["rerank"]) for b in res["builds"] for p in b["points"]})
    curves = {rr: bd.ann_points(res, rerank=rr) for rr in rrs}
    at95 = sorted(((bd.best_at(c, 0.95), rr) for rr, c in curves.items() if bd.best_at(c, 0.95)),
                  key=lambda x: -x[0].qps)
    title = "Re-rank factor trades throughput for recall"
    if len(at95) >= 2:
        b1, r1 = at95[0]
        close = [f"x{rr:g}" for bp, rr in at95[1:] if bp.qps >= 0.9 * b1.qps]
        title = (f"For 95% recall, re-rank factor x{r1:g} is fastest ({b1.qps:,.0f} QPS at efSearch {b1.ef})"
                 + (f"; {', '.join(close)} within 10%" if close else ""))
        min_ef = min(p.ef for c in curves.values() for p in c)
        over = [f"x{rr:g}" for rr, c in curves.items() if min(c, key=lambda p: p.ef).recall > 0.97]
        if over:
            title += f". Larger factors ({', '.join(over)}) already pass 0.97 recall at the smallest efSearch tested ({min_ef})"
    sub = ("SIFT1M | Nexus HNSWIndexSQ8, one saved graph (M=16, efConstruction=200), efSearch 16..256 per factor | the SQ8 "
           "traversal collects max(efSearch, k) x factor candidates, all re-scored with exact FP32 distances | single thread")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(), "Recall@10 (logit scale)", "Queries per second (log scale)")
    ax.set_yscale("log")
    ramp = T["ramp"]
    for i, rr in enumerate(rrs):
        c = curves[rr]
        col = ramp[min(i, len(ramp) - 1)]
        ax.plot([p.recall for p in c], [p.qps for p in c], color=col, lw=2.2, marker="o", ms=6, mec=T["ring"], mew=1.3,
                label=f"factor {rr:g}")
    ax.set_xscale("logit", one_half="0.5")
    hi = max(p.recall for c in curves.values() for p in c)
    lo = min(p.recall for c in curves.values() for p in c)
    ax.set_xlim(max(0.5, lo - 0.02), min(0.9995, 1 - (1 - hi) / 1.6))
    ticks = [t for t in (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(qps_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axvline(0.95, color=T["muted"], lw=1, ls=(0, (4, 4)))
    if at95:
        b1, r1 = at95[0]
        note(ax, C, b1.recall, b1.qps, f"x{r1:g}, efSearch {b1.ef}: {b1.qps:,.0f} QPS\nat recall {b1.recall:.3f}", (-150, -90))
    C.legend(ax, loc="lower left", ncol=2)
    better_arrow(ax, C, where=(0.88, 0.80))
    return C


@chart("write_path")
def write_chart(ms, theme, layout):
    m0 = ms[0]
    rows = []
    for idx in ("flat", "hnsw"):
        d = m0.result("write_path", f"default_{idx}")
        n = m0.result("write_path", f"nofsync_{idx}")
        if not d or not n:
            continue
        rows.append((idx, d["index_only"]["inserts_per_second"], n["engine_with_wal"]["inserts_per_second"],
                     d["engine_with_wal"]["inserts_per_second"], d, n))
    if not rows:
        return None
    h = next((r for r in rows if r[0] == "hnsw"), rows[0])
    fs_lat = h[4]["engine_with_wal"]["insert_latency"]["p50_us"]
    title = (f"Durability costs one fsync per insert: {h[3]:,.0f} inserts/s with fsync vs {h[2]:,.0f} without "
             f"(HNSW, P50 {fs_lat / 1000:.1f} ms per insert)")
    sub = (f"10,000 SIFT vectors, single thread | index only = no WAL | WAL without fsync = NEXUS_WAL_NO_FSYNC build (not durable) | "
           f"WAL + fsync = default engine | fsync cost is platform-specific: {m0.runtime} guest, "
           f"{m0.env.get('work_dir_filesystem', '?').split(' on ')[0]} on a virtual disk")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.axes(C.rect(pad_l=2.75 if layout == "wide" else 3.0), "Inserts per second (log scale)", "")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(qps_fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    labels, ys = [], []
    y = 0
    fs = 12.5 if layout == "tall" else 11
    cats = [("index only (no WAL)", 1, T["exact"] if theme == "light" else T["ink2"]),
            ("WAL, no fsync", 2, T["axis"] if theme == "light" else T["muted"]),
            ("WAL + fsync (default)", 3, T["nexus"])]
    for r in rows:
        for lab, k, col in cats:
            ax.barh(y, r[k], height=0.7, color=col if k != 3 else T["nexus"], zorder=3)
            ax.text(r[k], y, f"  {r[k]:,.0f}/s", va="center", color=T["ink"], fontsize=fs)
            labels.append(f"{'FlatIndex' if r[0] == 'flat' else 'HNSWIndex'}: {lab}")
            ys.append(y)
            y -= 1
        y -= 0.6
    ax.set_yticks(ys, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(r[3] for r in rows) / 3, max(r[1] for r in rows) * 12)
    return C


@chart("architecture")
def arch_chart(ms, theme, layout):
    title = "How Nexus answers a query and survives a crash"
    sub = ("Writes: WAL record (CRC32-C, fsync) before the in-memory index; checkpoints snapshot the index atomically and "
           "truncate the WAL | Reads: greedy descent through the HNSW layers, beam search on layer 0, optional SQ8 traversal "
           "with exact FP32 re-ranking")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    ax = C.fig.add_axes(C.rect(pad_l=0, pad_b=0), facecolor=T["surface"])
    ax.axis("off")
    W, H = (16, 7.2) if layout == "wide" else (10, 13)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    fs = 12 if layout == "wide" else 13

    def box(x, y, w, h, text, col, sub=None):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                            fc=T["surface"], ec=col, lw=2.2, zorder=3))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text, ha="center", va="center", color=T["ink"],
                fontsize=fs, fontweight="bold", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", color=T["ink2"], fontsize=fs - 2, zorder=4)

    def arrow(x0, y0, x1, y1, text=None, col=None, off=(0, 0.18)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col or T["muted"], lw=1.8, shrinkA=2, shrinkB=2), zorder=2)
        if text:
            ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center", va="bottom", color=T["ink2"],
                    fontsize=fs - 2)

    def layers(cx, cy, w, h):
        import random
        rnd = random.Random(3)
        levels = [(cy + 2 * h, 4), (cy + h, 9), (cy, 22)]
        pts = {}
        for li, (y, n) in enumerate(levels):
            ax.add_patch(patches.Ellipse((cx, y), w, h * 0.55, fc=T["grid"], ec=T["axis"], lw=1, alpha=0.6, zorder=1))
            xs = [cx - w * 0.42 + w * 0.84 * (i + rnd.uniform(0.1, 0.9)) / n for i in range(n)]
            ys = [y + rnd.uniform(-h * 0.12, h * 0.12) for _ in xs]
            pts[li] = list(zip(xs, ys))
            for a in range(len(xs)):
                for b in (a + 1, a + 2):
                    if b < len(xs) and (b == a + 1 or rnd.random() < 0.4):
                        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=T["axis"], lw=1, zorder=2)
            ax.scatter(xs, ys, s=28, color=T["nexus"], zorder=3, edgecolor=T["ring"], lw=0.8)
            ax.text(cx + w / 2 + 0.1, y, f"layer {2 - li}", color=T["ink2"], fontsize=fs - 2, va="center")
        path = [pts[0][1], pts[1][4], pts[2][11], pts[2][12], pts[2][13]]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=T["hnswlib"] if False else T["ink"], lw=1.6), zorder=4)
        return pts

    if layout == "wide":
        box(0.2, 5.6, 2.6, 1.0, "insert(id, v)", T["ink2"])
        box(3.6, 5.6, 3.2, 1.0, "WAL append", T["nexus"], "CRC32-C frame + fsync")
        box(7.6, 5.6, 3.0, 1.0, "HNSW add", T["nexus"], "exclusive lock")
        box(11.4, 5.6, 4.3, 1.0, "checkpoint", T["nexus"], "atomic snapshot, then WAL cleared")
        arrow(2.8, 6.1, 3.6, 6.1)
        arrow(6.8, 6.1, 7.6, 6.1)
        ax.text(9.1, 5.35, "acknowledged after both steps", ha="center", va="top", color=T["ink2"], fontsize=fs - 2)
        arrow(10.6, 6.1, 11.4, 6.1)
        box(0.2, 3.9, 2.6, 1.0, "restart", T["ink2"])
        box(3.6, 3.9, 3.2, 1.0, "WAL replay", T["good"], "verify CRC, cut torn tail")
        arrow(2.8, 4.4, 3.6, 4.4)
        arrow(5.2, 4.9, 5.2, 5.6)
        arrow(6.8, 4.4, 8.6, 5.6, "re-insert")
        box(0.2, 1.3, 2.6, 1.0, "search(q, k)", T["ink2"])
        layers(5.6, 0.7, 4.6, 1.15)
        arrow(2.8, 1.8, 3.3, 2.6)
        box(9.7, 1.3, 2.8, 1.0, "SQ8 traversal", T["sq8"], "1-byte codes")
        box(13.0, 1.3, 2.8, 1.0, "exact re-rank", T["sq8"], "FP32, ef x factor")
        arrow(8.9, 0.75, 9.7, 1.55)
        ax.text(9.3, 0.55, "optional (HNSWIndexSQ8)", ha="left", va="top", color=T["ink2"], fontsize=fs - 2)
        arrow(12.5, 1.8, 13.0, 1.8)
        ax.text(5.6, 3.5, "greedy descent, then beam search (efSearch) on layer 0", ha="center", color=T["ink2"], fontsize=fs - 1)
    else:
        box(0.3, 11.4, 4.2, 1.0, "insert(id, v)", T["ink2"])
        box(5.4, 11.4, 4.3, 1.0, "WAL append", T["nexus"], "CRC32-C frame + fsync")
        box(5.4, 9.8, 4.3, 1.0, "HNSW add", T["nexus"], "exclusive lock")
        box(5.4, 8.2, 4.3, 1.0, "checkpoint", T["nexus"], "atomic snapshot, WAL cleared")
        arrow(4.5, 11.9, 5.4, 11.9)
        arrow(7.55, 11.4, 7.55, 10.8)
        arrow(7.55, 9.8, 7.55, 9.2)
        box(0.3, 9.8, 4.2, 1.0, "restart", T["ink2"])
        box(0.3, 8.2, 4.2, 1.0, "WAL replay", T["good"], "verify CRC, cut torn tail")
        arrow(2.4, 9.8, 2.4, 9.2)
        arrow(4.5, 8.7, 5.4, 10.1, "re-insert", off=(-0.5, 0.1))
        box(0.3, 6.2, 4.2, 1.0, "search(q, k)", T["ink2"])
        layers(5.0, 2.8, 6.5, 1.25)
        arrow(2.4, 6.2, 3.0, 5.5)
        box(0.3, 0.3, 4.2, 1.0, "SQ8 traversal", T["sq8"], "1-byte codes")
        box(5.4, 0.3, 4.3, 1.0, "exact re-rank", T["sq8"], "FP32, ef x factor")
        arrow(4.5, 0.8, 5.4, 0.8)
        arrow(4.0, 2.4, 2.4, 1.3, "optional", off=(-0.6, 0))
    return C


@chart("hero")
def hero_chart(ms, theme, layout):
    m0 = ms[0]
    dc = bd.ann_points(m0.result("distance_computations", "sift_nexus_hnsw"))
    kc = m0.result("crash_recovery", "kill_cycles")
    fx = fix_data(m0)
    if not dc or not kc or not fx:
        return None
    n_base = m0.result("distance_computations", "sift_nexus_hnsw")["dataset"]["base_rows"]
    a = bd.min_dist_at(dc, 0.95)
    s = kc["summary"]
    ef = 64 if 64 in fx["after"] else sorted(fx["after"])[0]
    bmin, amin = min(fx["before"][ef]), min(fx["after"][ef])
    nx = bd.pareto(bd.all_grid(m0, "sift", "nexus"))
    p95 = bd.best_at(nx, 0.95)
    title = (f"Nexus: a C++20 vector search engine, measured: {n_base / a.dist:,.0f}x fewer distance computations than brute force, "
             f"{s['passed']}/{s['cycles']} crash recoveries verified")
    sub = (f"SIFT1M / SIFT-10K | single thread | {p95.qps:,.0f} QPS at recall {p95.recall:.3f} on one core | "
           f"worst-build recall after the graph fix {amin:.3f} (was {bmin:.2f})")
    C = Canvas(theme, layout, title, sub, footer(ms))
    T = C.T
    tall = layout == "tall"
    tiles = [
        (f"{n_base / a.dist:,.0f}x", f"fewer distance computations than a\nfull scan at {a.recall:.0%} recall (SIFT1M)", "dist"),
        (f"{amin:.3f}", f"worst-build Recall@10 after the\nneighbor-selection fix (was {bmin:.2f})", "fix"),
        (f"{s['passed']}/{s['cycles']}", f"SIGKILL crash/restart cycles with\n{s['lost_acked_total']} acknowledged inserts lost", "crash"),
    ]
    for i, (big, label, kind) in enumerate(tiles):
        if tall:
            r = C.rect(0, 1 - (i + 1) / 3, 1, 1 / 3, pad_l=0.1, pad_b=0.25, pad_t=0.1)
        else:
            r = C.rect(i / 3, 0, 1 / 3, 1, pad_l=0.15, pad_b=0.1, pad_r=0.25)
        bg = C.fig.add_axes(r, facecolor=T["surface"])
        bg.axis("off")
        bg.add_patch(patches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                            fc=T["surface"], ec=T["grid"], lw=1.2, transform=bg.transAxes))
        if tall:
            bg.text(0.05, 0.62, big, fontsize=58, fontweight="bold", color=T["ink"], transform=bg.transAxes, va="center")
            bg.text(0.05, 0.25, label, fontsize=16, color=T["ink2"], transform=bg.transAxes, va="center")
            ir = [r[0] + r[2] * 0.56, r[1] + r[3] * 0.12, r[2] * 0.4, r[3] * 0.76]
        else:
            bg.text(0.06, 0.88, big, fontsize=40, fontweight="bold", color=T["ink"], transform=bg.transAxes, va="center")
            bg.text(0.06, 0.70, label, fontsize=12, color=T["ink2"], transform=bg.transAxes, va="center")
            ir = [r[0] + r[2] * 0.12, r[1] + r[3] * 0.08, r[2] * 0.8, r[3] * 0.48]
        ax = C.fig.add_axes(ir, facecolor=T["surface"])
        if kind == "dist":
            ax.bar([0, 1], [n_base, a.dist], color=[T["exact"], T["nexus"]], width=0.6)
            ax.set_yscale("log")
            ax.set_xticks([0, 1], ["full scan", "HNSW"])
            ax.yaxis.set_major_formatter(FuncFormatter(qps_fmt))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif kind == "fix":
            import random
            rnd = random.Random(1)
            for j, (v, col) in enumerate((("before", T["exact"]), ("after", T["nexus"]))):
                vals = fx[v][ef]
                ax.scatter([j + rnd.uniform(-0.12, 0.12) for _ in vals], vals, s=40, color=col, edgecolor=T["ring"], lw=1)
            ax.set_xticks([0, 1], ["before", "after"])
            ax.set_xlim(-0.6, 1.6)
        else:
            cyc = kc["cycles"]
            cols = 25
            for k, c in enumerate(cyc):
                ax.add_patch(patches.Rectangle((k % cols + 0.1, -(k // cols) + 0.1), 0.8, 0.8,
                                               color=T["good"] if c["ok"] else T["critical"], lw=0))
            ax.set_xlim(0, cols)
            ax.set_ylim(-(len(cyc) // cols), 1)
            ax.set_aspect("equal")
            ax.axis("off")
        if kind != "crash":
            ax.tick_params(colors=T["muted"], labelcolor=T["ink2"], labelsize=11 if not tall else 13, length=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(T["axis"])
            ax.grid(True, axis="y", color=T["grid"], lw=0.8)
            ax.set_axisbelow(True)
    return C


# ----------------------------------------------------------------------------- main
def render(names, out: Path, ms) -> dict[str, list[Path]]:
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name in names:
        f = CHARTS[name]
        paths = []
        for theme in ("light", "dark"):
            for layout in ("wide", "tall"):
                try:
                    C = f(ms, theme, layout)
                except Exception as e:  # noqa: BLE001
                    print(f"  {name} {theme} {layout}: FAILED {type(e).__name__}: {e}")
                    C = None
                if C is None:
                    continue
                paths += C.save(out, name, theme)
        if paths:
            made[name] = paths
            print(f"  {name}: {len(paths)} files")
        else:
            print(f"  {name}: skipped (data missing)")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bd.REPO / "docs" / "figures"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    ms = bd.load_machines()
    if not ms:
        print("no results found under results/")
        return 1
    print("machines:", ", ".join(f"{m.machine_id} ({m.date})" for m in ms))
    render(a.only or list(CHARTS), Path(a.out), ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
