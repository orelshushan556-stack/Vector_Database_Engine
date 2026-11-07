#!/usr/bin/env python3
"""Run the Nexus benchmark suite end to end on this machine (Windows, Linux, WSL2, Docker).

  python scripts/run_all.py --machine-id <slug> --data-dir data

Steps: detect hardware -> build the benchmark variants -> run the suites one at a time
(never two timing-sensitive runs concurrently) -> write raw JSON under
results/<machine_id>/<date>/<suite>/. Every JSON file carries an `env` block (CPU, OS,
compiler, flags, git commit, dataset checksums, exact command line).

Use --resume to skip tasks whose JSON already exists (e.g. after a crash or reboot).
Only the standard library is required; datasets come from scripts/download_data.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"
EXE = ".exe" if IS_WIN else ""

# The pre-fix revision: parent of 2bf962f ("restore graph connectivity"), which replaced
# farthest-neighbor eviction with the HNSW Algorithm-4 heuristic (README section 2.2).
FIX_COMMIT = "2bf962f"
PREFIX_COMMIT = "2bf962f^"

DATASETS = {
    "siftsmall": dict(base="siftsmall_base.fvecs", query="siftsmall_query.fvecs",
                      gt="siftsmall_groundtruth.ivecs", metric="l2"),
    "sift": dict(base="sift_base.fvecs", query="sift_query.fvecs", gt="sift_groundtruth.ivecs",
                 metric="l2", learn="sift_learn.fvecs"),
    "glove100": dict(base="glove100_base.fvecs", query="glove100_query.fvecs",
                     gt="glove100_groundtruth.ivecs", metric="ip"),
}

# Order: short suites first, then the SIFT1M/GloVe suites that depend on saved snapshots.
ALL_SUITES = ["sanity_start", "before_after_fix", "crash_recovery", "write_path", "recall_qps",
              "distance_computations", "build_memory", "concurrency", "ablation", "vs_hnswlib", "sanity_end"]


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str] | str, check=True, capture=True, cwd=None, env=None, timeout=None) -> str:
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, timeout=timeout,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None,
                       text=True, errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{(r.stdout or '')[-3000:]}")
    return r.stdout or ""


def try_sh(cmd, **kw) -> str:
    try:
        return sh(cmd, check=False, **kw).strip()
    except Exception:  # noqa: BLE001
        return ""


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(r\)|\(tm\)|cpu|processor|with radeon.*|\d+-core", " ", s)
    s = re.sub(r"\b\d+(th|nd|rd|st) gen\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"-+", "-", s)
    return s.replace("intel-core-", "intel-").replace("amd-ryzen-", "amd-ryzen-")


# ----------------------------------------------------------------------------------------
# hardware / environment detection
# ----------------------------------------------------------------------------------------
def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def detect_env(data_dir: Path, work_dir: Path) -> dict:
    env: dict = {"python": platform.python_version(), "platform": platform.platform()}
    proc_version = read("/proc/version")
    env["runtime"] = ("wsl2" if "microsoft" in proc_version.lower() else
                      "docker" if Path("/.dockerenv").exists() else "native")
    env["os"] = platform.system()
    if env["os"] == "Linux":
        osr = read("/etc/os-release")
        m = re.search(r'PRETTY_NAME="([^"]+)"', osr)
        env["os_release"] = m.group(1) if m else ""
        env["kernel"] = platform.release()
    else:
        env["os_release"] = platform.platform()

    cpu: dict = {}
    if env["os"] == "Linux":
        info = read("/proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", info)
        cpu["model"] = m.group(1).strip() if m else platform.processor()
        m = re.search(r"vendor_id\s*:\s*(.+)", info)
        cpu["vendor"] = m.group(1).strip() if m else ""
        flags = set((re.search(r"flags\s*:\s*(.+)", info) or [None, ""])[1].split())
        cpu["logical_cpus"] = os.cpu_count()
        cores = set()
        for blk in info.split("\n\n"):
            p = re.search(r"physical id\s*:\s*(\d+)", blk)
            c = re.search(r"core id\s*:\s*(\d+)", blk)
            if p and c:
                cores.add((p.group(1), c.group(1)))
        cpu["physical_cores_visible"] = len(cores) or None
        p_cpus = read("/sys/devices/cpu_core/cpus")
        e_cpus = read("/sys/devices/cpu_atom/cpus")
        cpu["hybrid_visible"] = bool(p_cpus and e_cpus)
        if cpu["hybrid_visible"]:
            cpu["p_core_cpus"] = p_cpus
            cpu["e_core_cpus"] = e_cpus
        mem = re.search(r"MemTotal:\s*(\d+)", read("/proc/meminfo"))
        env["ram_bytes"] = int(mem.group(1)) * 1024 if mem else None
    else:
        cpu["model"] = try_sh(["powershell", "-NoProfile", "-Command",
                               "(Get-CimInstance Win32_Processor).Name"]) or platform.processor()
        cpu["vendor"] = "GenuineIntel" if "intel" in cpu["model"].lower() else "AuthenticAMD" if "amd" in cpu["model"].lower() else ""
        cpu["logical_cpus"] = os.cpu_count()
        cpu["physical_cores_visible"] = int(try_sh(["powershell", "-NoProfile", "-Command",
                                                    "(Get-CimInstance Win32_Processor).NumberOfCores"]) or 0) or None
        flags = set()
        env["ram_bytes"] = int(try_sh(["powershell", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]) or 0) or None
        eff = windows_efficiency_classes()
        if eff:
            cpu["efficiency_class_by_cpu"] = eff
            cpu["hybrid_visible"] = len(set(eff)) > 1
    cpu["flags_checked"] = {f: (f in flags) if flags else None for f in ("avx2", "fma", "sse4_2", "avx512f")}
    env["cpu"] = cpu

    # Host details when running under WSL2 (the hypervisor hides the hybrid topology).
    if env["runtime"] == "wsl2":
        ps = shutil.which("powershell.exe")
        if ps:
            q = lambda c: try_sh([ps, "-NoProfile", "-Command", c], timeout=60)  # noqa: E731
            env["host"] = {
                "cpu_model": q("(Get-CimInstance Win32_Processor).Name"),
                "physical_cores": q("(Get-CimInstance Win32_Processor).NumberOfCores"),
                "logical_processors": q("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"),
                "ram_bytes": q("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
                "os": q("(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"),
                "power_scheme": q("powercfg /getactivescheme"),
                "disks": q("Get-PhysicalDisk | ForEach-Object { $_.FriendlyName + ' ' + $_.MediaType + ' ' + $_.BusType }"),
            }
    elif IS_WIN:
        env["power_scheme"] = try_sh(["powercfg", "/getactivescheme"])

    env["work_dir"] = str(work_dir)
    env["work_dir_filesystem"] = filesystem_of(work_dir)
    env["data_dir"] = str(data_dir)
    env["git_commit"] = try_sh(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    env["git_dirty"] = bool(try_sh(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "include", "benchmarks", "CMakeLists.txt"]))
    env["governor"] = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or None
    return env


def windows_efficiency_classes() -> list[int] | None:
    """EfficiencyClass per logical processor (higher = faster core) via GetSystemCpuSetInformation."""
    if not IS_WIN:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        n = wintypes.ULONG(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
            return None
        out, off = {}, 0
        while off < n.value:
            size = int.from_bytes(bytes(buf[off + 4:off + 8]), "little")
            # SYSTEM_CPU_SET_INFORMATION.CpuSet: Id(4) Group(2) LogicalProcessorIndex(1) CoreIndex(1)
            # LastLevelCacheIndex(1) NumaNodeIndex(1) EfficiencyClass(1)
            lp = buf[off + 8 + 6]
            eff = buf[off + 8 + 10]
            out[lp] = eff
            off += size
        return [out[i] for i in sorted(out)]
    except Exception:  # noqa: BLE001
        return None


def filesystem_of(path: Path) -> str:
    path = path.resolve()
    if IS_WIN:
        drive = path.drive.rstrip(":")
        return try_sh(["powershell", "-NoProfile", "-Command", f"(Get-Volume -DriveLetter {drive}).FileSystemType"]) or "unknown"
    best, fs = "", "unknown"
    for line in read("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) >= 3 and str(path).startswith(parts[1]) and len(parts[1]) > len(best):
            best, fs = parts[1], f"{parts[2]} on {parts[0]} (mount {parts[1]})"
    return fs


def choose_pin_cpu(env: dict) -> tuple[int, str]:
    cpu = env["cpu"]
    if cpu.get("hybrid_visible") and cpu.get("p_core_cpus"):
        first = int(re.split(r"[-,]", cpu["p_core_cpus"])[0])
        c = first + 2  # skip cpu0 (interrupt-heavy); stay on a P-core
        return c, f"P-core logical cpu {c} (Linux hybrid sysfs)"
    if cpu.get("efficiency_class_by_cpu"):
        eff = cpu["efficiency_class_by_cpu"]
        best = max(eff)
        cands = [i for i, e in enumerate(eff) if e == best]
        c = cands[2] if len(cands) > 2 else cands[0]
        return c, f"logical processor {c}, EfficiencyClass {best} (P-core)"
    # Non-hybrid or hidden topology: pick logical cpu 2 if it is the first SMT sibling.
    sib = read("/sys/devices/system/cpu/cpu2/topology/thread_siblings_list")
    c = 2 if (not sib or re.split(r"[-,]", sib)[0] == "2") else 0
    note = f"logical cpu {c}"
    if env["runtime"] == "wsl2":
        note += (" (WSL2 vCPU: Hyper-V schedules vCPUs on host logical processors, so the physical core"
                 " and P/E type cannot be pinned from inside the VM)")
    return c, note


def dataset_manifest(data_dir: Path, name: str) -> dict | None:
    p = data_dir / name / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------
VARIANTS = {
    "default": ["-DNEXUS_BENCH_HNSWLIB=ON", "-DNEXUS_BUILD_TESTS=OFF"],
    "count": ["-DNEXUS_COUNT_DISTANCES=ON", "-DNEXUS_BENCH_HNSWLIB=ON", "-DNEXUS_BUILD_TESTS=OFF"],
    "noprefetch": ["-DNEXUS_DISABLE_PREFETCH=ON", "-DNEXUS_BUILD_TESTS=OFF"],
    "nofsync": ["-DNEXUS_WAL_NO_FSYNC=ON", "-DNEXUS_BUILD_TESTS=OFF"],
}


def cmake_build(src: Path, build: Path, opts: list[str]) -> None:
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    sh(["cmake", "-S", str(src), "-B", str(build), *gen, "-DCMAKE_BUILD_TYPE=Release", *opts])
    sh(["cmake", "--build", str(build), "--parallel"])


def build_all(build_root: Path) -> dict:
    out = {}
    for name, opts in VARIANTS.items():
        log(f"build variant {name}")
        cmake_build(REPO, build_root / name, opts)
        out[name] = {"options": opts}
    cache = (build_root / "default" / "CMakeCache.txt").read_text(errors="replace")
    m = re.search(r"CMAKE_CXX_COMPILER:\w+=(.+)", cache)
    compiler = m.group(1).strip() if m else ""
    out["compiler_path"] = compiler
    out["compiler_version"] = try_sh([compiler, "--version"]).splitlines()[0] if compiler else ""
    flags = re.search(r"add_compile_options\(([^)]*-O3[^)]*)\)", (REPO / "CMakeLists.txt").read_text(encoding="utf-8-sig"))
    out["cxx_flags"] = (flags.group(1).strip() if flags else "") + " (CMAKE_BUILD_TYPE=Release: -O3 -DNDEBUG)"
    return out


def bin_path(build_root: Path, variant: str, name: str) -> str:
    return str(build_root / variant / f"{name}{EXE}")


# ----------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------
class Runner:
    def __init__(self, args, env, build_info):
        self.args = args
        self.env = env
        self.build_info = build_info
        self.data = Path(args.data_dir).resolve()
        self.work = Path(args.work_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.build_root = Path(args.build_root).resolve()
        self.out_root = REPO / "results" / args.machine_id / args.date
        self.pin = env["pinning"]["cpu"]
        self.snap = self.work / "snapshots"
        self.snap.mkdir(exist_ok=True)

    def ds(self, name: str) -> dict:
        d = DATASETS[name]
        root = self.data / name
        r = {k: str(root / v) for k, v in d.items() if k != "metric"}
        r["metric"] = d["metric"]
        r["available"] = all((root / d[k]).exists() for k in ("base", "query", "gt"))
        return r

    def out_path(self, suite: str, task: str) -> Path:
        return self.out_root / suite / f"{task}.json"

    def done(self, suite: str, task: str) -> bool:
        return self.args.resume and self.out_path(suite, task).exists()

    def save(self, suite: str, task: str, cmd, result, extra: dict | None = None) -> None:
        p = self.out_path(suite, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"suite": suite, "task": task, "machine_id": self.args.machine_id,
               "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "command": cmd, "env": self.env, "build": self.build_info}
        if extra:
            doc.update(extra)
        doc["result"] = result
        p.write_text(json.dumps(doc, indent=1))
        log(f"  wrote {p.relative_to(REPO)}")

    def run_json(self, suite: str, task: str, cmd: list[str], extra: dict | None = None, timeout=None) -> dict | None:
        if self.done(suite, task):
            log(f"skip {suite}/{task} (exists)")
            return json.loads(self.out_path(suite, task).read_text())["result"]
        tmp = self.work / f"{suite}__{task}.tmp.json"
        tmp.unlink(missing_ok=True)
        full = cmd + ["--out", str(tmp)]
        log(f"run {suite}/{task}: {' '.join(Path(full[0]).name if i == 0 else c for i, c in enumerate(full))}")
        logf = self.work / f"{suite}__{task}.log"
        t0 = time.time()
        with open(logf, "w") as lf:
            r = subprocess.run(full, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        if r.returncode != 0 or not tmp.exists():
            log(f"  FAILED rc={r.returncode}; tail of log:\n{logf.read_text(errors='replace')[-2000:]}")
            self.record_gap(suite, task, f"exit code {r.returncode}")
            return None
        result = json.loads(tmp.read_text())
        tmp.unlink()
        x = dict(extra or {})
        x["wall_seconds"] = round(time.time() - t0, 3)
        self.save(suite, task, full, result, x)
        return result

    def record_gap(self, suite: str, task: str, reason: str) -> None:
        p = self.out_root / "gaps.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        gaps = json.loads(p.read_text()) if p.exists() else []
        gaps.append({"suite": suite, "task": task, "reason": reason,
                     "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        p.write_text(json.dumps(gaps, indent=1))

    def ds_extra(self, name: str) -> dict:
        return {"dataset": name, "dataset_manifest": dataset_manifest(self.data, name)}

    # ------------------------------------------------------------------ ANN building block
    def ann(self, suite, task, engine, dsname, index="hnsw", M=16, efc=200, efs=None, reranks=None,
            builds=1, repeats=3, variant="default", save=None, load=None, max_queries=None, pin=True):
        d = self.ds(dsname)
        if not d["available"]:
            self.record_gap(suite, task, f"dataset {dsname} not available")
            return None
        exe = bin_path(self.build_root, variant, "nexus_bench" if engine == "nexus" else "hnswlib_bench")
        if not Path(exe).exists():
            self.record_gap(suite, task, f"binary missing: {exe}")
            return None
        cmd = [exe, "ann", "--base", d["base"], "--query", d["query"], "--gt", d["gt"], "--metric", d["metric"],
               "--M", str(M), "--efc", str(efc), "--k", "10", "--builds", str(builds), "--repeats", str(repeats)]
        if engine == "nexus":
            cmd += ["--index", index]
        if efs:
            cmd += ["--ef", ",".join(map(str, efs))]
        if reranks:
            cmd += ["--rerank", ",".join(map(str, reranks))]
        if save:
            cmd += ["--save", str(save)]
        if load:
            cmd += ["--load", str(load)]
        if max_queries:
            cmd += ["--max-queries", str(max_queries)]
        if pin:
            cmd += ["--pin", str(self.pin)]
        extra = self.ds_extra(dsname)
        extra.update({"engine": engine, "index": index if engine == "nexus" else "hnswlib", "M": M, "efc": efc,
                      "variant": variant, "builds": builds, "repeats": repeats})
        return self.run_json(suite, task, cmd, extra)

    def snapshot(self, engine, dsname, index, M, efc) -> Path:
        ext = "hnswlib" if engine == "hnswlib" else "nexus"
        return self.snap / f"{dsname}_{engine}_{index}_M{M}_efc{efc}.{ext}"


# ----------------------------------------------------------------------------------------
# suites
# ----------------------------------------------------------------------------------------
EF_SMALL = [10, 16, 24, 32, 48, 64, 100, 150, 200]
EF_LARGE = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]


def suite_sanity(r: Runner, tag: str) -> None:
    """Fixed-graph drift probe: build (once) and save a SIFT-10K graph, then time it at a fixed ef."""
    snap = r.snap / "sanity_siftsmall_hnsw.nexus"
    if tag == "start" or not snap.exists():
        if not snap.exists():
            r.ann(f"sanity_{tag}", "build", "nexus", "siftsmall", save=snap, efs=[64], repeats=1)
    for i in range(3):
        r.ann(f"sanity_{tag}", f"probe_{i}", "nexus", "siftsmall", load=snap, efs=[16, 64], repeats=50)


def suite_recall_qps(r: Runner, quick: bool, cfg: dict) -> None:
    s = "recall_qps"
    if quick:
        r.ann(s, "siftsmall_hnsw", "nexus", "siftsmall", builds=1, efs=[10, 32, 64], repeats=3)
        return
    # SIFT-10K: >= 5 independent builds per index
    r.ann(s, "siftsmall_flat", "nexus", "siftsmall", index="flat", repeats=20)
    r.ann(s, "siftsmall_qflat", "nexus", "siftsmall", index="qflat", repeats=20)
    r.ann(s, "siftsmall_hnsw", "nexus", "siftsmall", builds=5, efs=EF_SMALL, repeats=20)
    r.ann(s, "siftsmall_sq8", "nexus", "siftsmall", index="sq8", builds=5, efs=EF_SMALL, reranks=[1, 2, 5], repeats=20)
    r.ann(s, "siftsmall_hnswlib", "hnswlib", "siftsmall", builds=5, efs=EF_SMALL, repeats=20)
    # SIFT1M: >= 3 independent builds per graph index; build 0 saved for the load-based suites.
    M, efc = cfg["M"], cfg["efc"]
    r.ann(s, "sift_flat", "nexus", "sift", index="flat", repeats=1, max_queries=1000)
    r.ann(s, "sift_qflat", "nexus", "sift", index="qflat", repeats=1, max_queries=1000)
    r.ann(s, "sift_hnsw", "nexus", "sift", M=M, efc=efc, builds=3, efs=EF_LARGE, repeats=2,
          save=r.snapshot("nexus", "sift", "hnsw", M, efc))
    r.ann(s, "sift_sq8", "nexus", "sift", index="sq8", M=M, efc=efc, builds=3, efs=EF_LARGE, reranks=[1, 2, 5],
          repeats=1, save=r.snapshot("nexus", "sift", "sq8", M, efc))
    r.ann(s, "sift_hnswlib", "hnswlib", "sift", M=M, efc=efc, builds=3, efs=EF_LARGE, repeats=2,
          save=r.snapshot("hnswlib", "sift", "hnsw", M, efc))
    # GloVe-100 (angular via inner product on normalized vectors)
    gM, gefc = cfg["glove_M"], cfg["glove_efc"]
    gefs = EF_LARGE + cfg.get("ef_glove_extra", [])
    r.ann(s, "glove100_flat", "nexus", "glove100", index="flat", repeats=1, max_queries=1000)
    r.ann(s, "glove100_hnsw", "nexus", "glove100", M=gM, efc=gefc, builds=1, efs=gefs, repeats=1,
          save=r.snapshot("nexus", "glove100", "hnsw", gM, gefc))
    r.ann(s, "glove100_hnswlib", "hnswlib", "glove100", M=gM, efc=gefc, builds=1, efs=gefs, repeats=1,
          save=r.snapshot("hnswlib", "glove100", "hnsw", gM, gefc))


def suite_distance_computations(r: Runner, cfg: dict) -> None:
    """Same saved graphs, loaded by the counting builds: counts reflect search only."""
    s = "distance_computations"
    for dsname, M, efc in (("sift", cfg["M"], cfg["efc"]), ("glove100", cfg["glove_M"], cfg["glove_efc"])):
        for engine, index in (("nexus", "hnsw"), ("nexus", "sq8"), ("hnswlib", "hnsw")):
            if dsname == "glove100" and index == "sq8":
                continue
            snap = r.snapshot(engine, dsname, index, M, efc)
            if not snap.exists():
                r.record_gap(s, f"{dsname}_{engine}_{index}", f"snapshot missing: {snap.name}")
                continue
            r.ann(s, f"{dsname}_{engine}_{index}", engine, dsname, index=index, M=M, efc=efc, load=snap,
                  efs=EF_LARGE + (cfg.get("ef_glove_extra", []) if dsname == "glove100" else []), reranks=[2] if index == "sq8" else None, repeats=0, variant="count")
    # SIFT-10K with fresh builds (cheap): exact scan cost = base size, for reference
    r.ann(s, "siftsmall_nexus_hnsw", "nexus", "siftsmall", builds=1, efs=EF_SMALL, repeats=0, variant="count")
    r.ann(s, "siftsmall_hnswlib_hnsw", "hnswlib", "siftsmall", builds=1, efs=EF_SMALL, repeats=0, variant="count")


def suite_vs_hnswlib(r: Runner, cfg: dict) -> None:
    """Tuning grid for both engines: every (M, efConstruction) pair, one build each, ef sweep.
    The comparison picks, per recall target, the best measured QPS of each engine."""
    s = "vs_hnswlib"
    for dsname, grid in (("sift", cfg["grid_sift"]), ("glove100", cfg["grid_glove"])):
        for M, efc in grid:
            for engine in ("nexus", "hnswlib"):
                task = f"{dsname}_{engine}_M{M}_efc{efc}"
                efs = EF_LARGE + (cfg.get("ef_glove_extra", []) if dsname == "glove100" else [])
                r.ann(s, task, engine, dsname, M=M, efc=efc, builds=1, efs=efs, repeats=1)
    # Optional FAISS HNSWFlat (Python), if importable
    faiss_script = REPO / "scripts" / "bench_faiss.py"
    for dsname, (M, efc) in (("sift", (cfg["M"], cfg["efc"])), ("glove100", (cfg["glove_M"], cfg["glove_efc"]))):
        task = f"{dsname}_faiss_M{M}_efc{efc}"
        if r.done(s, task):
            continue
        d = r.ds(dsname)
        try:
            import importlib.util
            if importlib.util.find_spec("faiss") is None or importlib.util.find_spec("numpy") is None:
                r.record_gap(s, task, "faiss not installed")
                continue
        except Exception:  # noqa: BLE001
            r.record_gap(s, task, "faiss not importable")
            continue
        cmd = [sys.executable, str(faiss_script), "--base", d["base"], "--query", d["query"], "--gt", d["gt"],
               "--metric", d["metric"], "--M", str(M), "--efc", str(efc),
               "--ef", ",".join(map(str, EF_LARGE + (cfg.get("ef_glove_extra", []) if dsname == "glove100" else []))),
               "--pin", str(r.pin)]
        extra = r.ds_extra(dsname)
        extra.update({"engine": "faiss", "index": "IndexHNSWFlat", "M": M, "efc": efc})
        r.run_json(s, task, cmd, extra)


def suite_before_after_fix(r: Runner, builds: int) -> None:
    s = "before_after_fix"
    wt = r.build_root / "worktree_prefix"
    if not wt.exists():
        sh(["git", "-C", str(REPO), "worktree", "add", "--detach", str(wt), PREFIX_COMMIT])
    prefix_sha = sh(["git", "-C", str(wt), "rev-parse", "HEAD"]).strip()
    fix_sha = sh(["git", "-C", str(REPO), "rev-parse", FIX_COMMIT]).strip()
    head_sha = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    harness = REPO / "benchmarks" / "fix_compare"
    d = r.ds("siftsmall")
    for label, root, sha in (("before", wt, prefix_sha), ("after", REPO, head_sha)):
        b = r.build_root / f"fix_{label}"
        cmake_build(harness, b, [f"-DENGINE_ROOT={root.as_posix()}"])
        exe = str(b / f"fix_bench{EXE}")
        for index in ("hnsw", "sq8"):
            cmd = [exe, "--base", d["base"], "--query", d["query"], "--gt", d["gt"], "--index", index,
                   "--builds", str(builds), "--ef", "10,16,32,64,100", "--repeats", "5", "--pin", str(r.pin)]
            extra = r.ds_extra("siftsmall")
            extra.update({"version": label, "engine_commit": sha, "fix_commit": fix_sha, "index": index,
                          "M": 16, "efc": 200})
            r.run_json(s, f"{label}_{index}", cmd, extra)


def suite_crash_recovery(r: Runner, cycles: int) -> None:
    s = "crash_recovery"
    task = "kill_cycles"
    exe = bin_path(r.build_root, "default", "nexus_bench")
    if not r.done(s, task):
        wal_dir = r.work / "crash"
        if wal_dir.exists():
            shutil.rmtree(wal_dir)
        wal_dir.mkdir()
        wal = wal_dir / "crash.wal"
        rng = random.Random(20261107)
        acked_total = 0
        results = []
        t_start = time.time()
        for c in range(cycles):
            p = subprocess.Popen([exe, "crash-writer", "--wal", str(wal), "--dim", "128"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            first = p.stdout.readline().split()
            if not first or first[0] != "READY":
                raise RuntimeError("crash-writer did not start")
            start_size = int(first[1])
            delay = rng.uniform(0.02, 0.6)
            time.sleep(delay)
            p.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
            out, _ = p.communicate()
            last = None
            n_acks = 0
            for line in out.splitlines():
                if line.startswith("A "):
                    last = int(line[2:])
                    n_acks += 1
            if last is not None:
                acked_total = last + 1
            inject = rng.randint(1, 532) if c % 4 == 3 else 0  # every 4th cycle: synthetic torn write
            vout = r.work / "crash_verify.json"
            vout.unlink(missing_ok=True)
            subprocess.run([exe, "crash-verify", "--wal", str(wal), "--dim", "128", "--acked", str(acked_total),
                            "--inject-torn-bytes", str(inject), "--out", str(vout)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            v = json.loads(vout.read_text())
            v.update({"cycle": c, "kill_after_seconds": round(delay, 4), "acks_this_cycle": n_acks,
                      "recovered_at_start": start_size,
                      "kill_signal": "TerminateProcess" if IS_WIN else "SIGKILL"})
            results.append(v)
            if not v["ok"]:
                log(f"  cycle {c}: VERIFY FAILED {v}")
            # an unacknowledged-but-durable insert is a valid outcome; the next writer continues after it
            acked_total = max(acked_total, v["recovered"])
            if c % 25 == 0:
                log(f"  crash cycle {c}/{cycles}: acked={acked_total} ok={v['ok']}")
        summary = {"cycles": cycles, "passed": sum(1 for x in results if x["ok"]),
                   "lost_acked_total": sum(x["lost_acked"] for x in results),
                   "torn_tails_seen": sum(1 for x in results if x["torn_tail_before_recovery"]),
                   "torn_tails_truncated": sum(1 for x in results if x["tail_truncated"]),
                   "natural_torn_tails": sum(1 for x in results if x["torn_tail_before_recovery"] and not x["injected_torn_bytes"]),
                   "injected_torn_tails": sum(1 for x in results if x["injected_torn_bytes"]),
                   "unacked_but_durable_cycles": sum(1 for x in results if x["unacked_but_durable"]),
                   "final_records": results[-1]["recovered"] if results else 0,
                   "wall_seconds": round(time.time() - t_start, 2),
                   "wal_filesystem": filesystem_of(wal_dir)}
        r.save(s, task, f"scripts/run_all.py crash loop ({cycles} cycles) using {Path(exe).name} crash-writer/crash-verify",
               {"summary": summary, "cycles": results})
    # replay time vs record count; WAL written by the no-fsync build to keep generation fast,
    # replay cost does not depend on how the file was written
    r.run_json(s, "replay_time", [bin_path(r.build_root, "nofsync", "nexus_bench"), "wal-replay",
                                  "--dir", str(r.work / "replay"), "--records", "10000,100000,1000000",
                                  "--hnsw-max", "100000", "--repeats", "3"],
               {"filesystem": filesystem_of(r.work)})


def suite_write_path(r: Runner) -> None:
    s = "write_path"
    d = r.ds("sift")
    if not d["available"]:
        d = r.ds("siftsmall")
    for variant in ("default", "nofsync"):
        for index in ("flat", "hnsw"):
            r.run_json(s, f"{variant}_{index}", [bin_path(r.build_root, variant, "nexus_bench"), "write-path",
                                                 "--base", d["base"], "--n", "10000", "--index", index,
                                                 "--dir", str(r.work / "write_path"), "--pin", str(r.pin)],
                       {"variant": variant, "filesystem": filesystem_of(r.work)})


def suite_build_memory(r: Runner) -> None:
    """Build time / RSS at 1M come from the recall_qps SIFT1M runs (one process per index type,
    so RSS deltas are not polluted by other indexes). Here: snapshot load times."""
    s = "build_memory"
    exe = bin_path(r.build_root, "default", "nexus_bench")
    for snap in sorted(r.snap.glob("sift_nexus_*.nexus")):
        r.run_json(s, f"snapshot_load_{snap.stem}", [exe, "snapshot-load", "--snapshot", str(snap), "--repeats", "3"],
                   {"snapshot": snap.name})


def suite_concurrency(r: Runner, cfg: dict) -> None:
    s = "concurrency"
    d = r.ds("sift")
    snap = r.snapshot("nexus", "sift", "hnsw", cfg["M"], cfg["efc"])
    if not snap.exists():
        r.record_gap(s, "all", "SIFT1M snapshot missing")
        return
    threads = [1, 2, 4, 6, 8, 10, 12, 16]
    exe = bin_path(r.build_root, "default", "nexus_bench")
    for task, frac in (("read_only", 0.0), ("mix_90_10", 0.1)):
        cmd = [exe, "concurrency", "--snapshot", str(snap), "--query", d["query"], "--threads",
               ",".join(map(str, threads)), "--seconds", "5", "--ef", str(cfg["concurrency_ef"]),
               "--wal", str(r.work / "concurrency.wal"), "--write-frac", str(frac)]
        if frac > 0:
            cmd += ["--insert-base", d["learn"]]
        r.run_json(s, task, cmd, {"threads_unpinned": True, "filesystem": filesystem_of(r.work)})


def suite_ablation(r: Runner, cfg: dict) -> None:
    s = "ablation"
    r.run_json(s, "simd_kernels", [bin_path(r.build_root, "default", "nexus_bench"), "simd", "--pin", str(r.pin)])
    snap = r.snapshot("nexus", "sift", "hnsw", cfg["M"], cfg["efc"])
    if snap.exists():
        efs = [16, 32, 64, 128, 256]
        # interleave A/B runs to spread any thermal or background drift over both variants
        for i in range(2):
            for variant in ("default", "noprefetch"):
                r.ann(s, f"prefetch_{variant}_{i}", "nexus", "sift", load=snap, efs=efs, repeats=2, variant=variant)
    else:
        r.record_gap(s, "prefetch", "SIFT1M HNSW snapshot missing")
    sq8 = r.snapshot("nexus", "sift", "sq8", cfg["M"], cfg["efc"])
    if sq8.exists():
        r.ann(s, "rerank_sweep", "nexus", "sift", index="sq8", load=sq8, efs=[16, 32, 64, 128, 256],
              reranks=[1, 1.5, 2, 3, 5, 8], repeats=1)
    else:
        r.record_gap(s, "rerank_sweep", "SIFT1M SQ8 snapshot missing")


# ----------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine-id", default=None, help="slug of the CPU model (auto if omitted)")
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--work-dir", default=None, help="scratch for snapshots/WALs; must be a local disk")
    ap.add_argument("--build-root", default=str(REPO / "build-bench"))
    ap.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    ap.add_argument("--quick", action="store_true", help="CI smoke: SIFT-10K recall only")
    ap.add_argument("--resume", action="store_true", help="skip tasks whose JSON already exists")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="results sub-folder (default: today)")
    ap.add_argument("--crash-cycles", type=int, default=200)
    ap.add_argument("--fix-builds", type=int, default=10)
    ap.add_argument("--config", default=str(REPO / "benchmarks" / "config.json"))
    args = ap.parse_args()
    args.work_dir = args.work_dir or str(Path(args.build_root) / "work")

    cfg = json.loads(Path(args.config).read_text())
    data_dir, work_dir = Path(args.data_dir).resolve(), Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    env = detect_env(data_dir, work_dir)
    if not args.machine_id:
        args.machine_id = slugify(env.get("host", {}).get("cpu_model") or env["cpu"]["model"])
    pin, note = choose_pin_cpu(env)
    env["pinning"] = {"cpu": pin, "note": note}
    log(f"machine_id={args.machine_id} runtime={env['runtime']} cpu={env['cpu']['model']} pin={pin}")
    missing = [f for f, ok in env["cpu"]["flags_checked"].items() if ok is False and f != "avx512f"]
    if missing:
        log(f"ERROR: CPU lacks {missing}; the engine requires AVX2/FMA/SSE4.2")
        return 1

    build_root = Path(args.build_root).resolve()
    if args.skip_build and (build_root / "build_info.json").exists():
        build_info = json.loads((build_root / "build_info.json").read_text())
    else:
        build_info = build_all(build_root)
        (build_root / "build_info.json").write_text(json.dumps(build_info, indent=1))
    r = Runner(args, env, build_info)
    (r.out_root).mkdir(parents=True, exist_ok=True)
    (r.out_root / "environment.json").write_text(json.dumps({"env": env, "build": build_info, "config": cfg,
                                                              "argv": sys.argv}, indent=1))

    for suite in args.suites:
        log(f"=== suite {suite}")
        try:
            if suite == "sanity_start":
                suite_sanity(r, "start")
            elif suite == "recall_qps":
                suite_recall_qps(r, args.quick, cfg)
            elif args.quick:
                continue
            elif suite == "distance_computations":
                suite_distance_computations(r, cfg)
            elif suite == "vs_hnswlib":
                suite_vs_hnswlib(r, cfg)
            elif suite == "before_after_fix":
                suite_before_after_fix(r, args.fix_builds)
            elif suite == "crash_recovery":
                suite_crash_recovery(r, args.crash_cycles)
            elif suite == "write_path":
                suite_write_path(r)
            elif suite == "build_memory":
                suite_build_memory(r)
            elif suite == "concurrency":
                suite_concurrency(r, cfg)
            elif suite == "ablation":
                suite_ablation(r, cfg)
            elif suite == "sanity_end":
                suite_sanity(r, "end")
        except Exception as e:  # noqa: BLE001 - keep going with the other suites
            log(f"suite {suite} failed: {e}")
            r.record_gap(suite, "*", str(e)[:2000])
    log("all suites finished")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
