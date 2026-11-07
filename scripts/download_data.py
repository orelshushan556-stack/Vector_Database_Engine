#!/usr/bin/env python3
"""Download and prepare benchmark datasets (portable: Windows, Linux, WSL, macOS).

Datasets (written as .fvecs/.ivecs under <data-dir>/<name>/):
  siftsmall  SIFT-10K  (INRIA Texmex)            L2       base 10,000   queries 100
  sift       SIFT1M    (INRIA Texmex)            L2       base 1,000,000 queries 10,000
  glove100   GloVe-100 angular (ann-benchmarks)   cosine   base 1,183,514 queries 10,000

GloVe is converted from the ann-benchmarks HDF5 file. Vectors are L2-normalized during
conversion (cosine distance on unit vectors equals 1 - dot product, so neighbor order is
unchanged); the ground truth is the file's own `neighbors` table. Normalization needs numpy
and h5py; the SIFT datasets need only the standard library.

Each dataset directory gets a manifest.json with the source URL and SHA-256 of every file.
A failed download is recorded in <data-dir>/download_status.json and does not abort the
other datasets.

Usage:
  python scripts/download_data.py --data-dir data [--datasets siftsmall sift glove100]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

TEXMEX = "ftp://ftp.irisa.fr/local/texmex/corpus/"
ANN = "http://ann-benchmarks.com/"

DATASETS = {
    "siftsmall": {
        "kind": "texmex",
        "urls": [TEXMEX + "siftsmall.tar.gz"],
        "hdf5_fallback": None,
        "files": {"base": "siftsmall_base.fvecs", "query": "siftsmall_query.fvecs",
                  "gt": "siftsmall_groundtruth.ivecs", "learn": "siftsmall_learn.fvecs"},
        "metric": "l2",
    },
    "sift": {
        "kind": "texmex",
        "urls": [TEXMEX + "sift.tar.gz"],
        "hdf5_fallback": ANN + "sift-128-euclidean.hdf5",
        "files": {"base": "sift_base.fvecs", "query": "sift_query.fvecs",
                  "gt": "sift_groundtruth.ivecs", "learn": "sift_learn.fvecs"},
        "metric": "l2",
    },
    "glove100": {
        "kind": "hdf5",
        "urls": [ANN + "glove-100-angular.hdf5"],
        "hdf5_fallback": None,
        "files": {"base": "glove100_base.fvecs", "query": "glove100_query.fvecs",
                  "gt": "glove100_groundtruth.ivecs"},
        "metric": "angular",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, retries: int = 3) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url} (attempt {attempt})", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "nexus-bench-downloader"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, last = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 10:
                        pct = f"{100 * done / total:.0f}%" if total else f"{done >> 20} MiB"
                        print(f"    {pct}", flush=True)
                        last = time.time()
            os.replace(tmp, dst)
            return
        except Exception as e:  # noqa: BLE001 - report and retry any transport error
            print(f"    failed: {e}", flush=True)
            if tmp.exists():
                tmp.unlink()
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def write_fvecs(path: Path, arr) -> None:
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0] = np.frombuffer(struct.pack("<i", d), dtype=np.float32)[0]
    out[:, 1:] = arr
    out.tofile(path)


def write_ivecs(path: Path, arr) -> None:
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = arr
    out.tofile(path)


def convert_hdf5(h5_path: Path, out_dir: Path, files: dict, normalize: bool) -> None:
    import h5py
    import numpy as np
    with h5py.File(h5_path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
    if normalize:
        for a in (train, test):
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # zero vectors stay zero
            a /= norms
    write_fvecs(out_dir / files["base"], train)
    write_fvecs(out_dir / files["query"], test)
    write_ivecs(out_dir / files["gt"], neighbors)


def prepare(name: str, data_dir: Path, keep_archives: bool) -> dict:
    spec = DATASETS[name]
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    needed = [out_dir / spec["files"][k] for k in ("base", "query", "gt")]
    if manifest_path.exists() and all(p.exists() for p in needed):
        print(f"[{name}] already prepared")
        return json.loads(manifest_path.read_text())

    dl_dir = data_dir / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    source = None
    errors = []

    if spec["kind"] == "texmex":
        for url in spec["urls"]:
            archive = dl_dir / url.rsplit("/", 1)[1]
            try:
                if not archive.exists():
                    fetch(url, archive)
                with tarfile.open(archive, "r:gz") as tar:
                    for m in tar.getmembers():
                        base = os.path.basename(m.name)
                        if m.isfile() and base in spec["files"].values():
                            src = tar.extractfile(m)
                            with open(out_dir / base, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                source = url
                if not keep_archives:
                    archive.unlink(missing_ok=True)
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{url}: {e}")
        if source is None and spec["hdf5_fallback"]:
            url = spec["hdf5_fallback"]
            h5 = dl_dir / url.rsplit("/", 1)[1]
            try:
                if not h5.exists():
                    fetch(url, h5)
                convert_hdf5(h5, out_dir, spec["files"], normalize=False)
                source = url
            except Exception as e:  # noqa: BLE001
                errors.append(f"{url}: {e}")
    else:
        url = spec["urls"][0]
        h5 = dl_dir / url.rsplit("/", 1)[1]
        try:
            if not h5.exists():
                fetch(url, h5)
            convert_hdf5(h5, out_dir, spec["files"], normalize=(spec["metric"] == "angular"))
            source = url
            if not keep_archives:
                h5.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")

    if source is None:
        raise RuntimeError("; ".join(errors))

    manifest = {
        "dataset": name,
        "source": source,
        "metric": spec["metric"],
        "normalized": spec["kind"] == "hdf5" and spec["metric"] == "angular",
        "files": {role: {"name": fn, "sha256": sha256(out_dir / fn), "bytes": (out_dir / fn).stat().st_size}
                  for role, fn in spec["files"].items() if (out_dir / fn).exists()},
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] ready from {source}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    status_path = data_dir / "download_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for name in args.datasets:
        try:
            m = prepare(name, data_dir, args.keep_archives)
            status[name] = {"ok": True, "source": m["source"]}
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            status[name] = {"ok": False, "error": str(e)}
        status_path.write_text(json.dumps(status, indent=2))
    failed = [n for n, s in status.items() if not s["ok"]]
    print("download status:", json.dumps(status, indent=2))
    return 1 if failed and len(failed) == len(args.datasets) else 0


if __name__ == "__main__":
    sys.exit(main())
