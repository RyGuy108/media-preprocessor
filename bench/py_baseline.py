"""Python baseline: the same pipeline (same algorithm, same thresholds, same OpenCV calls)
written the way a Python team would ship it: OpenCV's Python bindings plus a
concurrent.futures.ProcessPoolExecutor.

Modes:
  naive  - cap.read() on every frame, OpenCV's default threading. What most scripts do.
  tuned  - grab()/retrieve() so only analyzed frames are converted to BGR, one decode
           thread and cv2.setNumThreads(1) per process (the pool supplies parallelism).
           This is the C++ OpenCV backend's algorithm line for line.

Failure handling is what ProcessPoolExecutor gives you: exceptions inside a job are caught
and reported; a worker that dies (segfault, OOM kill) breaks the pool and every pending
job fails with BrokenProcessPool; a hung job holds its process until the batch watchdog
(--batch-timeout) gives up and kills the pool.

    python bench/py_baseline.py --list corpus/test/inputs.txt --mode tuned --report out/py.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Same defaults as include/mhprep/*.h
ANALYSIS_WIDTH, MIN_SCORE, RATIO, WINDOW, MIN_SCENE_SEC = 256, 12.0, 2.0, 8, 0.5
STRIDE, KEYFRAME_EVERY, FACE_SIZE, DUP_RADIUS, COLOR_TOL, JPEG_Q = 2, 3, 512, 8, 14, 90
FACE_CONF, FACE_MIN = 0.6, 48


class InputError(Exception):
    pass


# ----------------------------------------------------------------------------------------
# Building blocks (mirror src/phash.cpp, src/scene.cpp, src/faces.cpp)
# ----------------------------------------------------------------------------------------
def phash(img: np.ndarray) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = cv2.dct(small)[:8, :8].flatten()
    median = np.partition(v[1:], 31)[31]
    bits = v > median
    return int(sum(1 << i for i in range(64) if bits[i]))


def color_sig(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (4, 4), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2Lab).reshape(-1).astype(np.int32)


def color_distance(a: np.ndarray, b: np.ndarray) -> int:
    a, b = a.reshape(16, 3), b.reshape(16, 3)
    dl = a[:, 0] - b[:, 0]
    dl = np.abs(dl - dl.mean()) / 2.0
    return int((dl + np.abs(a[:, 1] - b[:, 1]) + np.abs(a[:, 2] - b[:, 2])).max())


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class DupIndex:
    """Linear scan; plenty for one job's few dozen assets."""

    def __init__(self):
        self.items = []  # (hash, sig, id)

    def find(self, h, sig):
        best = None
        for hh, ss, i in self.items:
            d = hamming(h, hh)
            if d <= DUP_RADIUS and color_distance(ss, sig) <= COLOR_TOL:
                if best is None or d < best[1] or (d == best[1] and i < best[0]):
                    best = (i, d)
        return best

    def add(self, h, sig, i):
        self.items.append((h, sig, i))


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lap = cv2.Laplacian(g, cv2.CV_16S)
    _, std = cv2.meanStdDev(lap)
    return float(std[0][0] ** 2)


def analysis_size(w: int, h: int) -> tuple[int, int]:
    if w <= ANALYSIS_WIDTH:
        return w, h
    hh = int(round(h * ANALYSIS_WIDTH / w))
    hh += hh & 1
    return ANALYSIS_WIDTH, max(2, hh)


class SceneDetector:
    def __init__(self, fps: float):
        self.min_gap = max(1, round(MIN_SCENE_SEC * fps))
        self.prev = None
        self.recent: list[float] = []
        self.last_cut = 0

    def push(self, idx: int, small: np.ndarray) -> bool:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        if self.prev is None:
            self.prev, self.last_cut = hsv, idx
            return False
        m = cv2.mean(cv2.absdiff(hsv, self.prev))
        score = (m[0] + m[1] + m[2]) / 3.0
        self.prev = hsv
        avg = sum(self.recent) / len(self.recent) if self.recent else 0.0
        if score >= MIN_SCORE and score >= RATIO * max(avg, 1.0) and idx - self.last_cut >= self.min_gap:
            self.last_cut = idx
            return True
        self.recent.append(score)
        if len(self.recent) > WINDOW:
            self.recent.pop(0)
        return False


class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(str(MODELS / "deploy.prototxt"),
                                            str(MODELS / "res10_300x300_ssd_iter_140000_fp16.caffemodel"))

    def detect(self, img: np.ndarray):
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.net.setInput(blob)
        det = self.net.forward()[0, 0]
        h, w = img.shape[:2]
        faces = []
        for row in det:
            conf = float(row[2])
            if conf < FACE_CONF:
                continue
            x1, y1, x2, y2 = int(row[3] * w), int(row[4] * h), int(row[5] * w), int(row[6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < FACE_MIN or y2 - y1 < FACE_MIN:
                continue
            faces.append(((x1, y1, x2 - x1, y2 - y1), conf))
        faces.sort(key=lambda f: -f[1])
        return faces


def crop_face(img, box, out=FACE_SIZE, margin=1.8):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = min(int(round(max(w, h) * margin)), img.shape[1], img.shape[0])
    x0 = min(max(int(round(cx - side / 2)), 0), img.shape[1] - side)
    y0 = min(max(int(round(cy - side / 2)), 0), img.shape[0] - side)
    interp = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (out, out), interpolation=interp)


# ----------------------------------------------------------------------------------------
# Pipeline (mirrors src/pipeline.cpp)
# ----------------------------------------------------------------------------------------
_faces: FaceDetector | None = None
_mode = "tuned"


def _init(mode: str):
    global _faces, _mode
    _mode = mode
    if mode == "tuned":
        cv2.setNumThreads(1)
    _faces = FaceDetector()


def _write(out_dir: Path, name: str, img, m) -> tuple[str, int]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    p = out_dir / name
    p.write_bytes(buf.tobytes())
    m["bytes_out"] += len(buf)
    return str(p), len(buf)


def _add_faces(img, scene, frame, prefix, out_dir, face_index, m, scene_faces):
    k = 0
    for box, conf in _faces.detect(img):
        crop = crop_face(img, box)
        a = {"kind": "face", "scene": scene, "frame": frame, "confidence": conf, "box": list(box)}
        h, sig = phash(crop), color_sig(crop)
        a["phash"] = f"{h:016x}"
        a["sharpness"] = sharpness(crop)
        aid = len(m["assets"])
        hit = face_index.find(h, sig)
        if hit:
            a["duplicate_of"], a["dup_distance"] = hit
        else:
            a["path"], a["bytes"] = _write(out_dir, f"{prefix}_face{k}.jpg", crop, m)
            k += 1
            face_index.add(h, sig, aid)
        m["assets"].append(a)
        if scene_faces is not None:
            scene_faces.append(aid)


def process_image(path: str, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"cannot decode image: {path}")
    m = {"input": path, "kind": "image", "assets": [], "scenes": [], "bytes_in": os.path.getsize(path),
         "bytes_out": 0}
    h = phash(img)
    sig = color_sig(img)
    m["assets"].append({"kind": "image", "path": path, "phash": f"{h:016x}",
                        "csig": bytes(sig.astype(np.uint8)).hex(), "sharpness": sharpness(img)})
    _add_faces(img, -1, -1, "img", out_dir, DupIndex(), m, None)
    m["timing_ms"] = {"total": (time.perf_counter() - t0) * 1000}
    return m


def process_video(path: str, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    if _mode == "tuned":
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    else:
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InputError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (0 < fps < 1000):
        fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise InputError(f"no video stream: {path}")
    asize = analysis_size(w, h)
    det = SceneDetector(fps)
    kf_index, face_index = DupIndex(), DupIndex()
    m = {"input": path, "kind": "video", "assets": [], "scenes": [], "bytes_in": os.path.getsize(path),
         "bytes_out": 0, "frames_decoded": 0, "frames_analyzed": 0}
    best = {"sharp": -1.0, "frame": -1, "img": None}
    scene_start, last_frame, analyzed_in_scene = 0, -1, 0
    kf_phase = KEYFRAME_EVERY // 2

    def finish_scene(end_frame):
        nonlocal best, analyzed_in_scene
        s = {"index": len(m["scenes"]), "start_frame": scene_start, "end_frame": end_frame,
             "start_sec": scene_start / fps, "end_sec": (end_frame + 1) / fps, "keyframe": -1,
             "faces": []}
        if best["img"] is not None:
            img = best["img"]
            hsh, sig = phash(img), color_sig(img)
            a = {"kind": "keyframe", "scene": s["index"], "frame": best["frame"],
                 "sharpness": best["sharp"], "phash": f"{hsh:016x}",
                 "csig": bytes(sig.astype(np.uint8)).hex()}
            aid = len(m["assets"])
            s["keyframe"] = aid
            prefix = f"s{s['index']:04d}"
            hit = kf_index.find(hsh, sig)
            if hit:
                a["duplicate_of"], a["dup_distance"] = hit
                m["assets"].append(a)
            else:
                a["path"], a["bytes"] = _write(out_dir, prefix + ".jpg", img, m)
                kf_index.add(hsh, sig, aid)
                m["assets"].append(a)
                _add_faces(img, s["index"], best["frame"], prefix, out_dir, face_index, m, s["faces"])
        m["scenes"].append(s)
        best = {"sharp": -1.0, "frame": -1, "img": None}
        analyzed_in_scene = 0

    while True:
        idx = last_frame + 1
        analyze = idx % STRIDE == 0
        if _mode == "naive":
            ok, frame = cap.read()  # decode + BGR conversion for every frame
            if not ok:
                break
        else:
            if not cap.grab():
                break
            frame = cap.retrieve()[1] if analyze else None
        m["frames_decoded"] += 1
        last_frame = idx
        if not analyze:
            continue
        m["frames_analyzed"] += 1
        small = cv2.resize(frame, asize, interpolation=cv2.INTER_AREA)
        if det.push(idx, small):
            finish_scene(idx - 1)
            scene_start = idx
        if analyzed_in_scene % KEYFRAME_EVERY == kf_phase or best["img"] is None:
            sh = sharpness(small)
            if sh > best["sharp"]:
                best = {"sharp": sh, "frame": idx, "img": frame.copy()}
        analyzed_in_scene += 1
    cap.release()
    if m["frames_decoded"] == 0:
        raise InputError(f"no decodable frames: {path}")
    finish_scene(last_frame)
    m["video"] = {"fps": fps, "width": w, "height": h, "frames": last_frame + 1,
                  "duration_sec": (last_frame + 1) / fps}
    dups = sum(1 for a in m["assets"] if "duplicate_of" in a)
    m["summary"] = {"scenes": len(m["scenes"]), "assets_unique": len(m["assets"]) - dups,
                    "assets_duplicate": dups, "faces": sum(1 for a in m["assets"] if a["kind"] == "face")}
    m["timing_ms"] = {"total": (time.perf_counter() - t0) * 1000}
    return m


def chaos_draw(inp: str, attempt: int, seed: int) -> float:
    """Same function as src/worker.cpp."""
    h = 1469598103934665603
    for ch in f"{inp}#{attempt}#{seed}".encode():
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 30
    h = (h * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 27
    h = (h * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 31
    return (h >> 11) / float(1 << 53)


def run_job(job_id: int, inp: str, out_root: str, tenant: str) -> dict:
    chaos = os.environ.get("MHPREP_CHAOS")
    if chaos:
        kv = dict(p.split("=") for p in chaos.split(","))
        r = chaos_draw(inp, 1, int(kv.get("seed", 0)))
        if r < float(kv.get("crash", 0)):
            os.kill(os.getpid(), signal.SIGSEGV)
        if r < float(kv.get("crash", 0)) + float(kv.get("hang", 0)):
            while True:
                signal.pause()
    out_dir = Path(out_root) / tenant / str(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(inp).suffix.lower() in IMAGE_EXT:
        return process_image(inp, out_dir)
    return process_video(inp, out_dir)


# ----------------------------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--mode", choices=["naive", "tuned"], default="tuned")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="out/py")
    ap.add_argument("--report")
    ap.add_argument("--batch-timeout", type=float, default=3600)
    a = ap.parse_args()

    jobs = []
    for line in open(a.list):
        line = line.rstrip("\n")
        if line and not line.startswith("#"):
            tenant, inp = line.split("\t", 1) if "\t" in line else ("default", line)
            jobs.append({"id": len(jobs) + 1, "input": inp, "tenant": tenant, "status": "queued"})

    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.mode,))
    futs = {ex.submit(run_job, j["id"], j["input"], a.out, j["tenant"]): j for j in jobs}
    pending = set(futs)
    tenant_idx: dict[str, list] = {}
    deadline = t0 + a.batch_timeout
    while pending:
        done, pending = wait(pending, timeout=max(0.0, deadline - time.perf_counter()),
                             return_when=FIRST_COMPLETED)
        if not done:
            break  # watchdog
        for f in done:
            j = futs[f]
            j["latency_ms"] = (time.perf_counter() - t0) * 1000
            try:
                m = f.result()
            except InputError as e:
                j.update(status="failed", error=str(e), error_kind="input")
                continue
            except BrokenProcessPool as e:
                j.update(status="failed", error=f"BrokenProcessPool: {e}", error_kind="crash")
                continue
            except Exception as e:  # noqa: BLE001
                j.update(status="failed", error=repr(e), error_kind="internal")
                continue
            # Cross-upload dedup in the parent, in completion order (same rule as C++).
            idx = tenant_idx.setdefault(j["tenant"], [])
            dups = 0
            for i, asset in enumerate(m["assets"]):
                if asset["kind"] not in ("keyframe", "image") or "duplicate_of" in asset:
                    continue
                h = int(asset["phash"], 16)
                sig = np.frombuffer(bytes.fromhex(asset["csig"]), dtype=np.uint8).astype(np.int32)
                best = None
                for hh, ss, owner in idx:
                    d = hamming(h, hh)
                    if d <= DUP_RADIUS and color_distance(ss, sig) <= COLOR_TOL and (best is None or d < best[1]):
                        best = (owner, d)
                if best:
                    asset["cross_duplicate_of"] = {"job": best[0][0], "asset": best[0][1], "distance": best[1]}
                    dups += 1
                else:
                    idx.append((h, sig, (j["id"], i)))
            m.setdefault("summary", {})["assets_cross_job_duplicate"] = dups
            j.update(status="succeeded", manifest=m)
    wall = time.perf_counter() - t0
    if pending:  # watchdog fired: hung jobs never return
        for f in pending:
            futs[f].update(status="failed", error="batch watchdog expired", error_kind="timeout")
    # Tear the pool down hard: hung workers would otherwise keep the process (and any
    # pipe it writes to) alive forever.
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        p.kill()

    ok = [j for j in jobs if j["status"] == "succeeded"]
    frames = sum(j["manifest"].get("frames_decoded", 0) for j in ok)
    media = sum(j["manifest"].get("video", {}).get("duration_sec", 0) for j in ok)
    lat = sorted(j["latency_ms"] for j in jobs if "latency_ms" in j)
    summary = {"mode": a.mode, "jobs": len(jobs), "succeeded": len(ok), "failed": len(jobs) - len(ok),
               "failed_by_kind": {k: sum(1 for j in jobs if j.get("error_kind") == k)
                                  for k in ("input", "crash", "timeout", "internal")},
               "workers": a.workers, "wall_sec": wall, "frames_decoded": frames, "fps": frames / wall,
               "media_sec": media, "realtime_factor": media / wall,
               "latency_p50_ms": lat[len(lat) // 2] if lat else 0,
               "latency_p99_ms": lat[min(len(lat) - 1, int(len(lat) * 0.99))] if lat else 0}
    if a.report:
        with open(a.report, "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
    print(json.dumps(summary, indent=2))
    sys.stdout.flush()
    os._exit(0)  # don't wait on hung workers at interpreter exit


if __name__ == "__main__":
    main()
