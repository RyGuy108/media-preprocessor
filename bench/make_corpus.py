"""Build a benchmark corpus with exact ground truth.

Videos: each one is a sequence of single-shot segments cut from the source videos and
re-encoded as H.264 (libx264 defaults, so B-frames are present like most real uploads).
Because we place every cut ourselves, the true scene boundaries are known to the frame.
Segments are drawn only from intervals that ffmpeg's own scene detector (scdet, run at a
very sensitive threshold) considers a single shot, so a source's internal cuts don't leak
into the ground truth. About 20% of segments repeat an earlier segment of the same video
(shot / reverse-shot, recaps): those keyframes are true duplicates.

Images: originals plus near-duplicate variants (re-encoded, resized, cropped, brightened,
burst frames). Ground truth = which images belong to the same group.

    python bench/make_corpus.py --sources bench/sources --out corpus/test --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SCDET_THRESHOLD = 3.0   # very sensitive: anything ffmpeg flags is excluded from segments
GUARD_SEC = 0.25        # stay this far from a flagged change
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_IMAGES = re.compile(r"(screen|storyboard|trailer_frames|_cut_|demo_week)")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den,
            "frames": int(s["nb_read_packets"]), "duration": float(j["format"]["duration"])}


def single_shot_intervals(path: Path, info: dict) -> list[tuple[float, float]]:
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf",
         f"scdet=threshold={SCDET_THRESHOLD}", "-an", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    times = [float(t) for t in re.findall(r"lavfi\.scd\.time: ([0-9.]+)", log)]
    bounds = [0.0] + times + [info["duration"]]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        lo, hi = a + (GUARD_SEC if a > 0 else 0.1), b - GUARD_SEC
        if hi - lo >= 1.0:
            out.append((lo, hi))
    return out


def analyse_sources(src_dir: Path) -> list[dict]:
    cache = src_dir / "intervals.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sources = []
    for p in sorted((src_dir / "videos").iterdir()):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        info = probe(p)
        iv = single_shot_intervals(p, info)
        sources.append({"path": str(p), "name": p.name, **info, "intervals": iv})
        print(f"  {p.name}: {len(iv)} single-shot intervals, {sum(b - a for a, b in iv):.1f}s")
    cache.write_text(json.dumps(sources, indent=1))
    return sources


def extract_frames(src: str, start: float, n: int, w: int, h: int) -> bytes:
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", src, "-frames:v", str(n),
         "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    got = len(raw) // (w * h * 3)
    if got < n:  # pad by repeating the last frame (rare: segment ran into EOF)
        raw += raw[-w * h * 3:] * (n - got)
    return raw[: n * w * h * 3]


def build_video(task: dict) -> dict:
    w, h, fps = task["width"], task["height"], 30
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-threads", "1", task["out"]],
        stdin=subprocess.PIPE)
    frame = 0
    for seg in task["segments"]:
        enc.stdin.write(extract_frames(seg["source"], seg["start"], seg["frames"], w, h))
        seg["start_frame"] = frame
        frame += seg["frames"]
    enc.stdin.close()
    enc.wait()
    cuts = [s["start_frame"] for s in task["segments"][1:]]
    return {"video": task["out"], "width": w, "height": h, "fps": fps, "frames": frame,
            "cuts": cuts, "segments": task["segments"]}


def plan_videos(sources: list[dict], n: int, rng: random.Random, out: Path) -> list[dict]:
    pool = [(si, ii) for si, s in enumerate(sources) for ii, _ in enumerate(s["intervals"])]
    tasks = []
    for v in range(n):
        width, height = (1920, 1080) if rng.random() < 0.3 else (1280, 720)
        segs, last_key = [], None
        for k in range(rng.randint(6, 14)):
            if k >= 2 and rng.random() < 0.2:
                # Repeat an earlier (non-adjacent) segment: a genuine duplicate shot.
                cands = [s for s in segs[:-1] if s["key"] != last_key]
                if cands:
                    seg = dict(rng.choice(cands))
                    segs.append(seg)
                    last_key = seg["key"]
                    continue
            while True:
                si, ii = rng.choice(pool)
                if (si, ii) != last_key:
                    break
            a, b = sources[si]["intervals"][ii]
            dur = min(rng.uniform(1.0, 5.0), b - a)
            start = rng.uniform(a, b - dur)
            nframes = max(15, int(dur * sources[si]["fps"]))
            seg = {"source": sources[si]["path"], "start": round(start, 3), "frames": nframes,
                   "key": (si, ii), "group": f"{si}:{ii}:{start:.3f}"}
            segs.append(seg)
            last_key = (si, ii)
        for s in segs:
            s["key"] = list(s["key"])
        tasks.append({"out": str(out / "videos" / f"v{v:03d}.mp4"), "width": width,
                      "height": height, "segments": segs})
    return tasks


# ----------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------
def variant(img: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if kind == "jpeg_q35":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 35])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "resize_50":
        return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if kind == "crop_3pct":
        dx, dy = int(w * 0.03), int(h * 0.03)
        return img[dy:h - dy, dx:w - dx].copy()
    if kind == "brighter":
        return cv2.convertScaleAbs(img, alpha=1.08, beta=12)
    raise ValueError(kind)


def build_images(sources: list[dict], src_dir: Path, rng: random.Random, out: Path,
                 frames_per_source: int) -> dict:
    (out / "images").mkdir(parents=True, exist_ok=True)
    originals = []  # (name, img, burst_img or None)
    for p in sorted((src_dir / "images").iterdir()):
        if p.suffix.lower() in IMAGE_EXT and not SKIP_IMAGES.search(p.name):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                originals.append((p.stem, img, None))
    # Video stills: at most one per single-shot interval so originals are distinct shots.
    for si, s in enumerate(sources):
        ivs = list(s["intervals"])
        rng.shuffle(ivs)
        for a, b in ivs[:frames_per_source]:
            t = rng.uniform(a, max(a, b - 0.4))
            cap = cv2.VideoCapture(s["path"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok1, f1 = cap.read()
            for _ in range(int(0.2 * s["fps"]) - 1):
                cap.grab()
            ok2, f2 = cap.read()
            cap.release()
            if ok1:
                originals.append((f"{Path(s['name']).stem}_{t:.2f}", f1, f2 if ok2 else None))
    items = []
    kinds = ["jpeg_q35", "resize_50", "crop_3pct", "brighter", "burst"]
    for gi, (name, img, burst) in enumerate(originals):
        group = f"g{gi:03d}"
        items.append({"group": group, "variant": "original", "img": img, "src": name})
        if rng.random() < 0.4:
            for kind in rng.sample(kinds, rng.randint(1, 2)):
                if kind == "burst":
                    if burst is None:
                        continue
                    v = burst
                else:
                    v = variant(img, kind, rng)
                items.append({"group": group, "variant": kind, "img": v, "src": name})
    rng.shuffle(items)
    truth = []
    for i, it in enumerate(items):
        path = out / "images" / f"img_{i:04d}.jpg"
        cv2.imwrite(str(path), it["img"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth.append({"path": str(path), "group": it["group"], "variant": it["variant"],
                      "src": it["src"]})
    return {"images": truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="bench/sources")
    ap.add_argument("--out", default="corpus/test")
    ap.add_argument("--videos", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames-per-source", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    src, out = Path(a.sources), Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    print("analysing sources ...")
    sources = analyse_sources(src)
    tasks = plan_videos(sources, a.videos, rng, out)
    print(f"encoding {len(tasks)} videos ...")
    with ProcessPoolExecutor(a.jobs) as ex:
        videos = list(ex.map(build_video, tasks))
    images = build_images(sources, src, rng, out, a.frames_per_source)
    truth = {"seed": a.seed, "videos": videos, **images}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    nf = sum(v["frames"] for v in videos)
    ngroups = len({i["group"] for i in images["images"]})
    print(f"done: {len(videos)} videos, {nf} frames ({nf / 30 / 60:.1f} min), "
          f"{sum(len(v['cuts']) for v in videos)} cuts; {len(images['images'])} images in "
          f"{ngroups} groups")


if __name__ == "__main__":
    main()
