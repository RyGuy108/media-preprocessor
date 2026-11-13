"""Run every benchmark and write results/*.json.

    python bench/run_bench.py                 # everything (about 15 minutes on 4 cores)
    python bench/run_bench.py --only throughput,chaos,http

Prereqs: build/mhprep built in Release, corpus/test built by bench/make_corpus.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evaluate  # noqa: E402
import http_load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "mhprep"
CORPUS = ROOT / "corpus" / "test"
RESULTS = ROOT / "results"
WORK = ROOT / "bench" / ".work"
WORKERS = os.cpu_count()


def sh(cmd, env=None, timeout=None):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})},
                       timeout=timeout, cwd=ROOT)
    wall = time.perf_counter() - t0
    if p.returncode not in (0, 1):
        print(p.stderr[-2000:])
        raise SystemExit(f"command failed: {cmd}")
    return json.loads(p.stdout[p.stdout.index("{"):]), wall


def corpus_stats(truth):
    frames = sum(v["frames"] for v in truth["videos"])
    return {"videos": len(truth["videos"]), "video_frames": frames,
            "video_minutes": round(frames / 30 / 60, 2), "true_cuts": sum(len(v["cuts"]) for v in truth["videos"]),
            "images": len(truth["images"]),
            "image_groups": len({i["group"] for i in truth["images"]}),
            "bytes": sum(os.path.getsize(v["video"]) for v in truth["videos"]) +
                     sum(os.path.getsize(i["path"]) for i in truth["images"])}


def run_cpp(name, inputs, options=None, extra=None, env=None):
    out, rep = WORK / name, WORK / f"{name}.jsonl"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [str(BIN), "run", "--list", str(inputs), "--out", str(out), "--report", str(rep),
           "--quiet", "--workers", str(WORKERS)]
    if options:
        cmd += ["--options", json.dumps(options)]
    cmd += extra or []
    summary, wall = sh(cmd, env=env)
    return summary, wall, rep


def run_py(name, inputs, mode, extra=None, env=None):
    out, rep = WORK / name, WORK / f"{name}.jsonl"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "bench/py_baseline.py", "--list", str(inputs), "--mode", mode,
           "--out", str(out), "--report", str(rep), "--workers", str(WORKERS)] + (extra or [])
    summary, wall = sh(cmd, env=env)
    return summary, wall, rep


def throughput(truth, repeats):
    configs = [
        ("cpp", "C++ (FFmpeg decoder, skip non-ref frames)", lambda n: run_cpp(n, CORPUS / "inputs.txt")),
        ("cpp_dense", "C++ (FFmpeg decoder, every frame)",
         lambda n: run_cpp(n, CORPUS / "inputs.txt", {"skip_nonref": False})),
        ("cpp_opencv", "C++ (cv::VideoCapture)",
         lambda n: run_cpp(n, CORPUS / "inputs.txt", {"backend": "opencv"})),
        ("py_tuned", "Python (tuned: grab/retrieve, 1 thread per process)",
         lambda n: run_py(n, CORPUS / "inputs.txt", "tuned")),
        ("py_naive", "Python (cap.read(), default threads)",
         lambda n: run_py(n, CORPUS / "inputs.txt", "naive")),
    ]
    stats = corpus_stats(truth)
    out = {"corpus": stats, "workers": WORKERS, "repeats": repeats, "configs": {}}
    for key, label, fn in configs:
        walls, last = [], None
        for r in range(repeats):
            summary, _, rep = fn(f"tp_{key}")
            walls.append(summary["wall_sec"])
            last = (summary, rep)
            print(f"  {key} run {r + 1}: {summary['wall_sec']:.1f}s")
        summary, rep = last
        wall = statistics.median(walls)
        acc = evaluate.score(truth, evaluate.load_report(rep))
        out["configs"][key] = {
            "label": label, "wall_sec": round(wall, 2), "walls": [round(w, 2) for w in walls],
            "source_frames_per_sec": round(stats["video_frames"] / wall, 1),
            "realtime_factor": round(stats["video_minutes"] * 60 / wall, 1),
            "frames_decoded": summary["frames_decoded"], "failed": summary["failed"],
            "accuracy": acc,
        }
    base = out["configs"]
    for k in base:
        base[k]["speedup_vs_py_tuned"] = round(base["py_tuned"]["wall_sec"] / base[k]["wall_sec"], 2)
        base[k]["speedup_vs_py_naive"] = round(base["py_naive"]["wall_sec"] / base[k]["wall_sec"], 2)
    # What actually goes to the API, from the main C++ run
    rep = evaluate.load_report(WORK / "tp_cpp.jsonl")
    kf = kf_dup = faces = faces_dup = img_dup = 0
    bytes_out = 0
    for r in rep.values():
        m = r["manifest"]
        bytes_out += m.get("bytes_out", 0)
        for a in m["assets"]:
            dup = "duplicate_of" in a or "cross_duplicate_of" in a
            if a["kind"] == "keyframe":
                kf += 1
                kf_dup += dup
            elif a["kind"] == "face":
                faces += 1
                faces_dup += dup
            elif a["kind"] == "image":
                img_dup += dup
    video_bytes = sum(os.path.getsize(v["video"]) for v in truth["videos"])
    out["api_payload"] = {
        "video_bytes_in": video_bytes,
        "asset_bytes_out": bytes_out,
        "bytes_reduction_x": round(video_bytes / max(1, bytes_out), 1),
        "scenes": kf, "duplicate_shot_keyframes_dropped": kf_dup,
        "face_crops": faces, "duplicate_face_crops_dropped": faces_dup,
        "images": len(truth["images"]), "duplicate_images_flagged": img_dup,
        "api_calls_before": kf + len(truth["images"]),
        "api_calls_after": kf - kf_dup + len(truth["images"]) - img_dup,
    }
    p = out["api_payload"]
    p["api_calls_avoided_pct"] = round(100 * (1 - p["api_calls_after"] / p["api_calls_before"]), 1)
    return out


def make_poison(dirpath: Path, truth) -> list[str]:
    dirpath.mkdir(parents=True, exist_ok=True)
    src = Path(truth["videos"][0]["video"]).read_bytes()
    trunc = dirpath / "truncated.mp4"
    trunc.write_bytes(src[: len(src) // 3])  # moov atom is at the end: unplayable
    junk = dirpath / "junk.mp4"
    junk.write_bytes(random.Random(0).randbytes(200_000))
    empty = dirpath / "empty.jpg"
    empty.write_bytes(b"")
    return [str(trunc), str(junk), str(empty), str(dirpath / "missing.mp4")]


def chaos(truth, py_clean_wall):
    WORK.mkdir(parents=True, exist_ok=True)
    poison = make_poison(WORK / "poison", truth)
    stall = subprocess.Popen([sys.executable, "bench/stall_server.py", "--port", "8765", "--file",
                              truth["videos"][0]["video"]], cwd=ROOT, stdout=subprocess.DEVNULL)
    time.sleep(0.5)
    stalls = [f"http://127.0.0.1:8765/upload_{i}.mp4" for i in range(3)]
    lines = [l for l in (CORPUS / "inputs.txt").read_text().splitlines() if l]
    rng = random.Random(5)
    bad = [f"videos\t{p}" for p in poison + stalls]
    for b in bad:
        lines.insert(rng.randrange(len(lines)), b)
    lst = WORK / "chaos_inputs.txt"
    lst.write_text("\n".join(lines) + "\n")
    env = {"MHPREP_CHAOS": "crash=0.03,hang=0.02,seed=11"}
    valid = len(lines) - len(bad)
    res = {"jobs": len(lines), "valid_jobs": valid, "bad_inputs": len(bad),
           "faults": "3% of attempts segfault, 2% hang (MHPREP_CHAOS, keyed per attempt); "
                     "3 URLs whose server stalls mid-file; truncated, random-bytes, empty and missing files"}
    try:
        s, wall, rep = run_cpp("chaos_cpp", lst, env=env, extra=["--timeout-ms", "20000", "--max-attempts", "3"])
        r = evaluate.load_report(rep)
        ok_valid = sum(1 for k, v in r.items() if v["status"] == "succeeded")
        res["cpp"] = {"wall_sec": round(wall, 1), "succeeded": ok_valid,
                      "valid_completion_pct": round(100 * ok_valid / valid, 2),
                      "failed": s["failed"], "retries": s["retries"], "timeouts": s["timeouts"],
                      "crashes": s["crashes"], "input_errors": s["input_errors"],
                      "worker_spawns": s["worker_spawns"],
                      "failed_by_kind": _kinds(r)}
        print(f"  chaos cpp: {ok_valid}/{valid} valid jobs in {wall:.0f}s")
        watchdog = max(120.0, 6 * py_clean_wall)
        s, wall, rep = run_py("chaos_py", lst, "tuned", env=env, extra=["--batch-timeout", str(watchdog)])
        r = evaluate.load_report(rep)
        ok_valid = sum(1 for v in r.values() if v["status"] == "succeeded")
        res["python"] = {"wall_sec": round(wall, 1), "succeeded": ok_valid,
                         "valid_completion_pct": round(100 * ok_valid / valid, 2),
                         "failed": s["failed"], "watchdog_sec": watchdog, "failed_by_kind": _kinds(r)}
        print(f"  chaos py: {ok_valid}/{valid} valid jobs in {wall:.0f}s")
    finally:
        stall.kill()
    return res


def _kinds(report):
    out = {}
    for v in report.values():
        if v["status"] != "succeeded":
            out[v.get("error_kind", "?")] = out.get(v.get("error_kind", "?"), 0) + 1
    return out


def http_bench():
    port = 8080
    srv = subprocess.Popen([str(BIN), "serve", "--port", str(port), "--workers", str(WORKERS),
                            "--out", str(WORK / "serve")], cwd=ROOT, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                http_load.Client(port).req("GET", "/healthz")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        c = http_load.Client(port)
        videos = http_load.read_list(CORPUS / "videos.txt", "videos")
        ids = [c.req("POST", "/v1/jobs", {"input": v, "tenant": "sat"})[1]["id"] for v in videos * 3]
        images = WORK / "images.txt"
        images.write_text("\n".join(http_load.read_list(CORPUS / "inputs.txt", "photos")) + "\n")
        out = subprocess.run(["wrk", "-t2", "-c64", "-d20s", "--timeout", "5s", "-s", "bench/wrk_mix.lua",
                              f"http://127.0.0.1:{port}"], capture_output=True, text=True, cwd=ROOT,
                             env={**os.environ, "MHPREP_IMAGES": str(images), "MHPREP_MAX_ID": str(max(ids))}).stdout
        w = json.loads(out.split("WRK_JSON ", 1)[1].splitlines()[0])
        _, health = c.req("GET", "/healthz")
        cp = {"tool": "wrk -t2 -c64 -d20s, 99.8% GET status / 0.2% POST job",
              "pool_state": f"{WORKERS} workers busy decoding {len(ids)} queued video jobs",
              "requests": w["requests"], "rps": round(w["requests"] / (w["duration_us"] / 1e6)),
              "errors": w["errors"], "p50_ms": w["p50_us"] / 1000, "p90_ms": w["p90_us"] / 1000,
              "p99_ms": w["p99_us"] / 1000, "p999_ms": w["p999_us"] / 1000, "max_ms": w["max_us"] / 1000,
              "workers_busy_after": health.get("running")}
        print(f"  http control plane: {cp['rps']} rps, p99 {cp['p99_ms']} ms")
        while True:
            _, h = c.req("GET", "/healthz")
            if h.get("queued", 0) == 0 and h.get("running", 0) == 0:
                break
            time.sleep(0.5)
        e2e = http_load.end_to_end(port, http_load.read_list(CORPUS / "inputs.txt", "photos"), 20, 30)
        print(f"  http e2e: p50 {e2e['p50_ms']} ms, p99 {e2e['p99_ms']} ms")
        import http.client
        mc = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        mc.request("GET", "/metrics")
        prom = mc.getresponse().read().decode()
        return {"control_plane_under_saturation": cp, "end_to_end_images_20_per_sec": e2e,
                "metrics_sample": [l for l in prom.splitlines() if l.startswith("mhprep_") and "bucket" not in l]}
    finally:
        srv.terminate()
        srv.wait(timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="throughput,chaos,http")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    truth = json.loads((CORPUS / "truth.json").read_text())
    only = set(a.only.split(","))
    meta = {"cpu": subprocess.run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
                                  capture_output=True, text=True).stdout.strip(),
            "cores": os.cpu_count(), "date": time.strftime("%Y-%m-%d")}
    if "throughput" in only:
        print("throughput + accuracy ...")
        tp = throughput(truth, a.repeats)
        tp["machine"] = meta
        (RESULTS / "throughput.json").write_text(json.dumps(tp, indent=2))
    if "chaos" in only:
        print("chaos ...")
        py_wall = json.loads((RESULTS / "throughput.json").read_text())["configs"]["py_tuned"]["wall_sec"]
        (RESULTS / "chaos.json").write_text(json.dumps(chaos(truth, py_wall), indent=2))
    if "http" in only:
        print("http ...")
        (RESULTS / "http.json").write_text(json.dumps({**http_bench(), "machine": meta}, indent=2))


if __name__ == "__main__":
    main()
