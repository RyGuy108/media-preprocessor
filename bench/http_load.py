"""HTTP load test against `mhprep serve`.

Phase 1 (control plane): N client threads issue a 1:4 mix of POST /v1/jobs and
GET /v1/jobs/{id} for D seconds while every worker is busy decoding video. Measures
API latency, i.e. whether a saturated pool slows the API down.

Phase 2 (end to end): images are submitted at a fixed Poisson rate; each submitter
long-polls GET /v1/jobs/{id}?wait_ms= and records submit -> result latency.

    python bench/http_load.py --port 8080 --videos corpus/test/videos.txt --images corpus/test/inputs.txt
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import threading
import time


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * len(xs) + 0.5)) - 1))] if xs else 0.0


class Client:
    def __init__(self, port):
        self.c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)

    def req(self, method, path, body=None, retries=1):
        data = json.dumps(body) if body is not None else None
        hdr = {"Content-Type": "application/json"} if data else {}
        try:
            self.c.request(method, path, body=data, headers=hdr)
            r = self.c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        except (http.client.HTTPException, OSError):
            # Idle keep-alive connections get closed by the server; reconnect once.
            self.c.close()
            self.c = http.client.HTTPConnection(self.c.host, self.c.port, timeout=120)
            if retries > 0:
                return self.req(method, path, body, retries - 1)
            raise


def read_list(path, prefix):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith(prefix + "\t"):
            out.append(line.split("\t", 1)[1])
    return out


def control_plane(port, videos, images, threads, duration):
    c = Client(port)
    # Saturate the pool with real video work first (several passes over the corpus).
    for v in videos * 3:
        c.req("POST", "/v1/jobs", {"input": v, "tenant": "load-videos"})
    known = []
    lat = {"POST": [], "GET": []}
    errors = [0]
    lock = threading.Lock()
    stop = time.time() + duration

    def worker(seed):
        rng = random.Random(seed)
        cl = Client(port)
        while time.time() < stop:
            if rng.random() < 0.2 or not known:
                t0 = time.perf_counter()
                try:
                    st, body = cl.req("POST", "/v1/jobs", {"input": rng.choice(images),
                                                           "tenant": f"load-{seed % 4}"})
                except Exception:  # noqa: BLE001
                    errors[0] += 1
                    continue
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    lat["POST"].append(dt)
                    if st == 202:
                        known.append(body["id"])
                    else:
                        errors[0] += 1
            else:
                jid = rng.choice(known)
                t0 = time.perf_counter()
                try:
                    st, _ = cl.req("GET", f"/v1/jobs/{jid}?manifest=0")
                except Exception:  # noqa: BLE001
                    errors[0] += 1
                    continue
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    lat["GET"].append(dt)
                    if st != 200:
                        errors[0] += 1

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    allv = lat["POST"] + lat["GET"]
    return {"threads": threads, "duration_sec": round(wall, 1), "requests": len(allv),
            "rps": round(len(allv) / wall, 1), "errors": errors[0],
            "p50_ms": round(pct(allv, 50), 2), "p99_ms": round(pct(allv, 99), 2),
            "post_p99_ms": round(pct(lat["POST"], 99), 2), "get_p99_ms": round(pct(lat["GET"], 99), 2),
            "max_ms": round(max(allv), 2)}


def end_to_end(port, images, rate, duration):
    lat, fails = [], [0]
    lock = threading.Lock()
    threads = []

    def one(path):
        cl = Client(port)
        t0 = time.perf_counter()
        st, body = cl.req("POST", "/v1/jobs", {"input": path, "tenant": "e2e"})
        if st != 202:
            fails[0] += 1
            return
        st, rec = cl.req("GET", f"/v1/jobs/{body['id']}?wait_ms=120000&manifest=0")
        dt = (time.perf_counter() - t0) * 1000
        with lock:
            if rec.get("status") == "succeeded":
                lat.append(dt)
            else:
                fails[0] += 1

    rng = random.Random(1)
    t_end = time.time() + duration
    while time.time() < t_end:
        t = threading.Thread(target=one, args=(rng.choice(images),))
        t.start()
        threads.append(t)
        time.sleep(rng.expovariate(rate))
    [t.join() for t in threads]
    return {"rate_per_sec": rate, "jobs": len(threads), "succeeded": len(lat), "failed": fails[0],
            "p50_ms": round(pct(lat, 50), 1), "p99_ms": round(pct(lat, 99), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--rate", type=float, default=20)
    a = ap.parse_args()
    videos = read_list(a.videos, "videos")
    images = read_list(a.images, "photos")
    cp = control_plane(a.port, videos, images, a.threads, a.duration)
    # Let the video backlog drain so phase 2 measures an unloaded-but-busy service.
    cl = Client(a.port)
    while True:
        _, h = cl.req("GET", "/healthz")
        if h.get("queued", 0) == 0 and h.get("running", 0) == 0:
            break
        time.sleep(0.5)
    e2e = end_to_end(a.port, images, a.rate, a.duration)
    print(json.dumps({"control_plane_under_saturation": cp, "end_to_end_images": e2e}, indent=2))


if __name__ == "__main__":
    main()
