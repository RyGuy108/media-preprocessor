#!/usr/bin/env bash
# Smoke-test the runtime image: start the service, run a real image job and a chaos job
# end to end, then check that `docker stop` (SIGTERM through tini) drains cleanly.
#   docker build -t mhprep . && scripts/docker_smoke.sh mhprep
set -euo pipefail
IMAGE=${1:-mhprep}
PORT=${PORT:-18080}
WORK=$(mktemp -d)
trap 'docker rm -f mhprep-smoke >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
mkdir -p "$WORK/in" "$WORK/out"
chmod 777 "$WORK/out"

# A 320x180 gradient with a few blocks, written as a PNG with the standard library only.
python3 - "$WORK/in/frame.png" <<'PY'
import struct, sys, zlib
w, h = 320, 180
rows = b"".join(
    b"\0" + bytes(c for x in range(w)
                  for c in (x * 255 // w, y * 255 // h, (0, 128, 255)[(x // 40 + y // 45) % 3]))
    for y in range(h))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PY

# Run as the invoking user so the files it writes can be cleaned up afterwards.
docker run -d --name mhprep-smoke --user "$(id -u):$(id -g)" -p "$PORT:8080" -v "$WORK/in:/data/in:ro" \
  -v "$WORK/out:/data/out" "$IMAGE" serve --port 8080 --out /data/out --workers 2 >/dev/null

for _ in $(seq 1 60); do
  curl -fsS "localhost:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "localhost:$PORT/healthz"; echo

submit() { curl -fsS -XPOST "localhost:$PORT/v1/jobs" -d "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'; }
result() { curl -fsS "localhost:$PORT/v1/jobs/$1?wait_ms=60000"; }

img=$(submit '{"input": "/data/in/frame.png", "tenant": "smoke"}')
chaos=$(submit '{"input": "chaos://ok"}')
result "$img" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r["status"] == "succeeded", r
assets = r["manifest"]["assets"]
assert assets and assets[0]["kind"] == "image", assets
print("image job ok:", assets[0]["phash"], assets[0].get("path"))'
result "$chaos" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r["status"] == "succeeded", r
print("chaos job ok")'
ls -R "$WORK/out" | head

curl -fsS "localhost:$PORT/metrics" | grep -E '^mhprep_jobs_succeeded_total 2$'

docker stop -t 20 mhprep-smoke >/dev/null
code=$(docker inspect -f '{{.State.ExitCode}}' mhprep-smoke)
docker logs mhprep-smoke 2>&1 | tail -3
docker logs mhprep-smoke 2>&1 | grep -q "drained, bye" || { echo "no graceful drain"; exit 1; }
[ "$code" = 0 ] || { echo "exit code $code"; exit 1; }
echo "smoke test passed"
