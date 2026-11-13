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

# A 320x180 gradient with a few blocks, as a binary PPM (OpenCV reads it; no tools needed).
python3 - "$WORK/in/frame.ppm" <<'PY'
import sys
w, h = 320, 180
px = bytearray()
for y in range(h):
    for x in range(w):
        block = (x // 40 + y // 45) % 3
        px += bytes((x * 255 // w, y * 255 // h, (0, 128, 255)[block]))
open(sys.argv[1], "wb").write(b"P6\n%d %d\n255\n" % (w, h) + px)
PY

docker run -d --name mhprep-smoke -p "$PORT:8080" -v "$WORK/in:/data/in:ro" \
  -v "$WORK/out:/data/out" "$IMAGE" serve --port 8080 --out /data/out --workers 2 >/dev/null

for _ in $(seq 1 60); do
  curl -fsS "localhost:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "localhost:$PORT/healthz"; echo

submit() { curl -fsS -XPOST "localhost:$PORT/v1/jobs" -d "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'; }
result() { curl -fsS "localhost:$PORT/v1/jobs/$1?wait_ms=60000"; }

img=$(submit '{"input": "/data/in/frame.ppm", "tenant": "smoke"}')
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
