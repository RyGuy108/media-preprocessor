"""A media URL whose server stops sending mid-file: the classic way a fetch-and-decode
worker wedges forever (FFmpeg's HTTP reader has no read timeout by default).

    python bench/stall_server.py --port 8765 --file corpus/test/videos/v000.mp4
    -> http://127.0.0.1:8765/<anything>.mp4 sends the headers and 64 KB, then goes silent.
"""
import argparse
import http.server
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    data = open(a.file, "rb").read()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:65536])
            self.wfile.flush()
            time.sleep(10**6)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"stalling server on :{a.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
