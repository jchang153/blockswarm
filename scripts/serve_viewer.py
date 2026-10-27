#!/usr/bin/env python3
"""Serve the repo root over HTTP and open the BlockSwarm replay viewer.

Usage:
    python3 scripts/serve_viewer.py [LOG_PATH] [--port 8765] [--host 127.0.0.1] [--follow] [--no-open]

LOG_PATH is a .jsonl episode log inside the repo (default: the first *.jsonl in runs/demo/,
if any). The viewer opens at http://localhost:PORT/viewer/index.html?log=<path>.

--follow adds &follow=1 so the viewer polls the log every second and shows new events as the
engine writes them (the log may not exist yet; the viewer waits for it). The server supports
HTTP Range requests (bytes=N- and bytes=N-M) so the viewer only downloads new bytes.

Stop the server with Ctrl+C.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import socket
import sys
import urllib.parse
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


def free_port(host: str, preferred: int) -> int:
    """Return `preferred` if it is free on `host`, else the next free one (or any free port)."""
    for port in [*range(preferred, preferred + 20), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port found")


def default_log() -> Path | None:
    demo = REPO / "runs" / "demo"
    if demo.is_dir():
        logs = sorted(demo.glob("*.jsonl"))
        if logs:
            return logs[0]
    return None


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with no-store caching and single-range `Range: bytes=N-[M]` support."""

    def log_message(self, format: str, *args) -> None:
        # only log errors; a followed log that doesn't exist yet 404s every second, so skip those
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            if str(args[1]) == "404" and ".jsonl" in str(args[0]):
                return
            super().log_message(format, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always show the latest log
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):  # type: ignore[override]
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = RANGE_RE.match(rng.strip())
        if not m:
            return super().send_head()  # unsupported range syntax (e.g. multiple ranges): send it all
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f.seek(start)
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return _LimitedFile(f, length)


class _LimitedFile:
    """File wrapper that yields at most `remaining` bytes (copyfile reads until EOF)."""

    def __init__(self, f, remaining: int):
        self.f, self.remaining = f, remaining

    def read(self, n: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        n = self.remaining if n < 0 else min(n, self.remaining)
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self) -> None:
        self.f.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the BlockSwarm replay viewer.")
    ap.add_argument("log", nargs="?", help="episode .jsonl log (default: first file in runs/demo/)")
    ap.add_argument("--port", type=int, default=8765, help="preferred port (default 8765; the next free one is used if taken)")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1; 0.0.0.0 for all interfaces)")
    ap.add_argument("--follow", action="store_true", help="follow the log live (it may not exist yet)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    query = ""
    log: Path | None
    if args.log:
        log = Path(args.log).resolve()
        if not log.is_file() and not args.follow:
            print(f"error: log file not found: {args.log} (use --follow to wait for it)", file=sys.stderr)
            return 1
    else:
        log = default_log()
        if log is None:
            print("note: no log given and runs/demo/ has no .jsonl files; the viewer will ask for a file.")
    if log is not None:
        try:
            rel = log.relative_to(REPO)
        except ValueError:
            print(f"error: the log must be inside the repo ({REPO}) so the server can serve it", file=sys.stderr)
            return 1
        # path relative to the repo root, absolute in URL terms
        query = "?log=" + urllib.parse.quote("/" + rel.as_posix())
        if args.follow:
            query += "&follow=1"
            if not log.exists():
                print(f"note: {rel} doesn't exist yet; the viewer will wait for it.")

    port = free_port(args.host, args.port)
    handler = functools.partial(ViewerHandler, directory=str(REPO))
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    shown_host = "localhost" if args.host in ("127.0.0.1", "localhost", "0.0.0.0", "") else args.host
    url = f"http://{shown_host}:{port}/viewer/index.html{query}"
    print(f"Serving {REPO} at http://{shown_host}:{port}/ (listening on {args.host})")
    print(f"Viewer: {url}")
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
