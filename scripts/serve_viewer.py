#!/usr/bin/env python3
"""Serve the repo root over HTTP and open the BlockSwarm replay viewer.

Usage:
    python3 scripts/serve_viewer.py [LOG_PATH] [--port 8765] [--no-open]

LOG_PATH is a .jsonl episode log inside the repo (default: the first *.jsonl in runs/demo/,
if any). The viewer opens at http://localhost:PORT/viewer/index.html?log=<path>.
Stop the server with Ctrl+C.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import sys
import urllib.parse
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def free_port(preferred: int) -> int:
    """Return `preferred` if it is free on localhost, else the next free one (or any free port)."""
    for port in [*range(preferred, preferred + 20), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
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


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # only log errors
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always show the latest log
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the BlockSwarm replay viewer.")
    ap.add_argument("log", nargs="?", help="episode .jsonl log (default: first file in runs/demo/)")
    ap.add_argument("--port", type=int, default=8765, help="preferred port (default 8765)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    query = ""
    if args.log:
        log = Path(args.log).resolve()
        if not log.is_file():
            print(f"error: log file not found: {args.log}", file=sys.stderr)
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

    port = free_port(args.port)
    handler = functools.partial(QuietHandler, directory=str(REPO))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://localhost:{port}/viewer/index.html{query}"
    print(f"Serving {REPO} at http://localhost:{port}/")
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
