"""Continuously pull ESP32 JPEGs and serve fresh Nemotron zone results."""

import argparse
import base64
import json
import logging
import math
from pathlib import Path
import signal
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools.benchmark_vision import PROMPT, SCHEMA

MODEL = "nemotron3:33b"
UNKNOWN = [-1] * 5


def validate_zones(body):
    if not isinstance(body, dict) or body.get("done") is not True or body.get("done_reason") != "stop":
        raise ValueError("Incomplete model response")
    zones = json.loads(body["message"]["content"])
    if not (isinstance(zones, list) and len(zones) == 5
            and all(type(v) is int and v in (-1, 0, 1) for v in zones)):
        raise ValueError("Expected five integer zone values in [-1, 0, 1]")
    return zones


class State:
    def __init__(self, max_age_ms=2500, clock=time.monotonic):
        self.lock = threading.Lock()
        self.clock = clock
        self.max_age_ms = max_age_ms
        self.session_id = uuid.uuid4().hex
        self.frame_id = 0
        self.started = None
        self.zones = UNKNOWN.copy()
        self.status = "starting"
        self.error = None
        self.timing = {}
        self.polls = 0
        self.last_client = None
        self.last_poll = None

    def publish(self, started, zones, timing):
        with self.lock:
            self.frame_id += 1
            self.started = started
            self.zones = zones.copy()
            self.status = "ok"
            self.error = None
            self.timing = timing.copy()

    def fail(self, error):
        with self.lock:
            self.status = "error"
            self.error = type(error).__name__
            self.zones = UNKNOWN.copy()

    def snapshot(self, client=None):
        with self.lock:
            now = self.clock()
            if client is not None:
                self.polls += 1
                self.last_client = client
                self.last_poll = now
            age = math.ceil((now - self.started) * 1000) if self.started is not None else None
            ttl = max(0, self.max_age_ms - age) if age is not None else 0
            status = self.status
            if status == "ok" and ttl <= 0:
                status = "stale"
            valid = status == "ok"
            return {
                "session_id": self.session_id, "frame_id": self.frame_id,
                "model": MODEL, "status": status, "valid": valid,
                "zones": self.zones.copy() if valid else UNKNOWN.copy(),
                "ttl_ms": ttl if valid else 0, "frame_age_ms": age,
                "max_age_ms": self.max_age_ms, "error": self.error,
                "timing": self.timing.copy(),
                "poll_count": self.polls, "last_client_ip": self.last_client,
                "last_poll_age_ms": round((now - self.last_poll) * 1000) if self.last_poll is not None else None,
            }


def infer_once(camera_url, ollama_url, camera_timeout, model_timeout):
    started = time.monotonic()
    separator = "&" if "?" in camera_url else "?"
    with urllib.request.urlopen(camera_url + separator + "_cb=" + str(time.time_ns()), timeout=camera_timeout) as r:
        image = r.read(8 * 1024 * 1024 + 1)
    if len(image) > 8 * 1024 * 1024 or not image.startswith(b"\xff\xd8"):
        raise ValueError("Camera did not return a JPEG under 8 MiB")
    fetched = time.monotonic()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT,
                      "images": [base64.b64encode(image).decode("ascii")]}],
        "format": SCHEMA, "think": False, "stream": False, "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 64},
    }
    request = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=model_timeout) as r:
        body = json.load(r)
    zones = validate_zones(body)
    completed = time.monotonic()
    return started, zones, {
        "capture_ms": round((fetched - started) * 1000),
        "inference_ms": round((completed - fetched) * 1000),
        "total_ms": round((completed - started) * 1000),
    }


def worker(state, stop, args):
    while not stop.is_set():
        try:
            started, zones, timing = infer_once(args.camera_url, args.ollama_url, args.camera_timeout, args.model_timeout)
            state.publish(started, zones, timing)
            logging.info("%s", json.dumps({"frame_id": state.frame_id, "zones": zones, **timing}))
            stop.wait(args.interval)
        except Exception as error:
            # Invalidate the last result immediately; the HTTP endpoint remains responsive.
            state.fail(error)
            logging.warning("Capture/inference failed: %s", error)
            stop.wait(1)


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/firmware/ZoneClient.h", "/firmware/README.md"):
                file = Path(__file__).resolve().parent.parent / path.lstrip("/")
                try:
                    data = file.read_bytes()
                except FileNotFoundError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path not in ("/", "/health", "/command"):
                self.send_error(404)
                return
            body = json.dumps(state.snapshot(self.client_address[0] if path == "/command" else None)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-url", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-age-ms", type=int, default=2500)
    parser.add_argument("--camera-timeout", type=float, default=3)
    parser.add_argument("--model-timeout", type=float, default=30)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_age_ms <= 0 or min(args.camera_timeout, args.model_timeout) <= 0 or args.interval < 0:
        parser.error("timeouts and max-age must be positive; interval must be nonnegative")
    for url in (args.camera_url, args.ollama_url):
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            parser.error("URLs must use HTTP or HTTPS")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = State(args.max_age_ms)
    stop = threading.Event()
    server = ThreadingHTTPServer((args.bind, args.port), handler_for(state))
    server.daemon_threads = True
    threading.Thread(target=worker, args=(state, stop, args), daemon=True).start()

    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    logging.info("Serving %s:%s with %s", args.bind, args.port, MODEL)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
