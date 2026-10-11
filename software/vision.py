"""JPEG receiver, snapshot puller, and single-image YOLO baseline.

Run `python software/vision.py --help`. Torch/Ultralytics are only imported
when detection is enabled; receive-only mode needs Pillow.
"""
import argparse
import io
import hmac
import json
import threading
import time
import urllib.request
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

MAX_BYTES = 2_000_000
MAX_PIXELS = 12_000_000


def decode_jpeg(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("JPEG must contain 1 to 2,000,000 bytes")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError("Only JPEG images are accepted")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image dimensions exceed limit")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or incomplete JPEG") from exc


def zone_states(boxes, width, boundaries):
    """Map horizontal box overlap to six sectors using calibrated pixel edges."""
    zones = [0] * 6
    for x1, x2 in boxes:
        left, right = max(0, x1 / width), min(1, x2 / width)
        for i in range(6):
            if max(left, boundaries[i]) < min(right, boundaries[i + 1]):
                zones[i] = 1
    return zones


def parse_boundaries(value):
    edges = [float(x) for x in value.split(",")]
    if (len(edges) != 7 or edges[0] != 0 or edges[-1] != 1
            or not all(a < b for a, b in zip(edges, edges[1:]))):
        raise argparse.ArgumentTypeError("Provide seven increasing edges from 0 to 1")
    return edges


class Pipeline:
    def __init__(self, output, detect=False, model="yolo26n.pt", device="0",
                 classes="person,chair", confidence=0.35, boundaries=None,
                 max_result_age_ms=1000, rotate=0, save_disk=True,
                 predict_horizon=True):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.boundaries = boundaries or [i / 6 for i in range(7)]
        self.rotate = rotate
        self.save_disk = save_disk
        self.predict_horizon = predict_horizon
        self.predictor = None
        if predict_horizon:
            try:
                from software.predictor import TrajectoryPredictor
            except ImportError:
                from predictor import TrajectoryPredictor
            self.predictor = TrajectoryPredictor(boundaries=self.boundaries)
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.latest = None
        self.latest_started = None
        self.session_id = uuid.uuid4().hex
        self.command_sequence = 0
        self.last_command_poll = None
        self.max_result_age_ms = max_result_age_ms
        self.model = None
        self.device = device
        self.confidence = confidence
        self.class_names = {x.strip() for x in classes.split(",") if x.strip()}
        if not self.class_names:
            raise ValueError("At least one target class is required")
        if detect:
            from ultralytics import YOLO
            self.model = YOLO(model)
            known = set(self.model.names.values())
            if not self.class_names <= known:
                raise ValueError(f"Unsupported classes: {self.class_names - known}")
            # Warm up and fail immediately if the selected GPU/runtime cannot run.
            self.model.predict(Image.new("RGB", (640, 480)), device=device,
                               imgsz=640, verbose=False)

    def current_result(self):
        with self.state_lock:
            if self.latest is None:
                return {"status": "waiting", "zones": [-1] * 6}
            result = dict(self.latest)
            age_ms = (time.monotonic() - self.latest_started) * 1000
        result["observation_age_ms"] = round(age_ms, 2)
        result["remaining_valid_ms"] = max(0, int(self.max_result_age_ms - age_ms))
        if age_ms >= self.max_result_age_ms:
            result["status"] = "expired"
            result["zones"] = [-1] * 6
        return result

    def process(self, data, frame_id, observation_started=None, image_input=None):
        started = time.monotonic()
        if image_input is not None:
            image = image_input
            if hasattr(image, "shape"):  # NumPy ndarray
                width, height = image.shape[1], image.shape[0]
            else:
                width, height = image.width, image.height
        else:
            image = decode_jpeg(data)
            if self.rotate:
                image = image.rotate(-self.rotate, expand=True)
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=95)
                data = output.getvalue()
            width, height = image.width, image.height
        decoded = time.monotonic()
        objects, boxes = [], []
        annotated = None
        if self.model is not None:
            class_ids = [i for i, name in self.model.names.items()
                         if name in self.class_names]
            result = self.model.predict(image, device=self.device, imgsz=640,
                                        conf=self.confidence, classes=class_ids,
                                        verbose=False)[0]
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                objects.append({"class": result.names[int(box.cls.item())],
                                "confidence": round(float(box.conf.item()), 4),
                                "xyxy": xyxy})
                boxes.append((xyxy[0], xyxy[2]))
            if self.save_disk:
                # plot() returns a BGR array; convert to RGB for Pillow.
                annotated = Image.fromarray(result.plot()[:, :, ::-1])
                draw = ImageDraw.Draw(annotated)
                for edge in self.boundaries[1:-1]:
                    x = int(edge * width)
                    draw.line((x, 0, x, height), fill="yellow", width=2)
        inferred = time.monotonic()
        source_started = observation_started if observation_started is not None else started
        scenario = None
        if self.predictor is not None and (self.model is not None or objects):
            objects, scenario = self.predictor.update(
                objects, width, height, timestamp=source_started
            )
        zones_6 = zone_states(boxes, width, self.boundaries) if self.model is not None else [-1] * 6
        zones_3 = (
            [1 if (zones_6[0] == 1 or zones_6[1] == 1) else 0,
             1 if (zones_6[2] == 1 or zones_6[3] == 1) else 0,
             1 if (zones_6[4] == 1 or zones_6[5] == 1) else 0]
            if self.model is not None else [-1, -1, -1]
        )
        response = {
            "frame_id": frame_id,
            "status": "ok" if self.model is not None else "received_only",
            "zones": zones_6,
            "zones_3": zones_3,
            "objects": objects, "bytes": len(data),
            "width": width, "height": height,
            "decode_ms": round((decoded - started) * 1000, 2),
            "inference_ms": round((inferred - decoded) * 1000, 2),
            "sector_basis": "configured_pixel_boundaries",
        }
        if scenario is not None:
            response["scenario"] = scenario
            response["predicted_zones_1s"] = scenario["predicted_zones_1s"]
            response["predicted_zones_2s"] = scenario["predicted_zones_2s"]
            response["predicted_zones_3s"] = scenario["predicted_zones_3s"]
            response["predicted_zones_3_2s"] = scenario.get("predicted_zones_3_2s", zones_3)
        if self.save_disk:
            # Atomic replacement prevents viewers reading a partially written image.
            temp = self.output / "latest.tmp"
            temp.write_bytes(data)
            temp.replace(self.output / "latest.jpg")
            if annotated is not None:
                annotated.save(self.output / "annotated.tmp", format="JPEG")
                (self.output / "annotated.tmp").replace(self.output / "latest.annotated.jpg")
            (self.output / "latest.json").write_text(json.dumps(response, indent=2))
        response["processing_ms"] = round((time.monotonic() - started) * 1000, 2)
        source_started = observation_started if observation_started is not None else started
        response["observation_to_result_ms"] = round(
            (time.monotonic() - source_started) * 1000, 2)
        with self.state_lock:
            self.latest_started = source_started
            self.command_sequence += 1
            response["command_sequence"] = self.command_sequence
            self.latest = response
        return self.current_result()


def make_handler(pipeline, token=None, commands_only=False):
    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            if token is None:
                return True
            actual = self.headers.get("Authorization", "").encode()
            if hmac.compare_digest(actual, ("Bearer " + token).encode()):
                return True
            self.close_connection = True
            self.send(401, {"error": "Authentication required"})
            return False

        def send(self, code, body, content_type="application/json"):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.authorized():
                return
            parsed_url = urllib.parse.urlsplit(self.path)
            route_path = parsed_url.path
            if commands_only and route_path not in ("/health", "/command"):
                return self.send(404, {"error": "Not found"})
            if route_path == "/health":
                return self.send(200, {"ready": True, "detection": pipeline.model is not None})
            if route_path == "/latest.json":
                return self.send(200 if pipeline.latest else 404,
                                 pipeline.current_result())
            if route_path == "/command":
                with pipeline.state_lock:
                    pipeline.last_command_poll = time.monotonic()
                result = pipeline.current_result()
                valid = result["status"] == "ok"
                qs = urllib.parse.parse_qs(parsed_url.query)
                want_3 = qs.get("zones", ["6"])[0] == "3"
                z6 = result["zones"] if valid else [-1] * 6
                z3 = (
                    [1 if (z6[0] == 1 or z6[1] == 1) else 0,
                     1 if (z6[2] == 1 or z6[3] == 1) else 0,
                     1 if (z6[4] == 1 or z6[5] == 1) else 0]
                    if valid else [-1, -1, -1]
                )
                return self.send(200, {
                    "session_id": pipeline.session_id,
                    "frame_id": result.get("command_sequence", 0),
                    "status": result["status"], "valid": valid,
                    "zones": z3 if want_3 else z6,
                    "zones_3": z3,
                    "ttl_ms": min(2500, result.get("remaining_valid_ms", 0)) if valid else 0,
                })
            if route_path in ("/latest.jpg", "/latest.annotated.jpg"):
                path = pipeline.output / route_path[1:]
                if path.exists():
                    return self.send(200, path.read_bytes(), "image/jpeg")
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized():
                return
            if commands_only:
                return self.send(404, {"error": "Not found"})
            if self.path != "/infer":
                return self.send(404, {"error": "Not found"})
            # No inference queue: caller retries using a freshly captured image.
            if not pipeline.lock.acquire(blocking=False):
                self.close_connection = True
                return self.send(429, {"error": "Busy; retry with a fresh frame"})
            try:
                observation_started = time.monotonic()
                self.connection.settimeout(10)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("Content-Length must be 1 to 2,000,000")
                frame_id = self.headers.get("X-Frame-ID", "")
                if not frame_id or len(frame_id) > 64:
                    raise ValueError("X-Frame-ID is required, maximum 64 characters")
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("Incomplete upload")
                self.send(200, pipeline.process(data, frame_id, observation_started))
            except (ValueError, TimeoutError) as exc:
                self.close_connection = True
                self.send(400, {"error": str(exc)})
            except Exception:
                self.close_connection = True
                self.send(500, {"error": "Inference failed; no valid directional result"})
                import traceback
                traceback.print_exc()
            finally:
                pipeline.lock.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "image", "pull"])
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", default="0", help="GPU index; cpu for laptop debugging")
    parser.add_argument("--classes", default="person,chair")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--boundaries", type=parse_boundaries,
                        help="Seven normalized pixel edges; default equal image strips")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--path", help="Saved JPEG for image mode")
    parser.add_argument("--url", help="ESP32 JPEG snapshot URL, not MJPEG stream")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-result-age-ms", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=3.0, help="Snapshot fetch timeout in seconds")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=270,
                        help="Clockwise rotation before detection (default: 270 for 90° CW mount / 90° CCW correction)")
    parser.add_argument("--frames", type=int, default=0, help="Stop pull mode after N attempts; 0 runs until stopped")
    parser.add_argument("--token-file", help="Bearer token file; required for a LAN listener")
    parser.add_argument("--idle-without-client", action="store_true",
                        help="Pull only while /command has been polled within five seconds")
    args = parser.parse_args()
    if args.mode == "image" and not args.path:
        parser.error("image mode requires --path")
    if args.mode == "pull" and (not args.url or args.interval <= 0):
        parser.error("pull mode requires --url and a positive --interval")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.max_result_age_ms <= 0:
        parser.error("--max-result-age-ms must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.frames < 0:
        parser.error("--frames cannot be negative")
    token = Path(args.token_file).read_text().strip() if args.token_file else None
    if token is not None and (len(token) < 32 or any(c.isspace() for c in token)):
        parser.error("Token must be at least 32 characters with no whitespace")
    commands_only = args.host not in ("127.0.0.1", "localhost")
    if args.mode != "image" and commands_only and not token:
        parser.error("A LAN listener requires --token-file; use loopback for SSH testing")
    pipeline = Pipeline(args.output, args.detect, args.model, args.device,
                        args.classes, args.confidence, args.boundaries,
                        args.max_result_age_ms, rotate=args.rotate)
    if args.mode == "image":
        print(json.dumps(pipeline.process(Path(args.path).read_bytes(), "offline-1"), indent=2))
    elif args.mode == "serve":
        server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline, token, commands_only))
        print(f"Ready on {args.host}:{args.port}; detection={args.detect}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline, token, commands_only))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Pulling {args.url}; results on port {args.port}", flush=True)
        frame_id = 0
        session_id = uuid.uuid4().hex[:8]
        try:
            while args.frames == 0 or frame_id < args.frames:
                if args.idle_without_client:
                    with pipeline.state_lock:
                        last_poll = pipeline.last_command_poll
                    if last_poll is None or time.monotonic() - last_poll > 5:
                        time.sleep(0.25)
                        continue
                started = time.monotonic()
                frame_id += 1
                try:
                    parts = urllib.parse.urlsplit(args.url)
                    query = urllib.parse.parse_qsl(parts.query) + [("_cb", str(time.time_ns()))]
                    snapshot_url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
                    request = urllib.request.Request(snapshot_url, headers={"Cache-Control": "no-cache"})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        data = response.read(MAX_BYTES + 1)
                    with pipeline.lock:
                        result = pipeline.process(data, f"{session_id}-{frame_id}", started)
                    result["fetch_to_result_ms"] = round((time.monotonic() - started) * 1000, 2)
                except Exception as exc:
                    with pipeline.lock:
                        with pipeline.state_lock:
                            pipeline.latest_started = started
                            pipeline.latest = {"frame_id": f"{session_id}-{frame_id}", "status": "error",
                                               "error": str(exc), "zones": [-1] * 6}
                    result = pipeline.current_result()
                result["logged_at_unix_ms"] = int(time.time() * 1000)
                line = json.dumps(result)
                with (pipeline.output / "frames.jsonl").open("a") as log:
                    log.write(line + "\n")
                print(line, flush=True)
                time.sleep(max(0, args.interval - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
