#!/usr/bin/env python3
"""Benchmark five-zone vision responses against a local Ollama server.

Example: python3 tools/benchmark_vision.py --model nemotron3:33b image1.jpg image2.jpg
Each stdout line is a JSON result. Files repeat when --rounds exceeds one and
may benefit from caching. With --camera-url, each round fetches a new JPEG.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


PROMPT = (
    "Divide this image into five equal vertical zones, ordered left to right. "
    "For each zone output 1 if a person, chair, table, or box occupies it, "
    "0 if none is detected, or -1 if unassessable. Objects can occupy multiple "
    "zones. Return only a JSON array of five integers. Do not estimate distance."
)
SCHEMA = {
    "type": "array", "items": {"type": "integer", "enum": [-1, 0, 1]},
    "minItems": 5, "maxItems": 5,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--camera-url", help="HTTP(S) JPEG capture endpoint; fetched anew for each round")
    parser.add_argument("--model", default="nemotron3:33b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout <= 0:
        parser.error("rounds and timeout must be positive")
    if bool(args.images) == bool(args.camera_url):
        parser.error("provide image files OR --camera-url")
    if args.camera_url and urllib.parse.urlsplit(args.camera_url).scheme not in ("http", "https"):
        parser.error("camera URL must use HTTP or HTTPS")

    failed = False
    for round_index in range(args.rounds):
        for path in args.images or [args.camera_url]:
            result = {"model": args.model, "image": str(path), "round": round_index + 1}
            started = time.perf_counter()
            try:
                if args.camera_url:
                    separator = "&" if "?" in args.camera_url else "?"
                    capture_url = args.camera_url + separator + "_cb=" + str(time.time_ns())
                    with urllib.request.urlopen(capture_url, timeout=10) as response:
                        data = response.read(8 * 1024 * 1024 + 1)
                    if len(data) > 8 * 1024 * 1024 or not data.startswith(b"\xff\xd8"):
                        raise ValueError("Camera must return a JPEG of at most 8 MiB")
                    result["camera_fetch_ms"] = round((time.perf_counter() - started) * 1000, 2)
                else:
                    data = path.read_bytes()
                result["image_sha256"] = hashlib.sha256(data).hexdigest()
                result["image_bytes"] = len(data)
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": PROMPT,
                                  "images": [base64.b64encode(data).decode("ascii")]}],
                    "format": SCHEMA, "think": False, "stream": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0, "num_predict": 64},
                }
                request = urllib.request.Request(
                    args.url.rstrip("/") + "/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    body = json.load(response)
                if body.get("error"):
                    raise ValueError(body["error"])
                result["response"] = body.get("message", {}).get("content", "")
                result["done_reason"] = body.get("done_reason")
                for name in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
                    if name in body:
                        result[name + "_ms"] = round(body[name] / 1_000_000, 2)
                for name in ("prompt_eval_count", "prompt_eval_cached_count", "eval_count"):
                    if name in body:
                        result[name] = body[name]
                if body.get("done") is not True or body.get("done_reason") != "stop":
                    raise ValueError("Model did not finish normally; reject this response")
                zones = json.loads(result["response"])
                if not (isinstance(zones, list) and len(zones) == 5
                        and all(type(value) is int and value in (-1, 0, 1) for value in zones)):
                    raise ValueError("Expected exactly five integers drawn from -1, 0, 1")
                result["zones"] = zones
                result["valid"] = True
            except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as error:
                result.update(valid=False, error=str(error))
                failed = True
            result["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
            print(json.dumps(result), flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
