"""Bounded, sequential YOLO/Nemotron smoke test using identical JPEGs.

Run from the repository: python -m software.compare_models --help
No HTTP listener is opened. Pause any existing camera loop before a live run.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from software.vision import MAX_BYTES, Pipeline, decode_jpeg


def nemotron(data, url, timeout, zones, temperature=0, seed=None):
    # Five zones reproduces the teammate's prompt/classes; six aligns with YOLO.
    count = "five" if zones == 5 else "six"
    classes = "person, chair, table, or box" if zones == 5 else "person or chair"
    prompt = (
        f"Divide this image into {count} equal vertical zones, ordered left to right. "
        f"For each zone output 1 if a {classes} occupies it, "
        "0 if none is detected, or -1 if unassessable. Objects can occupy multiple "
        f"zones. Return only a JSON array of {count} integers. Do not estimate distance."
    )
    payload = {
        "model": "nemotron3:33b", "think": False, "stream": False,
        "keep_alive": "10m", "options": {"temperature": temperature, "num_predict": 64},
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(data).decode("ascii")]}],
        "format": {"type": "array", "items": {"type": "integer", "enum": [-1, 0, 1]},
                   "minItems": zones, "maxItems": zones},
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    request = urllib.request.Request(url.rstrip("/") + "/api/chat",
                                     json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    if body.get("done") is not True or body.get("done_reason") != "stop":
        raise ValueError("Incomplete model response")
    values = json.loads(body["message"]["content"])
    if not (isinstance(values, list) and len(values) == zones
            and all(type(v) is int and v in (-1, 0, 1) for v in values)):
        raise ValueError("Invalid zone response")
    return {"zones": values, "valid": True, "temperature": temperature, "seed": seed,
            "think": False, "thinking_chars": len(body["message"].get("thinking", "")),
            "ollama_total_ms": body.get("total_duration", 0) / 1e6,
            "load_ms": body.get("load_duration", 0) / 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera-url")
    source.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--nemotron-zones", type=int, choices=(5, 6), default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.timeout <= 0:
        parser.error("frames and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args.output / "yolo", detect=True, model=args.model, device=args.device)
    failed = False
    with (args.output / "comparison.jsonl").open("a") as log:
        for index, path in enumerate(args.images or [None] * args.frames):
            started = time.monotonic()
            row = {"frame": index + 1, "source": str(path or args.camera_url),
                   "nemotron_zone_count": args.nemotron_zones}
            try:
                if path:
                    data = path.read_bytes()
                else:
                    sep = "&" if "?" in args.camera_url else "?"
                    with urllib.request.urlopen(args.camera_url + sep + "_cb=" + str(time.time_ns()),
                                                timeout=3) as response:
                        data = response.read(MAX_BYTES + 1)
                decode_jpeg(data)
                row["fetch_ms"] = round((time.monotonic() - started) * 1000, 2)
                row["sha256"] = hashlib.sha256(data).hexdigest()
                (args.output / f"frame-{index + 1}.jpg").write_bytes(data)
                # Alternate order to expose warm-up/order effects; no concurrent inference.
                order = ("yolo", "nemotron") if index % 2 == 0 else ("nemotron", "yolo")
                row["order"] = order
                for backend in order:
                    begin = time.monotonic()
                    try:
                        if backend == "yolo":
                            result = pipeline.process(data, str(index + 1))
                            row[backend] = {"valid": result["status"] == "ok",
                                            "zones": result["zones"], "objects": result["objects"]}
                        else:
                            row[backend] = nemotron(data, args.ollama_url, args.timeout, args.nemotron_zones)
                    except Exception as error:
                        row[backend] = {"valid": False, "error": str(error)}
                        failed = True
                    row[backend]["processing_ms"] = round((time.monotonic() - begin) * 1000, 2)
                    row[backend]["fetch_plus_processing_ms"] = round(
                        row["fetch_ms"] + row[backend]["processing_ms"], 2)
                    # This excludes time waiting for the other backend: not live command TTL.
            except Exception as error:
                row["capture_error"] = str(error)
                failed = True
            line = json.dumps(row)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
