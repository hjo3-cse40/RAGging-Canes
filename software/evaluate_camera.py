"""Capture a fixed dataset, then compare six-zone outputs and repeatability.

Example: python -m software.evaluate_camera --camera-url http://CAMERA/capture \
    --frames 24 --repeats 3 --output artifacts/evaluation-001
Output directory must be new. Existing services should be paused externally.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import urllib.request

from software.compare_models import nemotron
from software.vision import MAX_BYTES, Pipeline, decode_jpeg


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['backend']].append(row)
    summary = {}
    for backend, results in groups.items():
        valid = [r for r in results if r.get('valid')]
        images = defaultdict(list)
        for row in results:
            images[row['sha256']].append(row)
        complete = [rs for rs in images.values() if len(rs) >= 2 and all(r.get('valid') for r in rs)]
        times = sorted(r['processing_ms'] for r in valid)
        summary[backend] = {
            'requests': len(results), 'valid': len(valid),
            'median_ms': statistics.median(times) if times else None,
            'p95_ms': times[math.ceil(.95 * len(times))-1] if times else None,
            'repeated_images_without_errors': len(complete),
            'stable_images': sum(len({tuple(r['zones']) for r in rs}) == 1 for rs in complete),
            'distinct_zone_outputs': dict(Counter(str(r['zones']) for r in valid)),
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--camera-url')
    source.add_argument('--images', type=Path, nargs='+')
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--interval', type=float, default=.5)
    p.add_argument('--temperatures', type=float, nargs='+', default=[0, .1])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='yolo26n.pt')
    p.add_argument('--device', default='0')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    p.add_argument('--timeout', type=float, default=30)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.frames < 1 or args.repeats < 2 or args.interval < 0 or args.timeout <= 0
            or any(not math.isfinite(t) or t < 0 for t in args.temperatures)):
        p.error('positive frames/timeout, repeats >= 2, nonnegative interval/temperatures required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    frames = []
    with (args.output/'captures.jsonl').open('w') as log:
        for i, path in enumerate(args.images or [None]*args.frames, 1):
            start = time.monotonic()
            row = {'frame': i, 'source': str(path or args.camera_url), 'captured_at_unix': time.time()}
            try:
                if path:
                    data = path.read_bytes()
                else:
                    sep = '&' if '?' in args.camera_url else '?'
                    with urllib.request.urlopen(args.camera_url+sep+'_cb='+str(time.time_ns()), timeout=3) as r:
                        data = r.read(MAX_BYTES+1)
                image = decode_jpeg(data)
                row.update(sha256=hashlib.sha256(data).hexdigest(), width=image.width, height=image.height,
                           file=f'frame-{i:03}.jpg', valid=True)
                (args.output/row['file']).write_bytes(data)
                frames.append(row)
            except Exception as exc:
                row.update(valid=False, error=str(exc))
            row['fetch_ms'] = round((time.monotonic()-start)*1000, 2)
            log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps({'capture': row}), flush=True)
            if not path:
                time.sleep(args.interval)
    if not frames:
        return 1
    # Identical bytes across backends/repeats; fixed seed; revisit images after other images.
    pipeline = Pipeline(args.output/'yolo', detect=True, model=args.model, device=args.device)
    configs = [('yolo', None)] + [(f'nemotron_t{t:g}', t) for t in dict.fromkeys(args.temperatures)]
    rows = []
    with (args.output/'results.jsonl').open('w') as log:
        for repeat in range(args.repeats):
            for index, frame in enumerate(frames):
                data = (args.output/frame['file']).read_bytes()
                shift = (repeat+index) % len(configs)
                for backend, temp in configs[shift:]+configs[:shift]:
                    row = {'frame': frame['frame'], 'sha256': frame['sha256'],
                           'repeat': repeat+1, 'backend': backend}
                    start = time.monotonic()
                    try:
                        if temp is None:
                            result = pipeline.process(data, str(frame['frame']))
                            row.update(valid=result['status']=='ok', zones=result['zones'], objects=result['objects'])
                            if repeat == 0:
                                shutil.copyfile(args.output/'yolo/latest.annotated.jpg',
                                                args.output/f"frame-{frame['frame']:03}.yolo.jpg")
                        else:
                            row.update(nemotron(data, args.ollama_url, args.timeout, 6, temp, args.seed))
                    except Exception as exc:
                        row.update(valid=False, error=str(exc))
                    row['processing_ms'] = round((time.monotonic()-start)*1000, 2)
                    rows.append(row)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    print(json.dumps(row), flush=True)
    summary = summarize(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(any(not r.get('valid') for r in rows) or len(frames) != len(args.images or [None]*args.frames))


if __name__ == '__main__':
    raise SystemExit(main())
