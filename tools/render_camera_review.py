"""Render saved evaluate_camera runs into an offline image/result review."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path


def render(root):
    sections = []
    for run in sorted(root.glob('run-*')):
        if not (run/'summary.json').exists():
            continue
        summary = json.loads((run/'summary.json').read_text())
        results = defaultdict(list)
        for line in (run/'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            results[row['frame']].append(row)
        body = [f'<section><h2>{html.escape(run.name)}</h2><table><tr><th>Backend</th><th>Median</th><th>p95</th><th>Stable images</th></tr>']
        for backend, s in summary.items():
            body.append(f'<tr><td>{html.escape(backend)}</td><td>{s["median_ms"]:.1f} ms</td><td>{s["p95_ms"]:.1f} ms</td><td>{s["stable_images"]}/{s["repeated_images_without_errors"]}</td></tr>')
        body.append('</table><div class="grid">')
        for line in (run/'captures.jsonl').read_text().splitlines():
            frame = json.loads(line)
            if not frame['valid']:
                continue
            rows = results[frame['frame']]
            body.append(f'<article><h3>Frame {frame["frame"]}</h3><div class="images"><a href="{run.name}/{frame["file"]}"><img loading="lazy" src="{run.name}/{frame["file"]}" alt="Original frame {frame["frame"]}"></a>')
            ann = run/f'frame-{frame["frame"]:03}.yolo.jpg'
            if ann.exists():
                body.append(f'<a href="{run.name}/{ann.name}"><img loading="lazy" src="{run.name}/{ann.name}" alt="YOLO detections and six strip boundaries"></a>')
            body.append('</div><p class="muted">Original · YOLO overlay</p>')
            for backend in summary:
                rr = [r for r in rows if r['backend'] == backend]
                distinct = len({str(r.get('zones')) for r in rr})
                body.append(f'<h4>{html.escape(backend)} <span class="{"bad" if distinct > 1 else "good"}">{"varied" if distinct > 1 else "stable"}</span></h4>')
                for r in rr:
                    output = str(r.get('zones', r.get('error', 'missing')))
                    body.append(f'<div class="result">Pass {r["repeat"]}: <code>{html.escape(output)}</code> <span class="muted">{r["processing_ms"]:.1f} ms</span></div>')
            body.append(f'<details><summary>Image hash</summary><code>{frame["sha256"]}</code></details></article>')
        body.append('</div></section>')
        sections.append(''.join(body))
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Camera model comparison</title>
<style>body{font:16px system-ui;margin:0;background:#f1f4f7;color:#182638}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:36px}h2{margin-top:44px}p{max-width:900px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}article{background:white;border:1px solid #dae1e8;border-radius:12px;padding:20px;min-width:0}.images{display:flex;gap:8px}.images a{width:50%}img{width:100%;height:230px;object-fit:contain;background:#e9edf1}h4{margin:16px 0 6px}.muted{color:#5c6b7a;font-size:13px}.good,.bad{font-size:12px;padding:3px 7px;border-radius:12px}.good{background:#d7f2e2}.bad{background:#ffe1ce}table{border-collapse:collapse;margin:20px 0}td,th{text-align:left;padding:10px 22px 10px 0;border-bottom:1px solid #ced7e0}.result{font-size:13px;margin:5px 0}code{overflow-wrap:anywhere}details{margin-top:18px;font-size:12px}@media(max-width:500px){main{padding:14px}.grid{grid-template-columns:1fr}.images{display:block}.images a{width:100%}}</style>
<main><h1>Camera model comparison</h1><p>Same saved image, three passes per configuration. Nemotron uses thinking off, fixed seed 42, and temperatures 0 / 0.1. Both backends target people and chairs in six vertical image strips. Click an image to inspect it at full size.</p><p><b>Stable means identical zone arrays in this experiment; it does not mean correct.</b> Processing times exclude camera transfer and motors. The first set mostly shows ceiling; the second contains sideways person/chair views; the third uses six upright copies. These are correlated samples, not a representative accuracy benchmark.</p>'''+''.join(sections)+'</main></html>'
    (root/'review.html').write_text(document)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
