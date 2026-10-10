# GitHub integration and model tests — October 10, 2026

## Integration update

The implementations have now been merged into one checkout, preserving both Git histories. The three shared-file conflicts were resolved: combined ignore rules, a shared root setup guide, and separate five-/six-zone firmware instructions. Detailed Nemotron instructions are in `gx10/README.md`. The report below describes the pre-merge investigation; its isolated directories remain benchmark evidence, not the primary development checkout. Both backend protocols remain unchanged.

Reviewed teammate commit `482528f` against local base `6d2bb01` and the uncommitted six-zone implementation. GitHub was fetched; the laptop branch was not merged, reset, committed, or pushed. Tests ran in an isolated combined snapshot, preserving both existing GX10 deployments.

## Conflicts and integration approach

| Area | Finding | Approach |
| --- | --- | --- |
| `README.md` | Three-way merge preview produces five conflict regions. | Reconcile status and setup sections; describe both backends and make six motors the current hardware contract. |
| `.gitignore` | Both sides independently added different files. | Union patterns, retaining local weights, artifacts, private token/header exclusions and teammate log/PID exclusions. |
| `firmware/README.md` | Both sides independently added different instructions. | Preserve five-zone instructions as legacy and document six-zone integration separately. |
| Python files | Teammate adds `gx10/`, `tools/`, and `tests/test_service.py`; local work adds `software/` and `tests/test_vision.py`. No same-file code collision. | Both implementations coexist and their combined tests pass. |
| Command protocol | Nemotron service and `ZoneClient.h` require **five** values; YOLO and `SixZoneClient.h` require **six**. | Do not switch a six-motor client directly to the five-zone server or pad an array. Parameterize Nemotron prompt, validation, unknown arrays, tests, and firmware contract before switching. |
| Classes/freshness/access | Nemotron requests person/chair/table/box, uses 2500 ms expiry, and currently has no bearer authentication. YOLO defaults to person/chair, 1000 ms expiry, and authenticated result-only LAN access. | Align classes for comparisons; preserve authentication and capture-age expiry in a shared six-zone endpoint. |

The isolated snapshot is `artifacts/integration-review` on this laptop and `/home/asus/RAGging-Canes/artifacts/review-482528f` on GX10. The GitHub service and benchmark source hashes match the original GX10 deployment. The local merge preview is `artifacts/integration-review/README.merge-preview.md`; it is diagnostic, not a resolved README.

Recommended sequence: checkpoint local work on a branch, integrate the teammate commit with the three documentation/ignore collisions resolved, retain both backends, then implement one six-zone protected command contract. Use YOLO for the first directional-loop trial based on measured speed, while retaining Nemotron for comparison. Validate left/center/right coverage on labeled camera scenes before connecting motors.

## Measurements

All model tests used the existing GX10 GPU runtime and `nemotron3:33b` through local Ollama. The existing Nemotron camera process was temporarily paused with an EXIT restoration trap during benchmarks and was verified running afterward. No firmware was flashed and no new persistent endpoint was launched.

| Test | Result |
| --- | --- |
| Combined software tests on GX10 | **10 passed**, including new idle/authentication capture test. |
| Fresh shared-image comparison, current backend contracts | 5 capture attempts, 3 complete JPEGs, 2 timeouts. YOLO processing 20.88–28.73 ms (median 24.72); Nemotron 708.38–724.74 ms (median 719.56). |
| Camera fetch for those 3 frames | 360.75, 1647.51, and 4228.02 ms. Socket timeout is inactivity-based, so total transfers can exceed 3 seconds. |
| Six-zone/person-chair Nemotron experiment | 4/4 syntactically valid results on the three saved fresh frames plus an earlier person image. Nemotron 718.14–770.07 ms; YOLO 18.60–36.52 ms. This only changes the benchmark prompt, not the deployed service. |
| Actual YOLO live pull with 1000 ms expiry | 5 attempts: 4 fresh results, 1 timeout yielding six unknowns and zero TTL. Successful fetch-to-result 380.88–515.08 ms (median 447.73). |
| Restored Nemotron service | PID 87164 running, `/health` valid, 5 zones; observed capture 370 ms + inference 671 ms, total 1040 ms, remaining TTL 1212 ms at poll time. Temporary port 8001 listener closed. |

On the earlier saved person image, YOLO returned `[0,0,0,1,1,1]` while six-zone Nemotron returned `[0,0,0,0,0,1]`. On the three current frames YOLO detected no person/chair; six-zone Nemotron marked the leftmost zone in two. These disagreements need image inspection and ground-truth labels; output validity is not accuracy. The native five-zone comparison also differs in classes and sector boundaries, so element-wise agreement is not meaningful.

The comparison runner feeds identical JPEG bytes sequentially to both models, alternates execution order, records SHA-256, and saves results. YOLO is warmed at startup; Nemotron was already resident. Saved-image runs may benefit from caching. `fetch_plus_processing_ms` excludes the other backend's runtime and is a component estimate, not a live command age or measured vibration latency. Tiny samples do not establish p95, accuracy, or reliable obstacle detection.

## Repeat the tests

SSH to `asus@169.233.72.164`, then run the combined tests without touching either deployment:

```bash
cd /home/asus/RAGging-Canes/artifacts/review-482528f
/home/asus/RAGging-Canes/.venv/bin/python -m unittest discover -s tests -v
```

For isolated timing, temporarily pause the original service and always restore it. Run the following block in a **bash** shell on GX10. It checks for exactly one matching process, uses a subshell EXIT trap, and bounds the benchmark to 90 seconds. If the service command has changed, inspect processes before adapting the match.

```bash
(
  cd /home/asus/RAGging-Canes/artifacts/review-482528f || exit 1
  mapfile -t service_pids < <(pgrep -f '^/usr/bin/python3 -u -m gx10.service ')
  if [ "${#service_pids[@]}" -ne 1 ]; then
    echo 'Expected exactly one existing Nemotron service; inspect processes first.'
    exit 1
  fi
  service_pid=${service_pids[0]}
  trap 'kill -CONT "$service_pid"' EXIT
  kill -STOP "$service_pid" || exit 1
  timeout 90 /home/asus/RAGging-Canes/.venv/bin/python -m software.compare_models \
    --camera-url http://169.233.72.178/capture --frames 5 \
    --model /home/asus/RAGging-Canes/yolo26n.pt --nemotron-zones 6 \
    --output "results-repeat-$(date +%s)"
)
```

Use `--nemotron-zones 5` to reproduce the teammate's original prompt/classes. Replace `--camera-url ... --frames 5` with `--images results-native/frame-1.jpg results-native/frame-3.jpg results-native/frame-4.jpg` to reuse saved images. A capture/model failure produces a nonzero exit code and is recorded; the restoration trap still runs. Changing camera addresses requires updating the URL.

Current hardware blocker remains the full camera `.ino` and six motor GPIO/driver mapping. The receive-only headers have not been compiled in that sketch or flashed. The protected idle LAN endpoint from the earlier handoff remains a separate deployment step; these tests did not activate it.
