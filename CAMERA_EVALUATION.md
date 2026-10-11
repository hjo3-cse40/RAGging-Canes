# Camera evaluation — October 10, 2026

Use YOLO26n for the fast directional loop based on this experiment. Nemotron
`nemotron3:33b` did not produce repeatable six-zone arrays on every identical
image, even with temperature 0, thinking disabled, and seed 42. This is an
observed repeatability/latency result, not a comprehensive accuracy ranking.

## Data and results

Captured 40 distinct JPEGs from 48 attempts: 22/24 in the original ceiling-facing
view and 18/24 while the camera was repositioned. The latter includes one ceiling
frame and 17 sideways person/chair scenes. The user confirmed the camera is
mounted sideways. Created six additional upright inputs by rotating selected
person/chair captures 90° clockwise and saving JPEG at quality 95. These six are
transformations of existing captures, not independent scenes.

Each of the 46 inputs was processed three times by each configuration, for **414
inference calls**. All returned syntactically valid six-zone outputs. Every
Nemotron request used `think: false`, seed 42, and a fresh single-image chat; all
276 Nemotron responses had zero returned thinking characters. Both methods
used person/chair targets and six equal vertical image strips. YOLO confidence
was 0.35. Backend order rotated between images/passes; later passes revisited
images after processing other images.

| Configuration | Median processing | Empirical p95 | Images with identical arrays in all 3 passes |
| --- | ---: | ---: | ---: |
| YOLO26n, GPU | 23.96 ms | 27.28 ms | 46/46 (100%) |
| Nemotron, temperature 0 | 755.88 ms | 794.25 ms | 29/46 (63.0%) |
| Nemotron, temperature 0.1 | 758.02 ms | 800.44 ms | 30/46 (65.2%) |

YOLO was approximately 32× faster by aggregate median. Temperature 0.1 did not
resolve Nemotron's instability; the one-image difference is not evidence of a
meaningful improvement.

| Set | Inputs | YOLO stable | Nemotron 0 stable | Nemotron 0.1 stable |
| --- | ---: | ---: | ---: | ---: |
| Original ceiling/partial-person view | 22 | 22 | 14 | 14 |
| Repositioned, sideways view | 18 | 18 | 12 | 12 |
| Upright copies | 6 | 6 | 3 | 4 |

The deployed `/home/asus/projects/RAGging-Canes/gx10/service.py` explicitly sets
**temperature 0 and thinking off already**; it does not explicitly set a seed.
The experiment adds a fixed seed, yet differences persist. For example, the same
`run-01/frame-013.jpg` produced all zeros, all unknowns, then all zeros at
temperature 0. The same `run-02/frame-003.jpg` changed from `[0,0,0,1,1,0]` to
`[0,0,0,1,1,1]`. These changes cannot be attributed to different camera bytes.
The underlying cause (runtime numerical behavior, caching, or another factor)
was not isolated. Ollama documents temperature and seed controls in its
[parameter reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx).

## Detection quality and limits

Visual review shows a person in 17 repositioned frames. YOLO detected a person
in 15 of those on each pass; it returned no detections on two visibly occupied
frames (`run-02/frame-014.jpg` and `frame-015.jpg`, corresponding to original
capture numbers 15 and 16). Nemotron reported occupied zones in those two frames.
Conversely, Nemotron often assigned a much narrower horizontal extent than the
visible person or YOLO's bounding box. Inspect the originals and overlays before
judging individual zone boundaries.

This is a narrow presence-level observation, not labeled bounding-box mAP or
validated directional accuracy. Samples show mostly one person, one chair, and
one room; no independent human zone labels, empty forward scenes, diverse
obstacles, controlled six-position sweep, or lighting coverage were collected.
Repeatability does not imply correctness. YOLO's stable zone arrays also do not
establish bitwise-identical boxes or determinism across machines/restarts.

Rotate into the wearer-aligned orientation before inference and strip mapping.
Raw sideways strips correspond to the wrong physical axis for left/right cues.
Upright tests establish that Nemotron's variability persists after orientation
correction. Re-encoding also changes JPEG bytes; this is not a pure rotation-only
ablation. The six upright samples do not include the two sideways YOLO misses.

## Timing and capture limitations

Processing includes decode, inference, and the existing YOLO pipeline's artifact
writes/annotation, or Nemotron's HTTP request and response validation. It excludes
camera fetch, sampling interval, other backend work, ESP32 polling, and motors.
YOLO warms at construction; Nemotron was already resident. First requests can
still include setup overhead. Saved-image tests may benefit from caching. Small,
correlated samples do not establish deployment p95 or scene-change latency.

The original capture set had two timeouts, with successful fetch median 546.47 ms
(range 414.59–830.97 ms). The repositioned set had six timeouts and median
708.34 ms for successful fetches. The latter collection overlapped the first
benchmark's inference, and its tail overlapped the restored production camera
loop; **do not use it as an isolated camera performance comparison**. Camera
transfer is still substantially slower than YOLO inference. Capture timestamps
mark request time, not sensor exposure or the age of camera-buffered frames.

The original service was paused with restoration traps for model benchmarks,
then restored. Final verification showed PID 87164 running and `/health` valid
with a fresh five-zone result. Neither production backend, firmware, nor camera
orientation settings were changed.

## Reproduce and inspect

- [Offline image/result review](artifacts/camera-evaluation-20261010/review.html)
- [Combined statistics](artifacts/camera-evaluation-20261010/combined-summary.json)
- [Runtime and model hashes](artifacts/camera-evaluation-20261010/environment.json)
- [Capture/repeat evaluator](software/evaluate_camera.py)
- [Commands and measurement definitions](software/README.md#capture-samples-and-check-repeatability)

Local evidence is under `artifacts/camera-evaluation-20261010/`; artifacts remain
ignored by Git. GX10 originals and isolated source are under
`/home/asus/RAGging-Canes/artifacts/camera-evaluation-20261010/`. Each run contains
source JPEGs, first-pass YOLO overlays, captures.jsonl, results.jsonl, config.json,
and summary.json. `upright-copies/manifest.json` links transformed images to the
original hashes. Regenerate the review with:

```bash
python3 tools/render_camera_review.py artifacts/camera-evaluation-20261010
```

Runtime: NVIDIA GB10, PyTorch 2.14.1+cu130, Ultralytics 8.4.175, Ollama 0.23.2.
YOLO SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Nemotron tag digest: `f6d8b7ff496ccc53429cc480ad53971d522b443ee4a5aa58a6da49e57acf42cf`.
All **12 software tests passed**, including request-setting/image preservation
and repeatability summary handling. Existing socket tests required loopback
permissions; their first sandboxed run was blocked at bind, then passed.

## Bird's-eye display and incremental mapping

A live 2D display is a good next step, with two stages:

1. **Current sensors:** show the exact processed camera image, six directional
   strips, frame age, and a separate ultrasonic forward-range indicator. Show
   camera objects as bearings with unknown range. Do not place them at invented
   metric coordinates or treat unobserved regions as empty.
2. **Metric local map:** add calibrated depth (stereo/depth camera or other range
   sensing) and pose estimation. Detect in the upright camera image, associate
   each detection with depth, transform into a common coordinate frame using the
   camera calibration and current pose, then update a rolling occupancy grid and
   dynamic-object tracks. IMU helps orientation; persistent mapping while walking
   additionally needs translation/odometry, such as visual-inertial tracking.

A single-camera ground-plane homography can produce an approximate floor view
when intrinsics, camera height/orientation, and a flat floor are known. It maps
points on that plane; elevated objects distort, and cropped people often have
no visible floor-contact point. A head-mounted camera's changing pitch/roll makes
a fixed warp unreliable. See [OpenCV's homography explanation](https://docs.opencv.org/4.11.0/d9/dab/tutorial_homography.html).

Detect on the camera image first, then project supported geometry into the map;
warping the entire scene before YOLO is not automatically easier for detection.
One ultrasonic range is a beam measurement, not a distance for every detected
person or chair. Associate it only when geometry supports that association.

For incremental updates, retain timestamp, sensor source, pose, confidence and
uncertainty with observations. Update the display when new measurements arrive,
decay old evidence, distinguish unknown/free/occupied cells, and expire moving
objects. Clear or re-localize the map when pose tracking fails. A forward camera
only observes its field of view; the rest of the surroundings stays unknown until
observed. Render smoothly if desired, but show the actual measurement age.

Recommended next implementation: upright YOLO + a local live sector/range view,
followed by depth/pose fusion for a metric bird's-eye map. Keep Nemotron as an
optional slower semantic channel. For stable live cues, evaluate tracking and
boundary hysteresis separately from inference repeatability, preserving expiry
and unknown states; smoothing must not silently keep old detections alive.
