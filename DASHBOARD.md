# YOLO spatial dashboard

Branch: `codex/yolo-birdseye-dashboard`.

The dashboard pairs the exact processed JPEG with YOLO boxes, six directional
strips, freshness, timing, and a continuously updating top-down display. It runs
locally; no hosted service, external front-end assets, build system, or JavaScript
package installation is required. It does not send motor commands.

## Implementation plan and delivered scope

1. Reuse the existing YOLO pipeline and add a loopback-only dashboard server.
2. Provide live capture, saved-result replay, and a self-contained simulated demo.
3. Show image/boxes and directional observations before calibration.
4. Add four-click floor calibration and project supported box-bottom centers into
   a metric floor view. Save calibration separately from camera captures.
5. Invalidate observations on age, error, pause, and disconnection. Never extend
   an old frame's life on resume.
6. Verify geometry, HTTP controls, browser interactions, and a real GX10 camera
   run; document the physical accuracy test still needed.

Delivered: all six software steps. Physical floor-distance accuracy remains
unverified because measured floor markers were not supplied. This is an updating
local observation map, not a persistent SLAM reconstruction of the room.

## Run on this Mac

From the repository root, with Pillow installed:

```bash
python3 -m software.dashboard --demo
```

Open **http://127.0.0.1:8080**. This mode generates explicitly labeled simulated
images/detections and needs no camera, GPU, or model weights. To replay the fresh
upright camera test collected in this session:

```bash
python3 -m software.dashboard \
  --replay artifacts/camera-evaluation-20261010/upright-live --interval 1
```

Replay consumes the first valid YOLO pass in an evaluator-compatible dataset.
Images and results are saved observations; the dashboard labels them recorded
replay. The age is time since replay presentation, not original capture age.
Dataset artifacts are local and ignored by Git, so a fresh clone uses `--demo`
until you copy or collect a dataset. Install preview-only dependencies in a venv
if needed with `python -m pip install 'Pillow>=11,<13'`.

Stop with Ctrl+C. Add `--port 8081` if 8080 is already occupied.

## Run live on GX10 and view from your laptop

Run in the checkout containing this branch. Use the existing verified GPU venv;
there is no need to reinstall it. Use one capture loop at a time and close the
ESP32 browser stream. If the legacy Nemotron service is active, the wrapper can
pause it and restore it when you exit:

```bash
DASHBOARD_PYTHON=/home/asus/RAGging-Canes/.venv/bin/python \
  bash tools/run_dashboard_gx10.sh --pause-nemotron \
  --camera-url http://169.233.72.178/capture \
  --model /home/asus/RAGging-Canes/yolo26n.pt --device 0 --port 8080
```

If no Nemotron service is running, omit `--pause-nemotron`. The wrapper requires
an exact single match for the known legacy launch command; it refuses ambiguous
matches. Other capture loops must be stopped separately. Ctrl+C stops this
wrapper's dashboard and restores the paused legacy process. Do not terminate
the wrapper with SIGKILL. Pausing Nemotron suspends its existing command endpoint;
use this as a bench test, not alongside an active wearable command client.

An isolated tested source copy is also available on this GX10 at:
`/home/asus/RAGging-Canes/artifacts/yolo-dashboard-20261010`.
Run the same command from that directory after the task's final source copy.

On your laptop, open a separate terminal:

```bash
ssh -N -L 8080:127.0.0.1:8080 asus@169.233.72.164
```

Then open **http://127.0.0.1:8080** on the laptop. If a local preview uses that port,
use `-L 8081:127.0.0.1:8080` and browse `http://127.0.0.1:8081` instead.
The server deliberately binds only `127.0.0.1`; it serves camera images, so use
the SSH tunnel rather than a public/LAN listener. Addresses reflect this session
and may change when devices join a different network.

The upright camera uses `--rotate 0` (default). `--rotate 90`, `180`, or `270`
rotates clockwise before YOLO and zone mapping if the mounting changes. Reset
calibration whenever orientation, camera position, zoom, or resolution changes.
Do not mirror the image unless you also account for physical left/right.

## Calibrate the bird's-eye floor view

Without calibration the fan is schematic: labels mean image strips, not measured
angles, and marker radius does **not** mean distance. No FOV measurement has been
assumed. A single camera cannot directly measure each object's depth.

For approximate floor positions:

1. Fix the camera upright on a stable support at the intended height. Tilt enough
   to see the floor and the feet/legs of targets. It must remain stationary.
2. Put four visible tape markers on a flat floor, forming a rectangle. Measure
   their X/Y coordinates in meters from the point directly beneath the camera.
   X increases to camera-right; Y increases forward along the camera's ground
   heading. Example only: near corners `(−1,1)`, `(1,1)` and far corners `(1,3)`,
   `(−1,3)` describe a 2 m wide rectangle from 1 to 3 m ahead.
3. Click **Calibrate floor**. Capture pauses so the image cannot change underneath
   your clicks. Click near-left, near-right, far-right, then far-left.
4. Replace the example coordinates with your measured values and click
   **Apply & resume**. The display switches to meters. Crossed, collinear, or
   invalid selections are rejected. Cancel starts fresh capture without changing
   the existing calibration.
5. Check targets at additional measured points inside the marked region. Clear
   and repeat calibration if the camera moves.

The current map covers X = −5…5 m and Y = 0…8 m. Calibration coordinates must
lie within that region (Y at least 0.1 m). The saved file is
`artifacts/dashboard/calibration.json` by default. It is not silently loaded on
restart. For exactly the same fixed camera pose, explicitly reuse it with:

```bash
python -m software.dashboard --camera-url http://ESP32_IP/capture \
  --calibration artifacts/dashboard/calibration.json
```

Projection uses a planar homography and the bottom-center of each box as a
possible floor contact. It excludes bottom-cropped boxes, image-aspect changes,
points outside the calibration polygon, and estimates outside the map. Visible
feet can still be occluded or the detector's box can be inaccurate, so positions
are labeled **estimates**. Lens distortion is not corrected in this first version.
Objects lacking a supported position remain detections in the camera/zone view
and are counted as unplaced; they are not plotted at a guessed distance.

## Test setup and acceptance checklist

Use bright, even lighting; a fixed upright camera; a flat floor; tape; a tape
measure; a person or chair; and a second observer to record results. Keep the
camera stream closed. Keep initial tests stationary.

| Test | Procedure | Check |
| --- | --- | --- |
| Orientation | Move a target to physical left/right | Camera boxes and zones move to the same side; no mirroring. |
| Coverage | Full person/chair, left/center/right, hold each 5–10 s | Detection and zones match visible objects. A centered box can span both middle zones. |
| Presence/absence | Enter, stand, leave for 10 s | Current detections update; no old marker persists after expiry. |
| Calibration | Four measured floor corners | Floor grid activates only after valid correspondence. |
| Position error | Five held-out floor points within the rectangle, 10 s each | Compare plotted coordinates with tape measurements; record median/max error. A provisional demo target is ≤0.5 m median, not an achieved specification. |
| Occlusion/cropping | Crop feet at the bottom or step outside the marked region | Detection remains visible but unsupported ground position is omitted. |
| Freshness | Disconnect camera/Wi-Fi or pause capture | Zones become unknown and map observations disappear; last image is visibly historical. |
| Recovery | Reconnect/resume | Only a newly processed frame repopulates the map. |
| Camera movement | Move camera after calibration | Clear calibration and recalibrate; automatic pose tracking is not implemented. |

Repeat with chairs, multiple people, uneven lighting, and partial occlusion only
after the fixed-camera baseline. Measure camera fetch and end-to-end age too:
fast inference cannot compensate for a stalled JPEG transfer.

## Verification from this session

- Upright capture: **19/24** complete JPEGs; YOLO median processing **7.25 ms** on
  the GX10. Five attempts failed. The images confirm upright orientation but
  mostly contain a close, cropped person, followed by room/chair views. They do
  not establish a controlled six-position test or ground-distance accuracy.
- The actual new dashboard server ran against the camera on GX10 and published
  fresh paired images/objects. Camera errors later occurred; observations were
  invalidated and cleared. Pause/resume freshness checks passed.
- Six dashboard tests cover ground projection, unsupported positions, invalid
  calibration, expiry, errors, in-flight pause rejection, HTTP controls, and
  cross-origin/host rejection, connection reuse, and unchanged-frame responses. The full working-tree suite passed **18 tests**
  (including two earlier benchmark tests not part of this dashboard change).
- Browser checks covered desktop and narrow layouts, replay, pause/resume, and
  the four-click calibration/apply/clear flow on synthetic input. No browser
  console errors were observed. The synthetic calibration was cleared afterward.
- Screenshots: `artifacts/dashboard-preview.jpg` (real recorded upright input)
  and `artifacts/dashboard-calibrated-demo.jpg` (explicitly simulated metric view).

## Architecture and next stages

`software/dashboard/server.py` owns one sequential capture worker. `/api/state`
returns the exact JPEG, boxes, timing, zones, calibration, and projected objects
in one snapshot. `/api/control` pauses/resumes acquisition, and
`/api/calibration` validates and saves the four correspondences. Browser polling
and a local expiry watchdog clear old observations even if the server stops
responding. The source worker retries capture failures; it never queues frames.

This map replaces its observations each frame; it does not infer free space from
missing detections and does not accumulate an unregistered room map. Observations
already identify their source and whether their position is estimated or
bearing-only. Future depth/range inputs should add timestamped measurements in
the same coordinate frame. Persistent mapping while moving needs camera pose
tracking and depth/geometry, followed by uncertainty-aware occupancy updates and
track expiry. No ultrasonic stream, IMU, depth sensor, automatic camera-motion
detection, or SLAM is connected in this version.

## Active live session (October 10)

At the user's request, the real room-facing camera dashboard was left running
on GX10, with a laptop SSH tunnel at **http://127.0.0.1:8080**. The old local replay
servers were stopped. The live GPU service is in the isolated GX10 source folder
above and writes `dashboard-live.log`; its output is `artifacts/live/` there.
The legacy Nemotron process is intentionally paused while this dashboard runs.
Fresh frame IDs, current YOLO boxes, and recovery after camera timeouts were
verified through the tunnel. JPEG transfer remains intermittent: the UI clears
expired observations and recovers when fresh images arrive.

The launcher PID is saved in `dashboard-live.pid` in the GX10 deployment folder.
To stop the session and restore Nemotron, read that file, verify its PID still
identifies `bash tools/run_dashboard_gx10.sh`, then send that PID SIGTERM.
Avoid reusing a PID from an earlier session.

If the laptop tunnel closes, recreate it with the SSH command above. The GX10
process was launched with nohup; it does not auto-start after a GX10 reboot.
A future session will have different PIDs. This session has **no room-distance
calibration** yet; use measured floor markers before enabling metric estimates.

An isolated `.venv-dashboard` was also installed on the laptop as a fallback
when SSH briefly failed. It is ignored by Git. No laptop detector is running.
To use it deliberately instead of GX10, stop the competing camera loop first,
then run `.venv-dashboard/bin/python -m software.dashboard --camera-url
http://ESP32_IP/capture --device cpu`; model weights may download on first use.
Laptop performance has not been benchmarked.

## Faster live capture

Live mode now captures again immediately after processing each frame (default
`--interval 0`). It reuses the camera HTTP connection, reconnects after failures,
and uses a 1.5-second socket inactivity timeout. The dashboard polls at 100 ms
and avoids retransmitting the JPEG when the frame ID has not changed. The camera
footer reports recent successful capture FPS; transfer timing appears beside
YOLO processing. Replay and demo retain a 0.5-second default interval.

The initial 18-second persistent-connection check observed nine distinct frames,
with the recent arrival rate around 0.47 FPS. YOLO processing was around 22 ms;
camera transfer and intermittent connection failures limited the live rate.
The ESP32 page advertises a stream on port 81, but that port refused connections.
Continuous video requires a working stream endpoint from the teammate's camera
firmware. No camera firmware, resolution, or calibration settings were changed.

After the teammate was asked to send more images, a 25.1-second retest observed
8 new frame arrivals (about 0.32 FPS). Median successful camera transfer was
534 ms and median YOLO processing was 22 ms. Timeouts persisted. Detailed samples
are saved locally in `artifacts/dashboard-live-retest.json`; this is an indicative
live check, not a controlled network benchmark.
