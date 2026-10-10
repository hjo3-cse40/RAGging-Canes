# Continuation memory — October 10, 2026

Read this file before continuing the implementation. The user explicitly requested saving this work for a new chat. No new chat was created. Do not restart setup or repeat approval questions already resolved below.

## Objective and constraints

- Build the first software implementation for a camera cap that provides directional vibration feedback.
- User constraint: forward coverage is limited to 180°, with **six vibration feedback devices/motors**, superseding the original README's five-zone design.
- Actual camera horizontal FOV is not measured. Six equal pixel strips are implemented, **not proven equal 30° sectors**. Calibrate pixel-to-angle mapping before claiming angular coverage.
- Map a box to every horizontal sector it overlaps; a centered box crossing the midpoint cues motors 3 and 4.
- Prefer fresh JPEG detection over SLAM/world mapping for v1. Keep short-lived state and reject expired observations. Keep ultrasonic sensing and motor timing independent of network work.
- First demo: stationary wearer and supported objects moved left/right. No camera-based distance is measured.
- User asked us to perform steps 1–5: receive JPEG on GX10, inspect locally, detect saved JPEG, connect live detection, prepare six-zone ESP32 return path.

## Access and machines

- Local repo: `/Users/samjo/RAGging-Canes`.
- GX10: `asus@169.233.72.164`, hostname `148V`, ARM64, NVIDIA GB10 GPU, driver **580.142**, DGX Spark OS 7.5.0.
- SSH works from this laptop with the existing key:
  `ssh -i /Users/samjo/.ssh/id_ed25519 asus@169.233.72.164`.
- User installed the laptop public key using `ssh-copy-id` after exiting the GX10 session. Do not ask for their password in chat or create a new SSH key unnecessarily.
- Camera: `http://169.233.72.178/capture`; settings endpoint `/status`. Browser page advertises MJPEG at port 81 `/stream`; we use snapshots.
- IPs are current network observations, not permanent configuration. Confirm if connectivity changes.
- Actual JPEGs are **640×480**, typically 18–39 KB in observed frames, rather than the initial 960 KB estimate.
- Connectivity has been intermittent: complete JPEGs, slow partial transfers, timeouts, and occasional no-route errors. User confirmed the camera is on. It subsequently returned HTTP 200 from both laptop and GX10. Do not equate network failure with camera power-off.

## Two different GX10 directories — preserve both

1. Existing deployment: `/home/asus/projects/RAGging-Canes`.
   - Contains uncommitted `gx10/service.py`, `tools/benchmark_vision.py`, `firmware/ZoneClient.h`, tests, and README changes from other work.
   - Runs **Nemotron `nemotron3:33b`** and **five-zone** results on `169.233.72.164:8000`.
   - Observed process **PID 87164**, command:
     `/usr/bin/python3 -u -m gx10.service --camera-url http://169.233.72.178/capture --bind 169.233.72.164 --port 8000`.
   - At handoff, PID 87164 was running (`Ssl`), not paused.
   - Do not overwrite or assume this checkout matches the laptop. Its five-zone client rejects six-element arrays.
2. New isolated deployment: `/home/asus/RAGging-Canes`.
   - Contains our `software/vision.py`, requirements, notes, six-zone header, and artifacts.
   - GPU venv: `.venv`; verified CPU baseline venv: `.venv-cpu`.
   - Reserved port **8001**, defaults to loopback.
   - At handoff **no port-8001 listener was running**. Bounded benchmarks shut down normally.
   - The local `vision.py` has a final `--idle-without-client` change that has **not yet been copied to GX10 or tested**. Other authentication/result-only changes were copied earlier.

## Implemented local files

- `software/vision.py`: Pillow JPEG validation; binary HTTP receiver; snapshot puller with cache-busting query; saved-image detection; warmed YOLO26n; person/chair allowlist; six box-overlap sectors; configurable normalized boundaries; response expiry; per-frame log; bounded `--frames`; annotated images; compact `/command`; optional bearer-token authentication.
- Default host `127.0.0.1`, default port `8001`, default device `0` (GPU), confidence `0.35`, pull interval 1 second, max observation age 1000 ms.
- JPEG limit 2,000,000 bytes and 12 million pixels. Invalid/partial images are rejected.
- Non-loopback listener requires `--token-file` and exposes only `/health` and `/command`, **no image routes or uploads**. Loopback permits `/latest.jpg`, `/latest.annotated.jpg`, `/latest.json`, and POST `/infer`.
- `/command`: `{session_id: 32-character UUID, frame_id: numeric sequence, status, valid, zones: six integers, ttl_ms}`. Invalid/expired returns six `-1` values and zero TTL.
- `--idle-without-client`: final local addition; only pulls when an authorized `/command` poll occurred within five seconds. Intended to avoid unnecessary camera traffic before firmware is attached. Validate it before deployment.
- Pull timing starts at fetch request start, not actual sensor exposure. `--timeout` is socket inactivity, not an absolute transfer deadline.
- `firmware/SixZoneClient.h`: receive-only adaptation of the existing five-zone header. Uses ArduinoJson 7, HTTPClient, WiFi, and a FreeRTOS task. Accepts optional bearer token, six values, session/frame IDs, TTL ≤2500 ms; subtracts polling RTT and never extends a repeated frame's deadline. Polls every 200 ms plus network time.
- Firmware header does **not drive GPIO**, has **not been compiled in the full sketch**, and has **not been flashed**. No full `.ino`, motor pin mapping, or driver information has been supplied.
- `.gitignore` ignores venvs, artifacts, weights, `private/`, and `firmware/PrivateConfig.h`.
- `tests/test_vision.py`: four tests passed after auth/result-only changes: JPEG HTTP receipt/retrieval and corruption rejection, busy rejection, calibrated-strip/center-pair mapping, expiry, compact command schema, unauthenticated rejection, and image-route exclusion. Final idle flag was added afterward.
- `software/README.md` and main README need their stale **GPU-unverified** statements updated using the evidence below; setup notes already contain CPU evidence.
- Local changes are uncommitted. No push, branch, PR, or firmware flash performed.

## Verified inference and measured results

GPU runtime successfully installed and tested:

- PyTorch **2.14.1+cu130**; torchvision **0.29.1+cu130**.
- Ultralytics **8.4.175**; Pillow **12.3.0**.
- An actual CUDA matrix operation succeeded on NVIDIA GB10.
- YOLO26n downloaded from official Ultralytics assets (`v8.4.0/yolo26n.pt`); hash recorded on GX10 in `artifacts/model.sha256`.
- Saved real ESP32 frame on CPU: ~98.9 ms total processing, right-side person detections, `[0,0,0,1,1,1]`.
- Same saved frame on GPU: ~43.07 ms processing, same sector output; first post-warm-up frame includes extra initial overhead.
- CPU live test: 8 attempts, 6 complete, 2 timeouts. Successful fetch-to-result median **523.375 ms**, range **494.54–561.69 ms**. Processing median **89.84 ms**. One live left-side person yielded `[1,1,1,0,0,0]`.
- GPU live test: 12 attempts, 11 complete, 1 timeout. Processing median **20.36 ms**; successful fetch-to-result median **640.86 ms**, range **501.90–743.15 ms**. Subsequent processing usually **8–24 ms**, first ~43 ms. Person detections included `[1,1,1,1,0,0]` and `[0,1,1,1,0,0]`, but some frames with a partial person had no detection and one broad prediction activated all six sectors.
- These small samples use different live scenes; **not a controlled accuracy or CPU/GPU end-to-end comparison**. Timing excludes sampling wait, command polling, and vibration. Do not claim ≤500 ms scene-change-to-vibration or reliable hazard detection.
- Dominant delay is camera/network delivery. No TensorRT export or fine-tuning done. Investigate stable camera delivery, orientation/full-person visibility, FOV calibration, and detection flicker before motor integration.

Artifacts:

- Local ignored `artifacts/`: JPEGs, `yolo-cpu-annotated.jpg`, `yolo-cpu-result.json`, CPU logs/lock, model hash. GPU copies completed: `gpu-live-results.jsonl`, `gpu-requirements.lock.txt`, and `yolo-gpu-annotated.jpg`.
- GX10 `artifacts/cpu-detection/`, `cpu-live/`, `gpu-detection/`, `gpu-live/` contain images/results; live `frames.jsonl` logs preserve measurements.
- GX10 `artifacts/gpu-requirements.lock.txt` captures local GPU package versions.
- Larger GPU installation initially timed out. Successful retry used:
  `.venv/bin/python -m pip install --timeout 60 --retries 5 'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130`
  then `pip install -r software/requirements.txt`. Do not reinstall the working runtime.

## User authorizations and approval state

- User explicitly authorized pausing the existing Nemotron service **for benchmarks, then restoring it**. We used SIGSTOP with an EXIT trap for SIGCONT. Benchmarks ended and the original process was restored. Do not leave it paused or stop it permanently without authorizing that change.
- An automatic approval review rejected an **unauthenticated listener bound to all interfaces**, because it would expose camera images/results to reachable devices. We used loopback for all tests and then implemented token authentication plus a result-only LAN surface.
- User subsequently explicitly approved: **"Enable the protected command endpoint"**, in response to a question stating that it exposes only six-zone commands/readiness, uses a token, and HTTP does not encrypt the token in transit. **Do not ask for this approval again.**
- The protected LAN endpoint was **not started before interruption**. No production token or `PrivateConfig.h` was generated. `/home/asus/RAGging-Canes/private` did not exist at the handoff check. Next chat may proceed within the approved scope.
- Never print tokens/passwords in chat or shell output, commit them, or add them to command arguments/logs. Generate a random token into a mode-0600 file in a private directory; use `--token-file`; store firmware token in an ignored local configuration header.
- Missing information: full ESP32 camera sketch path/repo and six motor GPIO/driver mapping. A clarification was asked; no answer received before handoff. Receiving arrays can be integrated without GPIO writes once the sketch is available; physical haptics require wiring validation.

## Continue here

1. Verify original service state and current camera IP/reachability. Preserve the existing checkout.
2. Update GPU verification/benchmark notes. Local artifact copies are complete. Re-run appropriate tests for the final idle flag, then copy current `vision.py` to the isolated GX10 directory.
3. Generate and store a private random token; enable the **already approved**, authenticated, result-only listener on GX10 port 8001. Suggested GPU launch (token does not exist yet):

   ```bash
   cd /home/asus/RAGging-Canes
   .venv/bin/python software/vision.py pull --detect --device 0 \
     --url http://169.233.72.178/capture --host 169.233.72.164 --port 8001 \
     --token-file private/command.token --idle-without-client
   ```

   Keep model loaded; no capture requests until command polling begins. Verify unauthenticated 401, authenticated six-zone response, image routes unavailable, and expiry. Do not display the token during testing.
4. Preserve the five-zone Nemotron service. Avoid two continuous camera loops while benchmarking; separately agree on the active backend when switching physical firmware. The idle listener avoids traffic until a six-zone client polls.
5. Obtain the working `.ino` and motor details; integrate `SixZoneClient.h` and the private token header. Compile before flashing; keep camera pin/PSRAM/Wi-Fi settings. Verify Serial output of six values before driving any motors.
6. Verify calibrated sector direction, middle-pair behavior, disconnect/expiry, ultrasonic independence, and capture-to-vibration latency on hardware. Treat no supported detection as distinct from a clear path.

This is a saved handoff, not a claim that firmware or the protected endpoint is deployed. No reminder/automation or new chat was created.

## Follow-up: teammate integration review and model tests

See `INTEGRATION_REVIEW.md` for the complete October 10 follow-up and repeatable commands. Fetched teammate commit `482528f`; main remains at `6d2bb01`, with local work preserved and no merge/commit/push. Three colliding files: README (five merge conflict regions), .gitignore, firmware/README.md. The implementations coexist but five-zone Nemotron is incompatible with six-zone firmware.

Added `software/compare_models.py` for bounded same-JPEG GPU YOLO/Nemotron comparisons, including optional six-zone/person-chair Nemotron prompting. Added an authenticated idle-pull regression test; all 10 combined tests passed on GX10. Updated stale GPU status documentation. Reviewed source was explicitly authorized for transfer and copied only into `/home/asus/RAGging-Canes/artifacts/review-482528f`, preserving both deployments.

Fresh comparison: 3/5 complete captures, YOLO median processing 24.72 ms versus Nemotron 719.56 ms; two capture timeouts. Six-zone comparison: 4/4 syntactically valid Nemotron outputs, but coverage disagrees with YOLO. Separate actual YOLO live pull: 4/5 fresh results, successful fetch-to-result median 447.73 ms; timeout returned unknown/zero TTL. PID 87164 restored and verified serving valid Nemotron results; port 8001 closed after tests. No new persistent listener or firmware change. Benchmark results are under the isolated review directory (`results-native`, `results-six`, `results-yolo-live`).
