# First JPEG and detector implementation

## Observed hardware connection (October 10, 2026)

- GX10 SSH: `asus@169.233.72.164`; laptop key authentication has been verified. GPU: NVIDIA GB10, driver 580.142.
- ESP32 snapshot: `http://169.233.72.178/capture` returned HTTP 200 and a valid 640×480 JPEG, 27,402 bytes.
- ESP32 camera page advertises an MJPEG stream on port 81. This implementation uses snapshots, not that stream.
- A second capture request timed out after eight seconds with 16,158 bytes of a partial image. Stable capture latency has not been established.
- A direct GX10 capture subsequently succeeded: 32,683 bytes in 1.067 seconds. A valid sample was copied back to the laptop and decoded. Earlier direct connection attempts failed, so one successful request does not establish reliability.
- These addresses are current demo-network observations, not fixed device configuration. Supply the current URL at launch.

## Verified CPU baseline

An isolated `/home/asus/RAGging-Canes/.venv-cpu` environment runs YOLO26n successfully on the GX10. A real saved ESP32 JPEG produced person detections and `[0,0,0,1,1,1]` in approximately 99 ms total processing. The camera frame contains a partial person and overlapping predictions; this is execution evidence, not an accuracy evaluation.

An eight-attempt live test with the original Nemotron process temporarily paused completed six frames and encountered two capture timeouts. Successful fetch-to-result times were 494.54–561.69 ms, median 523.375 ms; median image processing time was 89.84 ms. A live left-side person produced `[1,1,1,0,0,0]`. These are a small sample collected while GPU dependencies were downloading, exclude sampling wait and motors, and do not establish p95 performance or the 500 ms scene-change target. The original service was resumed after the test.

Use the verified baseline on GX10:

```bash
cd ~/RAGging-Canes
.venv-cpu/bin/python software/vision.py pull --detect --device cpu \
  --url http://169.233.72.178/capture --host 127.0.0.1 --port 8001
```

Avoid running both camera inference loops simultaneously when measuring latency. The separate `.venv` GPU environment is verified on NVIDIA GB10 (PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, Ultralytics 8.4.175). Preserve this working environment. See [integration review and repeatable model tests](../INTEGRATION_REVIEW.md) for the latest measurements.

## Connect and copy the software

Run these from a **local laptop terminal**, outside an SSH session:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub asus@169.233.72.164
ssh asus@169.233.72.164 'mkdir -p ~/RAGging-Canes/software'
scp /Users/samjo/RAGging-Canes/software/vision.py /Users/samjo/RAGging-Canes/software/requirements.txt asus@169.233.72.164:~/RAGging-Canes/software/
ssh asus@169.233.72.164
```

The first command asks for the GX10 password locally and installs the laptop's existing public key. Do not put passwords in files or chat.

## Step 1: pull and validate one JPEG on GX10

Inside the GX10 session:

```bash
cd ~/RAGging-Canes
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'Pillow>=11,<13'
mkdir -p artifacts
curl --fail --max-time 10 'http://169.233.72.178/capture' -o artifacts/capture.jpg
python software/vision.py image --path artifacts/capture.jpg
```

Expect `received_only`, image dimensions, byte size, and six `-1` values. These mean detection has not run. A truncated or non-JPEG response is rejected. The default artifact directory retains only the latest image/result, rather than recording the entire camera feed.

## Step 2: inspect it on the laptop

From a separate **local laptop terminal**:

```bash
scp asus@169.233.72.164:~/RAGging-Canes/artifacts/latest.jpg /Users/samjo/RAGging-Canes/artifacts/gx10-frame.jpg
```

Check orientation, blur, exposure, and whether the camera actually points at the demo objects. The first fetched sample mostly showed the ceiling; that is not a useful person/chair test scene.

## Step 3: detect a saved image on GX10

In the GX10 virtual environment:

```bash
python -m pip install -r software/requirements.txt
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
python software/vision.py image --detect --device 0 --path artifacts/capture.jpg
```

The model downloads on first use and stays loaded in live mode. The service warms the detector before accepting frames. CUDA failure is an environment problem, not evidence that the camera failed. Do not silently switch the GX10 performance benchmark to CPU.

Use a GB10-compatible ARM64 PyTorch/CUDA environment. If the native environment cannot execute GPU inference, use the official [Ultralytics DGX Spark guide](https://docs.ultralytics.com/guides/nvidia-dgx-spark) for the NVIDIA ARM64 runtime and verify it on the GX10. Preserve a working host environment rather than upgrading drivers blindly. Record the working package versions and model hash after the first successful GPU run. TensorRT is a later optimization.

Outputs: `artifacts/latest.jpg`, `artifacts/latest.annotated.jpg`, and `artifacts/latest.json`. Yellow lines show configured sector boundaries. Copy the annotated JPEG to the laptop with `scp` as above, substituting the annotated filename.

## Step 4: live snapshot detection

On GX10:

```bash
python software/vision.py pull --detect --device 0 \
  --url http://169.233.72.178/capture --interval 1 \
  --max-result-age-ms 1000 --host 127.0.0.1 --port 8001
```

There is one snapshot/inference cycle at a time. Slow cycles do not create a queue or catch-up requests. Snapshot sockets have a three-second inactivity timeout by default (`--timeout` changes this); this is not an absolute total-transfer deadline. Stop with Ctrl-C. Avoid leaving the camera browser stream active while collecting baseline timings; test streaming contention separately. Use `--frames 10` for a bounded measurement run; each attempt is recorded in `artifacts/frames.jsonl`.

Open an SSH tunnel from a local laptop terminal:

```bash
ssh -N -L 8001:127.0.0.1:8001 asus@169.233.72.164
```

Then inspect from the laptop:

- `http://127.0.0.1:8001/health`
- `http://127.0.0.1:8001/latest.jpg`
- `http://127.0.0.1:8001/latest.annotated.jpg`
- `http://127.0.0.1:8001/latest.json`

The HTTP JSON includes current expiry state. The JSON file is a saved diagnostic snapshot; use the HTTP endpoint for live cues. Startup/model loading is excluded from per-frame timing. `observation_to_result_ms` includes fetch time in pull mode; it does not establish scene-change-to-vibration latency. Server observation age begins at fetch request start, not the ESP32's actual sensor exposure. Camera buffering must be measured separately.

The service defaults to loopback access. A non-loopback listener requires `--token-file`, authenticates every request, and exposes only `/health` and the compact `/command` response. It does not expose images or accept uploads. Direct ESP32 polling requires a separately authorized LAN listener; bearer authentication does not encrypt HTTP traffic. For local receiver testing:

```bash
python software/vision.py serve --detect --device 0
```

POST binary JPEG to `/infer` with `Content-Type: image/jpeg`, `Content-Length`, and `X-Frame-ID`. No multipart encoding. Busy requests receive 429; retry with a fresh capture. This simple HTTP receiver requires Content-Length, not chunked uploads. It is intended for the local prototype network.

## Step 5: ESP32 six-motor contract

Existing ESP32 firmware and motor GPIO/driver details are not in this repository. No motor firmware has been flashed or electrically validated.

After arranging authorized LAN access, add a separate network task to the current snapshot-serving firmware that polls GX10 `/command`, initially every 100 ms. Keep ultrasonic sampling and motor scheduling in independent tasks; no network request should run in the motor timing loop.

The ESP32 should:

1. Accept only HTTP 200, `valid == true`, six zone values, and positive `ttl_ms`.
2. Subtract its measured polling round-trip time from the returned remaining lifetime. Ignore responses with no remaining lifetime.
3. Compare the command `session_id` and numeric `frame_id`; produce at most one directional pulse per new pair. Restarted services use a new session identifier.
4. Apply bounded pulses to the six motors, then stop; never hold a motor on while waiting for the next response.
5. Reject waiting, receive-only, failed, malformed, and expired results. `-1` is unknown, never clear.
6. Keep measured ultrasonic proximity independent and give it priority over directional cues.

For a future ESP32 upload implementation, retain the capture timestamp locally and expire the response based on that original timestamp, even if server timing reports a newer observation.

Default classes are person and chair. Default confidence threshold is 0.35, an initial tuning value. Standard weights do not detect arbitrary obstacles. No distance is inferred from bounding-box size.

Default boundaries are six equal **pixel strips**, not calibrated 30° sectors. Set `--boundaries 0,b1,b2,b3,b4,b5,1` using seven measured normalized pixel edges once the field of view and lens mapping are known. A box activates every strip it overlaps; a box straddling the midpoint activates motors 3 and 4.

## Verification and next measurements

Local software tests:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover HTTP receipt and saved image retrieval, corrupted-JPEG rejection, concurrent-request rejection, middle-pair mapping, and expiry. They do not prove model accuracy, GX10 compatibility, motor wiring, or capture-to-vibration timing.

Before increasing capture rate, measure median and p95 fresh-frame fetch, inference, and scene-change-to-vibration latency. Test moving person/chair objects in all six sectors, centered objects, edges, low light, camera turns, disconnection, and reconnect. Compare Nemotron and YOLO on the same cap images. Only retain the 500 ms scene-change target if the integrated measurements support it.

The existing Nemotron deployment at `/home/asus/projects/RAGging-Canes` uses port 8000 and five zones. It has not been replaced. The YOLO deployment uses `/home/asus/RAGging-Canes` and port 8001. See [six-zone client](../firmware/README.md) for the prepared receive-only adapter; physical motor integration remains unverified.
