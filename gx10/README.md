# Nemotron five-zone backend

This backend coexists with the six-zone YOLO backend in the [shared repository](../README.md). Run all commands from the repository root. It retains the existing five-zone protocol and does not drive motors. Use `ZoneClient.h` with this service; `SixZoneClient.h` requires six values and will reject these responses.

## How the current software works

There are two HTTP request loops. The GX10 requests images from the ESP32, and the ESP32 receiver requests results from the GX10 once that receiver is added to the camera sketch.

1. **The ESP32 serves the camera.** Its existing firmware provides a camera page, a `/capture` JPEG endpoint, and a video stream on port 81. The working camera sketch is on the other Mac and is not included in this repository.
2. **The GX10 fetches a snapshot.** `gx10/service.py` requests `/capture` and reads one JPEG. It uses snapshots rather than processing the browser's video stream.
3. **Nemotron processes that JPEG locally.** The service calls Ollama on the GX10 with `nemotron3:33b`, thinking disabled, a short prompt, and a JSON schema requiring five values. The prompt asks for people, chairs, tables, or boxes in five equal vertical image zones.
4. **The GX10 publishes the validated result.** It stores the latest array with a frame ID, session ID, timing, and expiry. `/command` returns that stored result immediately; polling it does not trigger another model request. The capture loop then processes its next frame.
5. **The ESP32 polls for the array.** `firmware/ZoneClient.h` starts a background task that requests the GX10's `/command` endpoint approximately every 200 ms plus request time. It validates the JSON and remaining lifetime, then prints new arrays to Serial Monitor. This step requires uploading the modified camera sketch first.
6. **Future code can use the array.** `zoneClient.read(zones)` returns the current values, or five `-1` values if unavailable or expired. There are no motor GPIO writes in the current module.

```text
GX10 -- GET /capture --> ESP32 camera
GX10 <-- JPEG ---------- ESP32 camera
GX10 --> local Ollama / nemotron3:33b --> five-zone array

After the receiver module is flashed:
ESP32 -- GET /command --> GX10
ESP32 <-- JSON array + freshness metadata -- GX10
ESP32 --> Serial Monitor on its USB-connected Mac
```

The Mac connected to the ESP32 programs the board and shows its serial output. It does not need to relay images or run the AI; the ESP32 and GX10 communicate directly over the network.

### Continuous service and return channel

On the GX10, run from the repository root (replace the address placeholders):

```bash
python3 -m gx10.service --camera-url http://ESP32_IP/capture \
  --bind GX10_IP --port 8000
```

The service uses Python's standard library and the existing local Ollama installation. It processes one frame at a time without building a frame queue. The model is fixed to `nemotron3:33b` with thinking disabled.

| Endpoint | Purpose |
| --- | --- |
| `GET /command` | Latest array, validity, remaining lifetime, and frame ID; polled by the ESP32. |
| `GET /health` | Same state plus timing and polling diagnostics, without counting as a command poll. |
| `GET /firmware/ZoneClient.h` | Download the ESP32 polling module on the other Mac. |
| `GET /firmware/README.md` | Download the sketch integration instructions. |

An example command response includes:

```json
{"frame_id":12,"model":"nemotron3:33b","status":"ok","valid":true,"zones":[0,1,1,0,0],"ttl_ms":1100}
```

The real response also includes a session ID, frame age, and timing fields. Freshness expires 2500 ms after the GX10 starts fetching the frame, including capture and inference time. A startup, error, or expired result has `valid: false`, `ttl_ms: 0`, and five `-1` values. The age cannot account for older frames buffered inside the camera. This timeout is a prototype setting, not a validated safety threshold. The HTTP endpoint stays available while inference is busy.

The original GX10 deployment was configured as the transient user service `ragging-canes.service`:

```bash
systemctl --user status ragging-canes
journalctl --user -u ragging-canes -f
systemctl --user stop ragging-canes
```

It remains running after the SSH command exits and restarts after process failures. It is not installed as a boot service; after a reboot, start it again using the Python command above. Stop the running instance before starting another copy on the same port.

**To finish the return path:** follow [the ESP32 integration instructions](../firmware/README.md) on the Mac connected to the board. Add `ZoneClient.h` to the existing camera sketch, create the client with `http://GX10_IP:8000/command`, and call `begin()` after camera/Wi-Fi setup. The module prints received arrays and supports `read()` for later use. It does not touch motor pins. Confirm actual receipt in Serial Monitor after upload.

### Check results from the Mac connected to the ESP32

First get this repository onto that Mac using the clone command above. If it is already cloned, run `git pull --ff-only` from its repository folder. Copy `firmware/ZoneClient.h` into the working camera sketch folder and follow [the upload steps](../firmware/README.md).

To check that the GX10 is serving arrays, replace `GX10_IP` with its current address:

```bash
curl http://GX10_IP:8000/command
```

To watch repeated responses, press Ctrl+C when finished:

```bash
while true; do
  curl --max-time 3 -sS http://GX10_IP:8000/command
  echo
  sleep 1
done
```

| Response field | How to read it |
| --- | --- |
| `zones` | Five entries ordered left, leftish, center, rightish, right. |
| `valid` | The response passed validation and remains within its freshness budget; not a guarantee of correct detection. |
| `status` | `starting`, `ok`, `stale`, or `error`. |
| `frame_id` | Successful inference counter within this service session; printed by the ESP32 receiver. |
| `session_id` | Changes when the service restarts, so frame counters from different sessions are distinguishable. |
| `frame_age_ms` | Milliseconds since the GX10 began fetching this image. |
| `ttl_ms` | Remaining lifetime; zero means the receiver must not use the array. |

To verify that the **physical ESP32 receives and parses** the array, use Arduino Serial Monitor or find its serial port in Terminal:

```bash
ls /dev/cu.*
```

Then open the actual `usbmodem` or `usbserial` port (replace the example name):

```bash
screen /dev/cu.usbmodemXXXX 115200
```

Use the baud rate configured by the existing camera sketch if it differs from 115200. Close Arduino Serial Monitor before using `screen`. Exit `screen` with Ctrl+A, then K, then Y. After the receiver module is uploaded, expect lines such as:

```text
Frame 12 zones: [0,1,1,0,0]
```

`curl` checks the GX10 endpoint; the serial line confirms the ESP32 parsed a response. The `/health` endpoint's `last_client_ip` and `poll_count` also help diagnose polling, but browser requests update those fields too.

### Compare detections with the camera view

Open `http://ESP32_IP/` and move a person or chair into the left, center, and right of the view, holding each position for several seconds. Look for corresponding zone changes in Serial Monitor. The browser image will usually be newer than the image the model processed, and intermittent camera failures can delay results further.

The current service does **not** save or display the exact processed JPEG alongside its frame ID. A matching frame ID connects the serial output to a model response, not to a particular frame visible in the browser. Exact image/result alignment needs a paired-image debug view, which is not implemented yet.

### Validation

Server checks:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover startup, valid results, expiry including inference time, stale polls, invalid model responses, error recovery, and the HTTP response contract.

### Run a vision benchmark on the GX10

The original GX10 deployment checkout is `/home/asus/projects/RAGging-Canes`. The benchmark uses Python's standard library and the existing local Ollama service; it needs no additional Python packages.

```bash
cd /home/asus/projects/RAGging-Canes
python3 tools/benchmark_vision.py --model nemotron3:33b --rounds 2 \
  /absolute/path/image1.jpg /absolute/path/image2.jpg
```

Omitting `--model` also selects `nemotron3:33b`. Each output line is JSON with the image hash, five-zone response, response validity, wall time, and available Ollama timing fields. Requests disable thinking and constrain output with a JSON schema using the [Ollama chat API](https://docs.ollama.com/api/chat). Invalid or incomplete responses make the script exit with a nonzero status.

The first request can include model loading. Later rounds repeat the same files and may benefit from caching. `valid: true` means the response format passed validation; it does not establish detection accuracy. File-mode timings exclude ESP32 capture, Wi-Fi transfer, and motor startup.

### Test live ESP32 frames

The current camera page identifies itself as **ESP32 OV2640** and exposes `/capture` for JPEG snapshots and `http://ESP32_IP:81/stream` for video. Replace `ESP32_IP` with the board's current address:

```bash
python3 tools/benchmark_vision.py --model nemotron3:33b --rounds 5 \
  --camera-url http://ESP32_IP/capture
```

This mode fetches a JPEG for each round, records its hash and fetch time, then runs inference. `wall_ms` includes the HTTP camera fetch and the Ollama request, but excludes motor output and does not measure the age of the image already buffered by the camera. Stop the continuous service before benchmarking to avoid competing inference and camera requests. The continuous service also pulls frames; its `/command` endpoint provides the return channel once the ESP32 polling module is flashed.

### Initial measurements — October 10, 2026

| Test | Observed wall time |
| --- | --- |
| Nemotron, first desktop-image request including model load | 7.31 s |
| Nemotron, two other desktop images with the model loaded | 0.70–0.74 s |
| Gemma, first desktop-image request including model load | 5.43 s |
| Gemma, two other desktop images with the model loaded | 0.93–0.96 s |
| Nemotron, three live ESP32 fetch-and-inference requests with the model loaded | 1.28–1.35 s |

Both models returned valid arrays for six desktop-image requests each. Three live captures had distinct hashes and valid arrays; JPEG fetches took 0.55–0.63 s and Ollama processing took 0.69–0.74 s. These small samples establish connectivity and response formatting, not detection accuracy or safe response times. They exceed the initial 200–500 ms target before motor latency is included.

An earlier live-camera attempt hit the 10-second capture timeout before the successful three-frame run. Camera availability under streaming and repeated capture still needs testing; a failed capture must not be interpreted as an empty scene.

During continuous-service testing, the camera also intermittently timed out or refused connections. The endpoint invalidated its results and recovered on subsequent successful captures. The full return path still needs a hardware test after the polling module is flashed.
