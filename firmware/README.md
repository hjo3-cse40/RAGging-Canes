# ESP32 camera streaming, motor, and detection clients

The current wearable uses three motors: left, center, right. For variable strength, follow [the haptic integration guide](../HAPTICS.md).

| Component / Backend | Header | Output / Role | Default port / Path |
| --- | --- | --- | --- |
| **Low-Latency 320×240 Stream** | `FastStreamServer.h` | Persistent MJPEG (`/stream`) + Binary WebSocket (`/ws`) at `FRAMESIZE_QVGA` | `80` (`/stream`, `/ws`, `/capture`, `/control`) & `81` |
| **Dashboard depth + decision pipeline** | `HapticClient.h` | 3 strengths, 0–100, schema v2 | `8002` (`/command`) |
| **YOLO `/command?zones=3`** | `ThreeZoneClient.h` | 3 binary detections (`[Left, Center, Right]`) | `8001` (`/command?zones=3`) |
| **YOLO legacy** | `SixZoneClient.h` | 6 binary detections | `8001` (`/command`) |
| **Nemotron legacy** | `ZoneClient.h` | 5 binary detections | `8000` (`/command`) |

`HapticMotors.h` is an optional active-high PWM motor-driver adapter for `HapticClient.h`. It needs explicit pin, LEDC channel/timer, frequency and duty calibration. No motor pins are selected by default. The full working camera sketch remains on the Mac attached to the ESP32. The following sections document the low-latency stream server and retained binary clients.

---

## Low-Latency 320×240 (QVGA) Persistent MJPEG / WebSocket Stream (`FastStreamServer.h`)

`FastStreamServer.h` eliminates the ~550–700 ms per-frame HTTP `/capture` bottleneck and achieves **~45–85 ms frame delivery (~12–20 FPS)** by combining:
1. **`FRAMESIZE_QVGA` (`320×240`) + `jpeg_quality = 14`** (~4–7 KB per JPEG vs ~35 KB at VGA).
2. **`CAMERA_GRAB_LATEST` + 2 PSRAM framebuffers** so `esp_camera_fb_get()` always returns the newest exposure instead of a queued frame.
3. **`WiFi.setSleep(false)` (`WIFI_PS_NONE`)** to prevent ESP32 Wi-Fi modem sleep from adding 100–300 ms DTIM beacon delays.
4. **Persistent `GET /stream` (Multipart MJPEG) and `GET /ws` (RFC 6455 Binary WebSocket)** with `TCP_NODELAY` on port `80` (and port `81`), allowing single-port SSH tunnels (`-R 8117:ESP32_IP:80`) to stream continuously.

### Option A: Drop-in with existing `startCameraServer()` sketch
If your sketch already calls `startCameraServer()`, simply call `FastStreamServer::optimizeLiveSensor()` after `esp_camera_init()` and `WiFi.begin()`:

```cpp
#include "FastStreamServer.h"
#include "ThreeZoneClient.h"

// Before esp_camera_init(&config):
FastStreamServer::applyLowLatencyConfig(config, FRAMESIZE_QVGA, 14);

// In setup(), after WiFi is connected and camera is initialized:
FastStreamServer::optimizeLiveSensor(FRAMESIZE_QVGA, 14);
```

### Option B: Replace `startCameraServer()` with `FastStreamServer` (exposes `/stream` and `/ws` on port 80)

```cpp
#include "FastStreamServer.h"
FastStreamServer streamServer(80, 81);

// In setup() after esp_camera_init() and WiFi connection:
streamServer.begin(FRAMESIZE_QVGA, 14);
```

---

## Six-zone YOLO client

`SixZoneClient.h` adapts the existing GX10 checkout's five-zone polling client to six values. It is designed for the existing Arduino ESP32 camera sketch and ArduinoJson 7. It receives and prints direction results; it does not configure GPIO or drive motors.

The updated client has not yet been compiled against the full camera sketch or flashed. The working camera sketch and motor pin/driver configuration have not been supplied in this workspace.

After authorized direct ESP32-to-GX10 service access has been arranged, copy the header into the existing camera sketch folder. Add globally:

```cpp
#include "SixZoneClient.h"
SixZoneClient zoneClient("http://GX10_IP:8001/command", GX10_COMMAND_TOKEN);
```

Replace `GX10_IP` with the actual address and define `GX10_COMMAND_TOKEN` in a local, uncommitted configuration header. The same token must be configured on the authorized GX10 listener using `--token-file`. This simple bearer-token transport is for the controlled prototype network; it does not encrypt HTTP traffic. After the existing Wi-Fi connection and camera server startup in `setup()`:

```cpp
if (!zoneClient.begin()) {
  Serial.println("Could not start six-zone receiver");
}
```

The separate task polls every 200 ms plus HTTP time. It checks six integer zone states, session/frame identifiers, validity, and remaining lifetime. The entire HTTP round trip is deducted from the lifetime, and repeat polls never extend a frame's original deadline.

Existing `loop()` or a separate motor scheduler can read:

```cpp
int zones[6];
bool fresh = zoneClient.read(zones);
// fresh == false: all six entries are -1.
// Only use fresh results for bounded directional pulses.
```

Networking must remain separate from motor timing and ultrasonic sampling. Define the actual motor mapping and proximity priority after inspecting the wiring and motor driver code. Preserve the working camera sketch's board, PSRAM, partition, pin, and Wi-Fi settings.

Expected serial output for an object spanning the midpoint:

```text
Frame 12 zones: [0,0,1,1,0,0]
```

The YOLO service defaults to loopback access for SSH testing. The header cannot reach a loopback-bound service from the ESP32. The existing port-8000 Nemotron service and its five-value client remain separate and unmodified.

---

## Five-zone Nemotron client

This module adds a background polling task to the **existing working camera sketch**. It receives and prints arrays only; it does not configure GPIOs or drive motors. Keep the current camera setup, network configuration, and board settings.

## On the Mac connected to the ESP32

1. Keep a copy of the working camera sketch.
2. Install **ArduinoJson 7** by Benoit Blanchon using Arduino IDE's Library Manager. The module also uses WiFi and HTTPClient from the Espressif Arduino ESP32 core.
3. Copy `ZoneClient.h` into the same sketch folder as the main `.ino` file. The running GX10 service offers a copy at `http://GX10_IP:8000/firmware/ZoneClient.h`, with these instructions at `/firmware/README.md`. Replace `GX10_IP` with the GX10 address.
4. At the top of the main `.ino`, outside all functions, add:

   ```cpp
   #include "ZoneClient.h"
   ZoneClient zoneClient("http://GX10_IP:8000/command");
   ```

5. At the end of the existing `setup()`, after Wi-Fi is connected and `startCameraServer()` has been called, add:

   ```cpp
   if (!zoneClient.begin()) {
     Serial.println("Could not start GX10 zone receiver");
   }
   ```

6. Compile and upload the **existing camera sketch with these additions** using its current board/PSRAM/partition settings. Open Serial Monitor at the baud rate already used by that sketch (commonly 115200).

Expected output:

```text
Frame 12 zones: [0,1,1,0,0]
Frame 13 zones: [0,0,1,0,0]
```

The five entries are left, leftish, center, rightish, right. `1` means a target detection, `0` means no target detected, and `-1` means unknown. These are visual detections, not distances or safe-path instructions.

If the connection fails, the response is malformed, or its validity expires, the module reports `Zones unavailable/expired: [-1,-1,-1,-1,-1]`. It polls roughly every 200 ms plus HTTP request time. This does not make inference run at 5 FPS; polls may return the same frame and only new frames are normally printed.

## Read the latest array in your sketch

The background task handles networking; no polling call is needed in `loop()`. Later code can call:

```cpp
int zones[5];
bool fresh = zoneClient.read(zones);
// When fresh is false, zones contains five -1 values.
// No motor control is implemented in this module.
```

This accessor checks expiry even if a network request is blocked. The current client accepts a maximum server TTL of 2500 ms and subtracts the entire polling round-trip time. Repeated responses for the same frame never extend its original local deadline. Keep the global `zoneClient` alive for the lifetime of the sketch and call `begin()` only from setup.

## Verify the return path

Before uploading, open `http://GX10_IP:8000/command` on the other Mac to check network reachability. After uploading, check Serial Monitor and the GX10's `/health` endpoint. `last_client_ip` should match the ESP32 when it polls `/command`. Browser and manual command requests also update this field, so it alone is not proof that firmware parsed the response; confirm Serial output too.

The camera service itself cannot add this polling code remotely. The board must be flashed from the Mac that currently has its USB connection.

## Validation

`ZoneClient.h` compiled successfully in a minimal ESP32-S3 test sketch using Espressif Arduino core **3.3.12** and ArduinoJson **7.4.3**. The test sketch was not flashed. Compilation with your complete camera sketch and runtime receipt on the physical board remain to be checked on the Mac connected to the ESP32.

## References

- [Espressif Arduino Wi-Fi API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/wifi.html)
- [ArduinoJson type checks](https://arduinojson.org/v7/api/jsonvariant/is/)
