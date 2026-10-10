# Six-zone receive-only ESP32 client

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
