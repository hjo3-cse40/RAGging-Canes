# Receive the five-zone array on the ESP32

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
