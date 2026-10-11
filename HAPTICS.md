# Three-motor strength output

The dashboard pipeline now produces requested strengths in `[Left, Center, Right]` order. The GX10 combines decision confidence with the nearest fresh, confirmed object's estimated distance in each direction. The ESP32 receives three integers from 0–100 and can map them to calibrated PWM motor drivers.

This is implemented locally; it has not been deployed to the GX10 or flashed onto the physical cap. The existing camera sketch, wiring, and driver specifications are still required for hardware integration.

## Signal mapping

`software/haptics.py` implements:

```text
proximity = clamp((far_m - distance_m) / (far_m - near_m), 0, 1)
target_strength = round(100 * confidence * proximity)
```

Defaults: near = 0.5 m, far = 3 m; activate at confidence ≥ 0.65, deactivate at ≤ 0.45. Confidence between the thresholds preserves activation state. Strength rises with an 80 ms smoothing time constant and falls with a 250 ms time constant, updated on new source observations. The first accepted observation is applied directly. Missing/invalid inputs and expiry bypass smoothing. Losing a confirmed object or suppressing an implausible detection immediately zeros its output.

These settings are tuning defaults. Confidence values are backend-dependent scores, not calibrated physical risk: the Ollama backend derives them from hazard-class confidence, and the kinematic fallback uses 0/1. Distance is a monocular estimate. A percentage is a requested strength, not measured vibration amplitude.

Only currently observed, realistic tracks contribute. Predicted motion, occluded remembered objects, and named forecast pulse patterns remain dashboard information and do not drive motors in this version. For multiple objects in a direction, use the nearest. If any eligible object there has unknown depth, that motor is unavailable and receives zero; other motors remain independent. A clear direction is valid with zero output.

The decision carries the original observation's sequence, generation, timestamp, source mode, and per-object distances through asynchronous inference. A model result expires 2.5 seconds after capture starts by default, including capture and model time. Polling, dashboard refreshes, and model completion do not refresh that deadline. Slow decisions may therefore produce no motor output; choose a faster backend or measure/tune the freshness budget rather than renewing an old sample.

## Run on the GX10

Run from the repository with the environment used for the existing dashboard:

```bash
python -m software.dashboard \
  --camera-url http://ESP32_IP/capture \
  --decision-backend laya \
  --command-bind GX10_LAN_IP --command-port 8002
```

Replace both address placeholders. The existing dashboard remains loopback-only on port 8080. The new listener defaults to loopback too; `--command-bind` explicitly enables ESP32 access on the chosen LAN interface. Ports 8000/8001 and the old binary clients remain separate contracts. An optional `--command-token-file path/to/token` requires the same bearer token in the receiver. The listener exposes only `GET /command`, without dashboard controls or images.

Tuning flags: `--haptic-near`, `--haptic-far`, `--haptic-activate`, `--haptic-deactivate`, `--haptic-rise`, `--haptic-fall`, `--haptic-max-age` (seconds; maximum 2.5). Defaults are described above.

```bash
curl http://GX10_LAN_IP:8002/command
```

Example response (metadata values are illustrative):

```json
{
  "schema_version": 2,
  "session_id": "0123456789abcdef0123456789abcdef",
  "command_id": 42,
  "source_sequence": 19,
  "source_mode": "live",
  "valid": true,
  "status": "ok",
  "motor_valid_3": [true, true, true],
  "motor_strength_3": [54, 90, 0],
  "distance_m_3": [1.5, 0.5, 3.0],
  "confidence_3": [0.9, 0.9, 0.9],
  "ttl_ms": 500
}
```

`motor_valid_3` distinguishes unknown direction data from known zero output. `valid: false` or an expired lease means all output must stop. The dashboard displays the same strength calculation with confidence and distance. Simulation/replay can preview strengths, but the hardware endpoint always returns zero and `valid: false` for those sources, even if the UI is switched to live without a live camera feed.

## ESP32 integration

Requires ArduinoJson 7 and ESP32 Arduino core 3.x. Copy `firmware/HapticClient.h` to the working camera sketch. Add globally:

```cpp
#include "HapticClient.h"
HapticClient haptics("http://GX10_LAN_IP:8002/command");
// Optional second constructor argument: bearer token string.
```

After the existing camera and Wi-Fi setup, call `haptics.begin()` and check its return value. The task polls every 100 ms plus HTTP request time. New commands print:

```text
Motor strength % [L,C,R]: [54,90,0]
```

`haptics.read(strengths)` fills `int strengths[3]` and reports whether the command is fresh. Invalid directions are zero. On error, loss of Wi-Fi, or expiration, all outputs are zero. Repeated commands cannot extend a lease, change its values, or revive output invalidated by an error. A new command can resume output.

For active-high PWM-compatible motor drivers, also copy `firmware/HapticMotors.h`. It provides an independent 10 ms motor task, using explicit pins/channels and individual 8-bit minimum/maximum duty calibration. A disabled configuration compiles without selecting hardware pins:

```cpp
#include "HapticMotors.h"
const HapticMotorConfig motorConfig[3] = {}; // All pins disabled, maximum duty zero.
HapticMotors motors(haptics, motorConfig, 1000); // PWM Hz is an example, select for your driver.
// After haptics.begin(): check motors.begin().
```

Each configuration entry is `{pin, ledc_channel, minimum_duty, maximum_duty}`. Set these only after inspecting the actual driver circuit and camera sketch. Motors require suitable drivers; these pins carry control signals. Do not reuse the camera's XCLK LEDC channel **or timer**: distinct channel numbers can share a timer. Choose an unused timer/channel group at a frequency supported by the driver. This adapter is not an I²C haptic-driver or LRA driver implementation.

Zero strength maps to zero duty. Positive strength maps into the calibrated minimum-to-maximum duty range; therefore requested percentage and electrical duty are not identical. Wi-Fi loss or lease expiry is checked independently of a blocked HTTP request. Keep client/controller objects alive for the lifetime of the sketch.

## Verification

Run the Python suite with Pillow, NumPy and OpenCV installed (these vision dependencies are provided by the existing detector environment):

```bash
python -m unittest discover -s tests -v
```

Haptic tests cover proximity/confidence mapping, missing depth, multiple targets, hysteresis, smoothing, backend changes, delayed decisions, duplicates, ordering, generation changes, simulated sources, and the authenticated isolated HTTP endpoint. Both headers compile in a minimal ESP32-S3 sketch against core 3.3.12 / ArduinoJson 7.4.3. Compilation does not verify the complete camera sketch, PWM resource allocation, actual vibration, or hardware timeout behavior.

For bench verification: check left/center/right serial strengths first, configure one driver's bounded duty range, verify increasing strengths physically, then verify all three directions independently. Disconnect Wi-Fi and stop the GX10 service to confirm motors stop on disconnection/expiry. Integrate with the full camera sketch only after checking GPIO and LEDC ownership.
