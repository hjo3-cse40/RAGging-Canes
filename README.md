# RAGging Canes

A wearable obstacle-awareness prototype that turns camera observations into directional vibration cues. Built for the ASUS Local AI track at the UCSC AI Hackathon, October 10–11, 2026.

The concept is a baseball cap with a Freenove ESP32-S3 WROOM board, a camera, and vibration motors. Images travel over a local network to an ASUS Ascent GX10 for AI processing. The cap uses the results to indicate where objects appear relative to the wearer's head, while a separate ultrasonic sensor provides forward proximity feedback.

## Project status

This repository currently contains the project plan. Firmware, the GX10 service, training scripts, wiring diagrams, and an integrated demo have not been added yet.

Early manual image experiments on the GX10 produced useful scene descriptions. A detailed response took roughly six seconds; later testing with `nemotron3:33b` and thinking disabled felt nearly instant. These are informal observations, not end-to-end benchmarks. Fresh-image testing is still needed to distinguish live performance from model warm-up and repeated-image caching effects.

## Intended experience

The cap communicates the direction of visible objects through five positions:

**Left · Leftish · Center · Rightish · Right**

A vibration indicates where an object is detected. Cues are relative to the camera and the wearer's head, which may point differently from their walking direction. Forward proximity uses a distinct pulse pattern based on the ultrasonic reading.

The original hardware plan includes six motors: five directional motors and a possible rear status motor. The latest prototype discussion focuses on the five directional zones; the sixth motor's role remains optional. A rear status cue would indicate conditions such as a lost connection, not an object behind the wearer.

The first demo should have a stationary wearer identify the direction of objects moved within the camera view. This is an experimental awareness aid; silence does not establish that a path is clear, and the prototype is not a replacement for existing mobility aids.

## Planned architecture

```mermaid
flowchart LR
    Camera[Cap camera] --> ESP32[ESP32-S3]
    ESP32 -->|JPEG over local Wi-Fi| GX10[GX10: local vision inference]
    GX10 -->|Five-zone result and frame ID| ESP32
    Ultrasonic[Forward ultrasonic sensor] --> ESP32
    ESP32 --> Drivers[Motor drivers]
    Drivers --> Motors[Directional vibration motors]
```

The two feedback paths have separate responsibilities:

| Path | Purpose |
| --- | --- |
| Camera → GX10 → ESP32 → motors | Indicate the horizontal location of recognized objects. |
| Ultrasonic → ESP32 → motors | Provide measured forward proximity feedback without waiting for image inference or Wi-Fi. |

One ultrasonic reading does not provide a distance for every camera detection. Camera detections alone should not be treated as measured distance or a safe-navigation instruction.

The planned demo runs inference locally on the GX10 and communicates over the local network. A compact model running directly on the ESP32 is a possible future direction, not the current deployment plan.

## Hardware

| Component | Role / current detail |
| --- | --- |
| Baseball cap | Wearable mount for the camera and haptics. |
| Freenove ESP32-S3 WROOM board | Camera capture, networking, sensor input, and motor control. |
| Camera module | Forward-facing JPEG source; exact module and pin mapping still need documentation. |
| Five directional vibration motors | Bare two-wire motors; exact type, ratings, and driver circuit still need confirmation. |
| Optional sixth motor | Reserved in the original plan for a distinct status cue. |
| Ultrasonic sensor | Reported as a 3 V sensor; exact model, range, and signal levels remain to be verified. |
| Motor drivers and power supply | Must be selected for the actual motors; bare motors must not be powered directly from GPIO. |
| ASUS Ascent GX10 | Local model experimentation, inference, and planned training. |
| Development laptop and local network | Programming, debugging, and remote access to the GX10. |

An exact wiring map is pending identification of the camera, sensor, and motor parts, along with the board's available GPIOs.

## Vision and haptic output

The model candidates discussed and manually tried are `nemotron3:33b` and `gemma4:26b` through Ollama. Nemotron is the current candidate for further latency testing following the latest manual experiment. The final choice depends on fresh camera images, directional accuracy, and capture-to-vibration latency.

The proposed compact vision output is an array ordered left to right:

```json
[0, 1, 1, 0, 0]
```

| Value | Meaning |
| --- | --- |
| `1` | A target object is detected in this zone. |
| `0` | No target object was detected in this zone. |
| `-1` | The zone could not be assessed. |

An object may occupy multiple zones. Candidate demo objects include people, chairs, tables, and boxes; the final supported classes depend on the selected model and training dataset. Unknown results require distinct handling and must not be silently converted to a clear scene.

This is a proposed detection format, not an implemented API or electrical motor command. The service still needs response validation, frame identifiers, and command expiry. Haptic pulse patterns and priority between proximity, direction, and status cues also need to be defined.

## Latency plan

The initial engineering target discussed is approximately **200–500 ms from a scene change to noticeable vibration**. This has not been demonstrated on the integrated hardware.

- Keep the model loaded and disable thinking for short directional responses where supported.
- Process one image per request without accumulating conversation history.
- Compare 640×480 and 320×240 JPEG input for both missed detections and delay.
- Allow one inference request at a time and keep only the newest waiting frame.
- Return compact results and reject malformed, incomplete, or expired responses.
- Keep ultrasonic sensing and motor timing independent of network waits.
- If the vision-language models are too slow, evaluate a small object detector for the directional loop.

Benchmark with different images after warming up the model, including left, center, right, and empty scenes. Record missed detections, false detections, unstable zone changes, typical latency, and slow responses. Measure the full camera → network → inference → response → motor path, not just text generation time.

## Training workstream

The team identified training as a track requirement. The planned approach is to fine-tune a small detector on the GX10 using an existing labeled dataset, then evaluate it on images from the actual cap camera. This keeps data collection manageable within the hackathon.

1. Confirm the accepted training deliverable with the organizers.
2. Prove that the selected model can run on its intended deployment device.
3. Choose a narrow set of classes supported by the dataset.
4. Run one training epoch to estimate the time budget, then complete a bounded training run.
5. Save the checkpoint, dataset details, configuration, and training results.
6. Compare the baseline and trained model on held-out examples and real cap-camera images.

Inference with an existing model does not complete this workstream. No trained checkpoint or measured accuracy improvement is currently included in this repository.

## Getting started

Clone the repository using a GitHub account with access:

```bash
git clone git@github.com:hjo3-cse40/RAGging-Canes.git
cd RAGging-Canes
```

There are no application installation or launch commands yet. Before integration, put the laptop, ESP32, and GX10 on a local network that allows device-to-device communication. Remote access from the laptop to the GX10 is still pending confirmation; the earlier connection work did not establish a working session.

Once the GX10's username and network address are known, connect from the laptop with:

```bash
ssh USERNAME@GX10_IP
```

Replace both placeholders with the assigned values. Keep credentials and device-specific network settings out of the repository.

## Build milestones

- [ ] Confirm component models, power requirements, and GPIO assignments.
- [ ] Capture a usable image with the Freenove camera example.
- [ ] Drive one motor and verify ultrasonic readings independently.
- [ ] Establish laptop and ESP32 connectivity to the GX10.
- [ ] Benchmark candidate vision models on fresh cap-camera images.
- [ ] Implement a JPEG receiver and validated directional responses.
- [ ] Connect left/center/right feedback, then expand to five distinguishable zones.
- [ ] Add command expiry, connection-loss behavior, and sensor-error handling.
- [ ] Complete and document a GX10 training run.
- [ ] Measure the full pipeline and rehearse a stationary demonstration.
- [ ] Record a backup demo and preserve the working configuration.
