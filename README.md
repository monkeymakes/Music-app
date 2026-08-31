# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **TinyML** (default, fastest / accessible) | Core ML and/or ML Kit |
| **iOS phone** | MobileNet (optional) | Core ML / ML Kit |
| **iOS / LAN** | Registry-pinned model | **Server-side** `/v1/infer` |
| **Raspberry Pi** | **EfficientNet** | TFLite |

## Multi-Pi Streaming Observability (Wheelchair MVP)

Three wheelchair Pis stream telemetry to a central hub. Events are micro-batched and accumulated in **TensorBoard**. Cross-Pi **function calling** routes commands between sensor, voice, and vision nodes.

| Pi | Role | Streams |
| --- | --- | --- |
| **pi-sensors-01** | Distance + IMU | `distance_*_m`, `imu` |
| **pi-voice-01** | Wyoming/Aloha + mic | `wake_word_*`, `audio_level` |
| **pi-vision-01** | Stereo camera + 3D | `stereo_frame`, `depth`, `recognition` |

```bash
# Hub
pip install -r server/inference/requirements.txt
PYTHONPATH=. uvicorn server.inference.app:app --host 0.0.0.0 --port 8080

# Simulate all 3 Pis
python3 edge/wheelchair/run_all_pis_demo.py --duration 10

# TensorBoard
tensorboard --logdir runs/wheelchair_observability
```

See [`edge/wheelchair/README.md`](edge/wheelchair/README.md) for full setup.

```swift
MusicVisionClassifier(runtime: .coreML).prepare() // TinyML on-device
MusicVisionClassifier(runtime: .serverSide).prepare() // versioned remote infer
try await classifier.classifyBatch(images: photos) // batch still images
let stream = try await classifier.openStreamSession(batchSize: 8)
try await stream.pushFrame(cameraFrame)
```

```bash
# Model versioning
python3 ml/registry/model_registry.py list
python3 ml/registry/model_registry.py get-active music_vision_efficientnet

# Server-side inference
pip install -r server/inference/requirements.txt
uvicorn server.inference.app:app --reload --port 8080
```

See [`ml/registry/`](ml/registry/), [`server/inference/`](server/inference/), [`ml/tinyml/README.md`](ml/tinyml/README.md), [`ios/README.md`](ios/README.md).
