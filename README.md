# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **TinyML** (default, fastest / accessible) | Core ML and/or ML Kit |
| **iOS phone** | MobileNet (optional) | Core ML / ML Kit |
| **iOS / LAN** | Registry-pinned model | **Server-side** `/v1/infer` |
| **Raspberry Pi** | **EfficientNet** | TFLite |
| **Edge / LAN (optional)** | Frame metadata windows | **[Clink](edge/clink/)** (Flink-inspired C++ streams) |

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

See [`ml/registry/`](ml/registry/), [`server/inference/`](server/inference/), [`edge/clink/`](edge/clink/), [`docs/clink-stream-processing.md`](docs/clink-stream-processing.md), [`ml/tinyml/README.md`](ml/tinyml/README.md), [`ios/README.md`](ios/README.md).
