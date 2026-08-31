# Music-app

Share music with others.

## ML Vision (device-tiered)

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **TinyML** (default, fastest / accessible) | Core ML and/or ML Kit |
| **iOS phone** | MobileNet (optional) | Core ML / ML Kit |
| **iOS / LAN** | Registry-pinned model | **Server-side** `/v1/infer` |
| **Raspberry Pi / RHEL** | **EfficientNet** (quantized TFLite) | TFLite + Podman (Python only, no JS) |
| **Cloud / Mac lab** | SavedModel | TensorFlow Serving (Podman) |

```swift
MusicVisionClassifier(runtime: .coreML).prepare() // TinyML on-device
MusicVisionClassifier(runtime: .serverSide).prepare() // versioned remote infer
try await classifier.classifyBatch(images: photos) // batch still images
let stream = try await classifier.openStreamSession(
  batchSize: 8,
  samplingMode: .weightedReservoir,
  sampleWindow: 32
)
try await stream.pushFrame(cameraFrame, sampleWeight: sharpnessScore)
```

```bash
# Model versioning
python3 ml/registry/model_registry.py list
python3 ml/registry/model_registry.py get-active music_vision_efficientnet

# Server-side inference
pip install -r server/inference/requirements.txt
uvicorn server.inference.app:app --reload --port 8080
```

### Edge compression and Pi deployment

```bash
# Mac lab: compress EfficientNet for Pi
pip install -r ml/requirements-export-tflite.txt
python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic

# Pi / Red Hat: Podman inference (no JavaScript)
./edge/raspberry_pi/up.sh
```

See [`docs/edge-compression-and-deployment.md`](docs/edge-compression-and-deployment.md), [`docs/mac-tensorflow-lab.md`](docs/mac-tensorflow-lab.md), [`edge/raspberry_pi/`](edge/raspberry_pi/), [`ml/registry/`](ml/registry/), [`server/inference/`](server/inference/), [`serving/`](serving/), [`ml/tinyml/README.md`](ml/tinyml/README.md), [`ios/README.md`](ios/README.md).
