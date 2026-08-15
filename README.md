# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **TinyML** (default, fastest / accessible) | Core ML and/or ML Kit |
| **iOS phone** | MobileNet (optional) | Core ML / ML Kit |
| **iOS / LAN** | Registry-pinned model | **Server-side** `/v1/infer` |
| **Raspberry Pi** | **EfficientNet** | TFLite |

```swift
MusicVisionClassifier(runtime: .coreML).prepare() // TinyML on-device
MusicVisionClassifier(runtime: .serverSide).prepare() // versioned remote infer
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
