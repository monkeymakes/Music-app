# iOS Music Vision — TinyML + Core ML + ML Kit + Server-side

**iPhones default to TinyML** (compact MobileNet) for the fastest accessible
on-device vision. Run it through:

| Runtime | Model file | Stack |
| --- | --- | --- |
| **Core ML** (default) | `music_vision_tinyml.mlpackage` | Create ML / `convert_to_coreml.py --tinyml` |
| **ML Kit** | `music_vision_tinyml.tflite` | Google ML Kit ImageLabelingCustom |
| **Server** | Registry pin → `/v1/infer` | Versioned server-side inference |

**Raspberry Pis run EfficientNet** (see `edge/raspberry_pi/`).

## Layout

| Path | Role |
| --- | --- |
| `MusicApp/CoreML/CoreMLModelBridge.swift` | Apple Core ML + Vision |
| `MusicApp/MLKit/MLKitModelBridge.swift` | Google ML Kit TFLite |
| `MusicApp/ServerSide/` | HTTP client + version resolver |
| `MusicApp/MLKit/VisionDeploymentTarget.swift` | TinyML (iOS) vs EfficientNet (Pi) |
| `MusicApp/MLKit/MusicVisionClassifier.swift` | App API + Analytics + feedback |
| `../ml/registry/` | Model version channels |
| `../server/inference/` | Server-side inference API |
| `../ml/tinyml/` | TinyML docs |
| `../ml/create_ml/` | Create ML training folders |

## Setup

```bash
python3 ml/prepare_tinyml_model.py
python3 ml/convert_to_coreml.py --tinyml --input path/to/mobilenetv3_small
```

Bundle `music_vision_tinyml.mlpackage` (and optional `.tflite`) in MusicApp.

```swift
let classifier = MusicVisionClassifier(runtime: .coreML) // TinyML
classifier.prepare()

// Server-side (registry-backed version pins)
let server = MusicVisionClassifier(
  runtime: .serverSide,
  serverConfiguration: .init(
    baseURL: URL(string: "https://infer.example.com")!,
    modelName: "music_vision_tinyml",
    channel: "production"
  )
)
await server.prepareServerSide()
```

Model versioning: [`../ml/registry/README.md`](../ml/registry/README.md).
Server API: [`../server/inference/README.md`](../server/inference/README.md).
