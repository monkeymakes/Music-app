# iOS Music Vision — ML Kit + Core ML

**iPhones run MobileNet** for fast, accessible on-device vision. Choose:

| Runtime | Model file | Stack |
| --- | --- | --- |
| **Core ML** (default) | `music_vision_mobilenet.mlpackage` | Create ML / `coremltools` → Vision + Neural Engine |
| **ML Kit** | `music_vision_mobilenet.tflite` | Google ML Kit ImageLabelingCustom |

**Raspberry Pis run EfficientNet** (see `edge/raspberry_pi/`).

Firebase Analytics tracks model performance (`runtime=coreml|mlkit`); Firestore stores customer feedback.

## Why MobileNet on phones

Users who rely on VoiceOver, larger text, or other assistive features need
predictions that return quickly and are announced clearly. MobileNet keeps
latency and battery cost low compared with larger nets like EfficientNet.

## Layout

| Path | Role |
| --- | --- |
| `MusicApp/CoreML/CoreMLModelBridge.swift` | Apple Core ML + Vision bridge |
| `MusicApp/MLKit/MLKitModelBridge.swift` | Google ML Kit TFLite bridge |
| `MusicApp/MLKit/VisionInferenceRuntime.swift` | `.coreML` vs `.mlKit` |
| `MusicApp/MLKit/MusicVisionClassifier.swift` | App API + Analytics + feedback |
| `MusicApp/MLKit/MusicVisionAccessibility.swift` | VoiceOver / friendly labels |
| `MusicApp/Firebase/*` | Analytics + Firestore feedback |
| `MusicApp/Models/` | Drop `.mlpackage` and/or `.tflite` here |
| `Podfile` | ML Kit + Firebase (Core ML is system) |
| `../ml/create_ml/` | Create ML training folders |

## Xcode setup

1. Create an iOS App target named **MusicApp** (iOS 15+).
2. Add Swift files under `MusicApp/MLKit/`, `MusicApp/CoreML/`, and `MusicApp/Firebase/`.
3. Add Firebase `GoogleService-Info.plist` (from the example stub).
4. Enable Analytics + Firestore; `pod install`; open the workspace.
5. Bundle **Core ML**: `music_vision_mobilenet.mlpackage` (Create ML export).
6. Optional **ML Kit**: also bundle `music_vision_mobilenet.tflite`.
7. Prepare sidecars:

```bash
python3 ml/prepare_coreml_model.py --labels ml/labels.example.txt
python3 ml/prepare_mlkit_model.py --target ios_phone
```

## Usage

```swift
// Apple Core ML (Create ML export) — default
let classifier = MusicVisionClassifier(runtime: .coreML)
classifier.prepare()

// Or Google ML Kit (TFLite)
let mlKit = MusicVisionClassifier(runtime: .mlKit)
mlKit.prepare()

Task {
  let labels = try await classifier.classify(image: albumArt)
  try await classifier.submitFeedback(rating: .helpful, comment: "Correct")
}
```

## Create ML

See [`../ml/create_ml/README.md`](../ml/create_ml/README.md) for training and export.

## Firebase events

Params include `architecture`, `device_class`, and `runtime` (`coreml` or `mlkit`).

| Event | When |
| --- | --- |
| `music_vision_prepare` | Model load |
| `music_vision_inference` | After classify |
| `music_vision_feedback` | User rates prediction |
