# iOS ML Kit bridge (Music-app)

Bridges a custom AutoML / TensorFlow Lite vision model into **Google ML Kit** on iOS
via `LocalModel` → `CustomImageLabelerOptions` → `ImageLabeler`.

## Layout

| Path | Role |
| --- | --- |
| `MusicApp/MLKit/MLKitModelBridge.swift` | Core bridge (bundle / downloaded `.tflite` → ML Kit) |
| `MusicApp/MLKit/MusicVisionClassifier.swift` | App-facing SwiftUI-friendly API |
| `MusicApp/MLKit/RemoteModelDownloader.swift` | Optional remote model cache |
| `MusicApp/Models/` | Drop `music_vision.tflite` here |
| `Podfile` | `GoogleMLKit/ImageLabelingCustom` 8.0.0 |

## Xcode setup

1. Create an iOS App target named **MusicApp** (iOS 15+, SwiftUI or UIKit).
2. Add the Swift files under `MusicApp/MLKit/` to the target.
3. From this `ios/` directory:

```bash
pod install
open MusicApp.xcworkspace
```

4. Export / copy `music_vision.tflite` into `MusicApp/Models/` and enable
   **Copy Bundle Resources**.
5. Prepare labels / checklist from the repo root:

```bash
python3 ml/prepare_mlkit_model.py
```

## Usage

```swift
let classifier = MusicVisionClassifier()
classifier.prepare()

Task {
  let labels = try await classifier.classify(image: albumArt)
  // labels: [(text, confidence, index), ...]
}
```

For camera frames, use `classifyFrame(sampleBuffer:cameraPosition:)` on a
background queue with `AVCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = true`.

## Model sources

- **Bundled:** `music_vision.tflite` in the app bundle (offline, immediate).
- **Hosted:** download to Documents as `music_vision_remote.tflite`; the bridge
  prefers that path when present (see ML Kit custom model docs).
