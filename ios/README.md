# iOS ML Kit bridge (Music-app)

**iPhones run MobileNet** through Google ML Kit for fast, accessible on-device
vision. **Raspberry Pis run EfficientNet** (see `edge/raspberry_pi/`).

Firebase Analytics tracks model performance; Firestore stores customer feedback.

## Why MobileNet on phones

Users who rely on VoiceOver, larger text, or other assistive features need
predictions that return quickly and are announced clearly. MobileNet keeps
latency and battery cost low compared with larger nets like EfficientNet.

## Layout

| Path | Role |
| --- | --- |
| `MusicApp/MLKit/VisionDeploymentTarget.swift` | MobileNet (iOS) vs EfficientNet (Pi) |
| `MusicApp/MLKit/MLKitModelBridge.swift` | ML Kit bridge (defaults to MobileNet) |
| `MusicApp/MLKit/MusicVisionAccessibility.swift` | VoiceOver / friendly labels |
| `MusicApp/MLKit/MusicVisionClassifier.swift` | App API + Analytics + feedback |
| `MusicApp/Firebase/*` | Analytics + Firestore feedback |
| `MusicApp/Models/` | Drop `music_vision_mobilenet.tflite` here |
| `Podfile` | ML Kit + Firebase |
| `../edge/raspberry_pi/` | EfficientNet TFLite runner for Pi |

## Xcode setup

1. Create an iOS App target named **MusicApp** (iOS 15+).
2. Add Swift files under `MusicApp/MLKit/` and `MusicApp/Firebase/`.
3. Add Firebase `GoogleService-Info.plist` (from the example stub).
4. Enable Analytics + Firestore; `pod install`; open the workspace.
5. Bundle `music_vision_mobilenet.tflite` (Copy Bundle Resources).
6. `python3 ml/prepare_mlkit_model.py --target ios_phone`

## Usage

```swift
let classifier = MusicVisionClassifier(deploymentTarget: .iOSPhone) // MobileNet
classifier.prepare()

Task {
  let labels = try await classifier.classify(image: albumArt)
  // VoiceOver announces results automatically when enabled
  try await classifier.submitFeedback(rating: .helpful, comment: "Correct")
}
```

## Firebase events

Params now include `architecture=mobilenet` and `device_class=ios_phone`
(Pi jobs should tag `efficientnet` / `raspberry_pi`).

| Event | When |
| --- | --- |
| `music_vision_prepare` | Model load |
| `music_vision_inference` | After classify |
| `music_vision_feedback` | User rates prediction |
