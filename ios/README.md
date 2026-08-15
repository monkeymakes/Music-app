# iOS ML Kit bridge (Music-app)

Bridges a custom AutoML / TensorFlow Lite vision model into **Google ML Kit** on iOS
via `LocalModel` → `CustomImageLabelerOptions` → `ImageLabeler`, with **Firebase
Analytics** for model performance and **Cloud Firestore** for customer feedback.

## Layout

| Path | Role |
| --- | --- |
| `MusicApp/MLKit/MLKitModelBridge.swift` | Core bridge (bundle / downloaded `.tflite` → ML Kit) |
| `MusicApp/MLKit/MusicVisionClassifier.swift` | App API + Analytics hooks + feedback submit |
| `MusicApp/Firebase/ModelPerformanceAnalytics.swift` | Firebase Analytics events |
| `MusicApp/Firebase/ModelFeedbackService.swift` | Firestore `model_feedback` writes |
| `MusicApp/Firebase/FirebaseBootstrap.swift` | `FirebaseApp.configure()` |
| `MusicApp/Models/` | Drop `music_vision.tflite` here |
| `Podfile` | ML Kit + Firebase Analytics / Firestore / Storage |
| `GoogleService-Info.plist.example` | Copy → `GoogleService-Info.plist` from Firebase console |
| `firestore.rules.example` | Starter security rules for feedback |

## Xcode setup

1. Create an iOS App target named **MusicApp** (iOS 15+, SwiftUI or UIKit).
2. Add the Swift files under `MusicApp/MLKit/` and `MusicApp/Firebase/` to the target.
3. Create a Firebase iOS app, download `GoogleService-Info.plist`, add it to the target.
4. Enable **Analytics** and **Cloud Firestore** in the Firebase console.
5. From this `ios/` directory:

```bash
pod install
open MusicApp.xcworkspace
```

6. Export / copy `music_vision.tflite` into `MusicApp/Models/` (Copy Bundle Resources).
7. Optional: paste `firestore.rules.example` into Firestore Rules (then tighten).

## Firebase Analytics events

| Event | When | Useful params |
| --- | --- | --- |
| `music_vision_prepare` | Model load | `model_source`, `success` |
| `music_vision_inference` | After classify | `latency_ms`, `top_label`, `top_confidence`, `label_count`, `success` |
| `music_vision_feedback` | User rates prediction | `rating`, `top_label`, `has_comment` |

Register custom dimensions/metrics under **Analytics → Custom definitions**.

## Customer feedback (Firestore)

```swift
try await classifier.submitFeedback(rating: .helpful, comment: "Correct album genre")
// or .notHelpful
```

Documents land in collection `model_feedback` with inference labels, confidence,
latency, rating, and comment for model-quality review.

## Usage

```swift
_ = FirebaseBootstrap.configureIfNeeded()

let classifier = MusicVisionClassifier()
classifier.prepare()

Task {
  let labels = try await classifier.classify(image: albumArt)
  try await classifier.submitFeedback(rating: .helpful, comment: "Looks right")
}
```

For camera frames, use `classifyFrame(sampleBuffer:cameraPosition:)` on a
background queue. Analytics for frames are sampled (~1/30) to limit event volume.

## Model sources

- **Bundled:** `music_vision.tflite` in the app bundle (offline, immediate).
- **Hosted:** download to Documents as `music_vision_remote.tflite`; the bridge
  prefers that path when present (Firebase Storage helper in `RemoteModelDownloader`).
