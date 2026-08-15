# Music-app

Share music with others.

## Cloud ML Vision (iOS + Pi)

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **MobileNet** (fast / accessible) | **Core ML** (Create ML) and/or **ML Kit** (TFLite) |
| **Raspberry Pi** | **EfficientNet** (higher accuracy) | TFLite runtime |

MobileNet on phones keeps VoiceOver and assistive flows responsive. EfficientNet on Pis uses more compute where power is less constrained.

### iOS dual stack

| Path | Role |
| --- | --- |
| [`ios/MusicApp/CoreML/`](ios/MusicApp/CoreML/) | Apple Core ML + Vision (`VNCoreMLRequest`) |
| [`ios/MusicApp/MLKit/`](ios/MusicApp/MLKit/) | Google ML Kit custom TFLite bridge |
| [`ml/create_ml/`](ml/create_ml/README.md) | Create ML training data layout |
| [`ml/prepare_coreml_model.py`](ml/prepare_coreml_model.py) | Core ML labels + checklist |
| [`ml/convert_to_coreml.py`](ml/convert_to_coreml.py) | Keras/SavedModel → `.mlpackage` |
| [`ml/prepare_mlkit_model.py`](ml/prepare_mlkit_model.py) | ML Kit / Pi label sidecars |
| [`ios/MusicApp/Firebase/`](ios/MusicApp/Firebase/) | Analytics + Firestore feedback |

```swift
MusicVisionClassifier(runtime: .coreML) // Create ML → Neural Engine
MusicVisionClassifier(runtime: .mlKit)   // TFLite → ML Kit
```

```bash
python3 ml/prepare_coreml_model.py
python3 ml/prepare_mlkit_model.py
```
