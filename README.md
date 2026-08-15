# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Location |
| --- | --- | --- |
| **iOS phone** | **MobileNet** (fast / accessible) | [`ios/`](ios/README.md) |
| **Raspberry Pi** | **EfficientNet** (higher accuracy) | [`edge/raspberry_pi/`](edge/raspberry_pi/README.md) |

MobileNet on phones keeps VoiceOver and assistive flows responsive for users
with disabilities. EfficientNet on Pis uses more compute where power is less
constrained.

- iOS bridge: [`ios/MusicApp/MLKit/MLKitModelBridge.swift`](ios/MusicApp/MLKit/MLKitModelBridge.swift)
- Accessibility: [`ios/MusicApp/MLKit/MusicVisionAccessibility.swift`](ios/MusicApp/MLKit/MusicVisionAccessibility.swift)
- Firebase analytics / feedback: [`ios/MusicApp/Firebase/`](ios/MusicApp/Firebase/)
- Prepare both targets: `python3 ml/prepare_mlkit_model.py`
