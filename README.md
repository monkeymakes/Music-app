# Music-app

Share music with others.

## Cloud ML Vision (iOS)

On-device vision uses **Google ML Kit** with a custom `.tflite` model bridged on iOS:

- Bridge: [`ios/MusicApp/MLKit/MLKitModelBridge.swift`](ios/MusicApp/MLKit/MLKitModelBridge.swift)
- Setup: [`ios/README.md`](ios/README.md)
- Prepare labels / checklist: `python3 ml/prepare_mlkit_model.py`
