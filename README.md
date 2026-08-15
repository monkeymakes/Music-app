# Music-app

Share music with others.

## Cloud ML Vision

| Device | Model | Runtime |
| --- | --- | --- |
| **iOS phone** | **TinyML** (default, fastest / accessible) | Core ML and/or ML Kit |
| **iOS phone** | MobileNet (optional) | Core ML / ML Kit |
| **Raspberry Pi** | **EfficientNet** | TFLite |
| **Cloud / server** | Versioned SavedModel | **TensorFlow Serving** (Podman or Docker) |

```swift
MusicVisionClassifier(runtime: .coreML).prepare() // TinyML via Create ML / Core ML
MusicVisionClassifier(runtime: .mlKit).prepare()
```

```bash
python3 ml/prepare_tinyml_model.py
python3 ml/convert_to_coreml.py --tinyml --input path/to/model

# Server-side Serving (Podman preferred)
python3 ml/prepare_serving_model.py --export-stub --version 1
./serving/up.sh
python3 serving/client/predict_image.py --image path/to/album.jpg
```

See [`serving/README.md`](serving/README.md), [`ml/tinyml/README.md`](ml/tinyml/README.md), [`ios/README.md`](ios/README.md), [`ml/create_ml/README.md`](ml/create_ml/README.md).
