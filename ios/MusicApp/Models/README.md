# Music vision models

| Device | Architecture | Files |
| --- | --- | --- |
| **iOS phone (default)** | **TinyML** | `music_vision_tinyml.mlpackage` / `.tflite` |
| **iOS phone (optional)** | MobileNet | `music_vision_mobilenet.mlpackage` / `.tflite` |
| **Raspberry Pi** | EfficientNet | `edge/raspberry_pi/models/music_vision_efficientnet.tflite` |

```bash
python3 ml/prepare_tinyml_model.py
python3 ml/convert_to_coreml.py --tinyml --input path/to/model
```

```swift
MusicVisionClassifier(runtime: .coreML).prepare() // TinyML
```
