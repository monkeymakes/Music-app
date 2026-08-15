# TinyML (Music Vision)

**TinyML** is the default iOS architecture: a compact MobileNet-class model
(MobileNetV3-Small or Create ML compact classifier), optionally float16-quantized,
so VoiceOver and other assistive tech stay fast for users with disabilities.

| Device | Architecture |
| --- | --- |
| iOS phone (default) | **TinyML** |
| iOS phone (optional) | MobileNet |
| Raspberry Pi | EfficientNet |

## Prepare

```bash
python3 ml/prepare_tinyml_model.py
```

## Export Core ML

```bash
pip install -r ml/requirements-coreml.txt
python3 ml/convert_to_coreml.py --tinyml \
  --input path/to/mobilenetv3_small \
  --labels ml/labels.example.txt \
  --output ios/MusicApp/Models/music_vision_tinyml.mlpackage
```

Or train a **compact** Image Classifier in Create ML and export as
`music_vision_tinyml.mlpackage`.

## App

```swift
// iOS defaults to TinyML via VisionDeploymentTarget.iOSPhone
MusicVisionClassifier(runtime: .coreML).prepare()
MusicVisionClassifier(runtime: .mlKit).prepare()  // needs music_vision_tinyml.tflite
```
