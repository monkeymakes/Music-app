# Music vision models

## Device split

| Device | Architecture | Files |
| --- | --- | --- |
| **iOS phone** | **MobileNet** | `music_vision_mobilenet.mlpackage` (Core ML) and/or `.tflite` (ML Kit) |
| **Raspberry Pi** | **EfficientNet** | `edge/raspberry_pi/models/music_vision_efficientnet.tflite` |

MobileNet stays light so VoiceOver and other assistive tech stay responsive.
EfficientNet runs on the Pi where power and thermals allow a stronger network.

## iOS — Core ML (Create ML)

1. Train / export via [`../../../ml/create_ml/README.md`](../../../ml/create_ml/README.md).
2. Place `music_vision_mobilenet.mlpackage` in this folder.
3. Add it to the **MusicApp** target (Xcode compiles it to `.mlmodelc`).

```bash
python3 ml/prepare_coreml_model.py --labels ml/labels.example.txt
```

```swift
MusicVisionClassifier(runtime: .coreML).prepare()
```

## iOS — ML Kit (TFLite)

Add `music_vision_mobilenet.tflite` to the **MusicApp** target with
**Copy Bundle Resources** enabled.

Requirements:

- LiteRT / `.tflite` image classification model (MobileNet-family)
- Embedded metadata preferred
- Compatible with [ML Kit custom models](https://developers.google.com/ml-kit/custom-models)

```bash
python3 ml/prepare_mlkit_model.py --target ios_phone
```

```swift
MusicVisionClassifier(runtime: .mlKit).prepare()
```

## Remote updates

- ML Kit: download to Documents as `music_vision_mobilenet_remote.tflite`
- Core ML: download as `music_vision_mobilenet_remote.mlpackage`

`RemoteModelDownloader` + each bridge prefer the downloaded file when present.
