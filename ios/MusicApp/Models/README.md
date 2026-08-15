# Music vision models

## Device split

| Device | Architecture | File to place here / on Pi |
| --- | --- | --- |
| **iOS phone** | **MobileNet** | `music_vision_mobilenet.tflite` (this folder) |
| **Raspberry Pi** | **EfficientNet** | `edge/raspberry_pi/models/music_vision_efficientnet.tflite` |

MobileNet stays light so VoiceOver and other assistive tech stay responsive for
users with disabilities. EfficientNet runs on the Pi where power and thermals
allow a stronger network.

## iOS (ML Kit)

Add `music_vision_mobilenet.tflite` to the **MusicApp** target with
**Copy Bundle Resources** enabled.

Requirements:

- LiteRT / `.tflite` image classification model (MobileNet-family)
- Embedded metadata preferred
- Compatible with [ML Kit custom models](https://developers.google.com/ml-kit/custom-models)

```bash
python3 ml/prepare_mlkit_model.py --target ios_phone
```

## Remote updates

Download a newer MobileNet to Documents as `music_vision_mobilenet_remote.tflite`
(`RemoteModelDownloader`). The bridge prefers the downloaded file when present.
