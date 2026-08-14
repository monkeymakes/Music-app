# Music vision model (ML Kit / iOS)

Place your AutoML Vision Edge or TensorFlow Lite image classifier here as:

```text
music_vision.tflite
```

Then add it to the **MusicApp** Xcode target with **Copy Bundle Resources** enabled.

## Requirements for ML Kit

- LiteRT / `.tflite` image classification model
- Embedded metadata (input shape, labels, threshold) preferred
- Compatible with [ML Kit custom models](https://developers.google.com/ml-kit/custom-models)

## Prepare from this repo

From the repo root (Cloud Agent / local Python with Pillow):

```bash
python3 ml/prepare_mlkit_model.py --labels ml/labels.example.txt --out ios/MusicApp/Models
```

That writes label metadata notes and a checklist. Export the actual `.tflite`
from AutoML Vision Edge, TensorFlow Lite Model Maker, or TF Hub, then copy it
to `ios/MusicApp/Models/music_vision.tflite`.

## Remote updates

Download a newer model to the documents directory as `music_vision_remote.tflite`
(see `RemoteModelDownloader`). `MLKitModelBridge` prefers the downloaded file
when present, otherwise falls back to the bundled model.
