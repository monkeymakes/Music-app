# Music Vision model registry

Controls **which model version** each channel serves for on-device downloads and
**server-side inference**.

## Principles

1. **Versions are immutable** — never overwrite an artifact path; register `1.1.0`.
2. **Channels are pointers** — `production` / `staging` point at a version string.
3. **Rollback is a repoint** — `rollback --to 1.0.0` moves the channel back.

## Commands

```bash
python3 ml/registry/model_registry.py list

python3 ml/registry/model_registry.py get-active music_vision_efficientnet

python3 ml/registry/model_registry.py register music_vision_tinyml 1.1.0 \
  --artifact ios/MusicApp/Models/music_vision_tinyml.tflite \
  --labels ios/MusicApp/Models/music_vision_tinyml_labels.txt \
  --notes "Retrain with Aug labels"

python3 ml/registry/model_registry.py promote music_vision_tinyml 1.1.0 \
  --channel staging

python3 ml/registry/model_registry.py promote music_vision_tinyml 1.1.0 \
  --channel production

python3 ml/registry/model_registry.py rollback music_vision_tinyml \
  --channel production --to 1.0.0
```

## Models

| Name | Architecture | Default use |
| --- | --- | --- |
| `music_vision_tinyml` | TinyML | iOS phone (fast / accessible) |
| `music_vision_mobilenet` | MobileNet | Optional larger iOS path |
| `music_vision_efficientnet` | EfficientNet | Raspberry Pi + server-side |

The server reads this file at startup (and on reload) to resolve
`channel → version → artifact`. See [`server/inference/`](../../server/inference/).
