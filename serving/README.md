# TensorFlow Serving — Music Vision (cloud)

Serve the Music Vision classifier with **TensorFlow Serving** so phones and edge
devices can call a higher-capacity **EfficientNet** SavedModel over REST/gRPC.

| Device | Model | Runtime |
| --- | --- | --- |
| iOS phone | TinyML / MobileNet | Core ML / ML Kit (on-device) |
| Raspberry Pi | EfficientNet | TFLite |
| **Cloud / server** | **EfficientNet** | **TensorFlow Serving** |

Class labels stay identical across all three targets (`ml/labels.example.txt`).

## Layout

| Path | Role |
| --- | --- |
| `docker-compose.yml` | Runs `tensorflow/serving` with model config |
| `models.config` | Servable name `music_vision`, latest 2 versions |
| `models/music_vision/<version>/` | Versioned SavedModel tree |
| `client/predict_image.py` | REST predict helper |
| `../ml/prepare_serving_model.py` | Labels, manifest, stub / Keras export |

## 1. Prepare a SavedModel

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements-serving.txt

# Smoke-test stub (untrained, for wiring only):
python3 ml/prepare_serving_model.py --export-stub --version 1

# Or wrap your fine-tuned Keras model:
# python3 ml/prepare_serving_model.py --from-keras path/to/music_vision.keras --version 1
```

Expected tree:

```text
serving/models/music_vision/1/
  saved_model.pb
  variables/
serving/models/music_vision_labels.txt
serving/models/music_vision_serving_manifest.json
```

## 2. Start TensorFlow Serving

```bash
docker compose -f serving/docker-compose.yml up
```

Endpoints:

| Protocol | URL |
| --- | --- |
| Status | `GET http://localhost:8501/v1/models/music_vision` |
| Predict | `POST http://localhost:8501/v1/models/music_vision:predict` |
| gRPC | `localhost:8500` |

```bash
curl -s http://localhost:8501/v1/models/music_vision
```

## 3. Classify an image

```bash
pip install -r serving/client/requirements.txt
python3 serving/client/predict_image.py \
  --image path/to/album.jpg \
  --json-out /tmp/music_vision_serving.json
```

Stub models include an internal `Rescaling` layer — the client default
(`--preprocess model`) sends RGB floats in `[0, 255]`. For production exports
without built-in normalize, use `--preprocess minus_one_one`.

## Versioning

Drop a new export under `serving/models/music_vision/2/`. With
`model_config_file_poll_wait_seconds` set, TF Serving picks up the new version
without restarting the container. Clients that omit a version always hit the
latest.

## Notes

- Keep label order identical to iOS and Raspberry Pi exports.
- Do not commit large weight files; `serving/models/music_vision/` is gitignored.
- For production, put a reverse proxy / auth in front of `:8501` — the compose
  file binds locally for development only.
