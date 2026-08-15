# TensorFlow Serving for Music Vision (cloud / server)

Server-side inference for the shared Music Vision labels. Prefer **Podman**;
Docker Compose works with the same file.

| Runtime | Where | Model |
| --- | --- | --- |
| Core ML / ML Kit | iOS phone | TinyML (default) / MobileNet |
| TFLite | Raspberry Pi | EfficientNet |
| **TensorFlow Serving** | **Cloud / server** | Versioned SavedModel (`music_vision`) |

## Quick start

```bash
# 1) Export a versioned SavedModel (stub for smoke tests, or your trained model)
pip install -r ml/requirements-serving.txt
python3 ml/prepare_serving_model.py --export-stub --version 1
# python3 ml/prepare_serving_model.py --from-keras path/to/model.keras --version 1
# python3 ml/prepare_serving_model.py --from-saved-model path/to/saved_model --version 1

# 2) Start Serving (Podman preferred, Docker fallback)
chmod +x serving/up.sh
./serving/up.sh

# Or explicitly:
podman compose -f serving/docker-compose.yml up -d
# docker compose -f serving/docker-compose.yml up -d

# 3) Predict
python3 serving/client/predict_image.py --image path/to/album.jpg
```

Endpoints after `up`:

- REST: `http://127.0.0.1:8501/v1/models/music_vision`
- Predict: `POST http://127.0.0.1:8501/v1/models/music_vision:predict`
- gRPC: `127.0.0.1:8500`

## Model layout & versioning

```
serving/models/music_vision/
  1/                 # version 1 SavedModel
  2/                 # version 2 SavedModel (optional)
  labels.txt
  serving_manifest.json
```

`serving/models.config` loads `music_vision` and keeps the **latest 2** versions
(`model_version_policy.latest.num_versions: 2`). The compose file polls the
config every 30 seconds so you can drop a new version directory without
rebuilding the image.

Pin a version from the client:

```bash
python3 serving/client/predict_image.py --image album.jpg --version 2
```

## Preprocess contract

Same as the Pi EfficientNet runner:

1. Resize RGB to **224×224**
2. `float32` array with `(pixel / 127.5) - 1.0` → `[-1, 1]`
3. Batch as TF Serving `instances: [ array ]`

Label order must match `ml/labels.example.txt` (and iOS / Pi sidecars).

## Podman notes

- Install Podman + `podman-compose` **or** the `podman compose` plugin.
- Rootless Podman is fine for local smoke tests; publish ports `8500` / `8501`.
- SELinux hosts may need `:Z` on volume mounts; edit `docker-compose.yml` if needed.

## Stop

```bash
podman compose -f serving/docker-compose.yml down
# docker compose -f serving/docker-compose.yml down
```
