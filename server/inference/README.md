# Server-side Music Vision inference

HTTP API that runs versioned TFLite models from [`ml/registry/`](../../ml/registry/).

## Why server-side?

- Heavier models (EfficientNet) without draining phone battery
- Same registry pins for Pi edge boxes and cloud/LAN inference
- Promote / rollback model versions without shipping a new iOS build

## Quick start

```bash
pip install -r server/inference/requirements.txt
export ALLOW_MOCK_INFERENCE=1   # returns deterministic mock scores if .tflite missing
export PYTHONPATH=.
uvicorn server.inference.app:app --reload --host 0.0.0.0 --port 8080
```

From repo root so `server.inference` imports resolve (`PYTHONPATH=.`).

### Smoke test

```bash
PYTHONPATH=. python3 server/inference/smoke_test.py
```

### Call inference

```bash
curl -s http://127.0.0.1:8080/health | jq

curl -s http://127.0.0.1:8080/v1/models/music_vision_efficientnet/active | jq

curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -F image=@/path/to/album.jpg \
  -F model=music_vision_efficientnet \
  -F channel=production \
  -F top_k=5 | jq
```

### Batch inference

Upload multiple images in one request:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/infer/batch \
  -F images=@/path/to/a.jpg \
  -F images=@/path/to/b.jpg \
  -F model=music_vision_efficientnet \
  -F top_k=5 | jq
```

### Stream frame batching

Create a session, push camera frames, and let the server auto-flush when `batch_size` is reached (FIFO) or when a weighted reservoir window completes:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8080/v1/stream/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model":"music_vision_efficientnet","batch_size":4,"max_wait_ms":500}' \
  | jq -r .session_id)

curl -s -X POST "http://127.0.0.1:8080/v1/stream/sessions/$SESSION/frames" \
  -F image=@/path/to/frame.jpg | jq

curl -s -X POST "http://127.0.0.1:8080/v1/stream/sessions/$SESSION/flush" | jq
curl -s -X DELETE "http://127.0.0.1:8080/v1/stream/sessions/$SESSION"
```

#### Weighted reservoir sampling

For high-frequency camera streams, use Algorithm A-Res so the server keeps a fixed-size, weight-biased sample instead of inferring every frame:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8080/v1/stream/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"music_vision_efficientnet",
    "batch_size":4,
    "max_wait_ms":1000,
    "sampling_mode":"weighted_reservoir",
    "sample_window":16
  }' | jq -r .session_id)

# Prefer sharper / more salient frames with a higher sample_weight
curl -s -X POST "http://127.0.0.1:8080/v1/stream/sessions/$SESSION/frames" \
  -F image=@/path/to/frame.jpg -F sample_weight=3.5 | jq
```

- `batch_size` — reservoir capacity (how many frames are kept for each flush)
- `sample_window` — offer this many frames before auto-flush (default `batch_size * 4` when `0`)
- `max_wait_ms` — also auto-flush when the current window is older than this
- `sample_weight` — per-frame weight (`> 0`); higher weight ⇒ more likely to stay in the sample

On iOS, pass `samplingMode: .weightedReservoir` to `ServerSideStreamSession.open` / `MusicVisionClassifier.openStreamSession`, and `sampleWeight:` to `pushFrame`.

On iOS, use `ServerSideStreamSession` or `MusicVisionClassifier.openStreamSession()` for server-side camera batching, and `classifyBatch(images:)` for still-image batches.

Pin an exact version (ignore channel):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -F image=@/path/to/album.jpg \
  -F model=music_vision_tinyml \
  -F version=1.0.0 | jq
```

## Versioning flow

1. Train / export a new `.tflite`
2. `python3 ml/registry/model_registry.py register ... 1.1.0 --artifact ...`
3. Promote to `staging`, smoke-test via `/v1/infer`
4. Promote to `production`
5. `POST /v1/reload-registry` (or restart) so the server picks up the pin
6. Rollback with `model_registry.py rollback ... --to 1.0.0` if needed

## Env

| Variable | Default | Meaning |
| --- | --- | --- |
| `MUSIC_VISION_REGISTRY` | `ml/registry/registry.json` | Registry path |
| `MUSIC_VISION_DEFAULT_MODEL` | `music_vision_efficientnet` | Default `/v1/infer` model |
| `ALLOW_MOCK_INFERENCE` | `1` | Allow mock scores when artifact missing |
| `CORS_ALLOW_ORIGINS` | `*` | CORS origins |
| `MUSIC_VISION_MAX_BATCH_SIZE` | `32` | Max images per batch request |
| `MUSIC_VISION_STREAM_BATCH_SIZE` | `8` | Default stream session batch size / reservoir capacity |
| `MUSIC_VISION_STREAM_MAX_WAIT_MS` | `500` | Default max wait before auto-flush |
| `MUSIC_VISION_STREAM_IDLE_TTL_MS` | `60000` | Stream session idle expiry |
| `MUSIC_VISION_STREAM_SAMPLING_MODE` | `fifo` | Default `fifo` or `weighted_reservoir` |

Set `ALLOW_MOCK_INFERENCE=0` in production once real artifacts are present.
