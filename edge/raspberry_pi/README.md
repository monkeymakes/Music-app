# Raspberry Pi — EfficientNet Music Vision

Use **EfficientNet** (compressed TFLite) on Raspberry Pi for higher-accuracy edge inference.
iPhones keep **TinyML / MobileNet** via Core ML / ML Kit for fast, accessible on-device results.

## Why the split

| Device | Model | Why |
| --- | --- | --- |
| iOS phone | TinyML / MobileNet | Low latency / power for VoiceOver and assistive use |
| Raspberry Pi | EfficientNet (quantized TFLite) | More compute on a plugged-in Pi; compression keeps RAM/latency acceptable |
| Red Hat / Linux | Same TFLite via Podman | Python-only edge — **no JavaScript** |

## Layout

| Path | Role |
| --- | --- |
| `efficientnet_infer.py` | Classify a single image (CLI) |
| `benchmark.py` | Compare latency across threads / XNNPACK |
| `infer_server.py` | stdlib HTTP server for LAN inference |
| `tflite_runtime.py` | XNNPACK + threading + uint8 support |
| `Containerfile` / `compose.yaml` | Podman deployment for RHEL / Pi |
| `up.sh` | Start Podman stack (prefers Podman over Docker) |
| `models/` | Place `music_vision_efficientnet.tflite` here |

## Compress the model (on your Mac — see `docs/mac-tensorflow-lab.md`)

```bash
pip install -r ml/requirements-export-tflite.txt
python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic
# Or from your trained Keras model:
# python3 ml/export_efficientnet_tflite.py --from-keras model.keras --quantize int8 --calibration-dir ml/calibration_images/
```

## Setup on the Pi (bare metal)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r edge/raspberry_pi/requirements.txt
python3 ml/prepare_mlkit_model.py --target raspberry_pi
```

## Run

```bash
# CLI
python3 edge/raspberry_pi/efficientnet_infer.py \
  --image /home/pi/photos/album.jpg \
  --threads 4 \
  --json-out /tmp/music_vision_result.json

# Benchmark
python3 edge/raspberry_pi/benchmark.py \
  --image /home/pi/photos/album.jpg \
  --threads 1,2,4

# HTTP server (no JS)
python3 edge/raspberry_pi/infer_server.py --host 0.0.0.0 --port 8088
```

## Podman on Red Hat / Fedora / Pi OS

```bash
./edge/raspberry_pi/up.sh
curl http://127.0.0.1:8088/health
```

Full guide: [`docs/edge-compression-and-deployment.md`](../../docs/edge-compression-and-deployment.md)

Optional: upload `/tmp/music_vision_result.json` into the same Firestore
`model_feedback` / analytics pipeline your iOS app uses, tagging
`architecture=efficientnet` and `device_class=raspberry_pi`.
