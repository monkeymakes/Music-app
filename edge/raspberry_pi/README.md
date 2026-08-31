# Raspberry Pi — EfficientNet Music Vision

Use **EfficientNet** (TFLite) on Raspberry Pi for higher-accuracy edge inference.
iPhones keep **MobileNet** via ML Kit for fast, accessible on-device results.

## Why the split

| Device | Model | Why |
| --- | --- | --- |
| iOS phone | MobileNet | Low latency / power so VoiceOver and assistive use stay responsive |
| Raspberry Pi | EfficientNet | More compute headroom on a plugged-in Pi for better accuracy |

## Layout

| Path | Role |
| --- | --- |
| `efficientnet_infer.py` | Classify an image with EfficientNet TFLite |
| `models/` | Place `music_vision_efficientnet.tflite` here |
| `requirements.txt` | Pi Python deps |

## Setup on the Pi

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r edge/raspberry_pi/requirements.txt
```

Export or copy your fine-tuned EfficientNet classifier to:

```text
edge/raspberry_pi/models/music_vision_efficientnet.tflite
```

Prepare shared labels from the repo root:

```bash
python3 ml/prepare_mlkit_model.py --target raspberry_pi
```

## Run

```bash
python3 edge/raspberry_pi/efficientnet_infer.py \
  --image /home/pi/photos/album.jpg \
  --json-out /tmp/music_vision_result.json
```

Optional: upload `/tmp/music_vision_result.json` into the same Firestore
`model_feedback` / analytics pipeline your iOS app uses, tagging
`architecture=efficientnet` and `device_class=raspberry_pi`.
