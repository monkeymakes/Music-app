# Mac TensorFlow lab

Use your Mac (Apple Silicon or Intel) as the **model lab**: train, compress, and prove models before deploying `.tflite` artifacts to Raspberry Pis or Podman on Red Hat. The Pi and RHEL edge nodes run **Python only** — all heavy export work happens here.

## What the lab does

| Task | Tool | Output |
| --- | --- | --- |
| Compress EfficientNet for Pi | `ml/export_efficientnet_tflite.py` | `.tflite` + manifest |
| Compress TinyML for iOS | `ml/convert_to_coreml.py --tinyml` | `.mlpackage` (float16) |
| Prove server inference | `serving/up.sh` (Podman) | TF Serving REST :8501 |
| Internal TFLite distribution | See [internal-tflite-server-plan.md](notes/internal-tflite-server-plan.md) | nginx + Serving pod |

## Setup (≈15 min)

```bash
# Homebrew tooling
brew install podman podman-compose curl jq python@3.11

# Podman VM on Apple Silicon
export CONTAINERS_MACHINE_PROVIDER=applehv
podman machine init --cpus 4 --memory 8192 --disk-size 50 --now
podman machine start

# Python venv for export / conversion
cd /path/to/Music-app
python3.11 -m venv .venv-lab
source .venv-lab/bin/activate
pip install -r ml/requirements-export-tflite.txt
pip install -r ml/requirements-coreml.txt   # optional: iOS TinyML
pip install -r ml/requirements-serving.txt  # optional: TF Serving export
```

Persist the Podman provider in `~/.config/containers/containers.conf`:

```ini
[machine]
provider = "applehv"
```

## Export compressed Pi model

```bash
source .venv-lab/bin/activate

# Stub (no trained weights — good for pipeline smoke tests)
python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic

# Your trained model
python3 ml/export_efficientnet_tflite.py \
  --from-keras ~/Models/music_vision.keras \
  --quantize dynamic

# Prepare label sidecars
python3 ml/prepare_mlkit_model.py --target raspberry_pi
```

Copy `edge/raspberry_pi/models/` to your Pi or push the folder into a Podman volume on RHEL.

## Prove with TensorFlow Serving (Podman on Mac)

```bash
source .venv-lab/bin/activate
python3 ml/prepare_serving_model.py --export-stub --version 1
chmod +x serving/up.sh
./serving/up.sh

python3 serving/client/predict_image.py --image test/album.jpg
```

This uses the same preprocess contract as the Pi runner (224×224, [-1, 1] float).

## Dry-run Pi inference on Mac (x86_64 / arm64)

Before copying to a Pi, sanity-check the `.tflite` on your Mac:

```bash
pip install tensorflow  # full TF includes TFLite interpreter on Mac
python3 edge/raspberry_pi/efficientnet_infer.py \
  --image test/album.jpg \
  --threads 4
```

On Mac, XNNPACK may behave differently than on Pi ARM — use `benchmark.py` on the actual Pi for final tuning.

## Lab → edge workflow

1. **Train / fine-tune** on Mac (Create ML, Keras, or your pipeline).
2. **Compress** with `export_efficientnet_tflite.py` (dynamic or int8).
3. **Benchmark** optionally on Mac, then on Pi with `benchmark.py`.
4. **Deploy** via `scp`, git-lfs, or Podman volume to RHEL/Pi.
5. **Run** CLI (`efficientnet_infer.py`) or HTTP (`infer_server.py` / `up.sh`).

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `EfficientNetLite0` not found | Upgrade TF: `pip install -U 'tensorflow>=2.14,<2.17'` |
| Podman pull fails on ARM Mac | `podman pull --arch arm64 docker.io/tensorflow/serving:2.14.1` |
| int8 export fails | Add ≥10 images to `--calibration-dir` or use `--quantize dynamic` |
| Pi OOM on model load | Use `dynamic` or `int8`; limit container memory in `compose.yaml` |

See also [`docs/edge-compression-and-deployment.md`](edge-compression-and-deployment.md) for the full Red Hat / Podman edge story.
