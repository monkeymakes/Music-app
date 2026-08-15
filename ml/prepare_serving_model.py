#!/usr/bin/env python3
"""Prepare Music Vision artifacts for TensorFlow Serving.

Writes shared labels + a deployment manifest under serving/models/.
Optionally exports a versioned SavedModel that TF Serving can load:

  Cloud / server → EfficientNet (SavedModel) → serving/models/music_vision/<version>/

Does not train production weights. Use --export-stub for a smoke-test model,
or --from-keras to wrap an existing fine-tuned .keras / SavedModel export.

Examples:
  python3 ml/prepare_serving_model.py
  python3 ml/prepare_serving_model.py --export-stub --version 1
  python3 ml/prepare_serving_model.py --from-keras path/to/music_vision.keras --version 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODEL_NAME = "music_vision"
OUT_DIR = Path("serving/models")
DEFAULT_INPUT_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels (same order as iOS / Pi)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="SavedModel version directory (TF Serving loads numbered subdirs)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Square input size expected by the classifier",
    )
    parser.add_argument(
        "--export-stub",
        action="store_true",
        help="Export a tiny EfficientNet-style stub SavedModel for local smoke tests",
    )
    parser.add_argument(
        "--from-keras",
        type=Path,
        default=None,
        help="Path to a fine-tuned .keras or SavedModel directory to re-export",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Labels file not found: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not labels:
        raise SystemExit(f"No labels found in {path}")
    return labels


def write_sidecars(labels: list[str], version: int, input_size: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    version_dir = OUT_DIR / MODEL_NAME / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)

    labels_out = OUT_DIR / f"{MODEL_NAME}_labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    checklist = {
        "device_class": "cloud_server",
        "architecture": "efficientnet",
        "model_name": MODEL_NAME,
        "model_resource": f"{MODEL_NAME}/{version}/",
        "labels_file": labels_out.name,
        "label_count": len(labels),
        "labels": labels,
        "input_size": [input_size, input_size, 3],
        "preprocessing": {
            "client_default": "RGB float32 in [0, 255] when the SavedModel includes Rescaling",
            "alternate": "RGB float32 in [-1, 1] via (x / 127.5 - 1.0) for models without normalize",
        },
        "runtime": "TensorFlow Serving (Docker) REST :8501 / gRPC :8500",
        "endpoints": {
            "status": f"GET http://localhost:8501/v1/models/{MODEL_NAME}",
            "predict": f"POST http://localhost:8501/v1/models/{MODEL_NAME}:predict",
        },
        "next_steps": [
            f"Place or export a versioned SavedModel under {version_dir}/",
            "Keep class order identical to iOS TinyML/MobileNet and Pi EfficientNet",
            "docker compose -f serving/docker-compose.yml up",
            "python3 serving/client/predict_image.py --image path/to/photo.jpg",
        ],
    }

    manifest = OUT_DIR / f"{MODEL_NAME}_serving_manifest.json"
    manifest.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {labels_out}")
    print(f"Wrote {manifest}")
    return version_dir


def export_stub(version_dir: Path, num_classes: int, input_size: int) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "Install serving export deps: pip install -r ml/requirements-serving.txt"
        ) from exc

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), name="inputs")
    # Lightweight stand-in so Docker smoke tests work without production weights.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(inputs)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(
        x
    )
    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)

    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    @tf.function(input_signature=[tf.TensorSpec([None, input_size, input_size, 3], tf.float32, name="inputs")])
    def serving_fn(inputs_tensor):
        return {"probabilities": model(inputs_tensor)}

    tf.saved_model.save(model, str(version_dir), signatures={"serving_default": serving_fn})
    print(f"Exported stub SavedModel → {version_dir}")
def export_from_keras(source: Path, version_dir: Path) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "Install serving export deps: pip install -r ml/requirements-serving.txt"
        ) from exc

    if not source.exists():
        raise SystemExit(f"Keras / SavedModel path not found: {source}")

    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir() and (source / "saved_model.pb").exists():
        # Already a SavedModel — copy into the versioned serving tree.
        shutil.copytree(source, version_dir, dirs_exist_ok=True)
        print(f"Copied SavedModel → {version_dir}")
        return

    model = tf.keras.models.load_model(source)
    tf.saved_model.save(model, str(version_dir))
    print(f"Exported Keras model → {version_dir}")


def main() -> None:
    args = parse_args()
    if args.export_stub and args.from_keras:
        raise SystemExit("Use only one of --export-stub or --from-keras")

    labels = load_labels(args.labels)
    version_dir = write_sidecars(labels, args.version, args.input_size)

    if args.export_stub:
        export_stub(version_dir, len(labels), args.input_size)
    elif args.from_keras:
        export_from_keras(args.from_keras, version_dir)
    else:
        placeholder = version_dir / ".gitkeep"
        placeholder.touch(exist_ok=True)
        print(
            f"Sidecars ready. Add SavedModel files under {version_dir}/ "
            "or re-run with --export-stub / --from-keras"
        )

    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("Cloud → TensorFlow Serving (EfficientNet SavedModel)")


if __name__ == "__main__":
    main()
