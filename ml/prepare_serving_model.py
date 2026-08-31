#!/usr/bin/env python3
"""Export Music Vision as a versioned TensorFlow SavedModel for TF Serving.

Cloud / server inference uses versioned SavedModels under:

  serving/models/music_vision/<version>/

iOS stays on TinyML / MobileNet (Core ML / ML Kit). Raspberry Pi stays on
EfficientNet TFLite. Shared label order must match ml/labels.example.txt.

Examples:
  python3 ml/prepare_serving_model.py --export-stub --version 1
  python3 ml/prepare_serving_model.py --from-keras path/to/model.keras --version 2
  python3 ml/prepare_serving_model.py --from-saved-model path/to/saved_model --version 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_OUT = Path("serving/models/music_vision")
DEFAULT_LABELS = Path("ml/labels.example.txt")
INPUT_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Newline-separated class labels (same order as iOS / Pi)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="TF Serving model version directory number",
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=DEFAULT_OUT,
        help="Base path mounted into the serving container",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--export-stub",
        action="store_true",
        help="Write a small classifier stub for local serving smoke tests",
    )
    group.add_argument(
        "--from-keras",
        type=Path,
        help="Path to a .keras / .h5 classifier (softmax over Music Vision labels)",
    )
    group.add_argument(
        "--from-saved-model",
        type=Path,
        help="Path to an existing SavedModel directory to copy/version",
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


def version_dir(base: Path, version: int) -> Path:
    if version < 1:
        raise SystemExit("--version must be >= 1")
    return base / str(version)


def write_sidecars(base: Path, labels: list[str], version: int, source: str) -> None:
    labels_path = base / "labels.txt"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    manifest = {
        "device_class": "cloud_server",
        "architecture": "efficientnet_or_keras_classifier",
        "runtime": "tensorflow_serving",
        "model_name": "music_vision",
        "version": version,
        "input_size": [INPUT_SIZE, INPUT_SIZE, 3],
        "preprocess": "x/127.5 - 1.0  # float32 RGB -> [-1, 1]",
        "signature": "serving_default",
        "label_count": len(labels),
        "labels": labels,
        "labels_file": labels_path.as_posix(),
        "source": source,
        "rest": "POST http://127.0.0.1:8501/v1/models/music_vision:predict",
        "grpc": "127.0.0.1:8500",
        "compose": (
            "podman compose -f serving/docker-compose.yml up -d  "
            "# preferred; or docker compose / ./serving/up.sh"
        ),
        "next_steps": [
            "Keep label order identical to iOS TinyML/MobileNet and Pi EfficientNet",
            "Roll forward by exporting --version N+1; models.config keeps latest 2",
            "Smoke test: python3 serving/client/predict_image.py --image path/to.jpg",
        ],
    }
    manifest_path = base / "serving_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {labels_path}")
    print(f"Wrote {manifest_path}")


def export_stub(out: Path, num_classes: int) -> None:
    """Small CNN stub. Input float32 RGB in [-1, 1], output softmax."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="inputs")
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="music_vision_stub")
    out.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(out))


def export_from_keras(src: Path, out: Path) -> None:
    import tensorflow as tf

    if not src.exists():
        raise SystemExit(f"Keras model not found: {src}")
    model = tf.keras.models.load_model(src)
    out.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(out))


def copy_saved_model(src: Path, out: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"SavedModel directory not found: {src}")
    has_pb = (src / "saved_model.pb").exists() or (src / "saved_model.pbtxt").exists()
    if not has_pb and not (src / "variables").exists():
        raise SystemExit(f"Does not look like a SavedModel: {src}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    out = version_dir(args.out_base, args.version)
    args.out_base.mkdir(parents=True, exist_ok=True)

    if out.exists():
        shutil.rmtree(out)

    if args.export_stub:
        export_stub(out, len(labels))
        source = "stub_cnn_random_weights"
    elif args.from_keras:
        export_from_keras(args.from_keras, out)
        source = str(args.from_keras)
    else:
        copy_saved_model(args.from_saved_model, out)
        source = str(args.from_saved_model)

    write_sidecars(args.out_base, labels, args.version, source)
    print(f"SavedModel ready: {out}")
    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("Start serving: ./serving/up.sh   # Podman preferred, Docker fallback")


if __name__ == "__main__":
    main()
