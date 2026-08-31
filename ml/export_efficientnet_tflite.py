#!/usr/bin/env python3
"""Export Music Vision EfficientNet to compressed TFLite for Raspberry Pi.

Quantization modes shrink the model and speed up inference on small Pis:

  none     — float32 (largest, baseline accuracy)
  float16  — half-precision weights (≈2× smaller, minimal accuracy loss)
  dynamic  — dynamic-range int8 weights (good default for Pi without calibration)
  int8     — full integer quantization (smallest/fastest; needs calibration images)

Requires: pip install -r ml/requirements-export-tflite.txt

Examples:
  # Stub CNN for smoke tests (no trained weights needed)
  python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic

  # From your fine-tuned Keras model
  python3 ml/export_efficientnet_tflite.py \\
    --from-keras path/to/music_vision.keras \\
    --quantize int8 \\
    --calibration-dir ml/calibration_images/

  # From SavedModel
  python3 ml/export_efficientnet_tflite.py \\
    --from-saved-model path/to/saved_model \\
    --quantize float16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("edge/raspberry_pi/models/music_vision_efficientnet.tflite")
DEFAULT_LABELS = Path("ml/labels.example.txt")
INPUT_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Class labels (must match training order)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .tflite path",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "float16", "dynamic", "int8"],
        default="dynamic",
        help="Compression mode (default: dynamic — good Pi tradeoff without calibration)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory of RGB images for int8 calibration (required for --quantize int8)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Max images to use for int8 calibration",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--export-stub",
        action="store_true",
        help="Export EfficientNet-Lite0 stub resized to your label count",
    )
    group.add_argument(
        "--from-keras",
        type=Path,
        help="Fine-tuned Keras .keras / .h5 classifier",
    )
    group.add_argument(
        "--from-saved-model",
        type=Path,
        help="Existing SavedModel directory",
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


def build_stub_model(num_classes: int):
    """Small CNN stub for pipeline smoke tests (not production weights).

    Use --from-keras / --from-saved-model for a real EfficientNet classifier.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 5, strides=2, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="music_vision_efficientnet_stub")


def load_keras_model(path: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def saved_model_to_keras(path: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(str(path))


def preprocess_image(path: Path):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    return (array / 127.5) - 1.0


def calibration_generator(image_dir: Path, max_samples: int):
    import numpy as np

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in extensions
    )[:max_samples]
    if not paths:
        raise SystemExit(
            f"No calibration images under {image_dir}. "
            "Add RGB photos or use --quantize dynamic instead."
        )

    def representative_dataset():
        for path in paths:
            yield [np.expand_dims(preprocess_image(path), axis=0)]

    return representative_dataset, len(paths)


def convert_to_tflite(model, quantize: str, calibration_dir: Path | None, cal_samples: int):
    import tensorflow as tf

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.float32, name="image")
        ]
    )
    def serve_fn(image: tf.Tensor) -> tf.Tensor:
        return model(image, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serve_fn.get_concrete_function()]
    )

    if quantize == "none":
        pass
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "int8":
        if calibration_dir is None:
            raise SystemExit("--calibration-dir is required for --quantize int8")
        gen, count = calibration_generator(calibration_dir, cal_samples)
        print(f"Calibrating int8 with {count} images from {calibration_dir}")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise SystemExit(f"Unknown quantize mode: {quantize}")

    return converter.convert()


def write_manifest(output: Path, labels: list[str], quantize: str, size_bytes: int) -> None:
    manifest = output.parent / f"{output.stem}_export_manifest.json"
    payload = {
        "model": output.name,
        "architecture": "efficientnet",
        "device_class": "raspberry_pi",
        "quantization": quantize,
        "input_size": INPUT_SIZE,
        "preprocess": "RGB 224x224, (pixel/127.5)-1 → [-1,1]; uint8 models use 0-255",
        "label_count": len(labels),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        raise SystemExit(
            "TensorFlow is required for export. Install with:\n"
            "  pip install -r ml/requirements-export-tflite.txt"
        ) from None

    if args.export_stub:
        print(f"Building export stub CNN with {len(labels)} classes")
        model = build_stub_model(len(labels))
    elif args.from_keras:
        if not args.from_keras.is_file():
            raise SystemExit(f"Keras model not found: {args.from_keras}")
        model = load_keras_model(args.from_keras)
    else:
        if not args.from_saved_model.is_dir():
            raise SystemExit(f"SavedModel not found: {args.from_saved_model}")
        model = saved_model_to_keras(args.from_saved_model)

    print(f"Exporting TFLite (quantize={args.quantize}) → {args.output}")
    tflite_model = convert_to_tflite(
        model, args.quantize, args.calibration_dir, args.calibration_samples
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_model)
    size = len(tflite_model)
    print(f"Wrote {args.output} ({size / (1024 * 1024):.2f} MB)")
    write_manifest(args.output, labels, args.quantize, size)
    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("Deploy to Pi: edge/raspberry_pi/models/ or podman compose up in edge/raspberry_pi/")


if __name__ == "__main__":
    main()
