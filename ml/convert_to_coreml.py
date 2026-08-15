#!/usr/bin/env python3
"""Convert a trained image classifier into Core ML for Music-app iOS.

Supports:
  - TensorFlow SavedModel / Keras (.h5, SavedModel dir)
  - Optional TFLite via a SavedModel intermediate (preferred: train/export Keras)

Requires: pip install coremltools tensorflow pillow

Example:
  python3 ml/convert_to_coreml.py \\
    --input path/to/mobilenet_saved_model \\
    --labels ml/labels.example.txt \\
    --output ios/MusicApp/Models/music_vision_mobilenet.mlpackage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Keras .h5, SavedModel directory, or other coremltools-supported path",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels (same order as training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ios/MusicApp/Models/music_vision_mobilenet.mlpackage"),
        help="Output .mlpackage path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input spatial size (MobileNet default 224)",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label and not label.startswith("#")]
    if not labels:
        raise SystemExit(f"No labels found in {path}")
    return labels


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input model not found: {args.input}")

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit(
            "coremltools is required. Install with:\n"
            "  pip install 'coremltools>=7' tensorflow pillow"
        ) from None

    labels = load_labels(args.labels)
    print(f"Converting {args.input} → Core ML ({len(labels)} classes)…")

    # Prefer TensorFlow / Keras conversion path used by MobileNet exports.
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(args.input)
        mlmodel = ct.convert(
            model,
            convert_to="mlprogram",
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=(1, args.image_size, args.image_size, 3),
                    scale=1 / 255.0,
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            classifier_config=ct.ClassifierConfig(labels),
            minimum_deployment_target=ct.target.iOS15,
        )
    except Exception as keras_error:
        # Fallback: treat input as an already-traced / mil-compatible model path.
        print(f"Keras load failed ({keras_error}); trying generic ct.convert…", file=sys.stderr)
        mlmodel = ct.convert(
            str(args.input),
            convert_to="mlprogram",
            classifier_config=ct.ClassifierConfig(labels),
            minimum_deployment_target=ct.target.iOS15,
        )

    mlmodel.author = "Music-app"
    mlmodel.short_description = "Music Vision MobileNet (Core ML)"
    mlmodel.version = "1.0"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"Wrote {args.output}")
    print("Add the .mlpackage to the MusicApp Xcode target, then:")
    print("  MusicVisionClassifier(runtime: .coreML).prepare()")


if __name__ == "__main__":
    main()
