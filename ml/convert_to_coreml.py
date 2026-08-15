#!/usr/bin/env python3
"""Convert a trained image classifier into Core ML for Music-app iOS.

Supports:
  - TensorFlow SavedModel / Keras (.h5, SavedModel dir)
  - TinyML mode: smaller input + float16 weight quantization

Requires: pip install -r ml/requirements-coreml.txt

Examples:
  python3 ml/convert_to_coreml.py \\
    --input path/to/mobilenet_saved_model \\
    --labels ml/labels.example.txt \\
    --output ios/MusicApp/Models/music_vision_mobilenet.mlpackage

  # TinyML (default iOS path — smallest / fastest for accessibility)
  python3 ml/convert_to_coreml.py --tinyml \\
    --input path/to/mobilenetv3_small \\
    --labels ml/labels.example.txt \\
    --output ios/MusicApp/Models/music_vision_tinyml.mlpackage
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
        default=None,
        help="Output .mlpackage path (defaults depend on --tinyml)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input size (default 224, or 160 with --tinyml)",
    )
    parser.add_argument(
        "--tinyml",
        action="store_true",
        help="TinyML export: smaller default size + float16 quantization",
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


def maybe_quantize_tinyml(mlmodel, ct):  # type: ignore[no-untyped-def]
    """Apply float16 weight quantization for a smaller TinyML package."""
    try:
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="float16",
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        return ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    except Exception as exc:  # noqa: BLE001 — best-effort TinyML shrink
        print(f"TinyML float16 quantize skipped ({exc}); saving unquantized model.", file=sys.stderr)
        return mlmodel


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input model not found: {args.input}")

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit(
            "coremltools is required. Install with:\n"
            "  pip install -r ml/requirements-coreml.txt"
        ) from None

    image_size = args.image_size or (160 if args.tinyml else 224)
    if args.output is None:
        name = "music_vision_tinyml" if args.tinyml else "music_vision_mobilenet"
        args.output = Path(f"ios/MusicApp/Models/{name}.mlpackage")

    labels = load_labels(args.labels)
    mode = "TinyML" if args.tinyml else "Core ML"
    print(f"Converting {args.input} → {mode} ({len(labels)} classes, {image_size}px)…")

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(args.input)
        mlmodel = ct.convert(
            model,
            convert_to="mlprogram",
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=(1, image_size, image_size, 3),
                    scale=1 / 255.0,
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            classifier_config=ct.ClassifierConfig(labels),
            minimum_deployment_target=ct.target.iOS15,
        )
    except Exception as keras_error:
        print(f"Keras load failed ({keras_error}); trying generic ct.convert…", file=sys.stderr)
        mlmodel = ct.convert(
            str(args.input),
            convert_to="mlprogram",
            classifier_config=ct.ClassifierConfig(labels),
            minimum_deployment_target=ct.target.iOS15,
        )

    if args.tinyml:
        mlmodel = maybe_quantize_tinyml(mlmodel, ct)

    mlmodel.author = "Music-app"
    mlmodel.short_description = (
        "Music Vision TinyML (quantized MobileNet-class)"
        if args.tinyml
        else "Music Vision MobileNet (Core ML)"
    )
    mlmodel.version = "1.0"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"Wrote {args.output}")
    print("Add the .mlpackage to the MusicApp Xcode target, then:")
    print("  MusicVisionClassifier(runtime: .coreML).prepare()  # iOS defaults to TinyML")


if __name__ == "__main__":
    main()
