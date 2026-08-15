#!/usr/bin/env python3
"""Prepare TinyML sidecars for Music Vision on iOS.

TinyML = smallest/fastest on-device model (MobileNetV3-Small + optional int8)
so VoiceOver and assistive flows stay responsive.

Writes:
  ios/MusicApp/Models/music_vision_tinyml_*.txt/json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path("ios/MusicApp/Models")
MODEL_NAME = "music_vision_tinyml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels",
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
    labels = load_labels(args.labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_out = OUT_DIR / f"{MODEL_NAME}_labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    checklist = {
        "device_class": "ios_phone",
        "architecture": "tinyml",
        "runtime": "coreml_or_mlkit",
        "model_resource_coreml": f"{MODEL_NAME}.mlpackage",
        "model_resource_mlkit": f"{MODEL_NAME}.tflite",
        "labels_file": labels_out.name,
        "label_count": len(labels),
        "labels": labels,
        "why": (
            "TinyML minimizes latency and battery use so accessibility "
            "(VoiceOver, larger type) stays smooth for handicapped users."
        ),
        "base_network": "MobileNetV3-Small (or Create ML compact classifier)",
        "quantization": "Prefer float16 or int8 via ml/convert_to_coreml.py --tinyml",
        "next_steps": [
            "Train a compact Image Classifier in Create ML, or a MobileNetV3-Small Keras model",
            f"Export Core ML: python3 ml/convert_to_coreml.py --tinyml --input MODEL "
            f"--output {OUT_DIR / (MODEL_NAME + '.mlpackage')}",
            f"Or drop TFLite at {OUT_DIR / (MODEL_NAME + '.tflite')} for ML Kit",
            "MusicVisionClassifier(runtime: .coreML)  # defaults to TinyML on iOS",
        ],
    }

    manifest = OUT_DIR / f"{MODEL_NAME}_manifest.json"
    manifest.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {labels_out}")
    print(f"Wrote {manifest}")
    print(f"TinyML classes ({len(labels)}): {', '.join(labels)}")
    print("iOS default → TinyML | optional MobileNet | Pi → EfficientNet")


if __name__ == "__main__":
    main()
