#!/usr/bin/env python3
"""Prepare Create ML / Core ML sidecars for Music Vision on iOS.

Writes labels + a deployment checklist for Apple Core ML (Neural Engine):

  Create ML training data → music_vision_mobilenet.mlpackage
  → ios/MusicApp/Models/music_vision_mobilenet.mlpackage

Does not train networks. Use macOS Create ML (Xcode) or ml/convert_to_coreml.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path("ios/MusicApp/Models")
CREATE_ML_DIR = Path("ml/create_ml")
MODEL_NAME = "music_vision_mobilenet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels for the classifier",
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


def write_create_ml_layout(labels: list[str]) -> None:
    """Create empty per-class folders for Create ML Image Classifier training."""
    training = CREATE_ML_DIR / "TrainingData"
    testing = CREATE_ML_DIR / "TestingData"
    for root in (training, testing):
        for label in labels:
            (root / label).mkdir(parents=True, exist_ok=True)
    readme = CREATE_ML_DIR / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# Create ML training data\n\n"
            "Drop images into `TrainingData/<label>/` and optional\n"
            "`TestingData/<label>/`, then train an Image Classifier in Create ML.\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_create_ml_layout(labels)

    labels_out = OUT_DIR / f"{MODEL_NAME}_coreml_labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    checklist = {
        "device_class": "ios_phone",
        "architecture": "mobilenet",
        "runtime": "coreml",
        "model_resource": f"{MODEL_NAME}.mlpackage",
        "labels_file": labels_out.name,
        "label_count": len(labels),
        "labels": labels,
        "compute_units": "all (CPU + GPU + Neural Engine)",
        "accessibility": "VoiceOver announcements via MusicVisionAccessibility",
        "create_ml_training_data": str(CREATE_ML_DIR / "TrainingData"),
        "next_steps": [
            f"Train an Image Classifier in Create ML with folders under "
            f"{CREATE_ML_DIR / 'TrainingData'}",
            f"Export as {OUT_DIR / (MODEL_NAME + '.mlpackage')}",
            "Or convert a TF/Keras MobileNet: python3 ml/convert_to_coreml.py "
            "--input path/to/model",
            "Add the .mlpackage to the MusicApp Xcode target",
            "MusicVisionClassifier(runtime: .coreML).prepare()",
        ],
    }

    manifest = OUT_DIR / f"{MODEL_NAME}_coreml_manifest.json"
    manifest.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {labels_out}")
    print(f"Wrote {manifest}")
    print(f"Create ML folders under {CREATE_ML_DIR}")
    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("iOS Core ML → MobileNet (.mlpackage) | ML Kit remains available via .tflite")


if __name__ == "__main__":
    main()
