#!/usr/bin/env python3
"""Prepare label / packaging metadata for bridging a vision model to iOS ML Kit.

This does not train a network. It validates a labels file and writes the
sidecar notes ML Kit custom image labeling expects alongside a .tflite model
exported from AutoML Vision Edge, TFLite Model Maker, or TensorFlow Hub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels for the classifier",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ios/MusicApp/Models"),
        help="Output directory for iOS bundle resources",
    )
    parser.add_argument(
        "--model-name",
        default="music_vision",
        help="Bundled resource name without extension",
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
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_out = out_dir / f"{args.model_name}_labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    checklist = {
        "model_resource": f"{args.model_name}.tflite",
        "labels_file": labels_out.name,
        "label_count": len(labels),
        "labels": labels,
        "ml_kit_ios": {
            "pod": "GoogleMLKit/ImageLabelingCustom",
            "bridge": "ios/MusicApp/MLKit/MLKitModelBridge.swift",
            "local_model_api": "LocalModel(path:) → CustomImageLabelerOptions → ImageLabeler",
        },
        "next_steps": [
            f"Export a metadata-enabled .tflite classifier to {out_dir / (args.model_name + '.tflite')}",
            "Add the .tflite file to the MusicApp Xcode target (Copy Bundle Resources)",
            "Call MusicVisionClassifier.prepare() then classify(image:)",
        ],
    }

    manifest = out_dir / f"{args.model_name}_bridge_manifest.json"
    manifest.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {labels_out}")
    print(f"Wrote {manifest}")
    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("Place the exported .tflite next to these files, then rebuild the iOS app.")


if __name__ == "__main__":
    main()
