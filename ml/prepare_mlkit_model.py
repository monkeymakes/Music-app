#!/usr/bin/env python3
"""Prepare Music Vision label sidecars for MobileNet (iOS) and EfficientNet (Pi).

Does not train networks. Writes labels + a deployment checklist so you can drop
in architecture-specific .tflite files:

  iOS phone     → MobileNet     → ios/MusicApp/Models/music_vision_mobilenet.tflite
  Raspberry Pi  → EfficientNet  → edge/raspberry_pi/models/music_vision_efficientnet.tflite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = {
    "ios_phone": {
        "architecture": "mobilenet",
        "model_name": "music_vision_mobilenet",
        "out": Path("ios/MusicApp/Models"),
        "device_class": "ios_phone",
        "runtime": "Google ML Kit ImageLabelingCustom on iOS",
        "accessibility": "VoiceOver announcements via MusicVisionAccessibility",
    },
    "raspberry_pi": {
        "architecture": "efficientnet",
        "model_name": "music_vision_efficientnet",
        "out": Path("edge/raspberry_pi/models"),
        "device_class": "raspberry_pi",
        "runtime": "tflite-runtime EfficientNet classifier on Raspberry Pi",
        "accessibility": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels for the classifier",
    )
    parser.add_argument(
        "--target",
        choices=["ios_phone", "raspberry_pi", "all"],
        default="all",
        help="Which deployment target to prepare",
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


def prepare_target(target_key: str, labels: list[str]) -> None:
    meta = TARGETS[target_key]
    out_dir = meta["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = meta["model_name"]

    labels_out = out_dir / f"{model_name}_labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    checklist = {
        "device_class": meta["device_class"],
        "architecture": meta["architecture"],
        "model_resource": f"{model_name}.tflite",
        "labels_file": labels_out.name,
        "label_count": len(labels),
        "labels": labels,
        "runtime": meta["runtime"],
        "accessibility": meta["accessibility"],
        "next_steps": [
            f"Export a metadata-enabled {meta['architecture']} .tflite to "
            f"{out_dir / (model_name + '.tflite')}",
            "Keep class order identical across MobileNet and EfficientNet exports",
            (
                "Add the .tflite to the MusicApp Xcode target (Copy Bundle Resources)"
                if target_key == "ios_phone"
                else "Run edge/raspberry_pi/efficientnet_infer.py on the Pi"
            ),
        ],
    }

    manifest = out_dir / f"{model_name}_bridge_manifest.json"
    manifest.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")
    print(f"[{target_key}] Wrote {labels_out}")
    print(f"[{target_key}] Wrote {manifest}")


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    keys = list(TARGETS) if args.target == "all" else [args.target]
    for key in keys:
        prepare_target(key, labels)
    print(f"Classes ({len(labels)}): {', '.join(labels)}")
    print("iOS → MobileNet | Raspberry Pi → EfficientNet")


if __name__ == "__main__":
    main()
