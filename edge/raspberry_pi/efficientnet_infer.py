#!/usr/bin/env python3
"""Run Music Vision with EfficientNet TFLite on a Raspberry Pi.

Phones use MobileNet (iOS / ML Kit). Pis use EfficientNet for higher accuracy
on a plugged-in edge device. Supports compressed int8/float16 models exported
via ml/export_efficientnet_tflite.py with XNNPACK acceleration.

Example:
  python3 edge/raspberry_pi/efficientnet_infer.py \\
    --model edge/raspberry_pi/models/music_vision_efficientnet.tflite \\
    --labels ml/labels.example.txt \\
    --image /path/to/album.jpg \\
    --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.raspberry_pi.tflite_runtime import classify_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("edge/raspberry_pi/models/music_vision_efficientnet.tflite"),
        help="EfficientNet TFLite model path",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
        help="Newline-separated class labels",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--top-k", type=int, default=5, help="How many labels to print")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="TFLite interpreter threads (match Pi core count)",
    )
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK CPU delegate",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write inference JSON (for Firebase upload scripts)",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not labels:
        raise SystemExit(f"No labels in {path}")
    return labels


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Missing EfficientNet model: {args.model}\n"
            "Export one with: python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic"
        )
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    labels = load_labels(args.labels)
    result = classify_image(
        args.model,
        args.image,
        num_threads=args.threads,
        use_xnnpack=not args.no_xnnpack,
    )

    ranked = sorted(
        ((float(score), idx) for idx, score in enumerate(result.scores)),
        reverse=True,
    )[: args.top_k]

    results = []
    for score, idx in ranked:
        name = labels[idx] if idx < len(labels) else f"class_{idx}"
        results.append({"text": name, "confidence": score, "index": idx})
        print(f"{score:6.3f}  {name}")

    payload = {
        "architecture": "efficientnet",
        "device_class": "raspberry_pi",
        "model": str(args.model),
        "input_dtype": result.input_dtype,
        "latency_ms": result.latency_ms,
        "labels": results,
    }
    print(
        f"latency_ms={result.latency_ms:.1f} "
        f"architecture=efficientnet device=raspberry_pi dtype={result.input_dtype}"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
