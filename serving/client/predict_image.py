#!/usr/bin/env python3
"""REST client for Music Vision on TensorFlow Serving.

Sends a preprocessed image to:
  POST http://localhost:8501/v1/models/music_vision:predict

Example:
  python3 serving/client/predict_image.py --image path/to/album.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("serving/models/music_vision_labels.txt"),
        help="Newline-separated class labels",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8501/v1/models/music_vision:predict",
        help="TF Serving predict endpoint",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--preprocess",
        choices=["model", "minus_one_one"],
        default="model",
        help="'model' sends RGB [0,255] (model has Rescaling); "
        "'minus_one_one' sends (x/127.5)-1 for models without built-in normalize",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write inference JSON",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        fallback = Path("ml/labels.example.txt")
        if fallback.is_file():
            path = fallback
        else:
            raise SystemExit(f"Labels file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_image(path: Path, size: int, preprocess: str) -> list:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if preprocess == "minus_one_one":
        array = (array / 127.5) - 1.0
    # else: leave [0, 255] for models with an internal Rescaling layer
    return array.tolist()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    labels = load_labels(args.labels)
    instance = load_image(args.image, args.input_size, args.preprocess)
    payload = {"instances": [instance]}

    try:
        response = requests.post(args.url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach TensorFlow Serving at {args.url}: {exc}\n"
            "Start it with: docker compose -f serving/docker-compose.yml up"
        ) from exc

    if response.status_code != 200:
        raise SystemExit(f"Serving error {response.status_code}: {response.text}")

    body = response.json()
    predictions = body.get("predictions")
    if not predictions:
        raise SystemExit(f"Unexpected response: {body}")

    scores = np.asarray(predictions[0], dtype=np.float64).reshape(-1)
    if len(scores) != len(labels):
        print(
            f"Warning: {len(scores)} scores vs {len(labels)} labels",
            file=sys.stderr,
        )

    order = np.argsort(scores)[::-1][: args.top_k]
    results = [
        {
            "label": labels[i] if i < len(labels) else f"class_{i}",
            "score": float(scores[i]),
            "index": int(i),
        }
        for i in order
    ]

    print(f"image: {args.image}")
    for row in results:
        print(f"  {row['score']:.4f}  {row['label']}")

    out = {
        "architecture": "efficientnet",
        "device_class": "cloud_server",
        "runtime": "tensorflow_serving",
        "model": "music_vision",
        "image": str(args.image),
        "predictions": results,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
