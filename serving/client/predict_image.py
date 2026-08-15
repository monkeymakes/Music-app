#!/usr/bin/env python3
"""Call Music Vision TensorFlow Serving (REST) with an image.

Prefers the same preprocess as the Pi EfficientNet runner:
  float32 RGB resized to 224x224, then (x / 127.5) - 1.0

Example:
  python3 serving/client/predict_image.py --image path/to/album.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import requests
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Install client deps: pip install -r ml/requirements-serving.txt"
    ) from exc

DEFAULT_URL = "http://127.0.0.1:8501/v1/models/music_vision:predict"
DEFAULT_LABELS = Path("serving/models/music_vision/labels.txt")
FALLBACK_LABELS = Path("ml/labels.example.txt")
INPUT_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--url", default=DEFAULT_URL, help="TF Serving predict URL")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Labels file (defaults to serving manifest labels, then ml/labels.example.txt)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many labels to print")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional model version: .../music_vision/versions/N:predict",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write prediction JSON",
    )
    return parser.parse_args()


def resolve_labels(path: Path | None) -> list[str]:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend([DEFAULT_LABELS, FALLBACK_LABELS])
    for candidate in candidates:
        if candidate.is_file():
            labels = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if labels:
                return labels
    raise SystemExit("No labels file found")


def load_image(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"Image not found: {path}")
    image = Image.open(path).convert("RGB").resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return array.tolist()


def predict_url(base: str, version: int | None) -> str:
    if version is None:
        return base
    # Rewrite .../models/music_vision:predict → .../models/music_vision/versions/N:predict
    if ":predict" not in base:
        raise SystemExit("--version requires a URL ending with :predict")
    prefix, _ = base.rsplit(":predict", 1)
    return f"{prefix}/versions/{version}:predict"


def main() -> None:
    args = parse_args()
    labels = resolve_labels(args.labels)
    instances = [load_image(args.image)]
    url = predict_url(args.url, args.version)

    response = requests.post(url, json={"instances": instances}, timeout=60)
    if response.status_code != 200:
        print(response.text, file=sys.stderr)
        raise SystemExit(f"Serving error HTTP {response.status_code}")

    payload = response.json()
    preds = payload.get("predictions")
    if not preds:
        raise SystemExit(f"Unexpected response: {payload}")

    scores = preds[0]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    top = []
    for index, score in ranked[: max(1, args.top_k)]:
        label = labels[index] if index < len(labels) else f"class_{index}"
        top.append({"label": label, "score": float(score), "index": index})

    result = {
        "image": str(args.image),
        "url": url,
        "top": top,
        "label_count": len(labels),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
