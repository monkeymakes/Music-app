#!/usr/bin/env python3
"""Run Music Vision with EfficientNet TFLite on a Raspberry Pi.

Phones use MobileNet (iOS / ML Kit). Pis use EfficientNet for higher accuracy
on a plugged-in edge device.

Example:
  python3 edge/raspberry_pi/efficientnet_infer.py \\
    --model edge/raspberry_pi/models/music_vision_efficientnet.tflite \\
    --labels ml/labels.example.txt \\
    --image /path/to/album.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


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


def load_image(path: Path, size: tuple[int, int]):
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Install Pillow and numpy on the Pi: pip3 install pillow numpy"
        ) from exc

    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    # Common EfficientNet preprocessing: scale to [0, 1] then to [-1, 1]
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def run_tflite(model_path: Path, input_tensor):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise SystemExit(
                "Install tflite-runtime on the Pi: "
                "pip3 install tflite-runtime"
            ) from exc

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected = tuple(input_details[0]["shape"])
    if input_tensor.shape != expected:
        # Allow NHWC with dynamic batch when shape is [1,H,W,3]
        if len(expected) == 4 and expected[0] in (1, None):
            h, w = int(expected[1]), int(expected[2])
            if input_tensor.shape[1:3] != (h, w):
                raise SystemExit(
                    f"Input shape {input_tensor.shape} != model {expected}. "
                    "Re-export EfficientNet or resize accordingly."
                )

    if input_details[0]["dtype"].__name__ == "uint8":
        # Quantized models expect 0-255
        import numpy as np

        raw = ((input_tensor + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], raw)
    else:
        interpreter.set_tensor(input_details[0]["index"], input_tensor)

    started = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output, latency_ms


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Missing EfficientNet model: {args.model}\n"
            "Export music_vision_efficientnet.tflite into edge/raspberry_pi/models/"
        )
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    labels = load_labels(args.labels)

    # Default EfficientNet-Lite / B0 classifier input
    input_tensor = load_image(args.image, size=(224, 224))
    scores, latency_ms = run_tflite(args.model, input_tensor)

    ranked = sorted(
        ((float(score), idx) for idx, score in enumerate(scores)),
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
        "latency_ms": latency_ms,
        "labels": results,
    }
    print(f"latency_ms={latency_ms:.1f} architecture=efficientnet device=raspberry_pi")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
