#!/usr/bin/env python3
"""Benchmark compressed EfficientNet TFLite models on Raspberry Pi.

Compare latency across quantization modes and thread counts. Run on the Pi
after exporting models with ml/export_efficientnet_tflite.py.

Example:
  python3 edge/raspberry_pi/benchmark.py \\
    --model edge/raspberry_pi/models/music_vision_efficientnet.tflite \\
    --image test/album.jpg \\
    --threads 1,2,4 \\
    --warmup 3 \\
    --runs 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.raspberry_pi.tflite_runtime import (  # noqa: E402
    create_interpreter,
    preprocess_image,
    set_input_tensor,
    invoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("edge/raspberry_pi/models/music_vision_efficientnet.tflite"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--threads",
        default="1,2,4",
        help="Comma-separated thread counts to test",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK delegate",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write benchmark results as JSON",
    )
    return parser.parse_args()


def benchmark(
    model_path: Path,
    input_tensor,
    *,
    num_threads: int,
    use_xnnpack: bool,
    warmup: int,
    runs: int,
) -> dict:
    interpreter = create_interpreter(
        model_path, num_threads=num_threads, use_xnnpack=use_xnnpack
    )
    set_input_tensor(interpreter, input_tensor)

    for _ in range(warmup):
        invoke(interpreter)

    latencies = []
    for _ in range(runs):
        _, ms = invoke(interpreter)
        latencies.append(ms)

    return {
        "threads": num_threads,
        "xnnpack": use_xnnpack,
        "runs": runs,
        "latency_ms_mean": round(statistics.mean(latencies), 2),
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2
        ),
        "latency_ms_min": round(min(latencies), 2),
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    import numpy as np

    tensor = np.expand_dims(preprocess_image(args.image), axis=0)
    thread_counts = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
    use_xnnpack = not args.no_xnnpack

    results = []
    print(f"model={args.model.name} xnnpack={use_xnnpack} image={args.image.name}")
    print(f"{'threads':>8}  {'mean_ms':>8}  {'p50_ms':>8}  {'p95_ms':>8}  {'min_ms':>8}")
    print("-" * 48)

    for threads in thread_counts:
        row = benchmark(
            args.model,
            tensor,
            num_threads=threads,
            use_xnnpack=use_xnnpack,
            warmup=args.warmup,
            runs=args.runs,
        )
        results.append(row)
        print(
            f"{row['threads']:>8}  "
            f"{row['latency_ms_mean']:>8.2f}  "
            f"{row['latency_ms_p50']:>8.2f}  "
            f"{row['latency_ms_p95']:>8.2f}  "
            f"{row['latency_ms_min']:>8.2f}"
        )

    payload = {
        "model": str(args.model),
        "image": str(args.image),
        "xnnpack": use_xnnpack,
        "results": results,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
