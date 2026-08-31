#!/usr/bin/env python3
"""Stream stereo vision telemetry from the vision Pi to the observability hub.

Covers stereo frame capture, depth estimation, and 3D obstacle metrics.
Replace mock readers with picamera2 stereo capture and depth pipeline on the
wheelchair vision Pi.

Example:
  python3 edge/wheelchair/pi_vision_stream.py \\
    --hub http://192.168.1.100:8080 \\
    --device-id pi-vision-01
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from edge.wheelchair.telemetry_client import TelemetryStreamClient  # noqa: E402


def read_stereo_frame() -> dict[str, str | int | bool]:
    """Mock stereo capture — replace with picamera2 dual-camera pipeline."""
    return {
        "left_frame_id": str(uuid.uuid4()),
        "right_frame_id": str(uuid.uuid4()),
        "resolution": "640x480",
        "fps": 30,
        "captured": True,
    }


def read_depth_metrics() -> dict[str, float | bool]:
    """Mock stereo depth — replace with OpenCV stereo matching or neural depth."""
    mean_depth = random.uniform(0.5, 3.0)
    return {
        "min_depth_m": max(0.1, mean_depth - random.uniform(0.2, 0.8)),
        "max_depth_m": mean_depth + random.uniform(0.5, 2.0),
        "mean_depth_m": mean_depth,
        "obstacle_detected": mean_depth < 0.8,
        "confidence": random.uniform(0.7, 0.99),
    }


def read_recognition_scores() -> dict[str, float]:
    """Mock 3D object recognition scores — replace with on-Pi model inference."""
    return {
        "person": random.uniform(0.0, 0.95),
        "chair": random.uniform(0.0, 0.9),
        "doorway": random.uniform(0.0, 0.85),
        "obstacle": random.uniform(0.0, 0.8),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", required=True)
    parser.add_argument("--device-id", default="pi-vision-01")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--interval-ms", type=int, default=333)
    parser.add_argument("--duration", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TelemetryStreamClient(args.hub)
    client.open_session(
        device_id=args.device_id,
        pi_role="vision",
        batch_size=args.batch_size,
    )
    print(f"Streaming vision telemetry as {args.device_id}", file=sys.stderr)

    started = time.time()
    try:
        while True:
            client.push_event("stereo_frame", read_stereo_frame())
            client.push_event("depth", read_depth_metrics())
            client.push_event("recognition", read_recognition_scores())
            if args.duration > 0 and (time.time() - started) >= args.duration:
                break
            time.sleep(args.interval_ms / 1000)
    finally:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
        client.close()
        print("Vision stream closed", file=sys.stderr)


if __name__ == "__main__":
    main()
