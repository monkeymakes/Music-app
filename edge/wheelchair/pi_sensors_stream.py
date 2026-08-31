#!/usr/bin/env python3
"""Stream distance and IMU telemetry from the sensor Pi to the observability hub.

Simulates distant ultrasonic/IR sensors for MVP. Replace the mock readers with
real GPIO/I2C drivers on the wheelchair sensor Pi.

Example:
  python3 edge/wheelchair/pi_sensors_stream.py \\
    --hub http://192.168.1.100:8080 \\
    --device-id pi-sensors-01 \\
    --duration 30
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from edge.wheelchair.telemetry_client import TelemetryStreamClient  # noqa: E402


def read_distance_m(sensor_id: str) -> float:
    """Mock distance reader — replace with HC-SR04 or IR sensor driver."""
    base = {"front": 1.2, "left": 0.9, "right": 1.4, "rear": 2.0}[sensor_id]
    return max(0.1, base + random.uniform(-0.15, 0.15))


def read_imu() -> dict[str, float]:
    """Mock IMU reader — replace with MPU6050/BNO055 driver."""
    t = time.time()
    return {
        "pitch_deg": 2.0 * math.sin(t * 0.5),
        "roll_deg": 1.0 * math.cos(t * 0.3),
        "yaw_deg": (t * 10) % 360,
        "accel_z": 9.81 + random.uniform(-0.05, 0.05),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", required=True, help="Observability hub URL")
    parser.add_argument("--device-id", default="pi-sensors-01")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--duration", type=int, default=0, help="Seconds to run (0=forever)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TelemetryStreamClient(args.hub)
    client.open_session(
        device_id=args.device_id,
        pi_role="sensors",
        batch_size=args.batch_size,
    )
    print(f"Streaming sensor telemetry as {args.device_id}", file=sys.stderr)

    started = time.time()
    try:
        while True:
            for sensor_id in ("front", "left", "right", "rear"):
                client.push_event(
                    f"distance_{sensor_id}_m",
                    read_distance_m(sensor_id),
                    metadata={"sensor_id": sensor_id},
                )
            client.push_event("imu", read_imu())
            if args.duration > 0 and (time.time() - started) >= args.duration:
                break
            time.sleep(args.interval_ms / 1000)
    finally:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
        client.close()
        print("Sensor stream closed", file=sys.stderr)


if __name__ == "__main__":
    main()
