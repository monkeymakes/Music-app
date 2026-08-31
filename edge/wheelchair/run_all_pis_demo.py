#!/usr/bin/env python3
"""Run all three wheelchair Pi telemetry streams against the observability hub.

Useful for local MVP testing when you don't have three physical Pis yet.

Example:
  # Terminal 1: start the hub
  PYTHONPATH=. uvicorn server.inference.app:app --host 0.0.0.0 --port 8080

  # Terminal 2: simulate all three Pis
  python3 edge/wheelchair/run_all_pis_demo.py --hub http://127.0.0.1:8080 --duration 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="http://127.0.0.1:8080")
    parser.add_argument("--duration", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = [
        ("sensors", ROOT / "edge/wheelchair/pi_sensors_stream.py", ["--device-id", "pi-sensors-01"]),
        ("voice", ROOT / "edge/wheelchair/pi_voice_stream.py", ["--device-id", "pi-voice-01"]),
        ("vision", ROOT / "edge/wheelchair/pi_vision_stream.py", ["--device-id", "pi-vision-01"]),
    ]

    procs: list[subprocess.Popen] = []
    for name, script, extra in scripts:
        cmd = [
            sys.executable,
            str(script),
            "--hub",
            args.hub,
            "--duration",
            str(args.duration),
            *extra,
        ]
        print(f"Starting {name} stream: {' '.join(cmd)}", file=sys.stderr)
        procs.append(subprocess.Popen(cmd))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"One or more streams failed: {exit_codes}")
    print("All Pi streams completed", file=sys.stderr)


if __name__ == "__main__":
    main()
