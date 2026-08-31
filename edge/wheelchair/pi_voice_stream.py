#!/usr/bin/env python3
"""Stream voice/audio telemetry from the voice Pi to the observability hub.

Covers Wyoming and Aloha wake-word engines, microphone RMS levels, and
speech-activity detection. Replace mock readers with Wyoming protocol client
and PortAudio capture on the wheelchair voice Pi.

Example:
  python3 edge/wheelchair/pi_voice_stream.py \\
    --hub http://192.168.1.100:8080 \\
    --device-id pi-voice-01 \\
    --engine wyoming
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from edge.wheelchair.telemetry_client import TelemetryStreamClient  # noqa: E402


def read_audio_level() -> dict[str, float | int | bool]:
    """Mock mic analysis — replace with PortAudio + VAD on the Pi."""
    return {
        "rms_db": random.uniform(-45.0, -20.0),
        "speech_active": random.random() > 0.85,
        "sample_rate_hz": 16000,
    }


def read_wake_word(engine: str) -> dict[str, float | str | bool | None]:
    """Mock wake-word state — replace with Wyoming/Aloha client."""
    detected = random.random() > 0.95
    return {
        "engine": engine,
        "detected": detected,
        "confidence": random.uniform(0.0, 0.99) if detected else 0.0,
        "last_wake_word": "hey wheelchair" if detected else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", required=True)
    parser.add_argument("--device-id", default="pi-voice-01")
    parser.add_argument("--engine", choices=["wyoming", "aloha"], default="wyoming")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--duration", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TelemetryStreamClient(args.hub)
    client.open_session(
        device_id=args.device_id,
        pi_role="voice",
        batch_size=args.batch_size,
    )
    print(f"Streaming voice telemetry ({args.engine}) as {args.device_id}", file=sys.stderr)

    started = time.time()
    try:
        while True:
            client.push_event("audio_level", read_audio_level())
            client.push_event(
                f"wake_word_{args.engine}",
                read_wake_word(args.engine),
                metadata={"engine": args.engine},
            )
            if args.duration > 0 and (time.time() - started) >= args.duration:
                break
            time.sleep(args.interval_ms / 1000)
    finally:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
        client.close()
        print("Voice stream closed", file=sys.stderr)


if __name__ == "__main__":
    main()
