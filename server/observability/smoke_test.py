#!/usr/bin/env python3
"""Smoke tests for multi-Pi streaming observability."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_tensorboard_sink() -> None:
    from server.observability.tensorboard_sink import TensorBoardSink

    with tempfile.TemporaryDirectory() as tmp:
        sink = TensorBoardSink(log_dir=Path(tmp))
        written = sink.write_batch(
            [
                {
                    "device_id": "pi-sensors-01",
                    "pi_role": "sensors",
                    "metric": "distance_front_m",
                    "value": 1.2,
                    "timestamp_ms": 1000.0,
                },
                {
                    "device_id": "pi-voice-01",
                    "pi_role": "voice",
                    "metric": "audio_level",
                    "value": {"rms_db": -32.5, "speech_active": 0},
                    "timestamp_ms": 1000.0,
                },
            ]
        )
        assert written >= 2
        status = sink.status()
        assert status["total_batches_flushed"] == 1
        sink.close()


def test_telemetry_stream_manager() -> None:
    from server.observability.telemetry_stream import TelemetryStreamManager

    flushed_batches: list[list[dict]] = []

    def on_flush(events: list[dict]) -> int:
        flushed_batches.append(events)
        return len(events)

    mgr = TelemetryStreamManager(on_flush=on_flush)
    session = mgr.create_session(
        device_id="pi-vision-01",
        pi_role="vision",
        batch_size=2,
        max_wait_ms=500,
        idle_ttl_ms=60000,
    )

    _s, e1, should1 = mgr.add_event(session.session_id, metric="depth", value=1.5)
    assert should1 is False
    assert e1.metric == "depth"

    _s, e2, should2 = mgr.add_event(session.session_id, metric="depth", value=1.6)
    assert should2 is True

    result = mgr.flush_session(session.session_id, force=True)
    assert result is not None
    assert result.flushed_count == 2
    assert len(flushed_batches) == 1


def test_function_registry() -> None:
    from server.observability.function_registry import FunctionRegistry

    reg = FunctionRegistry()
    functions = reg.list_functions()
    names = {f["name"] for f in functions}
    assert "sensors.read_distance" in names
    assert "voice.get_wake_word" in names
    assert "vision.estimate_depth" in names

    result = reg.call(
        "sensors.read_distance",
        target_device_id="pi-sensors-01",
        arguments={"sensor_id": "front"},
    )
    assert result.error is None
    assert result.result["distance_m"] == 1.2

    missing = reg.call("nonexistent.function")
    assert missing.error is not None


def test_fastapi_observability_routes() -> None:
    from fastapi.testclient import TestClient

    from server.inference.app import app

    client = TestClient(app)

    health = client.get("/v1/observability/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["registered_functions"] >= 6

    session = client.post(
        "/v1/observability/stream/sessions",
        json={
            "device_id": "pi-sensors-01",
            "pi_role": "sensors",
            "batch_size": 2,
            "max_wait_ms": 500,
        },
    )
    assert session.status_code == 200
    session_id = session.json()["session_id"]

    event1 = client.post(
        f"/v1/observability/stream/sessions/{session_id}/events",
        json={"metric": "distance_front_m", "value": 1.1},
    )
    assert event1.status_code == 200
    assert event1.json()["auto_flushed"] is False

    event2 = client.post(
        f"/v1/observability/stream/sessions/{session_id}/events",
        json={"metric": "distance_front_m", "value": 1.2},
    )
    assert event2.status_code == 200
    assert event2.json()["auto_flushed"] is True
    assert event2.json()["flush"]["flushed_count"] == 2

    functions = client.get("/v1/observability/functions")
    assert functions.status_code == 200
    assert len(functions.json()["functions"]) >= 6

    call = client.post(
        "/v1/observability/functions/call",
        json={
            "function_name": "vision.estimate_depth",
            "target_device_id": "pi-vision-01",
            "arguments": {"region": "center"},
        },
    )
    assert call.status_code == 200
    assert call.json()["pi_role"] == "vision"
    assert "mean_depth_m" in call.json()["result"]

    deleted = client.delete(f"/v1/observability/stream/sessions/{session_id}")
    assert deleted.status_code == 200


def main() -> None:
    test_tensorboard_sink()
    print("ok tensorboard sink")
    test_telemetry_stream_manager()
    print("ok telemetry stream manager")
    test_function_registry()
    print("ok function registry")
    test_fastapi_observability_routes()
    print("ok fastapi observability routes")
    print("all observability smoke tests passed")


if __name__ == "__main__":
    main()
