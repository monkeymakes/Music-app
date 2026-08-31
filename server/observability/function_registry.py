"""Cross-Pi function calling registry for wheelchair observability MVP."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FunctionSpec:
    name: str
    pi_role: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class FunctionCallResult:
    call_id: str
    function_name: str
    target_device_id: str | None
    pi_role: str
    latency_ms: float
    result: dict[str, Any]
    error: str | None = None


class FunctionRegistry:
    """Registry of callable functions across wheelchair Pis."""

    def __init__(self) -> None:
        self._functions: dict[str, FunctionSpec] = {}
        self._register_defaults()

    def register(
        self,
        name: str,
        *,
        pi_role: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._functions[name] = FunctionSpec(
            name=name,
            pi_role=pi_role,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def list_functions(self, pi_role: str | None = None) -> list[dict[str, Any]]:
        specs = self._functions.values()
        if pi_role is not None:
            specs = [s for s in specs if s.pi_role == pi_role]
        return [
            {
                "name": spec.name,
                "pi_role": spec.pi_role,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in specs
        ]

    def call(
        self,
        function_name: str,
        *,
        target_device_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> FunctionCallResult:
        spec = self._functions.get(function_name)
        if spec is None:
            return FunctionCallResult(
                call_id=str(uuid.uuid4()),
                function_name=function_name,
                target_device_id=target_device_id,
                pi_role="unknown",
                latency_ms=0.0,
                result={},
                error=f"Unknown function '{function_name}'",
            )

        started = time.perf_counter()
        try:
            payload = {"target_device_id": target_device_id, **(arguments or {})}
            result = spec.handler(payload)
            latency_ms = (time.perf_counter() - started) * 1000
            return FunctionCallResult(
                call_id=str(uuid.uuid4()),
                function_name=function_name,
                target_device_id=target_device_id,
                pi_role=spec.pi_role,
                latency_ms=latency_ms,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 — surface to API caller
            latency_ms = (time.perf_counter() - started) * 1000
            return FunctionCallResult(
                call_id=str(uuid.uuid4()),
                function_name=function_name,
                target_device_id=target_device_id,
                pi_role=spec.pi_role,
                latency_ms=latency_ms,
                result={},
                error=str(exc),
            )

    def _register_defaults(self) -> None:
        """MVP function stubs for the three wheelchair Pi roles."""

        self.register(
            "sensors.read_distance",
            pi_role="sensors",
            description="Read ultrasonic/IR distance sensors on the sensor Pi",
            parameters={
                "type": "object",
                "properties": {
                    "sensor_id": {"type": "string", "description": "front|left|right|rear"},
                },
            },
            handler=_mock_sensors_read_distance,
        )
        self.register(
            "sensors.read_imu",
            pi_role="sensors",
            description="Read IMU orientation and acceleration from sensor Pi",
            parameters={"type": "object", "properties": {}},
            handler=_mock_sensors_read_imu,
        )
        self.register(
            "voice.get_wake_word",
            pi_role="voice",
            description="Check Wyoming/Aloha wake-word detection state on voice Pi",
            parameters={
                "type": "object",
                "properties": {
                    "engine": {"type": "string", "enum": ["wyoming", "aloha"]},
                },
            },
            handler=_mock_voice_get_wake_word,
        )
        self.register(
            "voice.analyze_audio_level",
            pi_role="voice",
            description="Analyze microphone RMS level and speech activity on voice Pi",
            parameters={"type": "object", "properties": {}},
            handler=_mock_voice_analyze_audio,
        )
        self.register(
            "vision.capture_stereo_frame",
            pi_role="vision",
            description="Capture stereo camera frame pair for 3D recognition on vision Pi",
            parameters={
                "type": "object",
                "properties": {
                    "resolution": {"type": "string", "default": "640x480"},
                },
            },
            handler=_mock_vision_capture_stereo,
        )
        self.register(
            "vision.estimate_depth",
            pi_role="vision",
            description="Run stereo depth estimation on vision Pi",
            parameters={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "center|full"},
                },
            },
            handler=_mock_vision_estimate_depth,
        )


def _mock_sensors_read_distance(args: dict[str, Any]) -> dict[str, Any]:
    sensor_id = args.get("sensor_id", "front")
    distances = {"front": 1.2, "left": 0.8, "right": 1.5, "rear": 2.1}
    return {
        "sensor_id": sensor_id,
        "distance_m": distances.get(sensor_id, 1.0),
        "unit": "meters",
        "device_id": args.get("target_device_id", "pi-sensors-01"),
    }


def _mock_sensors_read_imu(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "pitch_deg": 2.1,
        "roll_deg": -0.5,
        "yaw_deg": 45.0,
        "accel_x": 0.02,
        "accel_y": 0.01,
        "accel_z": 9.81,
    }


def _mock_voice_get_wake_word(args: dict[str, Any]) -> dict[str, Any]:
    engine = args.get("engine", "wyoming")
    return {
        "engine": engine,
        "detected": False,
        "last_wake_word": None,
        "confidence": 0.0,
    }


def _mock_voice_analyze_audio(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "rms_db": -32.5,
        "speech_active": False,
        "sample_rate_hz": 16000,
        "channels": 1,
    }


def _mock_vision_capture_stereo(args: dict[str, Any]) -> dict[str, Any]:
    resolution = args.get("resolution", "640x480")
    return {
        "left_frame_id": str(uuid.uuid4()),
        "right_frame_id": str(uuid.uuid4()),
        "resolution": resolution,
        "format": "jpeg",
        "captured": True,
    }


def _mock_vision_estimate_depth(args: dict[str, Any]) -> dict[str, Any]:
    region = args.get("region", "center")
    return {
        "region": region,
        "min_depth_m": 0.3,
        "max_depth_m": 4.5,
        "mean_depth_m": 1.8,
        "obstacle_detected": False,
    }
