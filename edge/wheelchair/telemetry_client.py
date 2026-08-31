#!/usr/bin/env python3
"""Shared telemetry streaming client for wheelchair Raspberry Pis.

Each Pi opens a telemetry stream session on the observability hub, pushes
metrics as they arrive, and relies on server-side micro-batching to flush
batches into TensorBoard.

Example:
  python3 edge/wheelchair/telemetry_client.py open \\
    --hub http://127.0.0.1:8080 \\
    --device-id pi-sensors-01 \\
    --pi-role sensors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import httpx
except ImportError as exc:
    raise SystemExit("Install httpx: pip3 install httpx") from exc


class TelemetryStreamClient:
    """HTTP client for /v1/observability stream endpoints."""

    def __init__(self, hub_url: str, timeout_s: float = 10.0) -> None:
        self.hub_url = hub_url.rstrip("/")
        self._client = httpx.Client(base_url=self.hub_url, timeout=timeout_s)
        self.session_id: str | None = None

    def open_session(
        self,
        *,
        device_id: str,
        pi_role: str,
        batch_size: int = 16,
        max_wait_ms: int = 500,
        idle_ttl_ms: int = 120000,
    ) -> str:
        response = self._client.post(
            "/v1/observability/stream/sessions",
            json={
                "device_id": device_id,
                "pi_role": pi_role,
                "batch_size": batch_size,
                "max_wait_ms": max_wait_ms,
                "idle_ttl_ms": idle_ttl_ms,
            },
        )
        response.raise_for_status()
        self.session_id = response.json()["session_id"]
        return self.session_id

    def push_event(
        self,
        metric: str,
        value: float | int | str | dict[str, Any],
        *,
        timestamp_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.session_id is None:
            raise RuntimeError("Call open_session() first")

        payload: dict[str, Any] = {"metric": metric, "value": value}
        if timestamp_ms is not None:
            payload["timestamp_ms"] = timestamp_ms
        if metadata:
            payload["metadata"] = metadata

        response = self._client.post(
            f"/v1/observability/stream/sessions/{self.session_id}/events",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def flush(self) -> dict[str, Any]:
        if self.session_id is None:
            raise RuntimeError("Call open_session() first")
        response = self._client.post(
            f"/v1/observability/stream/sessions/{self.session_id}/flush"
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        if self.session_id is not None:
            self._client.delete(f"/v1/observability/stream/sessions/{self.session_id}")
            self.session_id = None
        self._client.close()

    def call_function(
        self,
        function_name: str,
        *,
        target_device_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.post(
            "/v1/observability/functions/call",
            json={
                "function_name": function_name,
                "target_device_id": target_device_id,
                "arguments": arguments or {},
            },
        )
        response.raise_for_status()
        return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    open_cmd = sub.add_parser("open", help="Open a telemetry stream session")
    open_cmd.add_argument("--hub", required=True, help="Observability hub URL")
    open_cmd.add_argument("--device-id", required=True)
    open_cmd.add_argument("--pi-role", choices=["sensors", "voice", "vision"], required=True)
    open_cmd.add_argument("--batch-size", type=int, default=16)
    open_cmd.add_argument("--max-wait-ms", type=int, default=500)

    push_cmd = sub.add_parser("push", help="Push a single telemetry event")
    push_cmd.add_argument("--hub", required=True)
    push_cmd.add_argument("--session-id", required=True)
    push_cmd.add_argument("--metric", required=True)
    push_cmd.add_argument("--value", required=True, help="Numeric or JSON object value")
    push_cmd.add_argument("--value-type", choices=["float", "json"], default="float")

    call_cmd = sub.add_parser("call", help="Call a cross-Pi function")
    call_cmd.add_argument("--hub", required=True)
    call_cmd.add_argument("--function", required=True)
    call_cmd.add_argument("--device-id", default=None)
    call_cmd.add_argument("--args", default="{}", help="JSON arguments")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TelemetryStreamClient(args.hub)

    if args.command == "open":
        session_id = client.open_session(
            device_id=args.device_id,
            pi_role=args.pi_role,
            batch_size=args.batch_size,
            max_wait_ms=args.max_wait_ms,
        )
        print(json.dumps({"session_id": session_id}))
        return

    if args.command == "push":
        client.session_id = args.session_id
        if args.value_type == "json":
            value = json.loads(args.value)
        else:
            value = float(args.value)
        result = client.push_event(args.metric, value)
        print(json.dumps(result))
        return

    if args.command == "call":
        result = client.call_function(
            args.function,
            target_device_id=args.device_id,
            arguments=json.loads(args.args),
        )
        print(json.dumps(result))
        return


if __name__ == "__main__":
    main()
