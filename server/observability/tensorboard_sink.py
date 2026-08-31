"""TensorBoard sink for streaming telemetry from multiple wheelchair Pis."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


class TensorBoardSink:
    """Accumulates scalar and text telemetry into TensorBoard event files."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._writer: Any | None = None
        self._step_counters: dict[str, int] = {}
        self._total_events = 0
        self._total_batches = 0

    def _get_writer(self) -> Any:
        if self._writer is None:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]

            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        return self._writer

    def write_batch(self, events: list[dict[str, Any]]) -> int:
        """Write a batch of telemetry events. Returns count written."""
        if not events:
            return 0

        with self._lock:
            writer = self._get_writer()
            written = 0
            for event in events:
                device_id = event.get("device_id", "unknown")
                pi_role = event.get("pi_role", "unknown")
                metric = event.get("metric", "value")
                value = event.get("value")
                step_key = f"{device_id}/{metric}"
                step = self._step_counters.get(step_key, 0)
                tag = f"pi/{pi_role}/{device_id}/{metric}"

                if isinstance(value, (int, float)):
                    writer.add_scalar(tag, float(value), step)
                    self._step_counters[step_key] = step + 1
                    written += 1
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            sub_tag = f"{tag}/{sub_key}"
                            sub_step_key = f"{step_key}/{sub_key}"
                            sub_step = self._step_counters.get(sub_step_key, 0)
                            writer.add_scalar(sub_tag, float(sub_val), sub_step)
                            self._step_counters[sub_step_key] = sub_step + 1
                            written += 1
                elif isinstance(value, str):
                    writer.add_text(f"pi/{pi_role}/{device_id}/text/{metric}", value, step)
                    self._step_counters[step_key] = step + 1
                    written += 1

                ts_ms = event.get("timestamp_ms")
                if ts_ms is not None:
                    writer.add_scalar(
                        f"pi/{pi_role}/{device_id}/_ingest_lag_ms",
                        time.time() * 1000 - float(ts_ms),
                        step,
                    )

            writer.flush()
            self._total_events += written
            self._total_batches += 1
            return written

    def status(self) -> dict[str, Any]:
        return {
            "log_dir": str(self.log_dir),
            "total_events_written": self._total_events,
            "total_batches_flushed": self._total_batches,
            "active_metric_streams": len(self._step_counters),
        }

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
