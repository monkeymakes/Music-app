"""In-memory telemetry stream sessions that micro-batch events before TensorBoard flush."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TelemetryEvent:
    event_id: str
    device_id: str
    pi_role: str
    metric: str
    value: float | int | str | dict[str, Any]
    timestamp_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at_ms: float = field(default_factory=lambda: time.time() * 1000)


@dataclass
class TelemetryFlushResult:
    batch_id: str
    session_id: str
    flushed_count: int
    pending_count: int
    events_written: int
    events: list[dict[str, Any]]


@dataclass
class _TelemetrySession:
    session_id: str
    device_id: str
    pi_role: str
    batch_size: int
    max_wait_ms: int
    idle_ttl_ms: int
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    last_activity_ms: float = field(default_factory=lambda: time.time() * 1000)
    pending: list[TelemetryEvent] = field(default_factory=list)
    total_events_received: int = 0
    total_batches_flushed: int = 0

    def touch(self) -> None:
        self.last_activity_ms = time.time() * 1000

    def is_expired(self, now_ms: float | None = None) -> bool:
        now = now_ms if now_ms is not None else time.time() * 1000
        return (now - self.last_activity_ms) > self.idle_ttl_ms

    def should_auto_flush(self) -> bool:
        if not self.pending:
            return False
        if len(self.pending) >= self.batch_size:
            return True
        oldest = min(event.received_at_ms for event in self.pending)
        return (time.time() * 1000 - oldest) >= self.max_wait_ms


class TelemetryStreamManager:
    """Thread-safe store for multi-Pi telemetry micro-batching."""

    def __init__(self, on_flush: Callable[[list[dict[str, Any]]], int] | None = None) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _TelemetrySession] = {}
        self._on_flush = on_flush

    def create_session(
        self,
        *,
        device_id: str,
        pi_role: str,
        batch_size: int,
        max_wait_ms: int,
        idle_ttl_ms: int,
    ) -> _TelemetrySession:
        session = _TelemetrySession(
            session_id=str(uuid.uuid4()),
            device_id=device_id,
            pi_role=pi_role,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            idle_ttl_ms=idle_ttl_ms,
        )
        with self._lock:
            self._purge_expired_locked()
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> _TelemetrySession | None:
        with self._lock:
            self._purge_expired_locked()
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def add_event(
        self,
        session_id: str,
        *,
        metric: str,
        value: float | int | str | dict[str, Any],
        timestamp_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[_TelemetrySession, TelemetryEvent, bool]:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown telemetry session '{session_id}'")

            session.touch()
            event = TelemetryEvent(
                event_id=str(uuid.uuid4()),
                device_id=session.device_id,
                pi_role=session.pi_role,
                metric=metric,
                value=value,
                timestamp_ms=timestamp_ms if timestamp_ms is not None else time.time() * 1000,
                metadata=metadata or {},
            )
            session.pending.append(event)
            session.total_events_received += 1
            should_flush = session.should_auto_flush()
            return session, event, should_flush

    def flush_session(self, session_id: str, *, force: bool = False) -> TelemetryFlushResult | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown telemetry session '{session_id}'")

            if not session.pending:
                return None
            if not force and not session.should_auto_flush():
                return None

            events = session.pending
            session.pending = []
            session.touch()
            session.total_batches_flushed += 1
            pending_count = len(session.pending)

        event_dicts = [_event_to_dict(event) for event in events]
        events_written = 0
        if self._on_flush is not None:
            events_written = self._on_flush(event_dicts)

        return TelemetryFlushResult(
            batch_id=str(uuid.uuid4()),
            session_id=session_id,
            flushed_count=len(events),
            pending_count=pending_count,
            events_written=events_written,
            events=event_dicts,
        )

    def session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "session_id": session.session_id,
                "device_id": session.device_id,
                "pi_role": session.pi_role,
                "batch_size": session.batch_size,
                "max_wait_ms": session.max_wait_ms,
                "idle_ttl_ms": session.idle_ttl_ms,
                "pending_events": len(session.pending),
                "total_events_received": session.total_events_received,
                "total_batches_flushed": session.total_batches_flushed,
                "should_auto_flush": session.should_auto_flush(),
                "created_at_ms": session.created_at_ms,
                "last_activity_ms": session.last_activity_ms,
            }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            self._purge_expired_locked()
            return [self.session_status(sid) for sid in self._sessions if self.session_status(sid)]

    def _purge_expired_locked(self) -> None:
        now = time.time() * 1000
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.is_expired(now)
        ]
        for session_id in expired:
            del self._sessions[session_id]


def _event_to_dict(event: TelemetryEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "device_id": event.device_id,
        "pi_role": event.pi_role,
        "metric": event.metric,
        "value": event.value,
        "timestamp_ms": event.timestamp_ms,
        "metadata": event.metadata,
    }
