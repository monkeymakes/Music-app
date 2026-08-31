"""In-memory stream sessions that accumulate frames and flush batched inference."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from server.inference.engine import BatchInferenceItem, infer_images_batch


@dataclass
class StreamFrame:
    frame_id: str
    image_bytes: bytes
    client_index: int | None = None
    received_at_ms: float = field(default_factory=lambda: time.time() * 1000)


@dataclass
class StreamFlushResult:
    batch_id: str
    session_id: str
    flushed_count: int
    pending_count: int
    total_latency_ms: float
    engine_mode: str
    results: list[dict[str, Any]]


@dataclass
class _StreamSession:
    session_id: str
    model: str
    channel: str
    version: str | None
    top_k: int
    batch_size: int
    max_wait_ms: int
    idle_ttl_ms: int
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    last_activity_ms: float = field(default_factory=lambda: time.time() * 1000)
    pending: list[StreamFrame] = field(default_factory=list)
    total_frames_received: int = 0
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
        oldest = min(frame.received_at_ms for frame in self.pending)
        return (time.time() * 1000 - oldest) >= self.max_wait_ms


class StreamSessionManager:
    """Thread-safe store for camera/stream frame batching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _StreamSession] = {}

    def create_session(
        self,
        *,
        model: str,
        channel: str,
        version: str | None,
        top_k: int,
        batch_size: int,
        max_wait_ms: int,
        idle_ttl_ms: int,
    ) -> _StreamSession:
        session = _StreamSession(
            session_id=str(uuid.uuid4()),
            model=model,
            channel=channel,
            version=version,
            top_k=top_k,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            idle_ttl_ms=idle_ttl_ms,
        )
        with self._lock:
            self._purge_expired_locked()
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> _StreamSession | None:
        with self._lock:
            self._purge_expired_locked()
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def add_frame(
        self,
        session_id: str,
        *,
        image_bytes: bytes,
        client_index: int | None = None,
    ) -> tuple[_StreamSession, StreamFrame, bool]:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown stream session '{session_id}'")

            session.touch()
            frame = StreamFrame(
                frame_id=str(uuid.uuid4()),
                image_bytes=image_bytes,
                client_index=client_index,
            )
            session.pending.append(frame)
            session.total_frames_received += 1
            should_flush = session.should_auto_flush()
            return session, frame, should_flush

    def flush_session(
        self,
        session_id: str,
        *,
        labels: list[str],
        input_size: tuple[int, int],
        artifact_path: Any,
        allow_mock: bool,
        force: bool = False,
    ) -> StreamFlushResult | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown stream session '{session_id}'")

            if not session.pending:
                return None
            if not force and not session.should_auto_flush():
                return None

            frames = session.pending
            session.pending = []
            session.touch()

        batch = infer_images_batch(
            image_bytes_list=[frame.image_bytes for frame in frames],
            labels=labels,
            input_size=input_size,
            artifact_path=artifact_path,
            allow_mock=allow_mock,
            top_k_n=session.top_k,
        )

        with self._lock:
            active = self._sessions.get(session_id)
            pending_count = len(active.pending) if active is not None else 0
            if active is not None:
                active.total_batches_flushed += 1

        results: list[dict[str, Any]] = []
        for frame, item in zip(frames, batch.results, strict=True):
            results.append(_frame_result_dict(frame, item))

        return StreamFlushResult(
            batch_id=str(uuid.uuid4()),
            session_id=session_id,
            flushed_count=len(results),
            pending_count=pending_count,
            total_latency_ms=batch.total_latency_ms,
            engine_mode=batch.engine_mode,
            results=results,
        )

    def session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "session_id": session.session_id,
                "model": session.model,
                "channel": session.channel,
                "version": session.version,
                "top_k": session.top_k,
                "batch_size": session.batch_size,
                "max_wait_ms": session.max_wait_ms,
                "idle_ttl_ms": session.idle_ttl_ms,
                "pending_frames": len(session.pending),
                "total_frames_received": session.total_frames_received,
                "total_batches_flushed": session.total_batches_flushed,
                "should_auto_flush": session.should_auto_flush(),
                "created_at_ms": session.created_at_ms,
                "last_activity_ms": session.last_activity_ms,
            }

    def _purge_expired_locked(self) -> None:
        now = time.time() * 1000
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.is_expired(now)
        ]
        for session_id in expired:
            del self._sessions[session_id]


def _frame_result_dict(frame: StreamFrame, item: BatchInferenceItem) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "client_index": frame.client_index,
        "index": item.index,
        "latency_ms": item.latency_ms,
        "labels": [label.as_dict() for label in item.labels],
    }
