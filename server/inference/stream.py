"""In-memory stream sessions that accumulate frames and flush batched inference.

Supports FIFO batching (default) and weighted reservoir sampling (A-Res) so
high-frequency camera streams can keep a fixed-size, weight-biased sample
before inference instead of flushing every frame in arrival order.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from server.inference.engine import BatchInferenceItem, infer_images_batch
from server.inference.reservoir import WeightedReservoir

SamplingMode = Literal["fifo", "weighted_reservoir"]

VALID_SAMPLING_MODES: frozenset[str] = frozenset({"fifo", "weighted_reservoir"})


@dataclass
class StreamFrame:
    frame_id: str
    image_bytes: bytes
    client_index: int | None = None
    weight: float = 1.0
    received_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    seq: int = 0
    selected: bool = True


@dataclass
class StreamFlushResult:
    batch_id: str
    session_id: str
    flushed_count: int
    pending_count: int
    total_latency_ms: float
    engine_mode: str
    results: list[dict[str, Any]]
    sampling_mode: str
    frames_considered: int
    reservoir_stats: dict[str, int] | None = None


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
    sampling_mode: SamplingMode = "fifo"
    sample_window: int = 0
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    last_activity_ms: float = field(default_factory=lambda: time.time() * 1000)
    pending: list[StreamFrame] = field(default_factory=list)
    reservoir: WeightedReservoir[StreamFrame] | None = None
    total_frames_received: int = 0
    total_batches_flushed: int = 0
    frames_since_flush: int = 0
    window_started_at_ms: float | None = None
    _next_seq: int = 0

    def touch(self) -> None:
        self.last_activity_ms = time.time() * 1000

    def is_expired(self, now_ms: float | None = None) -> bool:
        now = now_ms if now_ms is not None else time.time() * 1000
        return (now - self.last_activity_ms) > self.idle_ttl_ms

    def effective_sample_window(self) -> int:
        if self.sample_window > 0:
            return self.sample_window
        # Default: observe 4× capacity before forcing a weighted flush.
        return max(self.batch_size * 4, self.batch_size)

    def pending_count(self) -> int:
        if self.sampling_mode == "weighted_reservoir" and self.reservoir is not None:
            return len(self.reservoir)
        return len(self.pending)

    def should_auto_flush(self) -> bool:
        if self.sampling_mode == "weighted_reservoir":
            return self._should_auto_flush_reservoir()
        return self._should_auto_flush_fifo()

    def _should_auto_flush_fifo(self) -> bool:
        if not self.pending:
            return False
        if len(self.pending) >= self.batch_size:
            return True
        oldest = min(frame.received_at_ms for frame in self.pending)
        return (time.time() * 1000 - oldest) >= self.max_wait_ms

    def _should_auto_flush_reservoir(self) -> bool:
        if self.reservoir is None or len(self.reservoir) == 0:
            return False
        if self.frames_since_flush >= self.effective_sample_window():
            return True
        if self.window_started_at_ms is None:
            return False
        return (time.time() * 1000 - self.window_started_at_ms) >= self.max_wait_ms

    def take_pending_frames(self) -> tuple[list[StreamFrame], int, dict[str, int] | None]:
        """Drain the current sample window. Returns frames, considered count, stats."""
        if self.sampling_mode == "weighted_reservoir" and self.reservoir is not None:
            frames = self.reservoir.items()
            considered = self.frames_since_flush
            stats = self.reservoir.stats()
            self.reservoir.clear()
            self.frames_since_flush = 0
            self.window_started_at_ms = None
            return frames, considered, stats

        frames = self.pending
        considered = len(frames)
        self.pending = []
        self.frames_since_flush = 0
        self.window_started_at_ms = None
        return frames, considered, None


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
        sampling_mode: SamplingMode = "fifo",
        sample_window: int = 0,
    ) -> _StreamSession:
        if sampling_mode not in VALID_SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {sorted(VALID_SAMPLING_MODES)}"
            )
        if sample_window < 0:
            raise ValueError("sample_window must be >= 0")

        session = _StreamSession(
            session_id=str(uuid.uuid4()),
            model=model,
            channel=channel,
            version=version,
            top_k=top_k,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            idle_ttl_ms=idle_ttl_ms,
            sampling_mode=sampling_mode,
            sample_window=sample_window,
            reservoir=(
                WeightedReservoir(batch_size)
                if sampling_mode == "weighted_reservoir"
                else None
            ),
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
        sample_weight: float = 1.0,
    ) -> tuple[_StreamSession, StreamFrame, bool]:
        if sample_weight <= 0:
            raise ValueError("sample_weight must be > 0")

        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown stream session '{session_id}'")

            session.touch()
            session._next_seq += 1
            if session.window_started_at_ms is None:
                session.window_started_at_ms = time.time() * 1000

            frame = StreamFrame(
                frame_id=str(uuid.uuid4()),
                image_bytes=image_bytes,
                client_index=client_index,
                weight=sample_weight,
                seq=session._next_seq,
            )
            session.total_frames_received += 1
            session.frames_since_flush += 1

            if session.sampling_mode == "weighted_reservoir":
                assert session.reservoir is not None
                selected = session.reservoir.offer(frame, weight=sample_weight)
                frame.selected = selected
            else:
                session.pending.append(frame)
                frame.selected = True

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

            if session.pending_count() == 0:
                return None
            if not force and not session.should_auto_flush():
                return None

            frames, considered, reservoir_stats = session.take_pending_frames()
            sampling_mode = session.sampling_mode
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
            pending_count = active.pending_count() if active is not None else 0
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
            sampling_mode=sampling_mode,
            frames_considered=considered,
            reservoir_stats=reservoir_stats,
        )

    def session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            status: dict[str, Any] = {
                "session_id": session.session_id,
                "model": session.model,
                "channel": session.channel,
                "version": session.version,
                "top_k": session.top_k,
                "batch_size": session.batch_size,
                "max_wait_ms": session.max_wait_ms,
                "idle_ttl_ms": session.idle_ttl_ms,
                "sampling_mode": session.sampling_mode,
                "sample_window": session.effective_sample_window(),
                "pending_frames": session.pending_count(),
                "total_frames_received": session.total_frames_received,
                "total_batches_flushed": session.total_batches_flushed,
                "frames_since_flush": session.frames_since_flush,
                "should_auto_flush": session.should_auto_flush(),
                "created_at_ms": session.created_at_ms,
                "last_activity_ms": session.last_activity_ms,
            }
            if session.reservoir is not None:
                status["reservoir"] = session.reservoir.stats()
            return status

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
        "sample_weight": frame.weight,
        "labels": [label.as_dict() for label in item.labels],
    }
