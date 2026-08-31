"""FastAPI routes for multi-Pi streaming observability."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.observability.function_registry import FunctionRegistry
from server.observability.telemetry_stream import TelemetryStreamManager
from server.observability.tensorboard_sink import TensorBoardSink

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TB_LOG_DIR = Path(
    os.environ.get("OBSERVABILITY_TENSORBOARD_LOG_DIR", str(ROOT / "runs" / "wheelchair_observability"))
)
DEFAULT_TELEMETRY_BATCH_SIZE = int(os.environ.get("OBSERVABILITY_TELEMETRY_BATCH_SIZE", "16"))
DEFAULT_TELEMETRY_MAX_WAIT_MS = int(os.environ.get("OBSERVABILITY_TELEMETRY_MAX_WAIT_MS", "500"))
DEFAULT_TELEMETRY_IDLE_TTL_MS = int(os.environ.get("OBSERVABILITY_TELEMETRY_IDLE_TTL_MS", "120000"))

tensorboard_sink = TensorBoardSink(log_dir=DEFAULT_TB_LOG_DIR)
telemetry_streams = TelemetryStreamManager(on_flush=tensorboard_sink.write_batch)
function_registry = FunctionRegistry()

router = APIRouter(prefix="/v1/observability", tags=["observability"])


class ObservabilityHealthResponse(BaseModel):
    status: str
    tensorboard_log_dir: str
    total_events_written: int
    total_batches_flushed: int
    active_sessions: int
    registered_functions: int


class TelemetrySessionCreateRequest(BaseModel):
    device_id: str = Field(description="Unique Pi identifier, e.g. pi-sensors-01")
    pi_role: str = Field(description="sensors | voice | vision")
    batch_size: int = DEFAULT_TELEMETRY_BATCH_SIZE
    max_wait_ms: int = DEFAULT_TELEMETRY_MAX_WAIT_MS
    idle_ttl_ms: int = DEFAULT_TELEMETRY_IDLE_TTL_MS


class TelemetrySessionResponse(BaseModel):
    session_id: str
    device_id: str
    pi_role: str
    batch_size: int
    max_wait_ms: int
    idle_ttl_ms: int
    pending_events: int = 0
    total_events_received: int = 0
    total_batches_flushed: int = 0


class TelemetryEventRequest(BaseModel):
    metric: str
    value: float | int | str | dict
    timestamp_ms: float | None = None
    metadata: dict = Field(default_factory=dict)


class TelemetryEventResponse(BaseModel):
    event_id: str
    session_id: str
    pending_events: int
    auto_flushed: bool
    flush: TelemetryFlushResponse | None = None


class TelemetryFlushResponse(BaseModel):
    batch_id: str
    session_id: str
    flushed_count: int
    pending_events: int
    events_written: int


class FunctionCallRequest(BaseModel):
    function_name: str
    target_device_id: str | None = None
    arguments: dict = Field(default_factory=dict)


class FunctionCallResponse(BaseModel):
    call_id: str
    function_name: str
    target_device_id: str | None
    pi_role: str
    latency_ms: float
    result: dict
    error: str | None = None


@router.get("/health", response_model=ObservabilityHealthResponse)
def observability_health() -> ObservabilityHealthResponse:
    tb_status = tensorboard_sink.status()
    sessions = telemetry_streams.list_sessions()
    return ObservabilityHealthResponse(
        status="ok",
        tensorboard_log_dir=tb_status["log_dir"],
        total_events_written=tb_status["total_events_written"],
        total_batches_flushed=tb_status["total_batches_flushed"],
        active_sessions=len(sessions),
        registered_functions=len(function_registry.list_functions()),
    )


@router.post("/stream/sessions", response_model=TelemetrySessionResponse)
def create_telemetry_session(body: TelemetrySessionCreateRequest) -> TelemetrySessionResponse:
    if body.pi_role not in ("sensors", "voice", "vision"):
        raise HTTPException(status_code=400, detail="pi_role must be sensors, voice, or vision")
    if body.batch_size < 1 or body.batch_size > 256:
        raise HTTPException(status_code=400, detail="batch_size must be 1..256")

    session = telemetry_streams.create_session(
        device_id=body.device_id,
        pi_role=body.pi_role,
        batch_size=body.batch_size,
        max_wait_ms=body.max_wait_ms,
        idle_ttl_ms=body.idle_ttl_ms,
    )
    return TelemetrySessionResponse(
        session_id=session.session_id,
        device_id=session.device_id,
        pi_role=session.pi_role,
        batch_size=session.batch_size,
        max_wait_ms=session.max_wait_ms,
        idle_ttl_ms=session.idle_ttl_ms,
    )


@router.get("/stream/sessions", response_model=list[TelemetrySessionResponse])
def list_telemetry_sessions() -> list[TelemetrySessionResponse]:
    return [TelemetrySessionResponse(**s) for s in telemetry_streams.list_sessions()]


@router.get("/stream/sessions/{session_id}", response_model=TelemetrySessionResponse)
def get_telemetry_session(session_id: str) -> TelemetrySessionResponse:
    status = telemetry_streams.session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown telemetry session '{session_id}'")
    return TelemetrySessionResponse(**status)


@router.post("/stream/sessions/{session_id}/events", response_model=TelemetryEventResponse)
def push_telemetry_event(session_id: str, body: TelemetryEventRequest) -> TelemetryEventResponse:
    try:
        _session, event, should_flush = telemetry_streams.add_event(
            session_id,
            metric=body.metric,
            value=body.value,
            timestamp_ms=body.timestamp_ms,
            metadata=body.metadata,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    flush_response: TelemetryFlushResponse | None = None
    if should_flush:
        try:
            flushed = telemetry_streams.flush_session(session_id, force=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if flushed is not None:
            flush_response = TelemetryFlushResponse(
                batch_id=flushed.batch_id,
                session_id=flushed.session_id,
                flushed_count=flushed.flushed_count,
                pending_events=flushed.pending_count,
                events_written=flushed.events_written,
            )

    status = telemetry_streams.session_status(session_id)
    pending = status["pending_events"] if status else 0
    return TelemetryEventResponse(
        event_id=event.event_id,
        session_id=session_id,
        pending_events=pending,
        auto_flushed=flush_response is not None,
        flush=flush_response,
    )


@router.post("/stream/sessions/{session_id}/flush", response_model=TelemetryFlushResponse)
def flush_telemetry_session(session_id: str) -> TelemetryFlushResponse:
    try:
        flushed = telemetry_streams.flush_session(session_id, force=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if flushed is None:
        raise HTTPException(status_code=409, detail="No pending events to flush")

    return TelemetryFlushResponse(
        batch_id=flushed.batch_id,
        session_id=flushed.session_id,
        flushed_count=flushed.flushed_count,
        pending_events=flushed.pending_count,
        events_written=flushed.events_written,
    )


@router.delete("/stream/sessions/{session_id}")
def delete_telemetry_session(session_id: str) -> dict:
    deleted = telemetry_streams.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown telemetry session '{session_id}'")
    return {"deleted": True, "session_id": session_id}


@router.get("/functions")
def list_functions(pi_role: str | None = None) -> dict:
    return {"functions": function_registry.list_functions(pi_role=pi_role)}


@router.post("/functions/call", response_model=FunctionCallResponse)
def call_function(body: FunctionCallRequest) -> FunctionCallResponse:
    result = function_registry.call(
        body.function_name,
        target_device_id=body.target_device_id,
        arguments=body.arguments,
    )
    if result.error and result.error.startswith("Unknown function"):
        raise HTTPException(status_code=404, detail=result.error)

    return FunctionCallResponse(
        call_id=result.call_id,
        function_name=result.function_name,
        target_device_id=result.target_device_id,
        pi_role=result.pi_role,
        latency_ms=result.latency_ms,
        result=result.result,
        error=result.error,
    )
