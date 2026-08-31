#!/usr/bin/env python3
"""Server-side Music Vision inference with registry-backed model versioning.

Run:
  pip install -r server/inference/requirements.txt
  uvicorn server.inference.app:app --reload --port 8080

Endpoints:
  GET  /health
  GET  /v1/models
  GET  /v1/models/{name}/active?channel=production
  POST /v1/infer  (multipart: image=..., optional model/channel/version/top_k)
  POST /v1/infer/batch  (multipart: images=..., optional model/channel/version/top_k)
  POST /v1/stream/sessions
  GET  /v1/stream/sessions/{session_id}
  POST /v1/stream/sessions/{session_id}/frames
  POST /v1/stream/sessions/{session_id}/flush
  DELETE /v1/stream/sessions/{session_id}
  POST /v1/reload-registry
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.inference.engine import infer_image, infer_images_batch, load_labels
from server.inference.registry_loader import ModelRegistry, ResolvedModel
from server.inference.stream import StreamSessionManager
from server.observability.routes import router as observability_router

ROOT = Path(__file__).resolve().parents[2]
ALLOW_MOCK = os.environ.get("ALLOW_MOCK_INFERENCE", "1") == "1"
DEFAULT_MODEL = os.environ.get("MUSIC_VISION_DEFAULT_MODEL", "music_vision_efficientnet")
REGISTRY_PATH = Path(
    os.environ.get("MUSIC_VISION_REGISTRY", str(ROOT / "ml" / "registry" / "registry.json"))
)
MAX_BATCH_SIZE = int(os.environ.get("MUSIC_VISION_MAX_BATCH_SIZE", "32"))
DEFAULT_STREAM_BATCH_SIZE = int(os.environ.get("MUSIC_VISION_STREAM_BATCH_SIZE", "8"))
DEFAULT_STREAM_MAX_WAIT_MS = int(os.environ.get("MUSIC_VISION_STREAM_MAX_WAIT_MS", "500"))
DEFAULT_STREAM_IDLE_TTL_MS = int(os.environ.get("MUSIC_VISION_STREAM_IDLE_TTL_MS", "60000"))

app = FastAPI(
    title="Music Vision Server-Side Inference",
    version="1.1.0",
    description="Versioned TFLite inference with batch and stream frame processing.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry(path=REGISTRY_PATH, root=ROOT)
stream_sessions = StreamSessionManager()
app.include_router(observability_router)


class HealthResponse(BaseModel):
    status: str
    registry_updated_at: str | None
    allow_mock_inference: bool
    default_model: str
    max_batch_size: int
    stream_batch_size: int


class InferResponse(BaseModel):
    inference_id: str
    model: str
    channel: str
    version: str
    architecture: str
    device_class: str
    runtime: str = "server_side"
    engine_mode: str = Field(description="tflite or mock")
    latency_ms: float
    labels: list[dict]


class BatchInferItemResponse(BaseModel):
    index: int
    inference_id: str
    latency_ms: float
    labels: list[dict]


class BatchInferResponse(BaseModel):
    batch_id: str
    model: str
    channel: str
    version: str
    architecture: str
    device_class: str
    runtime: str = "server_side"
    engine_mode: str
    count: int
    total_latency_ms: float
    results: list[BatchInferItemResponse]


class StreamSessionCreateRequest(BaseModel):
    model: str = DEFAULT_MODEL
    channel: str = "production"
    version: str | None = None
    top_k: int = 5
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE
    max_wait_ms: int = DEFAULT_STREAM_MAX_WAIT_MS
    idle_ttl_ms: int = DEFAULT_STREAM_IDLE_TTL_MS


class StreamSessionResponse(BaseModel):
    session_id: str
    model: str
    channel: str
    version: str | None
    top_k: int
    batch_size: int
    max_wait_ms: int
    idle_ttl_ms: int
    pending_frames: int = 0
    total_frames_received: int = 0
    total_batches_flushed: int = 0


class StreamFrameResponse(BaseModel):
    frame_id: str
    session_id: str
    pending_frames: int
    auto_flushed: bool
    flush: StreamFlushResponse | None = None


class StreamFlushResponse(BaseModel):
    batch_id: str
    session_id: str
    flushed_count: int
    pending_frames: int
    total_latency_ms: float
    engine_mode: str
    results: list[dict]


def _resolve_model(
    model: str,
    channel: str,
    version: str | None,
) -> ResolvedModel:
    pinned = version.strip() if version else None
    if pinned == "":
        pinned = None
    try:
        return registry.resolve(model, channel=channel, version=pinned)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _validate_top_k(top_k: int) -> None:
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1..50")


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be 1..{MAX_BATCH_SIZE}",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        registry_updated_at=registry.updated_at,
        allow_mock_inference=ALLOW_MOCK,
        default_model=DEFAULT_MODEL,
        max_batch_size=MAX_BATCH_SIZE,
        stream_batch_size=DEFAULT_STREAM_BATCH_SIZE,
    )


@app.get("/v1/models")
def list_models() -> dict:
    return {"models": registry.list_models(), "updated_at": registry.updated_at}


@app.get("/v1/models/{name}/active")
def active_model(name: str, channel: str = "production") -> dict:
    try:
        resolved = registry.resolve(name, channel=channel)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return resolved.public_dict()


@app.post("/v1/reload-registry")
def reload_registry() -> dict:
    try:
        registry.reload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"reloaded": True, "updated_at": registry.updated_at}


@app.post("/v1/infer", response_model=InferResponse)
async def infer(
    image: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    channel: str = Form("production"),
    version: str | None = Form(None),
    top_k: int = Form(5),
) -> InferResponse:
    _validate_top_k(top_k)
    resolved = _resolve_model(model, channel, version)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        labels = load_labels(resolved.labels_path)
        results, latency_ms, mode = infer_image(
            image_bytes=image_bytes,
            labels=labels,
            input_size=resolved.input_size,
            artifact_path=resolved.artifact_path,
            allow_mock=ALLOW_MOCK,
            top_k_n=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface runtime errors to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InferResponse(
        inference_id=str(uuid.uuid4()),
        model=resolved.name,
        channel=resolved.channel,
        version=resolved.version,
        architecture=resolved.architecture,
        device_class=resolved.device_class,
        engine_mode=mode,
        latency_ms=latency_ms,
        labels=[label.as_dict() for label in results],
    )


@app.post("/v1/infer/batch", response_model=BatchInferResponse)
async def infer_batch(
    images: list[UploadFile] = File(...),
    model: str = Form(DEFAULT_MODEL),
    channel: str = Form("production"),
    version: str | None = Form(None),
    top_k: int = Form(5),
) -> BatchInferResponse:
    _validate_top_k(top_k)
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(images)} exceeds max {MAX_BATCH_SIZE}",
        )

    resolved = _resolve_model(model, channel, version)
    image_bytes_list: list[bytes] = []
    for upload in images:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty image upload in batch")
        image_bytes_list.append(payload)

    try:
        labels = load_labels(resolved.labels_path)
        batch = infer_images_batch(
            image_bytes_list=image_bytes_list,
            labels=labels,
            input_size=resolved.input_size,
            artifact_path=resolved.artifact_path,
            allow_mock=ALLOW_MOCK,
            top_k_n=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return BatchInferResponse(
        batch_id=str(uuid.uuid4()),
        model=resolved.name,
        channel=resolved.channel,
        version=resolved.version,
        architecture=resolved.architecture,
        device_class=resolved.device_class,
        engine_mode=batch.engine_mode,
        count=len(batch.results),
        total_latency_ms=batch.total_latency_ms,
        results=[
            BatchInferItemResponse(
                index=item.index,
                inference_id=str(uuid.uuid4()),
                latency_ms=item.latency_ms,
                labels=[label.as_dict() for label in item.labels],
            )
            for item in batch.results
        ],
    )


@app.post("/v1/stream/sessions", response_model=StreamSessionResponse)
def create_stream_session(body: StreamSessionCreateRequest) -> StreamSessionResponse:
    _validate_top_k(body.top_k)
    _validate_batch_size(body.batch_size)
    if body.max_wait_ms < 50 or body.max_wait_ms > 10000:
        raise HTTPException(status_code=400, detail="max_wait_ms must be 50..10000")
    if body.idle_ttl_ms < 1000 or body.idle_ttl_ms > 3600000:
        raise HTTPException(status_code=400, detail="idle_ttl_ms must be 1000..3600000")

    session = stream_sessions.create_session(
        model=body.model,
        channel=body.channel,
        version=body.version,
        top_k=body.top_k,
        batch_size=body.batch_size,
        max_wait_ms=body.max_wait_ms,
        idle_ttl_ms=body.idle_ttl_ms,
    )
    return StreamSessionResponse(
        session_id=session.session_id,
        model=session.model,
        channel=session.channel,
        version=session.version,
        top_k=session.top_k,
        batch_size=session.batch_size,
        max_wait_ms=session.max_wait_ms,
        idle_ttl_ms=session.idle_ttl_ms,
    )


@app.get("/v1/stream/sessions/{session_id}", response_model=StreamSessionResponse)
def get_stream_session(session_id: str) -> StreamSessionResponse:
    status = stream_sessions.session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'")
    return StreamSessionResponse(**status)


@app.post("/v1/stream/sessions/{session_id}/frames", response_model=StreamFrameResponse)
async def push_stream_frame(
    session_id: str,
    image: UploadFile = File(...),
    client_index: int | None = Form(None),
) -> StreamFrameResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame upload")

    try:
        session, frame, should_flush = stream_sessions.add_frame(
            session_id,
            image_bytes=image_bytes,
            client_index=client_index,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    flush_response: StreamFlushResponse | None = None
    if should_flush:
        resolved = _resolve_model(session.model, session.channel, session.version)
        try:
            labels = load_labels(resolved.labels_path)
            flushed = stream_sessions.flush_session(
                session_id,
                labels=labels,
                input_size=resolved.input_size,
                artifact_path=resolved.artifact_path,
                allow_mock=ALLOW_MOCK,
                force=True,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if flushed is not None:
            flush_response = StreamFlushResponse(
                batch_id=flushed.batch_id,
                session_id=flushed.session_id,
                flushed_count=flushed.flushed_count,
                pending_frames=flushed.pending_count,
                total_latency_ms=flushed.total_latency_ms,
                engine_mode=flushed.engine_mode,
                results=flushed.results,
            )

    status = stream_sessions.session_status(session_id)
    pending = status["pending_frames"] if status else 0
    return StreamFrameResponse(
        frame_id=frame.frame_id,
        session_id=session_id,
        pending_frames=pending,
        auto_flushed=flush_response is not None,
        flush=flush_response,
    )


@app.post("/v1/stream/sessions/{session_id}/flush", response_model=StreamFlushResponse)
def flush_stream_session(session_id: str) -> StreamFlushResponse:
    session = stream_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'")

    resolved = _resolve_model(session.model, session.channel, session.version)
    try:
        labels = load_labels(resolved.labels_path)
        flushed = stream_sessions.flush_session(
            session_id,
            labels=labels,
            input_size=resolved.input_size,
            artifact_path=resolved.artifact_path,
            allow_mock=ALLOW_MOCK,
            force=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if flushed is None:
        raise HTTPException(status_code=409, detail="No pending frames to flush")

    return StreamFlushResponse(
        batch_id=flushed.batch_id,
        session_id=flushed.session_id,
        flushed_count=flushed.flushed_count,
        pending_frames=flushed.pending_count,
        total_latency_ms=flushed.total_latency_ms,
        engine_mode=flushed.engine_mode,
        results=flushed.results,
    )


@app.delete("/v1/stream/sessions/{session_id}")
def delete_stream_session(session_id: str) -> dict:
    deleted = stream_sessions.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'")
    return {"deleted": True, "session_id": session_id}
