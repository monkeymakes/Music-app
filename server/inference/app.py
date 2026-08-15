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
  POST /v1/reload-registry
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.inference.engine import infer_image, load_labels
from server.inference.registry_loader import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
ALLOW_MOCK = os.environ.get("ALLOW_MOCK_INFERENCE", "1") == "1"
DEFAULT_MODEL = os.environ.get("MUSIC_VISION_DEFAULT_MODEL", "music_vision_efficientnet")
REGISTRY_PATH = Path(
    os.environ.get("MUSIC_VISION_REGISTRY", str(ROOT / "ml" / "registry" / "registry.json"))
)

app = FastAPI(
    title="Music Vision Server-Side Inference",
    version="1.0.0",
    description="Versioned TFLite inference for Music Vision models.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry(path=REGISTRY_PATH, root=ROOT)


class HealthResponse(BaseModel):
    status: str
    registry_updated_at: str | None
    allow_mock_inference: bool
    default_model: str


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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        registry_updated_at=registry.updated_at,
        allow_mock_inference=ALLOW_MOCK,
        default_model=DEFAULT_MODEL,
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
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1..50")

    pinned = version.strip() if version else None
    if pinned == "":
        pinned = None

    try:
        resolved = registry.resolve(model, channel=channel, version=pinned)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
