"""Shared TFLite / mock inference helpers for Music Vision server-side serving."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceLabel:
    text: str
    confidence: float
    index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "index": self.index,
        }


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels not found: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not labels:
        raise ValueError(f"No labels in {path}")
    return labels


def preprocess_image(image_bytes: bytes, size: tuple[int, int]) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize(
        size, Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def run_tflite(model_path: Path, input_tensor: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Install tflite-runtime or tensorflow to run real models"
            ) from exc

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if input_details[0]["dtype"].__name__ == "uint8":
        raw = ((input_tensor + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], raw)
    else:
        interpreter.set_tensor(input_details[0]["index"], input_tensor)

    started = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output, latency_ms


def run_mock(labels: list[str], image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Deterministic fallback when the .tflite artifact is not present yet."""
    started = time.perf_counter()
    seed = sum(image_bytes[:64]) if image_bytes else 0
    rng = np.random.default_rng(seed)
    scores = rng.random(len(labels), dtype=np.float32)
    scores = scores / float(scores.sum())
    latency_ms = (time.perf_counter() - started) * 1000
    return scores, latency_ms


def top_k(scores: np.ndarray, labels: list[str], k: int) -> list[InferenceLabel]:
    order = np.argsort(scores)[::-1][:k]
    results: list[InferenceLabel] = []
    for idx in order:
        i = int(idx)
        name = labels[i] if i < len(labels) else f"class_{i}"
        results.append(InferenceLabel(text=name, confidence=float(scores[i]), index=i))
    return results


def infer_image(
    *,
    image_bytes: bytes,
    labels: list[str],
    input_size: tuple[int, int],
    artifact_path: Path | None,
    allow_mock: bool,
    top_k_n: int,
) -> tuple[list[InferenceLabel], float, str]:
    if artifact_path is not None and artifact_path.is_file():
        tensor = preprocess_image(image_bytes, input_size)
        scores, latency_ms = run_tflite(artifact_path, tensor)
        mode = "tflite"
    elif allow_mock:
        scores, latency_ms = run_mock(labels, image_bytes)
        mode = "mock"
    else:
        raise FileNotFoundError(
            f"Model artifact missing: {artifact_path}. "
            "Place a .tflite file or set ALLOW_MOCK_INFERENCE=1."
        )
    return top_k(scores, labels, top_k_n), latency_ms, mode
