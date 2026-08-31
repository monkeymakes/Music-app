"""Shared TFLite runtime helpers for Raspberry Pi Music Vision.

Optimizations:
  - XNNPACK delegate (CPU acceleration on aarch64 / armv7l)
  - Configurable thread count (match Pi core count)
  - float32 and uint8 input handling
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class InferenceResult:
    scores: Any
    latency_ms: float
    input_dtype: str


def create_interpreter(
    model_path: Path,
    *,
    num_threads: int = 4,
    use_xnnpack: bool = True,
):
    """Load a TFLite interpreter with optional XNNPACK acceleration."""
    try:
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            from tensorflow.lite.python.interpreter import load_delegate
        except ImportError as exc:
            raise SystemExit(
                "Install tflite-runtime on the Pi:\n"
                "  pip install tflite-runtime"
            ) from exc

    delegates = []
    if use_xnnpack:
        try:
            delegates.append(load_delegate("libXNNPACK.so"))
        except (ValueError, OSError):
            # XNNPACK may be built into the runtime or unavailable on this build.
            pass

    interpreter = Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
        experimental_delegates=delegates or None,
    )
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(path: Path, size: tuple[int, int] = (224, 224)):
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return (array / 127.5) - 1.0


def set_input_tensor(interpreter, input_tensor) -> str:
    """Set interpreter input; returns dtype name ('float32' or 'uint8')."""
    import numpy as np

    input_details = interpreter.get_input_details()
    dtype_name = input_details[0]["dtype"].__name__

    expected = tuple(input_details[0]["shape"])
    if input_tensor.shape != expected and len(expected) == 4:
        h, w = int(expected[1]), int(expected[2])
        if input_tensor.shape[1:3] != (h, w):
            raise ValueError(
                f"Input shape {input_tensor.shape} != model {expected}"
            )

    if dtype_name == "uint8":
        raw = ((input_tensor + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], raw)
    else:
        batch = input_tensor if input_tensor.ndim == 4 else np.expand_dims(input_tensor, 0)
        interpreter.set_tensor(input_details[0]["index"], batch.astype(np.float32))

    return dtype_name


def invoke(interpreter) -> tuple[Any, float]:
    """Run inference; return (output scores, latency_ms)."""
    output_details = interpreter.get_output_details()
    started = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Dequantize uint8 outputs if needed
    scale, zero_point = output_details[0].get("quantization", (0.0, 0))
    if output_details[0]["dtype"].__name__ == "uint8" and scale:
        import numpy as np

        output = (output.astype(np.float32) - zero_point) * scale

    return output, latency_ms


def classify_image(
    model_path: Path,
    image_path: Path,
    *,
    num_threads: int = 4,
    use_xnnpack: bool = True,
    input_size: tuple[int, int] = (224, 224),
) -> InferenceResult:
    """End-to-end single-image classification."""
    interpreter = create_interpreter(
        model_path, num_threads=num_threads, use_xnnpack=use_xnnpack
    )
    tensor = preprocess_image(image_path, size=input_size)
    batch = __import__("numpy").expand_dims(tensor, axis=0)
    dtype_name = set_input_tensor(interpreter, batch)
    scores, latency_ms = invoke(interpreter)
    return InferenceResult(scores=scores, latency_ms=latency_ms, input_dtype=dtype_name)
