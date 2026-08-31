#!/usr/bin/env python3
"""Minimal HTTP inference server for Raspberry Pi / Red Hat (no JavaScript).

Pure Python stdlib HTTP — suitable for Podman on RHEL, Fedora, or Pi OS.
Accepts POST /v1/infer with a JSON body or multipart image upload.

Example:
  python3 edge/raspberry_pi/infer_server.py \\
    --model edge/raspberry_pi/models/music_vision_efficientnet.tflite \\
    --labels ml/labels.example.txt \\
    --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.raspberry_pi.tflite_runtime import (  # noqa: E402
    create_interpreter,
    invoke,
    preprocess_image,
    set_input_tensor,
)


def load_labels(path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not labels:
        raise SystemExit(f"No labels in {path}")
    return labels


class InferState:
    def __init__(
        self,
        model_path: Path,
        labels: list[str],
        num_threads: int,
        use_xnnpack: bool,
    ) -> None:
        self.labels = labels
        self.interpreter = create_interpreter(
            model_path, num_threads=num_threads, use_xnnpack=use_xnnpack
        )
        self.input_details = self.interpreter.get_input_details()
        h, w = int(self.input_details[0]["shape"][1]), int(self.input_details[0]["shape"][2])
        self.input_size = (w, h)

    def predict_file(self, image_path: Path, top_k: int = 5) -> dict:
        import numpy as np

        tensor = np.expand_dims(preprocess_image(image_path, size=self.input_size), axis=0)
        dtype = set_input_tensor(self.interpreter, tensor)
        scores, latency_ms = invoke(self.interpreter)

        ranked = sorted(
            ((float(score), idx) for idx, score in enumerate(scores)),
            reverse=True,
        )[:top_k]

        labels_out = []
        for score, idx in ranked:
            name = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            labels_out.append({"text": name, "confidence": score, "index": idx})

        return {
            "architecture": "efficientnet",
            "device_class": "raspberry_pi",
            "input_dtype": dtype,
            "latency_ms": round(latency_ms, 2),
            "labels": labels_out,
        }


def make_handler(state: InferState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/health", "/v1/health"):
                self._json(200, {"status": "ok", "architecture": "efficientnet"})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/v1/infer", "/predict"):
                self._json(404, {"error": "not found"})
                return

            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""

            top_k = 5
            if "?" in self.path:
                qs = parse_qs(self.path.split("?", 1)[1])
                if "top_k" in qs:
                    top_k = int(qs["top_k"][0])

            try:
                if content_type.startswith("application/json"):
                    data = json.loads(raw.decode("utf-8"))
                    if "image_b64" in data:
                        suffix = ".jpg"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(base64.b64decode(data["image_b64"]))
                            tmp_path = Path(tmp.name)
                    elif "image_path" in data:
                        tmp_path = Path(data["image_path"])
                    else:
                        self._json(400, {"error": "JSON must include image_b64 or image_path"})
                        return
                    if "top_k" in data:
                        top_k = int(data["top_k"])
                elif content_type.startswith("multipart/form-data"):
                    # Minimal multipart: find first JPEG/PNG chunk after boundary
                    boundary = content_type.split("boundary=")[-1].strip()
                    parts = raw.split(f"--{boundary}".encode())
                    tmp_path = None
                    for part in parts:
                        if b"Content-Type: image" in part:
                            header, _, body = part.partition(b"\r\n\r\n")
                            if body.endswith(b"\r\n"):
                                body = body[:-2]
                            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                                tmp.write(body)
                                tmp_path = Path(tmp.name)
                            break
                    if tmp_path is None:
                        self._json(400, {"error": "multipart form must include an image field"})
                        return
                else:
                    self._json(
                        415,
                        {"error": "use application/json (image_b64) or multipart/form-data"},
                    )
                    return

                result = state.predict_file(tmp_path, top_k=top_k)
                self._json(200, result)
            except Exception as exc:  # noqa: BLE001 — surface inference errors to client
                self._json(500, {"error": str(exc)})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("edge/raspberry_pi/models/music_vision_efficientnet.tflite"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/labels.example.txt"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-xnnpack", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    labels = load_labels(args.labels)
    state = InferState(
        args.model,
        labels,
        num_threads=args.threads,
        use_xnnpack=not args.no_xnnpack,
    )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Music Vision Pi server listening on http://{args.host}:{args.port} "
        f"(threads={args.threads}, labels={len(labels)})",
        file=sys.stderr,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
