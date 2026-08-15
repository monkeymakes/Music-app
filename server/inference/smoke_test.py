#!/usr/bin/env python3
"""Smoke tests for model registry + server-side inference (mock mode)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CLI = ROOT / "ml" / "registry" / "model_registry.py"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_registry_cli() -> None:
    listed = run([sys.executable, str(REGISTRY_CLI), "list"])
    assert "music_vision_tinyml" in listed.stdout
    assert "music_vision_efficientnet" in listed.stdout

    active = run(
        [
            sys.executable,
            str(REGISTRY_CLI),
            "get-active",
            "music_vision_efficientnet",
            "--channel",
            "production",
        ]
    )
    payload = json.loads(active.stdout)
    assert payload["version"] == "1.0.0"
    assert payload["architecture"] == "efficientnet"

    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "registry.json"
        registry_path.write_text(
            (ROOT / "ml" / "registry" / "registry.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        artifact = Path(tmp) / "fake.tflite"
        artifact.write_bytes(b"FAKE_TFLITE")
        labels = Path(tmp) / "labels.txt"
        labels.write_text("a\nb\n", encoding="utf-8")

        run(
            [
                sys.executable,
                str(REGISTRY_CLI),
                "--registry",
                str(registry_path),
                "register",
                "music_vision_tinyml",
                "1.0.1-test",
                "--artifact",
                str(artifact),
                "--labels",
                str(labels),
                "--notes",
                "smoke",
            ]
        )
        run(
            [
                sys.executable,
                str(REGISTRY_CLI),
                "--registry",
                str(registry_path),
                "promote",
                "music_vision_tinyml",
                "1.0.1-test",
                "--channel",
                "staging",
            ]
        )
        staged = run(
            [
                sys.executable,
                str(REGISTRY_CLI),
                "--registry",
                str(registry_path),
                "get-active",
                "music_vision_tinyml",
                "--channel",
                "staging",
            ]
        )
        assert json.loads(staged.stdout)["version"] == "1.0.1-test"

        run(
            [
                sys.executable,
                str(REGISTRY_CLI),
                "--registry",
                str(registry_path),
                "rollback",
                "music_vision_tinyml",
                "--channel",
                "staging",
                "--to",
                "1.0.0",
            ]
        )
        rolled = run(
            [
                sys.executable,
                str(REGISTRY_CLI),
                "--registry",
                str(registry_path),
                "get-active",
                "music_vision_tinyml",
                "--channel",
                "staging",
            ]
        )
        assert json.loads(rolled.stdout)["version"] == "1.0.0"


def test_inference_engine_mock() -> None:
    from PIL import Image

    from server.inference.engine import infer_image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(20, 80, 160)).save(buf, format="JPEG")
    labels = ["instrument", "vinyl", "guitar"]
    results, latency_ms, mode = infer_image(
        image_bytes=buf.getvalue(),
        labels=labels,
        input_size=(160, 160),
        artifact_path=None,
        allow_mock=True,
        top_k_n=2,
    )
    assert mode == "mock"
    assert latency_ms >= 0
    assert len(results) == 2
    assert {r.text for r in results} <= set(labels)


def test_registry_loader() -> None:
    from server.inference.registry_loader import ModelRegistry

    reg = ModelRegistry()
    models = reg.list_models()
    assert any(m["name"] == "music_vision_tinyml" for m in models)
    resolved = reg.resolve("music_vision_tinyml", channel="production")
    assert resolved.version == "1.0.0"
    assert resolved.architecture == "tinyml"


def test_fastapi_infer_mock() -> None:
    from fastapi.testclient import TestClient
    from PIL import Image

    from server.inference.app import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert len(models.json()["models"]) >= 3

    active = client.get("/v1/models/music_vision_efficientnet/active")
    assert active.status_code == 200
    assert active.json()["version"] == "1.0.0"

    buf = BytesIO()
    Image.new("RGB", (32, 32), color=(200, 40, 40)).save(buf, format="JPEG")
    response = client.post(
        "/v1/infer",
        data={
            "model": "music_vision_efficientnet",
            "channel": "production",
            "top_k": "3",
        },
        files={"image": ("x.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine_mode"] == "mock"
    assert body["version"] == "1.0.0"
    assert len(body["labels"]) == 3
    assert body["runtime"] == "server_side"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    test_registry_cli()
    print("ok registry cli")
    test_registry_loader()
    print("ok registry loader")
    test_inference_engine_mock()
    print("ok inference engine mock")
    test_fastapi_infer_mock()
    print("ok fastapi infer mock")
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
