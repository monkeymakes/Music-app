"""Load and resolve versioned models from ml/registry/registry.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "ml" / "registry" / "registry.json"


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    channel: str
    version: str
    architecture: str
    device_class: str
    format: str
    input_size: tuple[int, int]
    artifact_path: Path
    labels_path: Path
    sha256: str | None
    notes: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "channel": self.channel,
            "version": self.version,
            "architecture": self.architecture,
            "device_class": self.device_class,
            "format": self.format,
            "input_size": list(self.input_size),
            "artifact_path": _rel_or_abs(self.artifact_path),
            "labels_path": _rel_or_abs(self.labels_path),
            "sha256": self.sha256,
            "notes": self.notes,
            "artifact_present": self.artifact_path.is_file(),
        }


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class ModelRegistry:
    def __init__(self, path: Path = DEFAULT_REGISTRY, root: Path = ROOT) -> None:
        self.path = path
        self.root = root
        self._data: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Registry not found: {self.path}")
        self._data = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def updated_at(self) -> str | None:
        return self._data.get("updated_at")

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for name, model in sorted((self._data.get("models") or {}).items()):
            out.append(
                {
                    "name": name,
                    "display_name": model.get("display_name"),
                    "architecture": model.get("architecture"),
                    "device_class": model.get("device_class"),
                    "format": model.get("format"),
                    "channels": model.get("channels") or {},
                    "versions": sorted((model.get("versions") or {}).keys()),
                }
            )
        return out

    def resolve(
        self,
        name: str,
        *,
        channel: str | None = "production",
        version: str | None = None,
    ) -> ResolvedModel:
        models = self._data.get("models") or {}
        if name not in models:
            raise KeyError(f"Unknown model '{name}'")
        model = models[name]
        versions = model.get("versions") or {}
        if version:
            if version not in versions:
                raise KeyError(f"Unknown version '{version}' for {name}")
            resolved_version = version
            resolved_channel = channel or "pin"
        else:
            channel = channel or "production"
            channels = model.get("channels") or {}
            if channel not in channels:
                raise KeyError(f"Channel '{channel}' not set on {name}")
            resolved_version = channels[channel]
            resolved_channel = channel
            if resolved_version not in versions:
                raise KeyError(
                    f"{name} channel {channel} points at missing version {resolved_version}"
                )

        entry = versions[resolved_version]
        size = model.get("input_size") or [224, 224]
        artifact = Path(entry["artifact_path"])
        labels = Path(entry["labels_path"])
        if not artifact.is_absolute():
            artifact = self.root / artifact
        if not labels.is_absolute():
            labels = self.root / labels

        return ResolvedModel(
            name=name,
            channel=resolved_channel,
            version=resolved_version,
            architecture=str(model.get("architecture") or ""),
            device_class=str(model.get("device_class") or ""),
            format=str(model.get("format") or "tflite"),
            input_size=(int(size[0]), int(size[1])),
            artifact_path=artifact,
            labels_path=labels,
            sha256=entry.get("sha256"),
            notes=str(entry.get("notes") or ""),
        )
