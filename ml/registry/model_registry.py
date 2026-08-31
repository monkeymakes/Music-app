#!/usr/bin/env python3
"""Versioned model registry for Music Vision.

Controls which artifact each deployment channel (production / staging) serves.
Never overwrite an existing version path — register a new semver instead, then
promote the channel pointer.

Examples:
  python3 ml/registry/model_registry.py list
  python3 ml/registry/model_registry.py get-active music_vision_tinyml
  python3 ml/registry/model_registry.py register music_vision_tinyml 1.1.0 \\
      --artifact ios/MusicApp/Models/music_vision_tinyml.tflite \\
      --labels ios/MusicApp/Models/music_vision_tinyml_labels.txt \\
      --notes "Retrained on Aug batch"
  python3 ml/registry/model_registry.py promote music_vision_tinyml 1.1.0 --channel production
  python3 ml/registry/model_registry.py rollback music_vision_tinyml --channel production --to 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "registry.json"
CHANNELS = ("production", "staging")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_model(data: dict[str, Any], name: str) -> dict[str, Any]:
    models = data.setdefault("models", {})
    if name not in models:
        raise SystemExit(f"Unknown model '{name}'. Known: {', '.join(sorted(models)) or '(none)'}")
    return models[name]


def cmd_list(data: dict[str, Any]) -> None:
    models = data.get("models") or {}
    if not models:
        print("No models registered.")
        return
    for name, model in sorted(models.items()):
        channels = model.get("channels") or {}
        versions = model.get("versions") or {}
        print(f"{name}")
        print(f"  architecture={model.get('architecture')} format={model.get('format')}")
        print(f"  channels: " + ", ".join(f"{c}→{v}" for c, v in sorted(channels.items())))
        print(f"  versions ({len(versions)}): {', '.join(sorted(versions))}")


def cmd_get_active(data: dict[str, Any], name: str, channel: str) -> None:
    model = require_model(data, name)
    channels = model.get("channels") or {}
    if channel not in channels:
        raise SystemExit(f"Channel '{channel}' not set on {name}")
    version = channels[channel]
    entry = (model.get("versions") or {}).get(version)
    if not entry:
        raise SystemExit(f"{name} channel {channel} points at missing version {version}")
    payload = {
        "model": name,
        "channel": channel,
        "version": version,
        "architecture": model.get("architecture"),
        "device_class": model.get("device_class"),
        "format": model.get("format"),
        "input_size": model.get("input_size"),
        **entry,
    }
    print(json.dumps(payload, indent=2))


def cmd_register(
    data: dict[str, Any],
    name: str,
    version: str,
    artifact: Path,
    labels: Path,
    notes: str,
    status: str,
) -> None:
    model = require_model(data, name)
    versions = model.setdefault("versions", {})
    if version in versions:
        raise SystemExit(
            f"Version {version} already exists for {name}. "
            "Register a new semver instead of overwriting."
        )
    artifact_rel = str(artifact if artifact.is_absolute() else artifact)
    labels_rel = str(labels if labels.is_absolute() else labels)
    artifact_abs = artifact if artifact.is_absolute() else ROOT / artifact
    entry = {
        "version": version,
        "status": status,
        "artifact_path": artifact_rel,
        "labels_path": labels_rel,
        "sha256": sha256_file(artifact_abs),
        "created_at": utc_now(),
        "notes": notes,
    }
    versions[version] = entry
    print(json.dumps({"registered": name, **entry}, indent=2))


def cmd_promote(data: dict[str, Any], name: str, version: str, channel: str) -> None:
    model = require_model(data, name)
    versions = model.get("versions") or {}
    if version not in versions:
        raise SystemExit(f"Unknown version {version} for {name}")
    previous = (model.get("channels") or {}).get(channel)
    model.setdefault("channels", {})[channel] = version
    versions[version]["status"] = "production" if channel == "production" else "staging"
    if previous and previous != version and previous in versions:
        if (model.get("channels") or {}).get("production") != previous and (
            model.get("channels") or {}
        ).get("staging") != previous:
            versions[previous]["status"] = "retired"
    print(
        json.dumps(
            {
                "model": name,
                "channel": channel,
                "previous": previous,
                "current": version,
            },
            indent=2,
        )
    )


def cmd_rollback(data: dict[str, Any], name: str, channel: str, to_version: str) -> None:
    cmd_promote(data, name, to_version, channel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to registry.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List models and channel pins")

    get_p = sub.add_parser("get-active", help="Show the version pinned to a channel")
    get_p.add_argument("model")
    get_p.add_argument("--channel", default="production", choices=CHANNELS)

    reg_p = sub.add_parser("register", help="Add an immutable new version")
    reg_p.add_argument("model")
    reg_p.add_argument("version")
    reg_p.add_argument("--artifact", type=Path, required=True)
    reg_p.add_argument("--labels", type=Path, required=True)
    reg_p.add_argument("--notes", default="")
    reg_p.add_argument(
        "--status",
        default="draft",
        choices=["draft", "staging", "production", "retired"],
    )

    prom_p = sub.add_parser("promote", help="Point a channel at a registered version")
    prom_p.add_argument("model")
    prom_p.add_argument("version")
    prom_p.add_argument("--channel", default="production", choices=CHANNELS)

    roll_p = sub.add_parser("rollback", help="Repoint a channel to a prior version")
    roll_p.add_argument("model")
    roll_p.add_argument("--channel", default="production", choices=CHANNELS)
    roll_p.add_argument("--to", dest="to_version", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_registry(args.registry)

    if args.command == "list":
        cmd_list(data)
        return

    if args.command == "get-active":
        cmd_get_active(data, args.model, args.channel)
        return

    if args.command == "register":
        cmd_register(
            data,
            args.model,
            args.version,
            args.artifact,
            args.labels,
            args.notes,
            args.status,
        )
        save_registry(args.registry, data)
        return

    if args.command == "promote":
        cmd_promote(data, args.model, args.version, args.channel)
        save_registry(args.registry, data)
        return

    if args.command == "rollback":
        cmd_rollback(data, args.model, args.channel, args.to_version)
        save_registry(args.registry, data)
        return

    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
