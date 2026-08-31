#!/usr/bin/env bash
# Start Music Vision Pi inference with Podman (preferred) or Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${ROOT}/edge/raspberry_pi/compose.yaml"
MODEL="${ROOT}/edge/raspberry_pi/models/music_vision_efficientnet.tflite"

if [[ ! -f "${MODEL}" ]]; then
  echo "Missing compressed model: ${MODEL}"
  echo "Export on your Mac first:"
  echo "  pip install -r ml/requirements-export-tflite.txt"
  echo "  python3 ml/export_efficientnet_tflite.py --export-stub --quantize dynamic"
  exit 1
fi

pick_compose() {
  if command -v podman >/dev/null 2>&1; then
    if podman compose version >/dev/null 2>&1; then
      echo "podman compose"
      return
    fi
    if command -v podman-compose >/dev/null 2>&1; then
      echo "podman-compose"
      return
    fi
  fi
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return
  fi
  echo ""
}

COMPOSE="$(pick_compose)"
if [[ -z "${COMPOSE}" ]]; then
  echo "Install Podman (preferred on RHEL) or Docker Compose."
  exit 1
fi

echo "Using: ${COMPOSE}"
# shellcheck disable=SC2086
cd "${ROOT}" && ${COMPOSE} -f "${COMPOSE_FILE}" up -d --build "$@"
echo "Pi inference: http://127.0.0.1:${PORT:-8088}/v1/infer"
echo "Health:       http://127.0.0.1:${PORT:-8088}/health"
