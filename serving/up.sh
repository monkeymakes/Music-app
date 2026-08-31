#!/usr/bin/env bash
# Start TensorFlow Serving with Podman (preferred) or Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT}/serving/docker-compose.yml"
MODEL_DIR="${ROOT}/serving/models/music_vision"

if [[ ! -d "${MODEL_DIR}" ]] || ! find "${MODEL_DIR}" -mindepth 2 -maxdepth 2 -name saved_model.pb 2>/dev/null | grep -q .; then
  echo "No versioned SavedModel under ${MODEL_DIR}/<version>/"
  echo "Export one first, e.g.:"
  echo "  pip install -r ml/requirements-serving.txt"
  echo "  python3 ml/prepare_serving_model.py --export-stub --version 1"
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
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
    return
  fi
  echo ""
}

COMPOSE="$(pick_compose)"
if [[ -z "${COMPOSE}" ]]; then
  echo "Neither Podman nor Docker Compose is available."
  echo "Install Podman (preferred) or Docker, then retry."
  exit 1
fi

echo "Using: ${COMPOSE}"
# shellcheck disable=SC2086
cd "${ROOT}" && ${COMPOSE} -f "${COMPOSE_FILE}" up -d "$@"
echo "REST http://127.0.0.1:8501  |  gRPC 127.0.0.1:8500"
echo "Try: python3 serving/client/predict_image.py --image path/to.jpg"
