#!/usr/bin/env bash
# Run a Music Vision Clink SQL pipeline from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="${1:-music_vision_batch}"
SQL_FILE="$ROOT/edge/clink/pipelines/${PIPELINE}.sql"
IMAGE="${CLINK_RUNTIME_IMAGE:-ghcr.io/orhaugh/clink-runtime:latest}"

if [[ ! -f "$SQL_FILE" ]]; then
  echo "Unknown pipeline: $PIPELINE" >&2
  echo "Available: music_vision_batch, music_vision_windows" >&2
  exit 1
fi

cd "$ROOT"

if command -v clink >/dev/null 2>&1; then
  exec clink run "$SQL_FILE"
fi

if command -v docker >/dev/null 2>&1; then
  exec docker run --rm -v "$ROOT:/work" -w /work "$IMAGE" run "/work/$SQL_FILE"
fi

echo "Install clink locally or Docker to run pipelines." >&2
echo "Quick try: docker run --rm -v \"\$PWD:/work\" -w /work $IMAGE run /work/edge/clink/pipelines/${PIPELINE}.sql" >&2
exit 1
