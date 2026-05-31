#!/usr/bin/env bash
#
# Convenience launcher for the Aegis retrieval API.
#
# It pins the HuggingFace/sentence-transformers model cache to a project-local
# directory (backend/.hf_cache). This avoids the common case where the global
# ~/.cache/huggingface directory is owned by another user (e.g. root) and the
# model download fails with a PermissionError.
#
# Usage:
#   cd backend
#   bash scripts/run.sh            # serve on :8000
#   PORT=8077 bash scripts/run.sh  # serve on a custom port
#
set -euo pipefail

# Resolve backend/ regardless of where the script is called from.
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BACKEND_DIR"

export HF_HOME="${HF_HOME:-$BACKEND_DIR/.hf_cache}"
PORT="${PORT:-8000}"

echo "Using model cache: $HF_HOME"
echo "LLM provider: ${AEGIS_PROVIDER:-mock} (fallback: ${AEGIS_FALLBACK_PROVIDER:-mock})"
echo "Starting Aegis retrieval API on http://127.0.0.1:$PORT (docs at /docs)"

exec ./venv/bin/uvicorn app.main:app --reload --port "$PORT"
