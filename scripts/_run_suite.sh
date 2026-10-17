#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SPARCE_GIT_SHA="${SPARCE_GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
set -o pipefail
uv run sparce run-suite --config configs/a4000.toml "$@" 2>&1 | tee "runs/experiments-$(date -u +%Y%m%dT%H%M%SZ).log"
