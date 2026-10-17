#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
set -o pipefail
uv run sparce run-suite --config configs/a4000.toml 2>&1 | tee "runs/experiments-$(date -u +%Y%m%dT%H%M%SZ).log"
