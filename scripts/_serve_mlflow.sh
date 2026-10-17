#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p mlruns/artifacts runs
exec uv run mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination "$ROOT/mlruns/artifacts" \
  --host 127.0.0.1 \
  --port 5000 >> runs/mlflow.log 2>&1
