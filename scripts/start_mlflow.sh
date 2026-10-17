#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v tmux >/dev/null || { echo 'tmux not installed' >&2; exit 1; }
command -v uv >/dev/null || { echo 'uv not installed' >&2; exit 1; }
mkdir -p "$ROOT/mlruns/artifacts" "$ROOT/runs"
if tmux has-session -t =sparce-mlflow 2>/dev/null; then
  echo 'Session sparce-mlflow already exists; leaving it unchanged.'
  exit 0
fi
tmux new-session -d -s sparce-mlflow -c "$ROOT" 'bash scripts/_serve_mlflow.sh'
echo 'MLflow started under tmux session sparce-mlflow, bound to 127.0.0.1:5000.'
echo 'Access from laptop: ssh -N -L 5000:127.0.0.1:5000 YOUR_SSH_ALIAS'
