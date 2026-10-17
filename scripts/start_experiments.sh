#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v tmux >/dev/null || { echo 'tmux not installed' >&2; exit 1; }
command -v uv >/dev/null || { echo 'uv not installed' >&2; exit 1; }
command -v docker >/dev/null || { echo 'docker (sandbox runtime) not installed' >&2; exit 1; }
if tmux has-session -t =sparce-exp 2>/dev/null; then
  echo 'Session sparce-exp already exists; refusing to launch duplicate.' >&2
  exit 1
fi
# Validate that the configuration and catalog load; this does NOT execute E1/E2/E3.
(cd "$ROOT" && uv run sparce run-suite --config configs/a4000.toml --dry-run >/dev/null)
tmux new-session -d -s sparce-exp -c "$ROOT" "bash scripts/_run_suite.sh $*"
echo 'Experiment runner launched in tmux session sparce-exp (pass --resume to continue an interrupted run).'
echo 'Attach with: tmux attach -t sparce-exp ; log: runs/experiments-*.log'
