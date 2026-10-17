#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 SSH_HOST_ALIAS" >&2
  exit 2
fi
# Remote checkout location is fixed by docs/REMOTE.md to avoid command injection.
ssh -- "$1" 'bash -lc "cd \$HOME/sparce-rvm-experiments && bash scripts/start_mlflow.sh && bash scripts/start_experiments.sh"'
