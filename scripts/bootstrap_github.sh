#!/usr/bin/env bash
set -euo pipefail
# Run FROM THE EXTRACTED REPOSITORY on a machine authenticated to GitHub as retromorph.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v gh >/dev/null || { echo 'Install GitHub CLI: https://cli.github.com' >&2; exit 1; }
command -v git >/dev/null || { echo 'Install git' >&2; exit 1; }
gh auth status >/dev/null || { echo 'Run gh auth login as retromorph first' >&2; exit 1; }
LOGIN="$(gh api user --jq '.login')"
if [ "$LOGIN" != retromorph ]; then
  echo "Authenticated as '$LOGIN', expected retromorph. Switch gh auth account before continuing." >&2
  exit 1
fi
if gh repo view retromorph/sparce-rvm-experiments >/dev/null 2>&1; then
  echo 'Repo already exists. Not modifying an existing GitHub repository.' >&2
  exit 1
fi
if [ ! -d .git ]; then
  git init -b main
fi
if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
  echo 'Set git user.name and user.email (local or global) before proceeding.' >&2
  exit 1
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m 'chore: initialize SPaRCE RVM experiment scaffold'
fi
gh repo create retromorph/sparce-rvm-experiments --private --source . --remote origin --push
printf '\nRepository created: https://github.com/retromorph/sparce-rvm-experiments\n'
