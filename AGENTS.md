# Contributor and coding-agent instructions

Read `AGENT_PROMPT.md` for the full end-to-end research assignment. `docs/EXPERIMENT_PROTOCOL.md` is the preregistered evaluation contract. Scientific validity and safe code execution override target pass rates.

Local minimum checks: `uv sync --group dev`, `uv run pytest -q`, `uv run sparce smoke --no-mlflow`, `uv run sparce run-suite --dry-run`. Commit `uv.lock` when generated. Generated data, checkpoints, MLflow stores and secrets stay ignored. Use scoped, version-pinned corrections; do not run generated code outside network-disabled containers; do not conflate ARD pruning with exact Bayesian inference.

GitHub repo target: `retromorph/sparce-rvm-experiments`; report goes to `reports/generated/SPARCE_POC_REPORT.md` as an artifact (ignored by default), optionally commit sanitized final report deliberately using `git add -f` after reviewing data governance.
