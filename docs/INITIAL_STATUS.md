# Initial scaffold status — 2026-10-17

This repository is a starter. **No E1/E2/E3 real LLM experiments have been run.**

Environment available when scaffold was prepared: Python 3.13.5, NumPy 2.3.5, SciPy 1.17, pytest 9.0; no configured remote GPU, no MLflow install and no GitHub repository-creation action. Running with `PYTHONPATH=src pytest -q` returned **12 passed**, and a synthetic sample `python -m sparce.cli smoke --no-mlflow` completed successfully. Tests do not establish validity on real code correction.

`uv lock` could not be produced in the preparation container because the project targets Python 3.11, the container has only Python 3.13, and DNS to download the standalone Python 3.11 build is unavailable. On a networked development or GPU host:

```
uv python install 3.11
uv lock
uv sync --group dev
uv run pytest -q
```

Commit the resulting `uv.lock` before benchmark runs. Validate the optional GPU dependency set against the installed driver, and pin the specific model and embedding revisions in `configs/a4000.toml` before measuring.

`uv run sparce run-suite --dry-run` prints the intended E1/E2/E3 contracts. Without `--dry-run`, it deliberately exits nonzero until the coding agent implements the actual suite. Do not call the synthetic demonstration a measured gain in production or in the benchmark.
