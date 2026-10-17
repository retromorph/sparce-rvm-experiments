# SPaRCE-RVM Experiments

**Status: starter repository, not completed LLM experiments.** Tested core inherited Laplace-ARD RVM implementation + policy + SQLite snapshot ledger; CPU synthetic smoke test only. The coding agent must implement the verified coding harness, frozen LLM integration, baselines, and the three go/no-go studies described below.

SPaRCE = **S**parse **P**robabilistic **R**elevance **C**orrective **E**xperiences: external, scoped corrections for a frozen coding model. Instead of retaining every mistake, learn a small set of relevance vectors using approximate empirical-Bayes automatic relevance determination (ARD), conservatively activate only in validated contexts, and keep versioned reversible snapshots.

## The three decisions

| Experiment | Question | Primary evidence |
| --- | --- | --- |
| E1 — Verified repair | Does the memory help an LLM fix *unseen, executable* versioned API errors? | Paired verified pass@1 vs frozen base, rule/RAG, and kNN controls |
| E2 — Memory efficiency | Does ARD improve the repair–memory–compute Pareto frontier vs equally budgeted kNN, random, and clustering? | Capacity- and budget-matched accuracy vs active count / bytes / latency |
| E3 — Locality + continual safety | Do new updates avoid breaking previously correct tasks, and can updates be reversed? | Collateral regressions, old-version controls, retention, uncertainty calibration, rollback |

**Reference protocol:** [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). **Full agent prompt:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md). **Report template:** [`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md).

## Quick start (CPU smoke)

```bash
uv python install 3.11
uv sync --group dev
uv run pytest -q
uv run sparce doctor
uv run sparce smoke --no-mlflow
uv run sparce run-suite --dry-run
```

The first `uv sync` creates `uv.lock`; **commit the generated lockfile** for fully reproducible experiments. The CPU smoke is intentionally synthetic and cannot support a research go/no-go judgment.

## Remote NVIDIA A4000 (16GB)

Run `scripts/start_mlflow.sh` on the remote machine to start the **loopback-only** MLflow server under `tmux`. Run `scripts/start_experiments.sh` to start `tmux` experiment orchestration once `run-suite` has been implemented by the coding agent. Tunnel the MLflow UI with `ssh -N -L 5000:127.0.0.1:5000 YOUR_SSH_ALIAS`; browse `http://127.0.0.1:5000` locally. **Do not open TCP/5000 to the internet.** See [`docs/REMOTE.md`](docs/REMOTE.md) for commands, UFW guidance and configuration.

Reference model: `Qwen/Qwen2.5-Coder-3B-Instruct`, batch size 1, 2 candidates, max input tokens 2048, 256 new tokens; checkpoint/dtype choices must be validated against real GPU and model card. A4000 is 16GB VRAM; leave space for runtime, KV cache and embeddings. Use GPU memory guard and record observed peak VRAM. CPU PyTorch or GPU dependency choices are controlled by `uv sync --extra gpu`; verify CUDA compatibility explicitly.

## Source tree

- `src/sparce/rvm.py`: sparse Laplace-ARD pairwise ranker (inherited prototype; needs independent calibration).
- `src/sparce/policy.py`: exact scope/support/posterior-margin gate; `auto_apply` requires independent verification.
- `src/sparce/ledger.py`: SQLite memory snapshot history and non-destructive rollback.
- `src/sparce/evaluation.py`: elementary paired decision/regression diagnostics.
- `src/sparce/cli.py`: doctor, synthetic smoke, and suite contract placeholder.
- `configs/a4000.toml`: computational defaults, declared methodology, local tracking URI.
- `scripts/start_mlflow.sh`: tmux-managed local MLflow server.
- `scripts/start_experiments.sh`: tmux-managed suite runner (intentionally fails until real suite exists).
- `scripts/remote_launch.sh`: optional SSH wrapper to invoke server-side launch.
- `.github/workflows/ci.yml`: CPU-only baseline unit tests, independent of a GPU.

## What must be implemented by the coding agent

1. Frozen model adapter: generate K candidates, score using consistent normalized likelihoods, and embed **(request, candidate)** pairs without label leakage.
2. Independent, container-isolated, version-pinned code verifier. The verifier must not receive hidden benchmark answers at inference, and all baselines must use identical runtime verification budgets.
3. Benchmark ingestion with immutable raw records, deduplication, chronological and API-family-disjoint splits, controls, and dataset/license provenance.
4. Official-document RAG or transparent version-aware rules; full/capacity-matched kNN; random/cluster prototype controls; RVM ablations; optional editing baselines only where feasible.
5. Metrics, uncertainty estimation, paired bootstrap, MLflow artifact schemas, deterministic seeds, and one report documenting **negative** as well as positive results.

**Safety:** Benchmark code is untrusted. Do not run arbitrary generated code unsandboxed on the host. Use an isolated container with no network, non-root user, resource limits, and pinned dependencies. No secrets or raw private code in MLflow logs.

## Research links

- Original RVM: https://www.jmlr.org/papers/v1/tipping01a.html
- CodeSyncBench/data engine: https://github.com/Lucky-Wang-Chenlong/CodeSync
- SERAC: https://arxiv.org/abs/2206.06520
- GRACE: https://arxiv.org/abs/2211.11031
- WISE: https://arxiv.org/abs/2405.14768
- ReasoningBank: https://arxiv.org/abs/2509.25140
- Mistake Notebook Learning: https://aclanthology.org/2026.findings-acl.719/

The external baseline repositories are **references**, not copied dependencies. Review licensing before incorporation.
