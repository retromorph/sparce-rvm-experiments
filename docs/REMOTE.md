# Remote operation — SSH + tmux + uv + MLflow

Tested conceptually for Ubuntu with NVIDIA RTX A4000 16GB. Exact CUDA/PyTorch wheels depend on driver. No remote host is configured or accessed by this repository.

## 1. Prerequisites

Remote: SSH account with `git`, `uv`, NVIDIA driver (`nvidia-smi`), `tmux`, Python 3.11 (or `uv python install 3.11`), sufficient disk for models/MLflow artifacts, and a secure isolated-container runtime for untrusted generated code. Set up the key via your normal SSH process. Never commit private keys, SSH configs, access tokens or dataset credentials.

`~/.ssh/config` on local machine (example; replace placeholders):

```
Host sparce-gpu
    HostName YOUR_REMOTE_HOST
    User YOUR_REMOTE_USER
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```

## 2. Clone, dependencies and model

On remote:

```bash
git clone git@github.com:retromorph/sparce-rvm-experiments.git
cd sparce-rvm-experiments
uv python install 3.11
uv sync --group dev --extra gpu
uv run sparce doctor
uv run pytest -q
```

Check `nvidia-smi`, `uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'`. Match the installed PyTorch wheel to the available GPU driver; do not assume `uv sync --extra gpu` supplies a CUDA build. Pin downloaded model and embedding revisions and log them. Initial A4000 profile is `configs/a4000.toml`. Keep `batch_size=1`, K=2, max context 2048 initially and inspect peak VRAM. Run an OOM-free smoke before launching full studies.

## 3. Start MLflow privately

```bash
bash scripts/start_mlflow.sh
# session check:
tmux ls
# view logs:
tmux attach -t sparce-mlflow
```

The script starts `uv run mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination "$PWD/mlruns/artifacts" --host 127.0.0.1 --port 5000` inside a detached `tmux` session. (The actual script uses absolute artifact path resolved at startup.) Existing session is left alone.

**On local laptop:**

```bash
ssh -N -L 5000:127.0.0.1:5000 sparce-gpu
# open http://127.0.0.1:5000
```

`MLFLOW_TRACKING_URI=http://127.0.0.1:5000` should be set in the GPU process environment. Both server and GPU worker run on the same remote host. Do not expose MLflow publicly with `0.0.0.0`, `--allowed-hosts '*'`, `--cors-allowed-origins '*'` or `ufw allow 5000/tcp`. MLflow's default server may have no user authentication. SSH port forwarding avoids a public tracking endpoint. If shared access is later needed, deploy authenticated HTTPS behind a proxy or VPN.

If UFW is already enabled, retain SSH ingress access (`sudo ufw allow OpenSSH`) before changing it; only change firewall rules with an out-of-band recovery path. This repository does not modify UFW.

## 4. Start the suite

Checked on the actual host (2026-10-17): Ubuntu 26.04, NVIDIA RTX A4000 16 GB, driver 595.71.05 (CUDA 13.2), torch 2.14.1+cu130 from `uv sync --extra gpu`, docker 29 for the sandbox (no NVIDIA container toolkit needed: the verifier is CPU-only). Model and embedding revisions are pinned in `configs/a4000.toml`; pre-download them with `huggingface_hub.snapshot_download(repo, revision=...)` and run with `HF_HUB_OFFLINE=1`. Measured generation throughput: batch 12 → ~240 tok/s, 6.6 GB peak reserved VRAM.

Pilot (subset of clusters, separate work dir):

```bash
uv run sparce run-suite --config configs/a4000.toml --work-dir runs/pilot --limit-clusters 30
```

## 4b. Start suite after agent implementation

```bash
bash scripts/start_experiments.sh
# attach or stop:
tmux attach -t sparce-exp
tmux kill-session -t sparce-exp
```

Initial `run-suite` command intentionally fails without `--dry-run`, because the real experiment pipelines are an **agent deliverable**, not present in this boilerplate. Agent must implement, unit-test and first run a small-scale pilot. The tmux runner logs to `runs/experiment.log`. `scripts/remote_launch.sh sparce-gpu` invokes the server-side launch via SSH after checkout on the remote.

## 5. Restartability

- Run `uv lock` and commit `uv.lock` before final measurements.
- Every experiment has an immutable run ID and must checkpoint learned RVM snapshot + source trace IDs, data split hash and random state.
- On interruption, `run-suite --resume <run-id>` should verify compatibility and resume without reusing test labels.
- MLflow SQLite and artifacts live in `mlruns/`; copy this directory to encrypted backups if important. This prototype does not automate backups.

## 6. Monitoring and safety

`nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv`; `df -h`; `tmux ls`; `tail -f runs/experiment.log`. Sandbox any generated code, prohibit network, pin package versions, set wall time/memory/pid limits and use an unprivileged user. Prevent uploads of private code to public artifact stores.
