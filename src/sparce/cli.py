"""Command-line entry points for the research starter.

`run-suite` is intentionally not implemented yet: see AGENT_PROMPT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

from sparce.smoke import run_smoke


def _doctor() -> dict[str, object]:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "unset"),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_total_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        result["torch"] = "not installed (optional gpu dependency)"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = "unavailable"
    return result


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("run", "hardware", "model", "embedding", "rvm", "eval", "verifier"):
        if section not in config:
            raise ValueError(f"Missing configuration section [{section}]")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPaRCE research starter")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect environment and GPU availability")
    smoke = commands.add_parser("smoke", help="Run SYNTHETIC RVM plumbing demonstration")
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--no-mlflow", action="store_true")
    suite = commands.add_parser("run-suite", help="Real E1/E2/E3 suite (agent TODO)")
    suite.add_argument("--config", default="configs/a4000.toml")
    suite.add_argument("--dry-run", action="store_true")
    suite.add_argument("--resume", default=None)
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2))
        return 0
    if args.command == "smoke":
        results = run_smoke(args.seed)
        print(json.dumps(results, indent=2))
        if not args.no_mlflow:
            try:
                import mlflow
            except ImportError as exc:
                raise SystemExit("Install dependencies using `uv sync`. ") from exc
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            mlflow.set_experiment("sparce-synthetic-smoke")
            with mlflow.start_run(run_name="synthetic-smoke"):
                mlflow.set_tag("synthetic_only", "true")
                mlflow.log_param("seed", args.seed)
                mlflow.log_metrics({
                    k: float(v) for k, v in results.items()
                    if type(v) in (int, float)
                })
        return 0
    if args.command == "run-suite":
        config = _load_config(args.config)
        if args.dry_run:
            print(json.dumps({
                "status": "SCAFFOLD_ONLY",
                "experiments": config["run"]["experiment_ids"],
                "gpu": config["hardware"]["gpu_name"],
                "model": config["model"]["id"],
                "message": "Real dataset/model/verifier/baselines not yet implemented; see AGENT_PROMPT.md",
            }, indent=2))
            return 0
        print("NOT IMPLEMENTED: real E1/E2/E3 suite. Complete AGENT_PROMPT.md first.", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
