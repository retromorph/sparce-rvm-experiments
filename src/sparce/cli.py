"""Command-line entry points: doctor, synthetic smoke, prepare, run-suite, summarize."""

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
            capture_output=True, check=False, text=True, timeout=10,
        )
        result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = "unavailable"
    from sparce.sandbox import docker_available
    result["docker_available"] = docker_available()
    try:
        from sparce.dataset import load_catalog
        changes = load_catalog()
        result["catalog_changes"] = len(changes)
        result["catalog_programs"] = sum(len(c.programs) for c in changes)
    except Exception as exc:  # noqa: BLE001
        result["catalog_error"] = str(exc)[:200]
    return result


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("run", "hardware", "model", "embedding", "rvm", "eval", "verifier"):
        if section not in config:
            raise ValueError(f"Missing configuration section [{section}]")
    config.setdefault("dataset", {})
    config.setdefault("knn", {})
    return config


def _dry_run(config: dict) -> dict:
    from sparce.dataset import build_tasks, load_catalog
    try:
        changes = load_catalog()
        built = build_tasks(changes, variants=int(config["dataset"].get("variants", 2)))
        catalog = {"api_changes": len(changes), "programs": sum(len(c.programs) for c in changes),
                   "siblings": sum(len(c.siblings) for c in changes), **built.manifest}
    except Exception as exc:  # noqa: BLE001
        catalog = {"error": str(exc)[:300]}
    return {
        "status": "DRY_RUN",
        "experiments": config["run"]["experiment_ids"],
        "gpu": config["hardware"]["gpu_name"],
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "embedding": config["embedding"]["id"],
        "stages": ["prepare", "generate", "verify", "embed", "analyze", "report"],
        "catalog_preview": catalog,
        "work_dir": config["run"].get("work_dir", "runs/suite"),
        "message": "Nothing executed. Without --dry-run the suite builds docker environments, generates/scores candidates on the GPU, verifies in the sandbox and runs E1/E2/E3.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPaRCE research suite")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect environment, GPU, docker and catalog availability")
    smoke = commands.add_parser("smoke", help="Run SYNTHETIC RVM plumbing demonstration")
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--no-mlflow", action="store_true")
    suite = commands.add_parser("run-suite", help="Run the real E1/E2/E3 suite end to end")
    suite.add_argument("--config", default="configs/a4000.toml")
    suite.add_argument("--dry-run", action="store_true")
    suite.add_argument("--resume", action="store_true", help="continue an existing work dir (caches, MLflow run ids)")
    suite.add_argument("--work-dir", default=None, help="override [run].work_dir")
    suite.add_argument("--limit-clusters", type=int, default=None, help="pilot: subsample program clusters")
    suite.add_argument("--stages", default="prepare,generate,verify,embed,analyze,report")
    suite.add_argument("--no-mlflow", action="store_true")
    summ = commands.add_parser("summarize", help="Re-run analysis/report from cached candidates/verifications/embeddings")
    summ.add_argument("--config", default="configs/a4000.toml")
    summ.add_argument("--work-dir", default=None)
    summ.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2))
        return 0
    if args.command == "smoke":
        results = run_smoke(args.seed)
        print(json.dumps(results, indent=2))
        if not args.no_mlflow:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            mlflow.set_experiment("sparce-synthetic-smoke")
            with mlflow.start_run(run_name="synthetic-smoke"):
                mlflow.set_tag("synthetic_only", "true")
                mlflow.log_param("seed", args.seed)
                mlflow.log_metrics({k: float(v) for k, v in results.items() if type(v) in (int, float)})
        return 0
    if args.command in ("run-suite", "summarize"):
        config = _load_config(args.config)
        if args.command == "run-suite" and args.dry_run:
            print(json.dumps(_dry_run(config), indent=2))
            return 0
        from sparce.suite import run_suite
        work_dir = Path(args.work_dir or config["run"].get("work_dir", "runs/suite"))
        if args.command == "summarize":
            run_suite(config, work_dir=work_dir, resume=True, no_mlflow=args.no_mlflow, stages=("analyze", "report"))
            return 0
        limit = args.limit_clusters if args.limit_clusters is not None else int(config["run"].get("limit_clusters", 0))
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
        run_suite(config, work_dir=work_dir, resume=args.resume, no_mlflow=args.no_mlflow, limit_clusters=limit, stages=stages)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
