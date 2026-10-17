"""End-to-end E1/E2/E3 suite orchestration with MLflow tracking and plots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sparce.analysis import DataBundle, build_pairs, run_e1, run_e2, run_e3
from sparce.dataset import load_tasks
from sparce.pipeline import (Paths, build_environments, candidate_records, stage_embed, stage_generate,
                             stage_inventories, stage_prepare, stage_verify, verification_lookup)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf8") as fh:
            fh.write(line + "\n")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    if not out and args and args[0] == "rev-parse":
        return os.environ.get("SPARCE_GIT_SHA", "unknown")
    return out


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def environment_manifest(cfg: dict, extra: dict | None = None) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "hostname": platform.node(),
            "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "uv_lock_sha256": _sha256_file(Path("uv.lock")), "config": cfg,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = "not installed"
    info.update(extra or {})
    return info


# --------------------------------------------------------------------------- plots
def make_plots(results_dir: Path, e1: dict, e2: dict, e3: dict, log) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out: list[Path] = []
    plots = results_dir / "plots"
    plots.mkdir(exist_ok=True)
    seed0 = e1["seeds"][0]
    # E1 bar chart with CIs of delta vs base
    names, deltas, lows, highs = [], [], [], []
    for name, entries in e1["methods"].items():
        e = next((x for x in entries if x["seed"] == seed0), entries[0])
        rep = e["repair"]
        if "delta_vs_base" in rep:
            names.append(name); deltas.append(rep["delta_vs_base"] * 100)
            lows.append((rep["delta_vs_base"] - rep["delta_ci_low"]) * 100); highs.append((rep["delta_ci_high"] - rep["delta_vs_base"]) * 100)
    if names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, deltas, yerr=[lows, highs], capsize=4, color="#4c72b0")
        ax.axhline(0, color="k", lw=0.8); ax.axhline(5, color="r", ls="--", lw=0.8, label="E1 go threshold (+5pp)")
        ax.set_ylabel("paired Δ verified pass@1 vs frozen base (pp)"); ax.set_title("E1: held-out repair, cluster-paired 95% bootstrap CI")
        ax.legend(); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
        p = plots / "e1_delta_vs_base.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    # E2 Pareto: accuracy vs bytes
    rows = [r for r in e2["rows"] if r["seed"] == seed0]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in sorted({r["method"] for r in rows}):
            rs = sorted([r for r in rows if r["method"] == method], key=lambda r: r["bytes"]["total"])
            ax.plot([r["bytes"]["total"] for r in rs], [r["accuracy"] * 100 for r in rs], marker="o", label=method)
        front = sorted(e2["pareto_front"], key=lambda p: p["bytes"])
        ax.plot([p["bytes"] for p in front], [p["accuracy"] * 100 for p in front], "k--", lw=0.8, label="Pareto front")
        ax.set_xscale("log"); ax.set_xlabel("active memory bytes (prototypes + weights + covariance + metadata)")
        ax.set_ylabel("test verified pass@1 (%)"); ax.set_title("E2: accuracy vs active bytes (seed %d)" % seed0); ax.legend(fontsize=8)
        fig.tight_layout(); p = plots / "e2_accuracy_vs_bytes.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in sorted({r["method"] for r in rows}):
            rs = sorted([r for r in rows if r["method"] == method], key=lambda r: r["latency_p95_ms"])
            ax.plot([r["latency_p95_ms"] for r in rs], [r["accuracy"] * 100 for r in rs], marker="s", label=method)
        ax.set_xlabel("decision latency p95 (ms, CPU)"); ax.set_ylabel("test verified pass@1 (%)"); ax.set_title("E2: accuracy vs latency"); ax.legend(fontsize=8)
        fig.tight_layout(); p = plots / "e2_accuracy_vs_latency.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    # E3 stream
    stream = e3.get("stream", [])
    if stream:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = [r["checkpoint"] for r in stream]
        ax.plot(x, [r["test_repair"].get("pass_at_1", np.nan) * 100 for r in stream], marker="o", label="test repair pass@1")
        ax.plot(x, [r["collateral"]["rate"] * 100 for r in stream], marker="x", label="collateral rate (controls)")
        ax.plot(x, [r["retention_test_fixed"] * 100 for r in stream], marker="^", label="retention of first-checkpoint fixes")
        ax.set_xlabel("verified updates admitted (checkpoint)"); ax.set_ylabel("%"); ax.set_xscale("log"); ax.legend(fontsize=8)
        ax.set_title("E3: sequential verified memory updates"); fig.tight_layout()
        p = plots / "e3_stream.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
        cal = e3["methods"].get("rvm", {}).get("calibration_test_pairs", {})
        if cal.get("bins"):
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.plot([b["confidence"] for b in cal["bins"]], [b["accuracy"] for b in cal["bins"]], marker="o")
            for b in cal["bins"]:
                ax.annotate(str(b["n"]), (b["confidence"], b["accuracy"]), fontsize=7)
            ax.set_xlabel("predicted P(alternative better)"); ax.set_ylabel("observed frequency")
            ax.set_title(f"E3: RVM reliability (Brier {cal['brier']:.3f}, ECE {cal['ece']:.3f})"); fig.tight_layout()
            p = plots / "e3_reliability.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    log(f"plots: {len(out)} written to {plots}")
    return out


def write_csvs(results_dir: Path, e1: dict, e2: dict, e3: dict) -> list[Path]:
    out = []
    p = results_dir / "e1_summary.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "n", "n_clusters", "pass_at_1", "base_pass_at_1", "delta_vs_base", "ci_low", "ci_high", "fixes", "regressions", "activations", "active_count", "bytes_total", "latency_p95_ms", "known_family_pass_at_1", "heldout_family_pass_at_1", "controls_regressions"])
        for name, entries in e1["methods"].items():
            for e in entries:
                r = e["repair"]; bs = e["repair_by_subset"]
                w.writerow([name, e["seed"], r.get("n"), r.get("n_clusters"), r.get("pass_at_1"), r.get("base_pass_at_1"), r.get("delta_vs_base"), r.get("delta_ci_low"), r.get("delta_ci_high"), r.get("fixes"), r.get("regressions"), r.get("activations"), e["active_count"], e["bytes"].get("total"), r.get("latency_p95_ms"), bs.get("known_family", {}).get("pass_at_1"), bs.get("heldout_family", {}).get("pass_at_1"), e["controls"].get("regressions")])
    out.append(p)
    p = results_dir / "e2_rows.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "capacity", "seed", "active_count", "bytes_total", "bytes_prototypes", "bytes_covariance", "accuracy", "all_accuracy", "latency_p50_ms", "latency_p95_ms", "fit_seconds", "controls_regressions"])
        for r in e2["rows"]:
            w.writerow([r["method"], r["capacity"], r["seed"], r["active_count"], r["bytes"]["total"], r["bytes"].get("prototypes"), r["bytes"].get("covariance"), r["accuracy"], r["all_accuracy"], r["latency_p50_ms"], r["latency_p95_ms"], r["fit_seconds"], r.get("controls_regressions")])
    out.append(p)
    p = results_dir / "e3_stream.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "n_admitted", "n_quarantined", "promoted", "active_count", "bytes_total", "test_pass_at_1", "test_fixes", "test_regressions", "collateral_regressions", "collateral_base_correct", "collateral_rate", "wilson_upper_95", "retention_test_fixed", "brier", "ece", "snapshot_bytes"])
        for r in e3.get("stream", []):
            w.writerow([r["checkpoint"], r["n_admitted"], r["n_quarantined"], r["promoted"], r["active_count"], r["bytes"]["total"], r["test_repair"].get("pass_at_1"), r["test_repair"].get("fixes"), r["test_repair"].get("regressions"), r["collateral"]["n_regressions"], r["collateral"]["n_base_correct"], r["collateral"]["rate"], r["collateral"]["wilson_upper_95"], r["retention_test_fixed"], r["calibration"].get("brier"), r["calibration"].get("ece"), r.get("snapshot_bytes")])
    out.append(p)
    return out


# --------------------------------------------------------------------------- MLflow
class Tracker:
    def __init__(self, cfg: dict, enabled: bool, work_dir: Path, log):
        self.enabled = enabled
        self.log = log
        self.ids_path = work_dir / "mlflow_runs.json"
        self.ids = json.loads(self.ids_path.read_text()) if self.ids_path.exists() else {}
        self.mlflow = None
        if enabled:
            import mlflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["run"]["mlflow_uri"]))
            mlflow.set_experiment(cfg["run"]["mlflow_experiment"])
            self.mlflow = mlflow

    def run(self, name: str, tags: dict, params: dict, metrics: dict, artifacts: list[Path]) -> str | None:
        if not self.enabled:
            return None
        ml = self.mlflow
        with ml.start_run(run_name=name) as run:
            ml.set_tags({k: str(v)[:250] for k, v in tags.items()})
            ml.log_params({k: str(v)[:250] for k, v in params.items()})
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))}
            for i in range(0, len(clean), 90):
                ml.log_metrics(dict(list(clean.items())[i:i + 90]))
            for a in artifacts:
                if a.exists():
                    ml.log_artifact(str(a))
            self.ids[name] = run.info.run_id
        self.ids_path.write_text(json.dumps(self.ids, indent=2))
        self.log(f"mlflow: run {name} -> {self.ids[name]}")
        return self.ids[name]


def _flatten(prefix: str, d: dict, out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten(key, v, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


# --------------------------------------------------------------------------- suite
def run_suite(cfg: dict, *, work_dir: Path, resume: bool, no_mlflow: bool, limit_clusters: int = 0,
              stages: tuple[str, ...] = ("prepare", "generate", "verify", "embed", "analyze", "report"),
              llm_factory=None, embedder_factory=None, sandbox_images=None) -> dict:
    work_dir = Path(work_dir)
    if work_dir.exists() and not resume and any(work_dir.iterdir()):
        raise SystemExit(f"work dir {work_dir} exists; pass --resume to continue it or choose a new --work-dir")
    paths = Paths(work_dir)
    log = Logger(work_dir / "suite.log")
    manifest = environment_manifest(cfg)
    (work_dir / "environment_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"suite start: git={manifest['git_sha'][:12]} dirty={manifest['git_dirty']} work_dir={work_dir} resume={resume}")
    tracker = Tracker(cfg, not no_mlflow, work_dir, log)

    images = sandbox_images
    if "prepare" in stages:
        split_manifest = stage_prepare(cfg, paths, log, limit_clusters=limit_clusters, images=images)
    tasks, hidden = load_tasks(paths.data)
    split_manifest = json.loads((paths.data / "split_manifest.json").read_text())
    if images is None:
        from sparce.dataset import build_tasks, load_catalog
        from sparce.environments import environment_for
        env_detail = split_manifest["environments_detail"]
        from sparce.environments import EnvironmentSpec
        specs = {k: EnvironmentSpec(v["env_id"], v["python_version"], tuple(v["requirements"]), v["package"], v["package_version"]) for k, v in env_detail.items()}
        images = build_environments(specs, paths, log)

    if "generate" in stages:
        inventories = stage_inventories(cfg, paths, log, tasks, images)
        llm = llm_factory(cfg) if llm_factory else _default_llm(cfg)
        stage_generate(cfg, paths, log, tasks, llm, inventories)
        model_meta = llm.metadata()
        (work_dir / "model_manifest.json").write_text(json.dumps(model_meta, indent=2))
        del llm
    if "verify" in stages:
        stage_verify(cfg, paths, log, tasks, hidden, images)
    if "embed" in stages:
        embedder = embedder_factory(cfg) if embedder_factory else _default_embedder(cfg)
        stage_embed(cfg, paths, log, tasks, embedder)
        (work_dir / "embedding_manifest.json").write_text(json.dumps(embedder.metadata(), indent=2))

    results: dict = {}
    if "analyze" in stages:
        with np.load(paths.embeddings) as z:
            emb = {k: z[k] for k in z.files}
        cands = candidate_records(paths)
        ver = verification_lookup(paths, tasks, hidden)
        base_pairs = build_pairs(tasks, cands, ver, emb, "base")
        rag_pairs = build_pairs(tasks, cands, ver, emb, "rag")
        log(f"analyze: {len(base_pairs)} base pairs, {len(rag_pairs)} rag pairs")
        data = DataBundle(base_pairs, rag_pairs, cfg)
        seeds = [int(s) for s in cfg["run"]["seeds"]]
        e1 = run_e1(data, log, seeds)
        e2 = run_e2(data, log, seeds, e1["tuned"])
        e3 = run_e3(data, log, seeds, e1["tuned"], work_dir)
        for name, obj in (("e1", e1), ("e2", e2), ("e3", e3)):
            (paths.results / f"{name}.json").write_text(json.dumps(obj, indent=1, default=_json_default))
        csvs = write_csvs(paths.results, e1, e2, e3)
        plots = make_plots(paths.results, e1, e2, e3, log)
        results = {"e1": e1, "e2": e2, "e3": e3}
        common_tags = {"git_sha": manifest["git_sha"], "uv_lock_sha256": manifest["uv_lock_sha256"],
                       "tasks_sha256": split_manifest.get("tasks_sha256"), "model": cfg["model"]["id"], "model_revision": cfg["model"]["revision"],
                       "embedding": cfg["embedding"]["id"], "embedding_revision": cfg["embedding"]["revision"], "work_dir": str(work_dir)}
        common_params = {"seeds": cfg["run"]["seeds"], "num_candidates": cfg["model"]["num_candidates"], "max_new_tokens": cfg["model"]["max_new_tokens"],
                         "max_input_tokens": cfg["model"]["max_input_tokens"], "batch_size": cfg["model"]["batch_size"], "n_tasks": len(tasks),
                         "n_test_repair": e1["n_test_repair"], "n_controls": e3["n_controls"]}
        shared_artifacts = [work_dir / "environment_manifest.json", paths.data / "split_manifest.json", paths.catalog_validation,
                            work_dir / "model_manifest.json", work_dir / "embedding_manifest.json", paths.env_manifest]
        e1_metrics = {}
        for name, entries in e1["methods"].items():
            e = entries[0]
            e1_metrics.update(_flatten(f"{name}.repair", e["repair"]))
            e1_metrics.update(_flatten(f"{name}.controls", {"regressions": e["controls"].get("regressions", 0)}))
            e1_metrics[f"{name}.active_count"] = e["active_count"]; e1_metrics[f"{name}.bytes_total"] = e["bytes"].get("total", 0)
        e1_metrics.update({"candidate_recall_at_k": e1["candidate_recall_at_k"], "greedy_pass_at_1": e1["greedy_pass_at_1"],
                           "tokens_per_task_base": e1["tokens_per_task"]["base"], "cold_evidence_bytes": e1["cold_evidence_bytes"]})
        tracker.run("E1-verified-repair", {**common_tags, "experiment": "E1", "decision": e1["decision"]["status"]},
                    {**common_params, "tuned_rvm": e1["tuned"]["rvm"]["best"], "tuned_knn": e1["tuned"]["knn"]["best"]},
                    e1_metrics, shared_artifacts + [paths.results / "e1.json", csvs[0]] + [p for p in plots if "e1" in p.name])
        e2_metrics = {}
        for r in e2["rows"]:
            if r["seed"] == seeds[0]:
                key = f"{r['method']}.cap{r['capacity']}"
                e2_metrics[f"{key}.accuracy"] = r["accuracy"]; e2_metrics[f"{key}.bytes"] = r["bytes"]["total"]
                e2_metrics[f"{key}.active"] = r["active_count"]; e2_metrics[f"{key}.lat95"] = r["latency_p95_ms"]
        tracker.run("E2-sparse-pareto", {**common_tags, "experiment": "E2", "decision": e2["decision"]["status"]},
                    {**common_params, "capacities": e2["capacities"]}, e2_metrics,
                    shared_artifacts + [paths.results / "e2.json", csvs[1]] + [p for p in plots if "e2" in p.name])
        e3_metrics = _flatten("rvm", e3["methods"]["rvm"]["collateral"])
        for r in e3["stream"]:
            e3_metrics[f"ck{r['checkpoint']}.test_pass_at_1"] = r["test_repair"].get("pass_at_1", float("nan"))
            e3_metrics[f"ck{r['checkpoint']}.collateral_rate"] = r["collateral"]["rate"]
            e3_metrics[f"ck{r['checkpoint']}.retention"] = r["retention_test_fixed"]
        e3_metrics["rollback_exact"] = float(e3["rollback"].get("exact", False)); e3_metrics["false_submissions_admitted"] = e3["false_submissions"]["n_admitted"]
        tracker.run("E3-safety-stream-rollback", {**common_tags, "experiment": "E3", "decision": e3["decision"]["status"]},
                    {**common_params, "checkpoints": e3["checkpoints"]}, e3_metrics,
                    shared_artifacts + [paths.results / "e3.json", csvs[2]] + [p for p in plots if "e3" in p.name])
        log(f"decisions: E1={e1['decision']['status']} E2={e2['decision']['status']} E3={e3['decision']['status']}")
    if "report" in stages:
        from sparce.report import write_report
        out_dir = Path(cfg["run"]["output_dir"])
        exploratory = cfg["run"].get("exploratory_work_dir")
        report_path = write_report(cfg, work_dir, out_dir, tracker.ids, Path(exploratory) if exploratory else None)
        log(f"report written to {report_path}")
        results["report"] = str(report_path)
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _default_llm(cfg: dict):
    from sparce.llm import FrozenCausalLM
    m = cfg["model"]
    return FrozenCausalLM(m["id"], m["revision"], m.get("dtype", "float16"), int(m["max_input_tokens"]))


def _default_embedder(cfg: dict):
    from sparce.embedding import JointEmbedder
    e = cfg["embedding"]
    return JointEmbedder(e["id"], e["revision"], batch_size=int(e.get("batch_size", 32)))
