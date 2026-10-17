"""Resumable data/LLM/verifier pipeline stages with per-item caches.

Stages (each idempotent; re-running skips cached items):
  prepare  -> build benchmark, docker environments, validate catalog by execution, request tracebacks
  generate -> frozen-LLM candidates (+ pinned-API RAG candidates for val/test), length-normalised scores
  verify   -> isolated execution of every candidate against hidden tests
  embed    -> joint (request, candidate) embeddings
All caches live under ``<work_dir>/cache``. Hidden records are only read by ``prepare`` and ``verify``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from sparce.dataset import (BuildResult, HiddenRecord, SplitConfig, Task, build_tasks, leakage_audit,
                            load_catalog, load_tasks, read_jsonl, write_dataset, write_jsonl)
from sparce.embedding import error_headline, joint_text
from sparce.environments import BuiltImage, build_image, write_manifest
from sparce.rag import INVENTORY_SCRIPT, IMPORT_NAMES, ApiReferenceRetriever, inventory_test_code, parse_inventory
from sparce.sandbox import DockerSandbox, SandboxJob, SandboxLimits, SandboxResult

Log = Callable[[str], None]


def _sanitize_traceback(text: str, max_lines: int = 8) -> str:
    """Remove sandbox paths; keep the informative tail of the traceback."""
    text = re.sub(r"/work/[0-9]+/", "", text or "")
    text = re.sub(r'File "[^"]*/(solution|test_solution)\.py"', r'File "\1.py"', text)
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:])[-1200:]


def _h(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf8")); h.update(b"\x1e")
    return h.hexdigest()[:24]


class Paths:
    def __init__(self, work_dir: Path):
        self.work = Path(work_dir)
        self.data = self.work / "data"
        self.cache = self.work / "cache"
        self.results = self.work / "results"
        for d in (self.data, self.cache, self.results):
            d.mkdir(parents=True, exist_ok=True)
        self.env_manifest = self.data / "environments.json"
        self.catalog_validation = self.data / "catalog_validation.json"
        self.candidates = self.cache / "candidates.jsonl"
        self.rag_candidates = self.cache / "rag_candidates.jsonl"
        self.verify = self.cache / "verify.jsonl"
        self.embeddings = self.cache / "embeddings.npz"
        self.inventories = self.cache / "inventories.json"
        self.stage_log = self.work / "stages.json"

    def mark(self, stage: str, info: dict) -> None:
        state = json.loads(self.stage_log.read_text()) if self.stage_log.exists() else {}
        state[stage] = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}
        self.stage_log.write_text(json.dumps(state, indent=2))

    def done(self, stage: str) -> bool:
        state = json.loads(self.stage_log.read_text()) if self.stage_log.exists() else {}
        return stage in state


# --------------------------------------------------------------------------- environments / sandbox
def build_environments(result_envs: dict, paths: Paths, log: Log, workers: int = 3) -> dict[str, BuiltImage]:
    from concurrent.futures import ThreadPoolExecutor
    specs = list(result_envs.values())
    log(f"building/locating {len(specs)} docker environments")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        built = list(pool.map(build_image, specs))
    write_manifest(built, paths.env_manifest)
    for b in built:
        log(f"  env {b.spec.env_id}: {'ok ' + b.image_id[:19] if b.ok else 'FAILED ' + b.build_log_tail[-300:]}")
    return {b.spec.env_id: b for b in built}


def sandboxes_for(images: dict[str, BuiltImage], vcfg: dict) -> dict[str, DockerSandbox]:
    limits = SandboxLimits(walltime_seconds=int(vcfg.get("max_walltime_seconds", 30)),
                           memory_mb=int(vcfg.get("max_memory_mb", 2048)), pids=int(vcfg.get("max_pids", 128)),
                           cpus=float(vcfg.get("cpus", 2.0)))
    return {k: DockerSandbox(b.spec.image_tag, b.image_id, limits) for k, b in images.items() if b.ok}


# Test seam: CPU tests replace this with a factory of fake sandboxes (no docker).
SANDBOX_FACTORY = sandboxes_for


def run_jobs(sandboxes: dict[str, DockerSandbox], jobs_by_env: dict[str, list[SandboxJob]], vcfg: dict,
             log: Log) -> dict[str, SandboxResult]:
    out: dict[str, SandboxResult] = {}
    total = sum(len(v) for v in jobs_by_env.values())
    done = 0
    for env_id, jobs in jobs_by_env.items():
        sb = sandboxes.get(env_id)
        if sb is None:
            for j in jobs:
                out[j.job_id] = SandboxResult(j.job_id, "error", False, None, "environment image unavailable", 0.0, "none", "")
            continue
        start = time.perf_counter()
        for r in sb.run(jobs, batch_size=int(vcfg.get("batch_size", 24)), workers=int(vcfg.get("workers", 4))):
            out[r.job_id] = r
        done += len(jobs)
        log(f"  verified {done}/{total} jobs ({env_id}, {time.perf_counter() - start:.0f}s)")
    return out


# --------------------------------------------------------------------------- prepare
def stage_prepare(cfg: dict, paths: Paths, log: Log, *, limit_clusters: int = 0, images: dict | None = None) -> dict:
    if paths.done("prepare") and (paths.data / "tasks.jsonl").exists():
        log("prepare: cached")
        return json.loads((paths.data / "split_manifest.json").read_text())
    dcfg = cfg.get("dataset", {})
    changes = load_catalog()
    if not changes:
        raise RuntimeError("Catalog is empty: no src/sparce/catalog/*.py modules with CHANGES found")
    split = SplitConfig(train_fraction=float(dcfg.get("train_fraction", 0.45)), val_fraction=float(dcfg.get("val_fraction", 0.15)),
                        heldout_family_fraction=float(dcfg.get("heldout_family_fraction", 0.2)), seed=int(cfg["run"]["seed"]))
    result = build_tasks(changes, split, variants=int(dcfg.get("variants", 2)))
    if limit_clusters:
        keep = sorted({t.cluster_id for t in result.tasks})
        rng = np.random.default_rng(int(cfg["run"]["seed"]))
        keep = set(rng.choice(keep, size=min(limit_clusters, len(keep)), replace=False).tolist())
        result.tasks = [t for t in result.tasks if t.cluster_id in keep]
        ids = {t.task_id for t in result.tasks}
        result.hidden = [h for h in result.hidden if h.task_id in ids]
    log(f"prepare: {len(result.tasks)} candidate tasks from {len(changes)} API changes, {result.manifest['n_clusters']} clusters")
    images = images or build_environments(result.environments, paths, log)
    sandboxes = SANDBOX_FACTORY(images, cfg["verifier"])

    hidden = {h.task_id: h for h in result.hidden}
    jobs_by_env: dict[str, list[SandboxJob]] = defaultdict(list)
    job_keys: dict[str, str] = {}
    for t in result.tasks:
        h = hidden[t.task_id]
        for label, code in (("request", t.request_code), ("reference", h.reference_code)):
            key = _h(t.env_id, code, h.test_code)
            if key not in job_keys:
                job_keys[key] = key
                jobs_by_env[t.env_id].append(SandboxJob(key, code, h.test_code))
    log(f"prepare: executing {sum(len(v) for v in jobs_by_env.values())} catalog validation jobs in sandbox")
    results = run_jobs(sandboxes, jobs_by_env, cfg["verifier"], log)

    kept: list[Task] = []
    excluded: list[dict] = []
    for t in result.tasks:
        h = hidden[t.task_id]
        req = results[_h(t.env_id, t.request_code, h.test_code)]
        ref = results[_h(t.env_id, h.reference_code, h.test_code)]
        if t.kind == "repair":
            ok = req.status == "fail" and ref.status == "pass"
            why = f"request={req.status},reference={ref.status}"
        else:
            ok = req.status == "pass"
            why = f"request={req.status}"
        if not ok:
            excluded.append({"task_id": t.task_id, "cluster_id": t.cluster_id, "kind": t.kind, "env_id": t.env_id, "reason": why,
                             "stderr": _sanitize_traceback(req.stderr_tail if req.status != "pass" else ref.stderr_tail, 4)})
            continue
        kept.append(Task(**{**asdict(t), "request_status": req.status,
                            "request_stderr": _sanitize_traceback(req.stderr_tail) if req.status != "pass" else ""}))
    # A repair cluster is only valid if its variants are valid; keep controls regardless.
    result.tasks = kept
    ids = {t.task_id for t in kept}
    result.hidden = [hh for hh in result.hidden if hh.task_id in ids]
    from sparce.dataset import assign_splits
    result.tasks = assign_splits(result.tasks, split)
    manifest = write_dataset(result, paths.data)
    audit = leakage_audit(result.tasks)
    validation = {"n_candidate_tasks": len(excluded) + len(kept), "n_kept": len(kept), "n_excluded": len(excluded),
                  "excluded": excluded, "leakage_audit": audit,
                  "images": {k: {"image_id": v.image_id, "ok": v.ok} for k, v in images.items()}}
    paths.catalog_validation.write_text(json.dumps(validation, indent=2))
    manifest.update({"n_excluded_by_execution": len(excluded), "leakage_audit": audit,
                     "kinds": {k: sum(1 for t in kept if t.kind == k) for k in sorted({t.kind for t in kept})},
                     "splits": {s: sum(1 for t in kept if t.split == s and t.kind == "repair") for s in ("train", "val", "test")},
                     "test_subsets": {s: sum(1 for t in kept if t.split == "test" and t.kind == "repair" and t.subset == s)
                                      for s in ("known_family", "heldout_family")}})
    (paths.data / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"prepare: kept {len(kept)} tasks, excluded {len(excluded)}; repair split {manifest['splits']}; audit ok={audit['ok']}")
    paths.mark("prepare", {"n_tasks": len(kept), "n_excluded": len(excluded)})
    return manifest


# --------------------------------------------------------------------------- inventories (RAG corpus)
def stage_inventories(cfg: dict, paths: Paths, log: Log, tasks: list[Task], images: dict) -> dict[str, list[dict]]:
    if paths.inventories.exists():
        return json.loads(paths.inventories.read_text())
    sandboxes = SANDBOX_FACTORY(images, cfg["verifier"])
    jobs_by_env: dict[str, list[SandboxJob]] = defaultdict(list)
    env_pkg = {t.env_id: t.package for t in tasks}
    for env_id, package in env_pkg.items():
        if package == "python":
            continue
        jobs_by_env[env_id].append(SandboxJob(f"inv:{env_id}", INVENTORY_SCRIPT, inventory_test_code(package)))
    vcfg = {**cfg["verifier"], "max_walltime_seconds": 300}
    sandboxes = {k: (DockerSandbox(s.image, s.image_digest, SandboxLimits(walltime_seconds=300, memory_mb=2048)) if isinstance(s, DockerSandbox) else s)
                 for k, s in sandboxes.items()}
    results = run_jobs(sandboxes, jobs_by_env, vcfg, log)
    inv = {env_id: parse_inventory(results[f"inv:{env_id}"].artifact) for env_id in jobs_by_env}
    for env_id in env_pkg:
        inv.setdefault(env_id, [])
    paths.inventories.write_text(json.dumps(inv))
    log("inventories: " + ", ".join(f"{k}={len(v)}" for k, v in inv.items()))
    return inv


# --------------------------------------------------------------------------- generate
def prompt_for(task: Task, rag_context: str | None = None) -> str:
    from sparce.llm import build_user_prompt
    error = task.request_stderr if task.request_status and task.request_status != "pass" else None
    return build_user_prompt(task.package, task.package_version, task.python_version, task.request_code, error, rag_context)


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["task_id"]: r for r in read_jsonl(path)}


def stage_generate(cfg: dict, paths: Paths, log: Log, tasks: list[Task], llm, inventories: dict[str, list[dict]]) -> None:
    from sparce.llm import GenerationConfig
    mcfg = cfg["model"]
    gcfg = GenerationConfig(num_candidates=int(mcfg["num_candidates"]), max_new_tokens=int(mcfg["max_new_tokens"]),
                            temperature=float(mcfg.get("temperature", 0.7)), top_p=float(mcfg.get("top_p", 0.95)),
                            batch_size=int(mcfg["batch_size"]), seed=int(cfg["run"]["seed"]))
    for source, path, subset in (("base", paths.candidates, tasks),
                                 ("rag", paths.rag_candidates, [t for t in tasks if t.split == "test"])):
        cache = _load_cache(path)
        todo = [t for t in subset if t.task_id not in cache]
        log(f"generate[{source}]: {len(subset) - len(todo)} cached, {len(todo)} to generate (K={gcfg.num_candidates}, max_new={gcfg.max_new_tokens})")
        if not todo:
            continue
        prompts: list[str] = []
        contexts: list[str] = []
        for t in todo:
            ctx = None
            if source == "rag":
                retriever = ApiReferenceRetriever(inventories.get(t.env_id, []))
                ctx = retriever.format(retriever.retrieve(t.request_code, error_headline(t.request_stderr)))
            contexts.append(ctx or "")
            prompts.append(prompt_for(t, ctx))
        chunk = max(1, gcfg.batch_size * 8)
        started = time.perf_counter()
        for start in range(0, len(todo), chunk):
            sl = slice(start, start + chunk)
            gens = llm.generate_candidates([t.task_id for t in todo[sl]], prompts[sl], gcfg)
            flat_prompts = [p for p, row in zip(prompts[sl], gens) for _ in row]
            flat_texts = [g.text for row in gens for g in row]
            scores = llm.score_candidates(flat_prompts, flat_texts, batch_size=gcfg.batch_size)
            k = 0
            rows = []
            for t, row, ctx in zip(todo[sl], gens, contexts[sl]):
                cands = []
                for g in row:
                    cands.append({**g.to_dict(), "score": scores[k]})
                    k += 1
                rows.append({"task_id": t.task_id, "source": source, "prompt_sha": _h(prompts[todo.index(t)]),
                             "rag_context": ctx, "generation_config": gcfg.digest(), "candidates": cands,
                             "peak_vram_bytes": llm.peak_vram_bytes()})
            with path.open("a", encoding="utf8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
            elapsed = time.perf_counter() - started
            n_done = min(len(todo), start + chunk)
            log(f"generate[{source}]: {n_done}/{len(todo)} tasks, {elapsed/60:.1f} min elapsed, peak VRAM {llm.peak_vram_bytes()/1e9:.2f} GB")
    paths.mark("generate", {"generation_config": gcfg.digest()})


# --------------------------------------------------------------------------- verify
def candidate_records(paths: Paths) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for source, path in (("base", paths.candidates), ("rag", paths.rag_candidates)):
        for r in _load_cache(path).values():
            out[(r["task_id"], source)] = r
    return out


def stage_verify(cfg: dict, paths: Paths, log: Log, tasks: list[Task], hidden: dict[str, HiddenRecord], images: dict) -> None:
    sandboxes = SANDBOX_FACTORY(images, cfg["verifier"])
    cache: dict[str, dict] = {}
    if paths.verify.exists():
        for r in read_jsonl(paths.verify):
            cache[r["key"]] = r
    by_id = {t.task_id: t for t in tasks}
    cands = candidate_records(paths)
    jobs_by_env: dict[str, list[SandboxJob]] = defaultdict(list)
    pending: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for (task_id, source), rec in cands.items():
        t = by_id.get(task_id)
        if t is None:
            continue
        h = hidden[task_id]
        for i, c in enumerate(rec["candidates"]):
            key = _h(t.env_id, c["code"], h.test_code)
            pending[key].append((task_id, source, i))
            if key in cache or any(j.job_id == key for j in jobs_by_env[t.env_id][-3:]):
                continue
            if not any(j.job_id == key for j in jobs_by_env[t.env_id]):
                jobs_by_env[t.env_id].append(SandboxJob(key, c["code"], h.test_code))
    n_jobs = sum(len(v) for v in jobs_by_env.values())
    log(f"verify: {len(cache)} cached results, {n_jobs} new sandbox jobs for {len(pending)} unique candidates")
    if n_jobs:
        results = run_jobs(sandboxes, jobs_by_env, cfg["verifier"], log)
        with paths.verify.open("a", encoding="utf8") as fh:
            for key, r in results.items():
                row = {"key": key, "status": r.status, "syntax_ok": r.syntax_ok, "execution_ms": r.execution_ms,
                       "verifier_id": r.verifier_id, "evidence_hash": r.evidence_hash,
                       "stderr_headline": error_headline(r.stderr_tail)[:300]}
                cache[key] = row
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    counts = defaultdict(int)
    for key, uses in pending.items():
        counts[cache.get(key, {}).get("status", "missing")] += len(uses)
    log(f"verify: candidate outcomes {dict(counts)}")
    paths.mark("verify", {"outcomes": dict(counts)})


def verification_lookup(paths: Paths, tasks: list[Task], hidden: dict[str, HiddenRecord]) -> dict[tuple[str, str, int], dict]:
    cache = {r["key"]: r for r in read_jsonl(paths.verify)} if paths.verify.exists() else {}
    by_id = {t.task_id: t for t in tasks}
    out: dict[tuple[str, str, int], dict] = {}
    for (task_id, source), rec in candidate_records(paths).items():
        t = by_id.get(task_id)
        if t is None:
            continue
        for i, c in enumerate(rec["candidates"]):
            key = _h(t.env_id, c["code"], hidden[task_id].test_code)
            out[(task_id, source, i)] = cache.get(key, {"status": "missing", "evidence_hash": "", "execution_ms": 0.0, "syntax_ok": False, "stderr_headline": ""})
    return out


# --------------------------------------------------------------------------- embed
def embedding_key(task_id: str, source: str, index: int) -> str:
    return f"{task_id}|{source}|{index}"


def stage_embed(cfg: dict, paths: Paths, log: Log, tasks: list[Task], embedder) -> dict[str, np.ndarray]:
    existing: dict[str, np.ndarray] = {}
    if paths.embeddings.exists():
        with np.load(paths.embeddings) as z:
            existing = {k: z[k] for k in z.files}
    by_id = {t.task_id: t for t in tasks}
    keys, texts = [], []
    for (task_id, source), rec in candidate_records(paths).items():
        t = by_id.get(task_id)
        if t is None:
            continue
        for i, c in enumerate(rec["candidates"]):
            k = embedding_key(task_id, source, i)
            if k in existing:
                continue
            keys.append(k)
            texts.append(joint_text(t.package, t.package_version, t.request_code, t.request_stderr, c["code"]))
    log(f"embed: {len(existing)} cached, {len(texts)} new joint texts")
    if texts:
        vecs = embedder.embed(texts)
        for k, v in zip(keys, vecs):
            existing[k] = v.astype(np.float32)
    if texts or not paths.embeddings.exists():
        np.savez_compressed(paths.embeddings, **existing)
    paths.mark("embed", {"n": len(existing), **embedder.metadata()})
    return existing
