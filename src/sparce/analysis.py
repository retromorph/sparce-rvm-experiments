"""E1/E2/E3 analyses over cached candidates, verifier outcomes and embeddings.

Hyperparameters are selected on the *validation* split only; the test split is
touched once per (method, seed) after selection. All paired intervals resample
clusters (programs), never individual near-duplicate variants.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit

from sparce.dataset import Task
from sparce.ledger import SnapshotLedger
from sparce.memory import Decision, PairExample, cold_evidence_bytes, identifier_tokens, make_method
from sparce.stats import (brier_score, cluster_bootstrap_proportion_upper, expected_calibration_error,
                          paired_accuracy_delta_interval, percentile, wilson_upper_onesided_95)

CONTROL_KINDS = ("previously_correct", "old_version", "near_api", "near_api_old")


# --------------------------------------------------------------------------- pair construction
@dataclass
class TaskPair:
    ex: PairExample
    task: Task
    source: str
    base_index: int
    alt_index: int
    cand_pass: list[bool | None]
    tokens_prompt: int
    tokens_completion: int
    gen_ms: float
    verifier_ms: float
    extracted_ok: list[bool]

    @property
    def recall_at_k(self) -> bool:
        return any(p is True for p in self.cand_pass)

    @property
    def base_correct(self) -> bool:
        return self.ex.b_pass is True


def _status_to_bool(status: str) -> bool | None:
    return True if status == "pass" else False if status == "fail" else None


def build_pairs(tasks: Sequence[Task], cand_records: dict, ver: dict, emb: dict, source: str = "base") -> list[TaskPair]:
    from sparce.pipeline import embedding_key
    pairs: list[TaskPair] = []
    for t in tasks:
        rec = cand_records.get((t.task_id, source))
        if rec is None:
            continue
        cands = rec["candidates"]
        scores = [c.get("score") for c in cands]
        scores = [(-1e9 if (s is None or (isinstance(s, float) and math.isnan(s))) else float(s)) for s in scores]
        base_i = int(np.argmax(scores))  # argmax returns the first (greedy) on ties
        alt_i = 1 if base_i == 0 else 0
        statuses = [ver.get((t.task_id, source, i), {}) for i in range(len(cands))]
        cand_pass = [_status_to_bool(s.get("status", "missing")) for s in statuses]
        ea = emb.get(embedding_key(t.task_id, source, alt_i))
        eb = emb.get(embedding_key(t.task_id, source, base_i))
        if ea is None or eb is None:
            continue
        evidence = sum(len(c["code"]) + len(c["text"]) for c in cands) + len(t.request_code) + len(t.request_stderr) + 400
        ex = PairExample(
            task_id=t.task_id, cluster_id=t.cluster_id, family_id=t.family_id, kind=t.kind, split=t.split, subset=t.subset,
            scope=f"{t.package}=={t.package_version}", timestamp=t.timestamp, emb_a=np.asarray(ea), emb_b=np.asarray(eb),
            d0=float(scores[alt_i] - scores[base_i]) if scores[alt_i] > -1e8 else -5.0,
            a_pass=cand_pass[alt_i], b_pass=cand_pass[base_i],
            request_tokens=identifier_tokens(t.request_code), tokens_a=identifier_tokens(cands[alt_i]["code"]),
            tokens_b=identifier_tokens(cands[base_i]["code"]), evidence_bytes=int(evidence),
            evidence_hash="|".join(s.get("evidence_hash", "") for s in statuses))
        pairs.append(TaskPair(ex, t, source, base_i, alt_i, cand_pass,
                              sum(c["prompt_tokens"] for c in cands), sum(c["completion_tokens"] for c in cands),
                              sum(c["gen_ms"] for c in cands), sum(float(s.get("execution_ms", 0.0)) for s in statuses),
                              [bool(c.get("extracted_ok", False)) for c in cands]))
    return pairs


# --------------------------------------------------------------------------- evaluation primitives
@dataclass
class Outcome:
    task_id: str
    cluster_id: str
    kind: str
    subset: str
    base_correct: bool
    correct: bool
    recommended: bool
    reason: str
    probability: float
    label: int | None
    latency_ms: float
    margin: float
    lower: float


def evaluate(method, pairs: Sequence[TaskPair]) -> list[Outcome]:
    out: list[Outcome] = []
    for p in pairs:
        start = time.perf_counter()
        d = method.decide(p.ex)
        lat = (time.perf_counter() - start) * 1000.0
        pick_pass = p.ex.a_pass if d.recommend_alternative else p.ex.b_pass
        out.append(Outcome(p.ex.task_id, p.ex.cluster_id, p.ex.kind, p.ex.subset, p.base_correct, pick_pass is True,
                           d.recommend_alternative, d.reason, float(d.probability), p.ex.label, lat, float(d.margin), float(d.lower)))
    return out


def summarize(outcomes: Sequence[Outcome], *, n_boot: int = 2000, seed: int = 42, with_ci: bool = True) -> dict:
    n = len(outcomes)
    if n == 0:
        return {"n": 0}
    correct = np.array([o.correct for o in outcomes])
    base = np.array([o.base_correct for o in outcomes])
    clusters = [o.cluster_id for o in outcomes]
    res = {
        "n": n, "n_clusters": len(set(clusters)),
        "pass_at_1": float(correct.mean()), "base_pass_at_1": float(base.mean()),
        "n_correct": int(correct.sum()), "n_base_correct": int(base.sum()),
        "fixes": int((~base & correct).sum()), "regressions": int((base & ~correct).sum()),
        "activations": int(sum(o.recommended for o in outcomes)),
        "latency_p50_ms": percentile([o.latency_ms for o in outcomes], 50),
        "latency_p95_ms": percentile([o.latency_ms for o in outcomes], 95),
    }
    if with_ci and n >= 2 and len(set(clusters)) >= 2:
        point, lo, hi = paired_accuracy_delta_interval(base, correct, cluster_ids=clusters, seed=seed, n_boot=n_boot)
        res.update({"delta_vs_base": point, "delta_ci_low": lo, "delta_ci_high": hi})
    return res


def paired_delta(a: Sequence[Outcome], b: Sequence[Outcome], *, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired difference accuracy(a) - accuracy(b) on the shared task set."""
    bm = {o.task_id: o for o in b}
    shared = [(o, bm[o.task_id]) for o in a if o.task_id in bm]
    if len(shared) < 2:
        return {"n": len(shared)}
    x = [oa.correct for oa, _ in shared]
    y = [ob.correct for _, ob in shared]
    clusters = [oa.cluster_id for oa, _ in shared]
    if len(set(clusters)) < 2:
        return {"n": len(shared), "delta": float(np.mean(x) - np.mean(y))}
    point, lo, hi = paired_accuracy_delta_interval(y, x, cluster_ids=clusters, seed=seed, n_boot=n_boot)
    return {"n": len(shared), "delta": point, "ci_low": lo, "ci_high": hi}


def by_subset(outcomes: Sequence[Outcome], key: str, **kw) -> dict:
    groups: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        groups[getattr(o, key)].append(o)
    return {k: summarize(v, **kw) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------- tuning (validation only)
def _grid(cfg_section: dict, names: dict[str, str]) -> list[dict]:
    axes = {p: cfg_section.get(g, [None]) for p, g in names.items()}
    keys = list(axes)
    combos = []
    for values in itertools.product(*[axes[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, values) if v is not None})
    return combos


def tune(method_name: str, grid: list[dict], train: Sequence[TaskPair], val: Sequence[TaskPair], *, capacity=None, seed=0) -> dict:
    """Pick the grid point with the best validation accuracy over ALL kinds (repairs + controls)."""
    records = []
    best = None
    for params in grid:
        m = make_method(method_name, capacity=capacity, seed=seed, params=params)
        m.fit([p.ex for p in train], seed=seed)
        s = summarize(evaluate(m, val), with_ci=False)
        rec = {"params": params, "val_pass_at_1": s["pass_at_1"], "regressions": s["regressions"], "fixes": s["fixes"],
               "activations": s["activations"], "active_count": m.active_count}
        records.append(rec)
        key = (s["pass_at_1"], -s["regressions"])
        if best is None or key > best[0]:
            best = (key, params)
    return {"best": best[1] if best else {}, "grid": records}


# --------------------------------------------------------------------------- E1
@dataclass
class DataBundle:
    base: list[TaskPair]
    rag: list[TaskPair]
    cfg: dict

    def split(self, name: str, source: str = "base") -> list[TaskPair]:
        pool = self.base if source == "base" else self.rag
        return [p for p in pool if p.ex.split == name]


def _repair(pairs: Sequence[TaskPair]) -> list[TaskPair]:
    return [p for p in pairs if p.ex.kind == "repair"]


def _filter(outs: Sequence[Outcome], kinds: Sequence[str]) -> list[Outcome]:
    return [o for o in outs if o.kind in kinds]


def run_e1(data: DataBundle, log, seeds: Sequence[int]) -> dict:
    cfg = data.cfg
    n_boot = int(cfg["eval"]["bootstrap_replicates"])
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    rvm_grid = _grid(cfg["rvm"], {"length_scale": "length_scale_grid", "min_support": "min_support_grid",
                                  "z_score": "posterior_z_grid", "margin_threshold": "margin_threshold_grid"})
    knn_grid = _grid(cfg["knn"], {"k": "k_grid", "threshold": "threshold_grid", "min_support": "min_support_grid"})
    log(f"E1: tuning on validation (train={len(train)}, val={len(val)}, test={len(test)}); rvm grid {len(rvm_grid)}, knn grid {len(knn_grid)}")
    tuned = {"rvm": tune("rvm", rvm_grid, train, val, capacity=int(cfg["rvm"]["max_basis"])),
             "knn": tune("knn_full", knn_grid, train, val)}
    log(f"E1: tuned rvm={tuned['rvm']['best']} knn={tuned['knn']['best']}")
    results: dict = {"tuned": tuned, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                     "n_test_repair": len(_repair(test)), "n_test_repair_clusters": len({p.ex.cluster_id for p in _repair(test)}),
                     "methods": {}, "per_task": {}, "seeds": list(seeds)}
    test_repair = _repair(test)
    results["candidate_recall_at_k"] = float(np.mean([p.recall_at_k for p in test_repair])) if test_repair else float("nan")
    results["candidate_recall_at_k_known"] = float(np.mean([p.recall_at_k for p in test_repair if p.ex.subset == "known_family"]) if any(p.ex.subset == "known_family" for p in test_repair) else float("nan"))
    results["candidate_recall_at_k_heldout"] = float(np.mean([p.recall_at_k for p in test_repair if p.ex.subset == "heldout_family"]) if any(p.ex.subset == "heldout_family" for p in test_repair) else float("nan"))
    results["greedy_pass_at_1"] = float(np.mean([p.cand_pass[0] is True for p in test_repair])) if test_repair else float("nan")
    results["tokens_per_task"] = {"base": float(np.mean([p.tokens_prompt + p.tokens_completion for p in test])) if test else 0.0}
    results["gen_ms_per_task_p95"] = percentile([p.gen_ms for p in test], 95)
    results["verifier_ms_per_task_mean"] = float(np.mean([p.verifier_ms for p in test])) if test else 0.0
    results["train_events_with_pair_label"] = sum(1 for p in train if p.ex.label is not None)
    results["cold_evidence_bytes"] = cold_evidence_bytes([p.ex for p in train])

    base_outs = evaluate(make_method("base"), test)
    method_names = [m for m in cfg["eval"]["e1_methods"] if m != "rag"]
    if "rvm" in method_names and "knn_cap" in method_names:  # kNN at the RVM's active count needs the RVM first
        method_names.remove("rvm")
        method_names.insert(method_names.index("knn_cap"), "rvm")
    rvm_active = None
    for seed in seeds:
        for name in method_names:
            params = dict(tuned["rvm"]["best"]) if name.startswith("rvm") else (dict(tuned["knn"]["best"]) if name.startswith("knn") else {})
            capacity = int(cfg["rvm"]["max_basis"]) if name.startswith("rvm") else (rvm_active if name == "knn_cap" else None)
            if name == "knn_cap" and rvm_active is None:
                continue
            m = make_method(name, capacity=capacity, seed=seed, params=params)
            m.fit([p.ex for p in train], seed=seed)
            if name == "rvm":
                rvm_active = max(1, m.active_count)
            outs = evaluate(m, test)
            entry = {"seed": seed, "active_count": m.active_count, "bytes": m.bytes_breakdown(), "fit_seconds": m.fit_seconds,
                     "params": params, "capacity": capacity,
                     "repair": summarize(_filter(outs, ["repair"]), n_boot=n_boot, seed=seed),
                     "repair_by_subset": by_subset(_filter(outs, ["repair"]), "subset", n_boot=n_boot, seed=seed),
                     "controls": summarize(_filter(outs, CONTROL_KINDS), n_boot=n_boot, seed=seed),
                     "all": summarize(outs, n_boot=n_boot, seed=seed),
                     "reasons": dict(sorted(defaultdict(int, {o.reason: sum(1 for x in outs if x.reason == o.reason) for o in outs}).items()))}
            if hasattr(m, "fit_log"):
                entry["fit_log"] = m.fit_log
            results["methods"].setdefault(name, []).append(entry)
            results["per_task"].setdefault(name, {})[str(seed)] = [asdict(o) for o in outs]
            log(f"E1[{name} seed={seed}] repair pass@1={entry['repair'].get('pass_at_1', float('nan')):.3f} "
                f"(base {entry['repair'].get('base_pass_at_1', float('nan')):.3f}) fixes={entry['repair'].get('fixes')} "
                f"regr={entry['repair'].get('regressions')} active={m.active_count}")
    # RAG generator baseline (its own candidate set); paired against base picks on shared tasks
    rag_test = data.split("test", "rag")
    if rag_test:
        rag_outs = evaluate(make_method("base"), rag_test)
        rag_repair = _filter(rag_outs, ["repair"])
        base_map = {o.task_id: o for o in base_outs}
        paired = [Outcome(o.task_id, o.cluster_id, o.kind, o.subset, base_map[o.task_id].correct, o.correct, True, "rag",
                          o.probability, o.label, o.latency_ms, o.margin, o.lower) for o in rag_outs if o.task_id in base_map]
        entry = {"seed": seeds[0], "active_count": 0, "bytes": {"total": 0}, "fit_seconds": 0.0, "params": {},
                 "repair": summarize(_filter(paired, ["repair"]), n_boot=n_boot),
                 "repair_by_subset": by_subset(_filter(paired, ["repair"]), "subset", n_boot=n_boot),
                 "controls": summarize(_filter(paired, CONTROL_KINDS), n_boot=n_boot), "all": summarize(paired, n_boot=n_boot),
                 "tokens_per_task": float(np.mean([p.tokens_prompt + p.tokens_completion for p in rag_test])),
                 "rag_recall_at_k": float(np.mean([p.recall_at_k for p in rag_test if p.ex.kind == "repair"])) if rag_repair else float("nan")}
        results["methods"]["rag"] = [entry]
        results["per_task"]["rag"] = {str(seeds[0]): [asdict(o) for o in paired]}
        results["tokens_per_task"]["rag"] = entry["tokens_per_task"]
        log(f"E1[rag] repair pass@1={entry['repair'].get('pass_at_1', float('nan')):.3f} tokens/task={entry['tokens_per_task']:.0f}")
    # decision
    results["decision"] = e1_decision(results, cfg, seeds[0])
    return results


def e1_decision(results: dict, cfg: dict, seed: int) -> dict:
    n_min = int(cfg["eval"]["min_test_cases"])
    rvm = next((e for e in results["methods"].get("rvm", []) if e["seed"] == seed), None)
    if rvm is None:
        return {"status": "INCONCLUSIVE", "why": "no SPaRCE result"}
    rep = rvm["repair"]
    n = rep.get("n", 0)
    gain = rep.get("delta_vs_base", float("nan")) * 100
    lo = rep.get("delta_ci_low", float("nan")) * 100
    comps = {}
    for name in ("rules", "rag", "knn_full"):
        e = next((x for x in results["methods"].get(name, []) if x["seed"] == seed or name == "rag"), None)
        if e:
            comps[name] = e["repair"].get("pass_at_1", float("nan"))
    strongest = max(comps, key=comps.get) if comps else None
    margin_vs_strongest = (rep["pass_at_1"] - comps[strongest]) * 100 if strongest else float("nan")
    checks = {"n_ge_min": n >= n_min, "gain_ge_5pp": gain >= 5.0, "ci_lower_gt_0": lo > 0,
              "within_2pp_of_strongest_comparator": (margin_vs_strongest >= -2.0) if strongest else False}
    if not checks["n_ge_min"]:
        status = "INCONCLUSIVE"
    elif all(checks.values()):
        status = "GO"
    elif rep.get("delta_ci_high", 1.0) * 100 < 5.0:
        status = "NO-GO"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "n_test_repair": n, "min_n": n_min, "gain_pp": gain, "ci_low_pp": lo,
            "ci_high_pp": rep.get("delta_ci_high", float("nan")) * 100, "strongest_comparator": strongest,
            "strongest_comparator_pass_at_1": comps.get(strongest) if strongest else None,
            "margin_vs_strongest_pp": margin_vs_strongest, "checks": checks, "comparators": comps}


# --------------------------------------------------------------------------- E2
def _pareto_front(points: list[dict]) -> list[dict]:
    """Non-dominated set: maximise accuracy, minimise bytes and latency p95."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q["accuracy"] >= p["accuracy"] and q["bytes"] <= p["bytes"] and q["latency_p95_ms"] <= p["latency_p95_ms"]
                    and (q["accuracy"] > p["accuracy"] or q["bytes"] < p["bytes"] or q["latency_p95_ms"] < p["latency_p95_ms"])):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def run_e2(data: DataBundle, log, seeds: Sequence[int], tuned: dict) -> dict:
    cfg = data.cfg
    n_boot = int(cfg["eval"]["bootstrap_replicates"])
    train, test = data.split("train"), data.split("test")
    test_repair = _repair(test)
    caps = [int(c) for c in cfg["eval"]["capacities"]]
    pool_size = len(set((p.ex.scope, p.ex.emb_a.tobytes()) for p in train) | set((p.ex.scope, p.ex.emb_b.tobytes()) for p in train))
    caps = [c for c in caps if c <= max(pool_size, 8)]
    rows: list[dict] = []
    outs_store: dict[tuple, list[Outcome]] = {}
    full = make_method("knn_full", params=dict(tuned["knn"]["best"]))
    full.fit([p.ex for p in train], seed=seeds[0])
    full_outs = evaluate(full, test)
    full_rep = summarize(_filter(full_outs, ["repair"]), n_boot=n_boot)
    full_row = {"method": "knn_full", "capacity": None, "seed": seeds[0], "active_count": full.active_count,
                "bytes": full.bytes_breakdown(), "accuracy": full_rep["pass_at_1"], "all_accuracy": summarize(full_outs, with_ci=False)["pass_at_1"],
                "latency_p50_ms": full_rep["latency_p50_ms"], "latency_p95_ms": full_rep["latency_p95_ms"],
                "fit_seconds": full.fit_seconds, "controls_regressions": summarize(_filter(full_outs, CONTROL_KINDS), with_ci=False).get("regressions")}
    rows.append(full_row)
    outs_store[("knn_full", None, seeds[0])] = full_outs
    per_proto_bytes = full.bytes_breakdown()["total"] / max(1, full.active_count)
    for cap in caps:
        for seed in seeds:
            for name in cfg["eval"]["e2_methods"]:
                params = dict(tuned["rvm"]["best"]) if name.startswith("rvm") else dict(tuned["knn"]["best"])
                m = make_method(name, capacity=cap, seed=seed, params=params)
                t0 = time.perf_counter()
                m.fit([p.ex for p in train], seed=seed)
                fit_s = time.perf_counter() - t0
                # update cost: refit after one more verified event (warm data, cold start)
                t1 = time.perf_counter()
                m2 = make_method(name, capacity=cap, seed=seed, params=params)
                m2.fit([p.ex for p in train[:-1]], seed=seed)
                update_s = fit_s  # conservative: a full refit is the documented update procedure
                outs = evaluate(m, test)
                rep = summarize(_filter(outs, ["repair"]), n_boot=n_boot, seed=seed)
                row = {"method": name, "capacity": cap, "seed": seed, "active_count": m.active_count, "bytes": m.bytes_breakdown(),
                       "accuracy": rep["pass_at_1"], "all_accuracy": summarize(outs, with_ci=False)["pass_at_1"],
                       "latency_p50_ms": rep["latency_p50_ms"], "latency_p95_ms": rep["latency_p95_ms"], "fit_seconds": fit_s,
                       "update_seconds": update_s, "controls_regressions": summarize(_filter(outs, CONTROL_KINDS), with_ci=False).get("regressions"),
                       "delta_vs_base": rep.get("delta_vs_base"), "delta_ci_low": rep.get("delta_ci_low"), "delta_ci_high": rep.get("delta_ci_high")}
                rows.append(row)
                outs_store[(name, cap, seed)] = outs
                log(f"E2[{name} cap={cap} seed={seed}] acc={row['accuracy']:.3f} active={m.active_count} bytes={row['bytes']['total']} lat95={row['latency_p95_ms']:.2f}ms")
            # bytes-matched kNN: as many prototypes as fit in the RVM's active bytes
            rvm_row = next(r for r in rows if r["method"] == "rvm" and r["capacity"] == cap and r["seed"] == seed)
            bm_cap = max(1, int(rvm_row["bytes"]["total"] // max(1.0, per_proto_bytes)))
            m = make_method("knn_cap", capacity=bm_cap, seed=seed, params=dict(tuned["knn"]["best"]))
            m.fit([p.ex for p in train], seed=seed)
            outs = evaluate(m, test)
            rep = summarize(_filter(outs, ["repair"]), n_boot=n_boot, seed=seed)
            rows.append({"method": "knn_bytes_matched", "capacity": cap, "seed": seed, "active_count": m.active_count, "bytes": m.bytes_breakdown(),
                         "accuracy": rep["pass_at_1"], "all_accuracy": summarize(outs, with_ci=False)["pass_at_1"],
                         "latency_p50_ms": rep["latency_p50_ms"], "latency_p95_ms": rep["latency_p95_ms"], "fit_seconds": m.fit_seconds,
                         "controls_regressions": summarize(_filter(outs, CONTROL_KINDS), with_ci=False).get("regressions")})
            outs_store[("knn_bytes_matched", cap, seed)] = outs
    # paired comparisons rvm vs capacity-matched / bytes-matched kNN at each capacity (seed 0 CI; mean over seeds)
    comparisons = []
    for cap in caps:
        for comp in ("knn_cap", "knn_bytes_matched", "random", "medoids"):
            per_seed = []
            for seed in seeds:
                a = _filter(outs_store[("rvm", cap, seed)], ["repair"])
                b = _filter(outs_store[(comp, cap, seed)], ["repair"])
                per_seed.append(paired_delta(a, b, n_boot=n_boot, seed=seed))
            comparisons.append({"capacity": cap, "comparator": comp, "seed0": per_seed[0],
                                "mean_delta": float(np.mean([d.get("delta", float("nan")) for d in per_seed])),
                                "per_seed": per_seed})
    points = [{"method": r["method"], "capacity": r["capacity"], "seed": r["seed"], "accuracy": r["accuracy"],
               "bytes": r["bytes"]["total"], "latency_p95_ms": r["latency_p95_ms"], "active_count": r["active_count"]}
              for r in rows if r["seed"] == seeds[0]]
    front = _pareto_front(points)
    # decision
    improvement = [c for c in comparisons if c["comparator"] == "knn_cap" and c["seed0"].get("delta", -1) * 100 >= 2.0 and c["seed0"].get("ci_low", -1) > 0]
    improvement_bytes = [c for c in comparisons if c["comparator"] == "knn_bytes_matched" and c["seed0"].get("delta", -1) * 100 >= 2.0 and c["seed0"].get("ci_low", -1) > 0]
    rvm_on_front = any(p["method"] == "rvm" for p in front)
    compression = [r for r in rows if r["method"] == "rvm" and r["seed"] == seeds[0]
                   and r["accuracy"] >= full_row["accuracy"] - 0.01 and r["bytes"]["total"] <= 0.25 * full_row["bytes"]["total"]]
    status = "GO" if (improvement and improvement_bytes and rvm_on_front and compression) else "NO-GO"
    if len(test_repair) < int(cfg["eval"]["min_test_cases"]):
        status = "INCONCLUSIVE"
    return {"capacities": caps, "pool_size": pool_size, "rows": rows, "comparisons": comparisons, "pareto_front": front,
            "knn_full": full_row, "per_prototype_bytes": per_proto_bytes, "n_test_repair": len(test_repair),
            "decision": {"status": status, "improvement_capacities_count_matched": [c["capacity"] for c in improvement],
                         "improvement_capacities_bytes_matched": [c["capacity"] for c in improvement_bytes],
                         "rvm_non_dominated": rvm_on_front,
                         "compression_capacities": [r["capacity"] for r in compression],
                         "knn_full_accuracy": full_row["accuracy"], "knn_full_bytes": full_row["bytes"]["total"]},
            "per_task": {f"{k[0]}|{k[1]}|{k[2]}": [asdict(o) for o in v] for k, v in outs_store.items() if k[2] == seeds[0]}}


# --------------------------------------------------------------------------- E3
def _collateral(outs: Sequence[Outcome], seed: int, n_boot: int) -> dict:
    base_ok = [o for o in outs if o.base_correct]
    regress = [not o.correct for o in base_ok]
    n_bad = int(sum(regress))
    n = len(base_ok)
    res = {"n_base_correct": n, "n_regressions": n_bad, "rate": (n_bad / n) if n else float("nan"),
           "wilson_upper_95": wilson_upper_onesided_95(n_bad, n) if n else float("nan"),
           "cluster_bootstrap_upper_95": cluster_bootstrap_proportion_upper(regress, [o.cluster_id for o in base_ok], seed=seed, n_boot=n_boot) if n >= 2 else float("nan"),
           "spurious_activations": int(sum(o.recommended for o in outs)), "n_controls": len(outs)}
    return res


def _calibration(outs: Sequence[Outcome]) -> dict:
    labelled = [o for o in outs if o.label is not None]
    if len(labelled) < 10:
        return {"n": len(labelled), "note": "fewer than 10 labelled pairs"}
    p = [min(1.0, max(0.0, o.probability)) for o in labelled]
    y = [o.label for o in labelled]
    ece = expected_calibration_error(p, y)
    return {"n": len(labelled), "brier": brier_score(p, y), "ece": ece["ece"], "bins": ece["bins"]}


def admit_event(ex: PairExample) -> tuple[bool, str]:
    """Independent-verification admission check for a submitted memory event."""
    if not ex.verified:
        return False, "unverified_submission"
    if not ex.evidence_hash or "sha256:" not in ex.evidence_hash:
        return False, "missing_verifier_evidence"
    if ex.a_pass is None and ex.b_pass is None:
        return False, "no_verified_outcome"
    return True, "verified"


def run_e3(data: DataBundle, log, seeds: Sequence[int], tuned: dict, work_dir: Path) -> dict:
    cfg = data.cfg
    n_boot = int(cfg["eval"]["bootstrap_replicates"])
    seed = seeds[0]
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    controls = [p for p in val + test if p.ex.kind in CONTROL_KINDS]
    train_controls = [p for p in train if p.ex.kind in CONTROL_KINDS]
    rvm_params = dict(tuned["rvm"]["best"])
    knn_params = dict(tuned["knn"]["best"])
    cap = int(cfg["rvm"]["max_basis"])
    results: dict = {"n_controls": len(controls), "n_control_clusters": len({p.ex.cluster_id for p in controls}),
                     "n_train_controls": len(train_controls), "methods": {}}
    # 1. collateral on untouched controls for the tuned methods
    for name in ("rvm", "rvm_noscope", "rvm_nogate", "knn_full", "rules"):
        params = rvm_params if name.startswith("rvm") else (knn_params if name.startswith("knn") else {})
        m = make_method(name, capacity=cap if name.startswith("rvm") else None, seed=seed, params=params)
        m.fit([p.ex for p in train], seed=seed)
        outs = evaluate(m, controls)
        entry = {"collateral": _collateral(outs, seed, n_boot),
                 "collateral_by_kind": {k: _collateral([o for o in outs if o.kind == k], seed, n_boot) for k in CONTROL_KINDS},
                 "train_cluster_controls": _collateral(evaluate(m, train_controls), seed, n_boot) if train_controls else {},
                 "calibration_test_pairs": _calibration(evaluate(m, test))}
        results["methods"][name] = entry
        log(f"E3[{name}] collateral {entry['collateral']['n_regressions']}/{entry['collateral']['n_base_correct']} "
            f"rate={entry['collateral']['rate']:.4f} wilson95={entry['collateral']['wilson_upper_95']:.4f}")
    rag_test = [p for p in data.split("val", "rag") + data.split("test", "rag") if p.ex.kind in CONTROL_KINDS]
    if rag_test:
        base_map = {o.task_id: o for o in evaluate(make_method("base"), controls)}
        rag_outs = evaluate(make_method("base"), rag_test)
        paired = [Outcome(o.task_id, o.cluster_id, o.kind, o.subset, base_map[o.task_id].correct, o.correct, True, "rag", o.probability, o.label, o.latency_ms, o.margin, o.lower)
                  for o in rag_outs if o.task_id in base_map]
        results["methods"]["rag"] = {"collateral": _collateral(paired, seed, n_boot)}
    results["methods"]["base"] = {"calibration_test_pairs": _calibration(evaluate(make_method("base"), test))}

    # 2. sequential verified updates with gated admission, snapshots, retention, rollback
    stream = sorted(train, key=lambda p: (p.ex.timestamp, p.ex.task_id))
    checkpoints = [c for c in cfg["eval"]["update_checkpoints"] if c <= len(stream)]
    if checkpoints and checkpoints[-1] < len(stream) and len(stream) not in checkpoints:
        checkpoints.append(len(stream))
    ledger_path = work_dir / "results" / "e3_ledger.sqlite"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = SnapshotLedger(ledger_path)
    battery = test + controls
    base_val = summarize(evaluate(make_method("base"), val), with_ci=False)
    stream_rows = []
    promoted_versions: dict[int, int] = {}
    decisions_at: dict[int, list[dict]] = {}
    first_fixed: set[str] | None = None
    first_train_fixed: set[str] | None = None
    current = make_method("rvm", capacity=cap, seed=seed, params=rvm_params)
    current_state = None
    for ck in checkpoints:
        events = [p.ex for p in stream[:ck]]
        admitted, quarantined = [], []
        for e in events:
            ok, why = admit_event(e)
            (admitted if ok else quarantined).append((e, why))
        cand = make_method("rvm", capacity=cap, seed=seed, params=rvm_params)
        t0 = time.perf_counter()
        cand.fit([e for e, _ in admitted], seed=seed)
        fit_s = time.perf_counter() - t0
        val_s = summarize(evaluate(cand, val), with_ci=False)
        val_budget = max(1, int(0.01 * max(1, val_s["n"])))
        benefit_ok = val_s["pass_at_1"] >= base_val["pass_at_1"]
        safety_ok = val_s["regressions"] <= val_budget
        promote = benefit_ok and safety_ok and cand.active_count > 0
        row = {"checkpoint": ck, "n_admitted": len(admitted), "n_quarantined": len(quarantined), "fit_seconds": fit_s,
               "val_pass_at_1": val_s["pass_at_1"], "val_regressions": val_s["regressions"], "val_budget": val_budget,
               "benefit_ok": benefit_ok, "safety_ok": safety_ok, "promoted": promote, "active_count": cand.active_count,
               "bytes": cand.bytes_breakdown()}
        if promote:
            current = cand
            current_state = cand.snapshot()
            version = ledger.commit(current_state, reason=f"checkpoint_{ck}", evidence_ids=[e.task_id for e, _ in admitted])
            promoted_versions[ck] = version
            row["snapshot_version"] = version
            row["snapshot_bytes"] = ledger.payload_bytes(version)
        outs = evaluate(current, battery)
        rep = _filter(outs, ["repair"])
        ctl = [o for o in outs if o.kind in CONTROL_KINDS]
        fixed_now = {o.task_id for o in rep if o.correct and not o.base_correct}
        train_outs = evaluate(current, stream[: checkpoints[0]])
        train_fixed_now = {o.task_id for o in train_outs if o.correct}
        if first_fixed is None and promote:
            first_fixed, first_train_fixed = fixed_now, train_fixed_now
        row.update({"test_repair": summarize(rep, with_ci=False), "collateral": _collateral(ctl, seed, n_boot),
                    "calibration": _calibration(outs),
                    "retention_test_fixed": (len(fixed_now & first_fixed) / len(first_fixed)) if first_fixed else float("nan"),
                    "n_first_fixed": len(first_fixed) if first_fixed else 0,
                    "retention_first_events": (len(train_fixed_now & first_train_fixed) / len(first_train_fixed)) if first_train_fixed else float("nan")})
        decisions_at[ck] = [{"task_id": o.task_id, "recommended": o.recommended, "margin": o.margin, "lower": o.lower, "probability": o.probability} for o in outs]
        stream_rows.append(row)
        log(f"E3 stream ck={ck}: admitted={len(admitted)} promoted={promote} test pass@1={row['test_repair'].get('pass_at_1', float('nan')):.3f} "
            f"collateral={row['collateral']['n_regressions']}/{row['collateral']['n_base_correct']} retention={row['retention_test_fixed']:.3f}")

    # 3. false / unverified submissions must be quarantined
    n_false = int(cfg["eval"].get("false_submissions", 20))
    rng = np.random.default_rng(seed)
    false_events = []
    for p in rng.choice(stream, size=min(n_false, len(stream)), replace=False):
        e = p.ex
        false_events.append(PairExample(**{**asdict_pair(e), "a_pass": True, "b_pass": False, "verified": False, "evidence_hash": ""}))
    false_admitted = sum(1 for e in false_events if admit_event(e)[0])
    # ablation: what would happen without the admission check
    forced = make_method("rvm", capacity=cap, seed=seed, params=rvm_params)
    forced_events = [p.ex for p in stream] + [PairExample(**{**asdict_pair(e), "verified": True, "evidence_hash": "sha256:forged"}) for e in false_events]
    forced.fit(forced_events, seed=seed)
    forced_ctl = evaluate(forced, controls)
    results["false_submissions"] = {"n_submitted": len(false_events), "n_admitted": false_admitted,
                                    "without_admission_check_collateral": _collateral(forced_ctl, seed, n_boot)}

    # 4. exact rollback: restore the earliest promoted full posterior snapshot and replay the battery
    rollback = {"performed": False}
    if promoted_versions:
        first_ck = min(promoted_versions)
        last_ck = max(promoted_versions)
        new_version = ledger.rollback(promoted_versions[first_ck])
        _, state = ledger.load(new_version)
        restored = make_method("rvm", capacity=cap, seed=seed, params=rvm_params)
        restored.restore(state)
        replay = evaluate(restored, battery)
        recorded = {d["task_id"]: d for d in decisions_at[first_ck]}
        mismatches = [{"task_id": o.task_id, "recorded": recorded[o.task_id], "replay": {"recommended": o.recommended, "margin": o.margin, "lower": o.lower}}
                      for o in replay if not (recorded[o.task_id]["recommended"] == o.recommended and recorded[o.task_id]["margin"] == o.margin
                                              and recorded[o.task_id]["lower"] == o.lower)]
        exact = len(replay) - len(mismatches)
        last_recorded = {d["task_id"]: d for d in decisions_at[last_ck]}
        differs_from_last = sum(1 for o in replay if last_recorded[o.task_id]["recommended"] != o.recommended or last_recorded[o.task_id]["margin"] != o.margin)
        rollback = {"performed": True, "from_checkpoint": last_ck, "to_checkpoint": first_ck, "ledger_version": new_version,
                    "n_battery": len(replay), "n_exact_matches": exact, "exact": exact == len(replay),
                    "n_decisions_differing_from_last_checkpoint": differs_from_last, "mismatches": mismatches[:20],
                    "ledger_events": len(ledger.events()), "snapshot_bytes": ledger.payload_bytes(new_version)}
    ledger.close()
    results["stream"] = stream_rows
    results["checkpoints"] = checkpoints
    results["rollback"] = rollback
    results["decision"] = e3_decision(results, cfg)
    return results


def asdict_pair(e: PairExample) -> dict:
    d = asdict(e)
    return d


def e3_decision(results: dict, cfg: dict) -> dict:
    rvm = results["methods"].get("rvm", {}).get("collateral", {})
    upper = rvm.get("wilson_upper_95", float("nan"))
    cl_upper = rvm.get("cluster_bootstrap_upper_95", float("nan"))
    n_ctrl = rvm.get("n_base_correct", 0)
    stream = results.get("stream", [])
    ck100 = next((r for r in stream if r["checkpoint"] == 100), None)
    retention_loss_pp = (1.0 - ck100["retention_test_fixed"]) * 100 if ck100 and not math.isnan(ck100.get("retention_test_fixed", float("nan"))) else float("nan")
    rollback_exact = bool(results.get("rollback", {}).get("exact", False))
    false_ok = results.get("false_submissions", {}).get("n_admitted", 1) == 0
    powered = n_ctrl >= int(cfg["eval"]["min_negative_controls"])
    checks = {"collateral_upper_le_1pct": upper <= 0.01, "retention_loss_le_2pp_at_100": (retention_loss_pp <= 2.0) if not math.isnan(retention_loss_pp) else False,
              "rollback_exact": rollback_exact, "false_submissions_rejected": false_ok, "controls_powered": powered}
    if not powered or ck100 is None:
        status = "INCONCLUSIVE"
    elif all(checks.values()):
        status = "GO"
    elif (upper > 0.01 and rvm.get("rate", 0) > 0.01) or not rollback_exact or not false_ok or (not math.isnan(retention_loss_pp) and retention_loss_pp > 2.0):
        status = "NO-GO"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "collateral_rate": rvm.get("rate"), "collateral_wilson_upper": upper, "collateral_cluster_upper": cl_upper,
            "n_base_correct_controls": n_ctrl, "retention_loss_pp_at_100": retention_loss_pp, "checks": checks}
