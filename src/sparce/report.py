"""Generate reports/generated/SPARCE_POC_REPORT.md from result JSON files (template: docs/REPORT_TEMPLATE.md)."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path


def _f(x, nd=1, pct=True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.{nd}f}" if pct else f"{x:.{nd}f}"


def _ci(rep: dict) -> str:
    if "delta_vs_base" not in rep:
        return "n/a"
    return f"{rep['delta_vs_base']*100:+.1f} [{rep['delta_ci_low']*100:+.1f}, {rep['delta_ci_high']*100:+.1f}]"


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def failure_cases(work_dir: Path, e1: dict, n: int = 12) -> list[dict]:
    from sparce.dataset import read_jsonl
    tasks = {t["task_id"]: t for t in read_jsonl(work_dir / "data" / "tasks.jsonl")}
    seed = str(e1["seeds"][0])
    rvm = e1["per_task"].get("rvm", {}).get(seed, [])
    cands = {r["task_id"]: r for r in read_jsonl(work_dir / "cache" / "candidates.jsonl")}
    ver = {r["key"]: r for r in read_jsonl(work_dir / "cache" / "verify.jsonl")} if (work_dir / "cache" / "verify.jsonl").exists() else {}
    from sparce.pipeline import _h
    hidden = {h["task_id"]: h for h in read_jsonl(work_dir / "data" / "hidden" / "hidden.jsonl")}
    cases = []
    for o in rvm:
        if o["correct"] or o["kind"] != "repair":
            continue
        t = tasks[o["task_id"]]
        c = cands.get(o["task_id"], {}).get("candidates", [])
        statuses = [ver.get(_h(t["env_id"], x["code"], hidden[o["task_id"]]["test_code"]), {}).get("status", "missing") for x in c]
        heads = [ver.get(_h(t["env_id"], x["code"], hidden[o["task_id"]]["test_code"]), {}).get("stderr_headline", "") for x in c]
        if "pass" not in statuses:
            if any(s in ("timeout", "error") for s in statuses):
                category = "verifier_timeout_or_error"
            else:
                category = "correct_candidate_absent_from_K (generation ceiling)"
        elif o["reason"] == "scope_mismatch":
            category = "scope_gate_abstained (no memory partition for this exact scope)"
        elif o["recommended"]:
            category = "memory_selection_error (recommended a failing alternative)"
        else:
            category = f"memory_selection_miss (abstained: {o['reason']})"
        cases.append({"task_id": o["task_id"], "cluster": t["cluster_id"], "family": t["family_id"], "scope": f"{t['package']}=={t['package_version']}",
                      "subset": t["subset"], "category": category, "candidate_status": statuses, "candidate_errors": [h[:120] for h in heads],
                      "request_error": (t["request_stderr"].splitlines() or [""])[-1][:160], "base_correct": o["base_correct"],
                      "decision_reason": o["reason"], "margin": o["margin"], "lower": o["lower"], "provenance": t["change_url"]})
    # round-robin across failure categories, distinct families first
    by_cat: dict[str, list[dict]] = {}
    for c in sorted(cases, key=lambda c: (c["family"], c["cluster"])):
        by_cat.setdefault(c["category"], []).append(c)
    for cat, lst in by_cat.items():
        seen_f: set[str] = set()
        first = [c for c in lst if not (c["family"] in seen_f or seen_f.add(c["family"]))]
        by_cat[cat] = first + [c for c in lst if c not in first]
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop(0))
    return picked


def write_report(cfg: dict, work_dir: Path, out_dir: Path, mlflow_ids: dict, exploratory: Path | None = None) -> Path:
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = work_dir / "results"
    e1, e2, e3 = _load(res / "e1.json"), _load(res / "e2.json"), _load(res / "e3.json")
    env = _load(work_dir / "environment_manifest.json")
    split = _load(work_dir / "data" / "split_manifest.json")
    val = _load(work_dir / "data" / "catalog_validation.json")
    model = _load(work_dir / "model_manifest.json")
    emb = _load(work_dir / "embedding_manifest.json")
    envs = json.loads((work_dir / "data" / "environments.json").read_text()) if (work_dir / "data" / "environments.json").exists() else []
    stages = _load(work_dir / "stages.json")
    # copy artifacts next to the report
    for sub in ("plots",):
        src = res / sub
        if src.exists():
            shutil.copytree(src, out_dir / sub, dirs_exist_ok=True)
    for name in ("e1.json", "e2.json", "e3.json", "e1_summary.csv", "e2_rows.csv", "e3_stream.csv"):
        if (res / name).exists():
            shutil.copy(res / name, out_dir / name)
    for name in ("split_manifest.json", "catalog_validation.json", "environments.json"):
        if (work_dir / "data" / name).exists():
            shutil.copy(work_dir / "data" / name, out_dir / name)
    shutil.copy(work_dir / "environment_manifest.json", out_dir / "environment_manifest.json")

    seed0 = e1.get("seeds", [0])[0]
    d1, d2, d3 = e1.get("decision", {}), e2.get("decision", {}), e3.get("decision", {})
    overall = "NO-GO" if "NO-GO" in (d1.get("status"), d2.get("status"), d3.get("status")) else (
        "GO" if all(d.get("status") == "GO" for d in (d1, d2, d3)) else "INCONCLUSIVE")
    L: list[str] = []
    A = L.append
    A("# SPaRCE PoC — final research report\n")
    A(f"**Author:** coding agent (Claude Fable 5.1)  \n**Date:** {time.strftime('%Y-%m-%d')}  \n**Repository commit:** `{env.get('git_sha')}` (dirty={env.get('git_dirty')})  \n"
      f"**MLflow experiment:** `{cfg['run']['mlflow_experiment']}` at `{cfg['run']['mlflow_uri']}` (SSH-tunnelled; loopback only); run IDs: " + ", ".join(f"{k}=`{v}`" for k, v in mlflow_ids.items()) + "  \n"
      f"**Overall decision:** **{overall}** — E1 {d1.get('status')}, E2 {d2.get('status')}, E3 {d3.get('status')} (details in §9)\n")
    # 1
    rvm = (e1.get("methods", {}).get("rvm") or [{}])[0]
    rep = rvm.get("repair", {})
    A("## 1. Executive summary\n")
    A(f"We built an executable, version-pinned Python API-change repair benchmark ({split.get('n_tasks')} tasks, {split.get('n_clusters')} program clusters, "
      f"{split.get('n_families')} API families, {len(envs)} pinned Docker environments) from first-party programs, because CodeSyncBench ships no executable tests and no license (§3). "
      f"A frozen Qwen2.5-Coder-3B-Instruct generates K={cfg['model']['num_candidates']} candidates per task; every candidate is executed against hidden tests in a network-disabled, non-root container. "
      f"SPaRCE (scoped Laplace-ARD pairwise RVM with support and credible-margin gates) reranks the K candidates.\n")
    A(f"- **E1 (held-out verified repair):** SPaRCE pass@1 {_f(rep.get('pass_at_1'))}% vs frozen base {_f(rep.get('base_pass_at_1'))}% on N={rep.get('n')} test repairs "
      f"({rep.get('n_clusters')} clusters); paired Δ {_ci(rep)} pp (cluster bootstrap 95% CI). Candidate recall@K = {_f(e1.get('candidate_recall_at_k'))}% (reranking ceiling). "
      f"Strongest comparator: {d1.get('strongest_comparator')} at {_f(d1.get('strongest_comparator_pass_at_1'))}%. Decision: **{d1.get('status')}**.")
    A(f"- **E2 (sparse Pareto):** count-matched improvement at capacities {d2.get('improvement_capacities_count_matched')}, bytes-matched {d2.get('improvement_capacities_bytes_matched')}; "
      f"RVM non-dominated: {d2.get('rvm_non_dominated')}; compression target met at capacities {d2.get('compression_capacities')} (full-kNN {_f(d2.get('knn_full_accuracy'))}% at {d2.get('knn_full_bytes')} bytes). Decision: **{d2.get('status')}**.")
    coll = e3.get("methods", {}).get("rvm", {}).get("collateral", {})
    A(f"- **E3 (safety):** collateral regressions {coll.get('n_regressions')}/{coll.get('n_base_correct')} = {_f(coll.get('rate'), 2)}% (one-sided Wilson 95% upper {_f(coll.get('wilson_upper_95'), 2)}%, cluster-bootstrap upper {_f(coll.get('cluster_bootstrap_upper_95'), 2)}%); "
      f"retention loss at 100 updates {_f(d3.get('retention_loss_pp_at_100')/100 if d3.get('retention_loss_pp_at_100') is not None else None, 1)} pp; rollback exact: {e3.get('rollback', {}).get('exact')}; "
      f"false submissions admitted: {e3.get('false_submissions', {}).get('n_admitted')}/{e3.get('false_submissions', {}).get('n_submitted')}. Decision: **{d3.get('status')}**.")
    A("- What cannot be concluded: see §7 (power) and §9 (limitations). All thresholds were fixed in docs/EXPERIMENT_PROTOCOL.md v0.2 before test evaluation.\n")
    # 2
    A("## 2. Reproducibility manifest\n")
    A("| Item | Value |\n| --- | --- |")
    A(f"| Frozen LLM | `{cfg['model']['id']}` @ `{cfg['model']['revision']}` ({cfg['model']['dtype']}, {model.get('n_parameters', 'n/a')} params) |")
    A(f"| Embedding | `{cfg['embedding']['id']}` @ `{cfg['embedding']['revision']}` (dim {emb.get('dimension')}) |")
    A(f"| GPU / driver | {env.get('nvidia_smi')} |\n| torch / CUDA | {env.get('torch')} / {env.get('cuda')} (cuda_available={env.get('cuda_available')}) |")
    A(f"| Python / uv.lock | {env.get('python')} / `{env.get('uv_lock_sha256')}` |\n| Git commit | `{env.get('git_sha')}` |")
    A(f"| Benchmark | first-party executable catalog (MIT), {split.get('n_tasks')} tasks; tasks.jsonl `{split.get('tasks_sha256')}`; hidden.jsonl `{split.get('hidden_sha256')}` |")
    A(f"| Split policy | chronological by simulated stream timestamp at cluster level (train {cfg['dataset']['train_fraction']}, val {cfg['dataset']['val_fraction']}), {cfg['dataset']['heldout_family_fraction']} of API families held out entirely |")
    A(f"| Leakage audit | {json.dumps(split.get('leakage_audit', {}))} |")
    A(f"| Environments | {len(envs)} docker images: " + "; ".join(f"`{e['env_id']}`={e['image_id'][:19]}" for e in envs) + " |")
    A(f"| Verifier | docker, `--network none`, uid 65534, read-only rootfs, {cfg['verifier']['max_memory_mb']} MB, {cfg['verifier']['max_pids']} pids, {cfg['verifier']['max_walltime_seconds']} s/job |")
    A(f"| Generation budget | K={cfg['model']['num_candidates']} (greedy + sampled T={cfg['model']['temperature']}), max_new_tokens={cfg['model']['max_new_tokens']}, max_input={cfg['model']['max_input_tokens']}, batch={cfg['model']['batch_size']} |")
    A(f"| Seeds | {cfg['run']['seeds']} |\n| Peak VRAM (generation) | {_peak_vram(work_dir)} |\n| Stages | {json.dumps({k: v.get('finished_at') for k, v in stages.items()})} |")
    A(f"| Work dir / logs | `{work_dir}` (suite.log, cache/, results/) |")
    A("\nSetup and run commands:\n\n```bash\nuv sync --group dev --extra gpu\nbash scripts/start_mlflow.sh\nuv run sparce run-suite --config configs/a4000.toml            # full suite\nuv run sparce run-suite --config configs/a4000.toml --resume   # continue an interrupted run\nuv run sparce summarize --config configs/a4000.toml            # regenerate report/plots from caches\n```\n")
    # 3
    A("## 3. Methodology\n")
    A(f"**Benchmark.** CodeSyncBench (HF `LuckyVoyage/CodeSyncBench`) was assessed on 2026-10-17: no dataset card, no license, static QA/completion records only (`API_path`, `question`, `answer`, signatures), no executable tests. It was therefore not used. "
      f"Instead a first-party catalog of {len({t for t in split.get('environments', [])})}-environment, real, documented breaking changes (numpy 2.0, pandas 2.0, SciPy 1.12–1.15, scikit-learn 1.2–1.6, Pillow 10, networkx 3, Jinja 3, pydantic 2, CPython 3.12/3.13 removals) was authored, each with 6–8 distinct programs and 2 unchanged-sibling controls, and validated by execution (old code passes on the legacy pin, fails on the new pin; reference fix passes). "
      f"Execution validation excluded {val.get('n_excluded')} of {val.get('n_candidate_tasks')} generated tasks (catalog_validation.json). Variant 1 of each program applies consistent identifier renaming (tokenize-based) and shares the program's cluster id.\n")
    A(f"**Task kinds.** repair (old code on the new pin; the real traceback is shown), previously_correct (reference code on the new pin), old_version (old code on the legacy pin; out of memory scope), near_api / near_api_old (sibling programs using related unchanged APIs). Counts: {json.dumps(split.get('kinds'))}. Repair split sizes {json.dumps(split.get('splits'))}; test subsets {json.dumps(split.get('test_subsets'))}.\n")
    A("**Candidates and base score.** Candidate 0 = greedy, candidate 1 = nucleus sample (fixed task-derived seed); the assistant turn is prefilled with a code fence for every method. Base score = mean per-token log-likelihood of the response under the frozen model; the frozen base picks the higher score (ties → greedy). "
      "All rerankers see the same two candidates, the same score margin d0, and the same joint embedding (scope + error headline + unified diff request→candidate; MiniLM-L6-v2). Hidden tests and reference fixes never enter prompts, retrieval, embeddings or ranking.\n")
    A("**SPaRCE.** Per exact scope (package==version) a Laplace-ARD pairwise RVM: P(a≻b)=σ(d0+Σ_j w_j[k(q_a,z_j)−k(q_b,z_j)]), Gaussian ARD priors α_j, MAP + Laplace covariance, farthest-first pre-screen to max_basis, then ARD pruning (α>1e6). Gate: exact scope → support (max kernel to a retained vector ≥ min_support) → credible margin (mean − z·sd > threshold). Prototypes are verified candidate embeddings (pass → +, fail → −) from training events; pairs with one passing and one failing candidate supply labels. "
      f"Hyperparameters tuned on validation only: {json.dumps(e1.get('tuned', {}).get('rvm', {}).get('best'))}; kNN: {json.dumps(e1.get('tuned', {}).get('knn', {}).get('best'))}.\n")
    A("**Comparators.** B0 frozen base (K=1 greedy and K=2 score pick); B1 rules = exact-scope identifier substitution rules mined from verified training repairs; B2 RAG = prompt augmented with entries retrieved from the pinned package's own public API inventory (dumped inside the sandbox; generator baseline with its own candidates and larger token cost); B3 full kNN (similarity-weighted vote of verified neighbours, same scope gate); B4 kNN at the RVM's active count; B5 random prototypes; B6 k-means medoids; B7 RVM without scope; B8 RVM without credible gate; B9 SPaRCE. Verifier-assisted selection (run both candidates) equals recall@K and is reported separately as an upper bound.\n")
    A("**Deviations from the protocol.** (1) CodeSyncBench replaced by a first-party executable catalog (license/executability). (2) Stream timestamps are simulated (exponential offsets after the real release date, deterministic per cluster) because first-party programs have no observation time. (3) max_new_tokens 384 instead of 256 (whole-module rewrites), identical for all methods. (4) Seeds vary memory fitting/selection and bootstrap only; candidate generation is produced once per task with a fixed task-derived seed (GPU budget). (5) SERAC/GRACE/WISE/MEMOIR were not run (out of the 16 GB / 9 h budget). Recorded before test evaluation in docs/EXPERIMENT_PROTOCOL.md v0.2.\n")
    # 4
    A("## 4. E1 — Verified error repair\n")
    cov = e1.get("coverage", {})
    A(f"**Memory coverage of the test split.** Families with at least one training repair: {cov.get('families_with_train_repairs')} of {split.get('n_families')}; test repairs in such families (\"seen_family\"): {cov.get('coverage_test_subsets', {}).get('seen_family')} vs unseen {cov.get('coverage_test_subsets', {}).get('unseen_family')} "
      f"(nominal split labels were known={cov.get('nominal_test_subsets', {}).get('known_family')}, heldout={cov.get('nominal_test_subsets', {}).get('heldout_family')}). Headroom tasks (base wrong, some candidate passes): {e1.get('headroom_tasks')} overall, {e1.get('headroom_tasks_seen_family')} in seen families. "
      f"Base pass@1 by package: {json.dumps({k: round(v, 2) for k, v in e1.get('base_pass_at_1_by_package', {}).items()})}; test repairs by package {json.dumps(e1.get('test_repair_by_package', {}))}.\n")
    A(f"Test repairs N={e1.get('n_test_repair')} ({e1.get('n_test_repair_clusters')} clusters); train events {e1.get('n_train')} ({e1.get('train_events_with_pair_label')} with a pairwise label); val {e1.get('n_val')}. "
      f"Greedy-only (K=1) pass@1 = {_f(e1.get('greedy_pass_at_1'))}%; candidate recall@K = {_f(e1.get('candidate_recall_at_k'))}% (known families {_f(e1.get('candidate_recall_at_k_known'))}%, held-out families {_f(e1.get('candidate_recall_at_k_heldout'))}%). Reranking cannot exceed recall@K.\n")
    A("| Method | N | pass@1 % | paired Δ vs base pp [95% CI] | known-fam % | held-out-fam % | fixes / regressions | activations | tokens/task | p95 decision ms | control regressions |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, entries in e1.get("methods", {}).items():
        e = next((x for x in entries if x.get("seed") == seed0), entries[0])
        r = e["repair"]; bs = e.get("repair_by_subset", {})
        tokens = e.get("tokens_per_task", e1.get("tokens_per_task", {}).get("base"))
        A(f"| {name} | {r.get('n')} | {_f(r.get('pass_at_1'))} | {_ci(r)} | {_f(bs.get('seen_family', {}).get('pass_at_1'))} | {_f(bs.get('unseen_family', {}).get('pass_at_1'))} | {r.get('fixes')} / {r.get('regressions')} | {r.get('activations')} | {tokens:.0f} | {r.get('latency_p95_ms', 0):.2f} | {e.get('controls', {}).get('regressions')} |")
    A(f"| verifier-assisted (any of K passes) | {e1.get('n_test_repair')} | {_f(e1.get('candidate_recall_at_k'))} | — | {_f(e1.get('candidate_recall_at_k_known'))} | {_f(e1.get('candidate_recall_at_k_heldout'))} | — | — | — | +{e1.get('verifier_ms_per_task_mean', 0):.0f} ms verifier | — |")
    A(f"\nSeed sensitivity (mean ± sd of pass@1 over seeds {e1.get('seeds')}): " + "; ".join(
        f"{name} {_seed_stat(entries)}" for name, entries in e1.get("methods", {}).items()) + ".")
    A(f"\nE1 decision reasoning: N {'≥' if d1.get('checks', {}).get('n_ge_min') else '<'} {d1.get('min_n')}; gain {d1.get('gain_pp', float('nan')):+.1f} pp (CI low {d1.get('ci_low_pp', float('nan')):+.1f}); "
      f"margin vs strongest comparator ({d1.get('strongest_comparator')}) {d1.get('margin_vs_strongest_pp', float('nan')):+.1f} pp; checks {json.dumps(d1.get('checks'))} → **{d1.get('status')}**.\n")
    A("Decision reasons on test repairs (SPaRCE): " + json.dumps(rvm.get("reasons", {})) + ". Per-scope fit log: " + json.dumps(rvm.get("fit_log", {})) + "\n")
    A("![E1](plots/e1_delta_vs_base.png)\n")
    # 5
    A("## 5. E2 — Accuracy versus sparse memory cost\n")
    A(f"Prototype pool {e2.get('pool_size')} verified candidate embeddings; capacities {e2.get('capacities')}; test repairs N={e2.get('n_test_repair')}; per-prototype kNN bytes {e2.get('per_prototype_bytes', 0):.0f}. "
      f"Cold evidence store (raw texts + verifier records) = {e1.get('cold_evidence_bytes')} bytes, reported separately.\n")
    A("| Method | capacity | active | bytes total | prototype bytes | covariance bytes | pass@1 % | Δ vs base pp [CI] | p50/p95 ms | fit s | control regr. |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in e2.get("rows", []):
        if r["seed"] != seed0:
            continue
        ci = f"{r['delta_vs_base']*100:+.1f} [{r['delta_ci_low']*100:+.1f}, {r['delta_ci_high']*100:+.1f}]" if r.get("delta_vs_base") is not None else "—"
        A(f"| {r['method']} | {r['capacity'] if r['capacity'] is not None else 'full'} | {r['active_count']} | {r['bytes']['total']} | {r['bytes'].get('prototypes')} | {r['bytes'].get('covariance')} | {_f(r['accuracy'])} | {ci} | {r['latency_p50_ms']:.2f}/{r['latency_p95_ms']:.2f} | {r['fit_seconds']:.2f} | {r.get('controls_regressions')} |")
    A("\nPaired SPaRCE − comparator on test repairs (seed 0 CI; mean over seeds):\n\n| capacity | comparator | Δ pp [95% CI] | mean over seeds |\n| ---: | --- | ---: | ---: |")
    for c in e2.get("comparisons", []):
        s0 = c["seed0"]
        A(f"| {c['capacity']} | {c['comparator']} | {s0.get('delta', float('nan'))*100:+.1f} [{s0.get('ci_low', float('nan'))*100:+.1f}, {s0.get('ci_high', float('nan'))*100:+.1f}] | {c['mean_delta']*100:+.1f} |")
    A(f"\nPareto front (accuracy↑, bytes↓, p95 latency↓): " + "; ".join(f"{p['method']}@{p['capacity']} ({_f(p['accuracy'])}%, {p['bytes']} B)" for p in e2.get("pareto_front", [])) + ".")
    base_acc = (e1.get("methods", {}).get("base") or [{}])[0].get("repair", {}).get("pass_at_1")
    full_acc = d2.get("knn_full_accuracy")
    caveat = ""
    if base_acc is not None and full_acc is not None and abs(full_acc - base_acc) < 0.01:
        caveat = (f" **Caveat:** the compression target is met only trivially: full kNN itself is within {abs(full_acc - base_acc) * 100:.1f} pp of the memory-free base "
                  f"({_f(base_acc)}%), so 'within 1 pp of full kNN at ≤25% bytes' is satisfied by any method that does nothing; it is not evidence of useful compression.")
    A(f"\nE2 decision: {json.dumps(d2)} → **{d2.get('status')}**.{caveat} ARD-pruned vectors (active) are reported, not the screened pool; covariance bytes are O(M²) and included.\n")
    A("![E2 bytes](plots/e2_accuracy_vs_bytes.png)\n![E2 latency](plots/e2_accuracy_vs_latency.png)\n")
    # 6
    A("## 6. E3 — Regression, continual learning and rollback\n")
    A(f"Controls (val+test clusters, never used for fitting): N={e3.get('n_controls')} in {e3.get('n_control_clusters')} clusters; train-cluster controls reported separately.\n")
    A("| Method | base-correct controls | regressions | rate % | Wilson one-sided 95% upper % | cluster-bootstrap upper % | spurious activations | Brier (test pairs) | ECE |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, m in e3.get("methods", {}).items():
        c = m.get("collateral", {}); cal = m.get("calibration_test_pairs", {})
        A(f"| {name} | {c.get('n_base_correct', '—')} | {c.get('n_regressions', '—')} | {_f(c.get('rate'), 2)} | {_f(c.get('wilson_upper_95'), 2)} | {_f(c.get('cluster_bootstrap_upper_95'), 2)} | {c.get('spurious_activations', '—')} | {cal.get('brier', 'n/a') if isinstance(cal.get('brier'), str) else _f(cal.get('brier'), 3, False)} | {_f(cal.get('ece'), 3, False) if cal.get('ece') is not None else 'n/a'} |")
    rk = e3.get("methods", {}).get("rvm", {}).get("collateral_by_kind", {})
    A("\nSPaRCE collateral by control kind: " + "; ".join(f"{k}: {v.get('n_regressions')}/{v.get('n_base_correct')} (Wilson upper {_f(v.get('wilson_upper_95'), 2)}%)" for k, v in rk.items()) + ".")
    tc = e3.get("methods", {}).get("rvm", {}).get("train_cluster_controls", {})
    if tc:
        A(f" Train-cluster controls: {tc.get('n_regressions')}/{tc.get('n_base_correct')}.")
    A("\n\n**Sequential verified updates** (chronological training stream; admission gate on the untouched validation split: no accuracy loss vs base and ≤1% regressions):\n")
    A("| checkpoint | admitted | quarantined | promoted | active | bytes | test pass@1 % | fixes/regr. | collateral | retention of first fixes % | Brier | ECE | snapshot bytes |\n| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in e3.get("stream", []):
        tr = r["test_repair"]; cal = r.get("calibration", {})
        A(f"| {r['checkpoint']} | {r['n_admitted']} | {r['n_quarantined']} | {r['promoted']} | {r['active_count']} | {r['bytes']['total']} | {_f(tr.get('pass_at_1'))} | {tr.get('fixes')}/{tr.get('regressions')} | {r['collateral']['n_regressions']}/{r['collateral']['n_base_correct']} | {_f(r['retention_test_fixed'])} | {_f(cal.get('brier'), 3, False) if cal.get('brier') is not None else 'n/a'} | {_f(cal.get('ece'), 3, False) if cal.get('ece') is not None else 'n/a'} | {r.get('snapshot_bytes', '—')} |")
    fs = e3.get("false_submissions", {})
    A(f"\nFalse/unverified submissions: {fs.get('n_submitted')} injected (flipped labels, no verifier evidence); admitted = {fs.get('n_admitted')} (all quarantined). Ablation without the admission check: collateral {fs.get('without_admission_check_collateral', {}).get('n_regressions')}/{fs.get('without_admission_check_collateral', {}).get('n_base_correct')}.")
    first_fixed = next((r.get("n_first_fixed") for r in e3.get("stream", []) if r.get("promoted")), None)
    if first_fixed == 0:
        A("\nRetention is undefined (n/a): the first promoted snapshot fixed zero test repairs, so there were no 'old repairs' whose retention could be measured; the retention gate is therefore INCONCLUSIVE rather than passed.")
    rb = e3.get("rollback", {})
    A(f"\nRollback: restored the full joint posterior snapshot of checkpoint {rb.get('to_checkpoint')} from checkpoint {rb.get('from_checkpoint')} (ledger version {rb.get('ledger_version')}, {rb.get('snapshot_bytes')} bytes, non-destructive append); replayed battery of {rb.get('n_battery')} decisions: {rb.get('n_exact_matches')} exact matches (recommend flag, margin, lower bound) → exact = {rb.get('exact')}; {rb.get('n_decisions_differing_from_last_checkpoint')} decisions differ from the last checkpoint.")
    A(f"\nE3 decision: {json.dumps(d3)} → **{d3.get('status')}**. Posterior credibility is Laplace-approximate; the reliability diagram is an empirical check, not a guarantee.\n")
    A("![E3 stream](plots/e3_stream.png)\n![E3 reliability](plots/e3_reliability.png)\n")
    # 7
    A("## 7. Statistics\n")
    A(f"Paired percentile bootstrap over clusters (programs; near-duplicate variants never split across resamples), {cfg['eval']['bootstrap_replicates']} resamples, seed = method seed; one-sided Wilson 95% upper bounds for proportions plus a cluster-bootstrap upper bound because controls are dependent within clusters. "
      f"Hyperparameters chosen on validation only; test touched once per (method, seed). No multiplicity correction across the nine E1 comparators (selection limitation). "
      f"Actual N: test repairs {e1.get('n_test_repair')} (threshold {cfg['eval']['min_test_cases']}); base-correct controls {coll.get('n_base_correct')} (threshold {cfg['eval']['min_negative_controls']}). Raw numerators/denominators are in e1.json, e2.json, e3.json and the CSVs next to this report.\n")
    # 8
    A("## 8. Failure analysis\n")
    try:
        cases = failure_cases(work_dir, e1)
    except Exception as exc:  # noqa: BLE001
        cases = []
        A(f"(failure cases unavailable: {exc})")
    for i, c in enumerate(cases, 1):
        A(f"{i}. **{c['category']}** — cluster `{c['cluster']}` ({c['scope']}, {c['subset']}); request error `{c['request_error']}`; candidates {c['candidate_status']} with errors {c['candidate_errors']}; base correct={c['base_correct']}; decision `{c['decision_reason']}` (margin {c['margin']:.3f}, lower {c['lower']:.3f}); provenance {c['provenance']}")
    # 9
    A("\n## 9. Decisions and next steps\n")
    A("| Gate | Threshold (pre-data) | Observed | Confidence | Decision |\n| --- | --- | --- | --- | --- |")
    A(f"| E1 | ≥+5 pp vs base, CI low > 0, ≥−2 pp vs strongest comparator, N≥{cfg['eval']['min_test_cases']} | Δ {d1.get('gain_pp', float('nan')):+.1f} pp [{d1.get('ci_low_pp', float('nan')):+.1f}, {d1.get('ci_high_pp', float('nan')):+.1f}], vs {d1.get('strongest_comparator')} {d1.get('margin_vs_strongest_pp', float('nan')):+.1f} pp, N={d1.get('n_test_repair')} | cluster-paired bootstrap | **{d1.get('status')}** |")
    A(f"| E2 | ≥+2 pp vs capacity-matched kNN (CI low>0) at ≥1 budget, non-dominated, and ≤25% bytes within 1 pp of full kNN | count-matched wins {d2.get('improvement_capacities_count_matched')}, bytes-matched {d2.get('improvement_capacities_bytes_matched')}, non-dominated={d2.get('rvm_non_dominated')}, compression {d2.get('compression_capacities')} | seed-0 CI + 3 seeds | **{d2.get('status')}** |")
    A(f"| E3 | collateral upper ≤1%, retention loss ≤2 pp at 100, exact rollback, false submissions rejected, N≥{cfg['eval']['min_negative_controls']} | upper {_f(d3.get('collateral_wilson_upper'), 2)}%, retention loss {d3.get('retention_loss_pp_at_100')}, rollback {rb.get('exact')}, admitted {fs.get('n_admitted')} | Wilson + cluster bootstrap | **{d3.get('status')}** |")
    A(f"\n**Overall: {overall}.** Technical debt / risks: simulated stream timestamps; first-party benchmark authored by the same team (possible unconscious bias toward alias-rename changes); single frozen 3B model; MiniLM is a natural-language embedder; Laplace posterior may be over-confident under shift; rules baseline is mined from the same training events; no multiplicity correction. "
      "Next experiments: (1) generation-time steering so memory can add a candidate when recall@K is the ceiling; (2) a second embedding model (code-trained) and kernel ablation; (3) real observed-time stream from an organisation's CI logs with the same admission gate.\n")
    A(f"\n**Review packet.** Report: `{out_dir / 'SPARCE_POC_REPORT.md'}`; manifests: `{out_dir / 'environment_manifest.json'}`, `{out_dir / 'split_manifest.json'}`, `{out_dir / 'catalog_validation.json'}`; results: e1/e2/e3.json + CSVs + plots/ in the same directory; MLflow runs: " + ", ".join(f"{k}=`{v}`" for k, v in mlflow_ids.items()) + f"; logs: `{work_dir / 'suite.log'}`; per-task decisions inside e1.json/e2.json (`per_task`). Not a production-readiness claim.\n")
    if exploratory is not None and (Path(exploratory) / "results" / "e1.json").exists():
        L.extend(exploratory_section(cfg, Path(exploratory), out_dir))
    path = out_dir / "SPARCE_POC_REPORT.md"
    path.write_text("\n".join(L), encoding="utf8")
    return path


def exploratory_section(cfg: dict, work_dir: Path, out_dir: Path) -> list[str]:
    """Appendix for a post-hoc re-split run. Never eligible for GO; reported for diagnosis only."""
    res = work_dir / "results"
    e1, e2, e3 = _load(res / "e1.json"), _load(res / "e2.json"), _load(res / "e3.json")
    split = _load(work_dir / "data" / "split_manifest.json")
    sub = out_dir / "exploratory_v0.3"
    sub.mkdir(parents=True, exist_ok=True)
    for name in ("e1.json", "e2.json", "e3.json", "e1_summary.csv", "e2_rows.csv", "e3_stream.csv"):
        if (res / name).exists():
            shutil.copy(res / name, sub / name)
    if (res / "plots").exists():
        shutil.copytree(res / "plots", sub / "plots", dirs_exist_ok=True)
    if (work_dir / "data" / "split_manifest.json").exists():
        shutil.copy(work_dir / "data" / "split_manifest.json", sub / "split_manifest.json")
    seed0 = e1.get("seeds", [0])[0]
    cov = e1.get("coverage", {})
    L = ["\n## Appendix A. Exploratory re-split (protocol v0.3, post hoc — NOT eligible for GO)\n"]
    A = L.append
    A("The preregistered v0.2 split anchors simulated event times to each API change's real release date, so the global chronological cutoff sorted whole families by release date: the latest families (numpy 2.0, SciPy 1.14/1.15, CPython 3.13) fell entirely into test and memory had no training repair for most test tasks (§4). "
      "This appendix re-runs the identical cached candidates, verifier outcomes and embeddings under a *deployment-window* timing model (all events observed after a single adoption date, 2025-02-01, that post-dates every catalogued release, with the same per-cluster exponential offsets), which interleaves families in time while still never placing an event before its real release. "
      "The split rule, the held-out-family fraction, the hyperparameter grids, the budgets and the thresholds are unchanged. Because this was decided after seeing v0.2 test results, it is exploratory: it can explain the null result but cannot upgrade any decision.\n")
    A(f"Split: repairs train/val/test = {json.dumps(split.get('splits'))}; coverage: families with train repairs {cov.get('families_with_train_repairs')}, test repairs seen/unseen = {cov.get('coverage_test_subsets', {}).get('seen_family')}/{cov.get('coverage_test_subsets', {}).get('unseen_family')}; headroom {e1.get('headroom_tasks')} ({e1.get('headroom_tasks_seen_family')} in seen families); recall@K {_f(e1.get('candidate_recall_at_k'))}% (seen {_f(e1.get('candidate_recall_at_k_known'))}%, unseen {_f(e1.get('candidate_recall_at_k_heldout'))}%). "
      f"Tuned: rvm {json.dumps(e1.get('tuned', {}).get('rvm', {}).get('best'))}, knn {json.dumps(e1.get('tuned', {}).get('knn', {}).get('best'))}.\n")
    A("| Method | N | pass@1 % | paired Δ vs base pp [95% CI] | seen-fam % | unseen-fam % | fixes / regressions | activations | control regressions |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, entries in e1.get("methods", {}).items():
        e = next((x for x in entries if x.get("seed") == seed0), entries[0])
        r = e["repair"]; bs = e.get("repair_by_subset", {})
        A(f"| {name} | {r.get('n')} | {_f(r.get('pass_at_1'))} | {_ci(r)} | {_f(bs.get('seen_family', {}).get('pass_at_1'))} | {_f(bs.get('unseen_family', {}).get('pass_at_1'))} | {r.get('fixes')} / {r.get('regressions')} | {r.get('activations')} | {e.get('controls', {}).get('regressions')} |")
    d1, d2, d3 = e1.get("decision", {}), e2.get("decision", {}), e3.get("decision", {})
    A(f"\nWould-be gate outcomes under v0.3 (informational only): E1 {d1.get('status')} (Δ {d1.get('gain_pp', float('nan')):+.1f} pp [{d1.get('ci_low_pp', float('nan')):+.1f}, {d1.get('ci_high_pp', float('nan')):+.1f}], vs {d1.get('strongest_comparator')} {d1.get('margin_vs_strongest_pp', float('nan')):+.1f} pp); "
      f"E2 {d2.get('status')} (count-matched wins {d2.get('improvement_capacities_count_matched')}, bytes-matched {d2.get('improvement_capacities_bytes_matched')}, non-dominated {d2.get('rvm_non_dominated')}, compression {d2.get('compression_capacities')}); "
      f"E3 {d3.get('status')} (collateral upper {_f(d3.get('collateral_wilson_upper'), 2)}%, retention loss at 100 {d3.get('retention_loss_pp_at_100')}, rollback exact {e3.get('rollback', {}).get('exact')}).")
    rvm = (e1.get("methods", {}).get("rvm") or [{}])[0]
    A(f"\nSPaRCE decision reasons on the v0.3 test split: {json.dumps(rvm.get('reasons', {}))}. Per-scope fits: {json.dumps({k: (v.get('status'), v.get('n_pairs'), v.get('active')) for k, v in rvm.get('fit_log', {}).items()})}.")
    A("\n![Exploratory E1](exploratory_v0.3/plots/e1_delta_vs_base.png)\n![Exploratory E2](exploratory_v0.3/plots/e2_accuracy_vs_bytes.png)\n")
    return L


def _seed_stat(entries: list[dict]) -> str:
    vals = [e["repair"].get("pass_at_1") for e in entries if e.get("repair", {}).get("pass_at_1") is not None]
    if not vals:
        return "n/a"
    import numpy as np
    return f"{np.mean(vals)*100:.1f} ± {np.std(vals)*100:.1f}"


def _peak_vram(work_dir: Path) -> str:
    p = work_dir / "cache" / "candidates.jsonl"
    if not p.exists():
        return "n/a"
    peak = 0
    with p.open() as fh:
        for line in fh:
            try:
                peak = max(peak, int(json.loads(line).get("peak_vram_bytes", 0)))
            except (ValueError, json.JSONDecodeError):
                continue
    return f"{peak / 1e9:.2f} GB reserved" if peak else "n/a"
