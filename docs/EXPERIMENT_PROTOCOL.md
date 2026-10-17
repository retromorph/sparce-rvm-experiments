# Preregistered experiment protocol — SPaRCE PoC

**Version:** 0.1 (pre-data). **Owner:** coding agent; changes after holdout access must be documented and treated as exploratory. This file defines the three go/no-go studies. Synthetic smoke runs are not eligible evidence.

## Core prediction rule

Given request x, proposed candidate a and original candidate b, compute frozen normalized base score margin d0. Encode joint `(x,a)` and `(x,b)` in a versioned embedding model. The ARD relevance-vector basis is `phi_j = k(q_a,z_j)-k(q_b,z_j)` and `P(a better b)=sigmoid(d0 + phi @ w)`. Gaussian ARD prior uses independent precision alpha_j; fitted sparse posterior via MAP+Laplace. Route only if: verified scope matches, local support >= validation threshold, and `mean_margin - z * posterior_std > threshold`. Neither posterior confidence nor retrieval similarity independently certifies correctness. Deployments MUST run independent verification before applying nontrivial code changes.

Train/freeze the embedding model and the base LLM independent of holdouts. Log hashes of weights, tokenizer, prompts, data, environment and verifier. Do not update pretrained weights for primary SPaRCE study.

## Dataset and split principles

1. Start from CodeSyncBench (https://github.com/Lucky-Wang-Chenlong/CodeSync), inspect source/license and confirm labels and metadata. The benchmark alone does not guarantee executable harnesses; author independently reproducible, version-pinned tests for the selected tasks.
2. Each sample has an independent outcome, source timestamp, package and exact version, API family, request, original model answer, alternative, and execution trace. Exclude or disclose any items that cannot be verified.
3. Split chronologically: no later information in memory for earlier-time test tasks; additionally hold out **API families** when assessing transfer. Near-duplicate calls must remain grouped. Preserve sample IDs and a split manifest with SHA256 hash.
4. Include previously-correct tasks, old-version tasks, nearby APIs with distinct signatures and scope-mismatched controls. Do not infer success solely from compilation when behavior may be wrong.
5. Maintain strict division between **offline verifier labels** (evaluation only) and optional **online runtime verifier**. No use of hidden test cases, ground-truth patches, or test results in ranking unless all compared systems are given the same declared verifier budget.
6. Select hyperparameters on train/validation only, lock before touching final test. Run at least 3 seeds and paired comparisons. Underpowered experiments are **inconclusive**, not wins.

## Comparators and budget parity

B0: frozen LLM only (K=1 and fixed candidate budget K when compared against a reranker); B1: simple exact version-aware API rule or deterministic patch; B2: official documentation RAG; B3: full kNN candidate-level memory; B4: kNN at RVM's active count; B5: random memory at matched count; B6: k-means/medoid memory at matched count; B7: RVM without scope gate; B8: RVM with scope but no credible-margin gate; B9: full SPaRCE (scope, credible gate, external verification, snapshots). Adapt or omit comparators only if impossible, explain why. SERAC/GRACE/WISE/MEMOIR optional high-cost baselines if integrations are viable under the same 16GB constraints.

Use same base checkpoint, tokenizer, generation candidates or an explicit separate-generation protocol, prompt lengths, retrieval corpus, max examples, decoder seed, scoring normalization, verifier cost, total available memory bytes, latency definition and evaluation code. Report unadjusted candidate selection as well as any runtime-verifier-assisted endpoint separately.

## E1 — Validated error repair and transfer

**Question:** Can sparse corrective memory improve held-out, executable API repair vs the frozen base and strong retrieval/rule baselines?

Primary outcome: paired *verified* pass@1 on held-out tasks; also record generated candidate recall@K (an upper bound for candidate reranking) and pass@K. Subsets: unseen examples from known API families, held-out API families, near-version transfer, OOD APIs. All methods scored by identical independent executable tests.

Proposed go target (adjust *only before data access*): absolute gain >= 5 percentage points over budget-matched base and positive paired-bootstrap 95% CI lower bound, with no degradation vs the strongest feasible RAG/rules baseline beyond a prespecified 2-point tolerance. Explicitly report if the latter fails. Minimum target final test N>=200 executable examples; if missing, classify **inconclusive**.

E1 **GO** if these tests pass. **NO-GO** if a powered experiment rejects the claim or posterior model underperforms; otherwise **INCONCLUSIVE**.

## E2 — RVM sparsity and cost/accuracy Pareto frontier

**Question:** Does ARD gain accuracy at a fixed active memory footprint compared with full and capacity-matched kNN, random and clustering variants?

Sweep active budget `[8,16,32,64,128]` (skip values exceeding available candidate pool), fit multiple seeds, and compare all methods using **active count**, exact serialized parameter bytes, index bytes, raw evidence bytes (reported separately), online latency p50/p95, inference total tokens, update wall time, peak VRAM and performance. RVM covariance storage is O(M^2) and MUST count in bytes. Graph speed/memory Pareto frontiers and matched count; avoid framing reduced memory count alone as proof of lower RAM.

Proposed go target: at the same or lower active bytes and with matched candidate/verifier budget, SPaRCE must exceed capacity-matched kNN by >= 2 pp verified pass@1 with paired-bootstrap CI lower bound > 0 on at least one practically relevant memory budget **and** show a non-dominated point on the accuracy–memory–latency frontier. As separate compression target, it should retain within 1 pp of the full-kNN accuracy while requiring <= 25% of full-kNN active bytes. Failure of one target is reported, not hidden.

E2 **GO** only if both the improvement and compression targets pass; otherwise **NO-GO** or **INCONCLUSIVE** if data insufficient. If the ARD objective is inferior, report it rather than silently changing model family.

## E3 — Safety, scope locality, lifelong stability and rollback

**Question:** Can corrections be introduced without altering already-correct or out-of-scope behavior? Can every promoted snapshot be rolled back deterministically?

Construct negative controls (N>=1000 target) across previously-correct responses, old API versions, out-of-scope libraries and confusing near matches. Chronological stream has checkpoints at 10/50/100 updates, with 500 as stretch if dataset supports it. Log true error-to-correct gains, correct-to-error regressions, retained old repairs, false activation frequency, Brier score, ECE, inference uncertainty and memory changes. Include deliberately false/unverified memory submissions; they must be rejected or quarantined and never silently promoted.

The primary collateral regression rate is `#(baseline correct AND memory wrong) / #(baseline correct)`, with one-sided 95% upper CI; use Wilson or exact binomial CI for proportions, or appropriately clustered analysis when controls are dependent. Propose maximum 1% upper bound (only meaningful when N is adequate), <= 2 pp retention loss at 100 updates, and exact reproduction of previous snapshot decisions after rollback on a fixed test battery. `accuracy`, bytes, environment and latency all recorded. Posterior credible intervals should undergo independent calibration assessment; do not imply calibrated 95% coverage by construction.

E3 **GO** if collateral upper bound <=1%, retention <=2 pp and rollback exact; **NO-GO** for measured violation with adequate power; otherwise **INCONCLUSIVE**. If a verifier cannot confirm a proposed code change at runtime, the production-safe branch abstains or falls back, never treats RVM confidence alone as proof.

## Statistics and reporting

- Use **paired**, task/cluster-aware bootstrap confidence intervals (>=2000 resamples), 3+ seeds and explicit random state. For E3 one-sided upper binomial bound with dependence caveat. Include raw numerator/denominator and actual N.
- Record per-sample JSONL IDs, predictions, scope, confidence, verification result, token counts and timings (redact secrets); do not commit derived private data.
- Do not claim product readiness based on these PoC metrics. No-GO for any critical E3 safety target blocks deployable claim even if repair accuracy improves.
- Show negative outcomes; include comparisons at equal inference budget. Avoid cherry-picking model checkpoint, task families or hyperparameters using final test.
- Log MLflow run IDs, commit SHA, uv.lock hash, CUDA/driver info, GPUs, model/dataset revisions and artifact SHA256s. Attach raw (non-sensitive) CSV/JSON summaries, plots and the report.
- All go/no-go thresholds here are *proposed research design choices*, not empirically validated safety guarantees. If infeasible, record changes before test evaluation and note departures.
