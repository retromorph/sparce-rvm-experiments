# SPaRCE: Research and Production Design

**Design date:** 2026-10-17  
**Status:** Proposed system, with a standalone runnable mathematical reference  
**Application:** Continual error correction for frozen LLM coding agents (initial), later temporal factual QA  
**Claim to test:** Bayes/ARD-selected correction operators with explicit scope and validation improve *future verified task success per memory byte* with fewer collateral regressions than capacity-matched kNN retrieval and strong continual-editing baselines.

## 1. Scope and positioning

SPaRCE is a corrective **sidecar** that intervenes only when (i) there is independently verified historical evidence, (ii) a narrow applicability predicate matches the current environment, and (iii) the estimated corrected answer advantage clears a conservative confidence threshold. It is **not** a general conversational-memory system; it is a system for transferring verified fixes to future decisions under strict locality constraints.

Adjacent literature already contains sparse external edits and posterior skill-evolution. RVM sparsity *alone* is not the claimed innovation. Distinguish from [SERAC](https://arxiv.org/abs/2206.06520), [GRACE](https://arxiv.org/abs/2211.11031), [WISE](https://arxiv.org/abs/2405.14768), [MEMOIR](https://proceedings.neurips.cc/paper_files/paper/2025/hash/809a560bf60984be05180c071c352eaa-Abstract-Conference.html), [Mistake Notebook Learning](https://aclanthology.org/2026.findings-acl.719/), [Bayesian-Agent](https://arxiv.org/abs/2606.08348), and [TrustMem](https://arxiv.org/abs/2606.25161).

The first publishable result requires showing the value of the *combination* of ARD + out-of-scope controls + confidence gating + reversible validation, through ablations and causal comparisons.

## 2. Modeling contract

Represent each independent failure/repair event as:

```
CorrectionEvent {
  event_id: UUID
  timestamp: ISO8601
  tenant_id, domain, repository, library, api, exact_version: strings
  model_id, prompt_template_id, embedder_id, verifier_id: strings
  original_request_hash, failed_answer_hash, corrected_answer_hash: SHA256
  evidence_uri, test_artifact_uri, environment_hash: strings
  verifier_status: PASS | FAIL | UNKNOWN
  failure_taxonomy, code_ast_features, source_commit: optional
  activation_scope: validated predicate
  replaced_by, supersedes: optional event IDs
}
```

The raw trace and original verifier result must remain **immutable source evidence**. A derived memory may store an embedding, correction key, text rule, learnable RVM coefficient, uncertainty, and links to evidence. Evidence for a later contradiction yields a new event, not silent modification of the previous evidence.

Version, tenant, and dependency scope form a **hard gate**. Semantic similarity must not override explicit incompatibility. For software, verification includes an executable test harness, sandbox isolation, and API signature comparison where appropriate. Unverified LLM self-criticism may create a quarantined candidate but never activate a correction.

## 3. Learning objective

For labeled alternative/incumbent pairs (a,b) with verified preference label y, compute joint embeddings e(x,a), e(x,b), frozen-model score difference d0 = s0(x,a) - s0(x,b), and candidate relevance bases z_j. Define design matrix:

`Phi[n,j] = k(e(x_n,a_n),z_j) - k(e(x_n,b_n),z_j)`

The likelihood is:

`P(y_n=1|w) = sigmoid(d0_n + Phi[n,:] @ w)`

Each basis coefficient has an ARD prior `w_j ~ Normal(0, 1/alpha_j)`. Use Laplace empirical-Bayes optimization, not a claim of exact inference. Screen candidates using fast diversity/ANN retrieval, then ARD-select from a **bounded** working set. Use positive verified corrections to propose relevance bases, but train and validate on both *positive* and *negative / scope-boundary* examples. Preserve the negative examples in the evidence store even when they are not retained as active basis vectors.

For posterior approximation `w~N(mu,Sigma)`, a pairwise margin has mean `d0+Phi @ mu` and variance `Phi @ Sigma @ Phi.T`. Default to **abstention** on unknown context, inadequate support, a contradictory version, missing provenance, high uncertainty, or ambiguous verification.

For sequence generation after the first stage, a token-level extension uses a group-ARD logit adapter: `logits = frozen_logits + scope_gate * B @ sum_j u_j*k(h,z_j)` with `u_j in R^r`; *freeze B* for update-level reversibility and separately penalize changes to unrelated output distributions.

## 4. Training and update algorithm

```
ON_VERIFIED_FEEDBACK(event):
  1. Verify provenance, permissions, library/environment version, and test artifact.
  2. Append immutable evidence; reject stale or unverifiable inputs.
  3. Build candidate repair + near-miss negatives and out-of-scope contrasts.
  4. Add event as a quarantined candidate in its scope partition.
  5. Fit a bounded, warm-started Laplace-ARD candidate policy offline.
  6. Evaluate on a *separate* chronological validation window:
       repair success, non-target regressions, calibration, and cost.
  7. Accept only if utility and regression budgets pass the predetermined gate.
  8. Persist a full memory snapshot; canary before promotion.
  9. If canary regressions or drift exceed limits, restore the prior snapshot.
```

If the training set remains unchanged after a single update, a warm-started optimizer can be used, but **not** assumed to provide a Bayesian incremental update for free. Re-training or a principled posterior update is necessary when basis interactions change. Removing an RVM basis also changes the posterior over correlated surviving bases; do **not** treat independent subtraction of a coefficient as exact rollback.

## 5. Serving architecture

```
                  INFERENCE / HOT PATH
  request + resolved versioned environment
                     |
               Frozen LLM
                     |
        K=4 candidate answers + base scores
                     |
          joint request/answer embedder
                     |
          scope check / OOD check / support
                     |
           small RVM posterior scorer
                     |
       recommend alternative or abstain
                     |
        executable verifier (if available)
                     |
          deliver verified / base answer

                 LEARNING / COLD PATH
  trace + external feedback -> quarantine -> evidence event
       -> hard negatives -> ARD fitting -> blind validation
       -> shadow/canary -> versioned promotion/rollback
```

Production data plane should use a pinned-memory-snapshot per request, an authenticated tenant-aware service, an explicitly versioned encoder and base model, a fast ANN prefilter for domains with many possible prototypes, and a separate offline training job. An update controller exposes `stage`, `evaluate`, `promote`, `revoke`, and `rollback`; these are privileged actions, not public inference endpoints. Dynamic prompts and untrusted web content are never treated as authoritative verifier feedback.

Keep a hard kill switch returning the frozen-model response with unchanged scores. Production logs record why a memory activated, its ID and snapshot, scope, predicted lower margin, counterfactual base selection, and downstream result where measurable, with privacy-preserving evidence references.

## 6. Experimental design

### First benchmark: versioned Python API repairs

Start from [CodeSync / CodeSyncBench](https://proceedings.mlr.press/v267/wang25t.html) (220 updated APIs, six libraries, 3,300 static test cases) but construct **executable** and chronologically split repair tasks; synthetic pairs are not automatically verified executable programs. Store failed code, version-pinned corrected code, exact dependency resolution, and verified success/failure results.

*Training stream*: sequentially exposed verified mistakes, including positive repairs and hard negatives.  
*Validation*: later tasks from previously seen API families but disjoint repository contexts. May be used only for hyperparameter/model selection with documented adaptive reuse policy.  
*Locked test*: later release windows and held-out API families; require explicit old-version / sibling-API negatives. No golden solutions, future docs, or test expected outputs may be retrieved during inference.  
*Second task family*: temporally revised factual edits with CounterFact+/RippleEdits locality and implication probes, but without claiming executable truth verification in unconstrained open-world QA.

### Comparators (matched conditions)

| Category | Baseline |
|---|---|
| No memory | Frozen LLM + same candidate/sample and verifier budget |
| Simple reliable correction | Version-aware deterministic rule table |
| Retrieval | Current version official-document RAG; kNN memories, full and at matched memory budget |
| Memory compression | Random, k-medoids/farthest-first, frequency/recency, and an RVM without scope gating |
| Mistake-based agent memory | MNL and ReasoningBank (same generator and feedback privileges) |
| External / parametric editing | SERAC, GRACE, WISE, MEMOIR; optionally LoRA, with best-effort matched resource usage |
| Proposed ablations | RVM only; +hard scope; +uncertainty gate; +validated admission; +rollback |

Require equal maximum candidate count, base LLM, scorer access, verification budget, tokenizer/embeddings where possible, and comparable memory footprints. Avoid comparing a reranker with an unconstrained generator without accounting for its candidate-generation cost.

### Main measurements

- **Correction:** Pass@1 on previously unseen tasks, fixed test-budget solve rate, and error recurrence.
- **Locality:** Rate of baseline-correct tasks turned incorrect, nearest-miss versions, sibling APIs, and unrelated-query KL for optional logit adapter.
- **Transfer:** Same-API paraphrases, repository transfer, held-out API families, stale/contradictory updates.
- **Sparsity:** Number of active basis functions; serialized bytes, not only count; total evidence store bytes disclosed separately.
- **Bayesian quality:** Brier score, reliability diagrams, calibration by scope, selective risk vs coverage.
- **Operations:** p50/p95 added latency, total inference tokens, number of verifier executions, update cost, memory load latency, stale-snapshot rate, rollback correctness.
- **Continual learning:** Metrics at 10, 50, 100, 500 sequential repair events; report order variation and multiple random seeds.
- **Causal utility:** Paired same-task policy-on vs policy-off performance; record positive fixes and negative flips. Randomize shadow/canary assignment rather than comparing heterogeneous traffic.

Report bootstrap confidence intervals over independent repository/API groups and explicit paired statistics. Splitting random paraphrases across train/test is *not* valid evidence of novel error generalization.

### Proposed go/no-go targets — aspirational, not empirical results

| Gate | Proposed criterion |
|---|---|
| Demonstrated repair | Positive, statistically supported net gain over kNN at matched memory and compute on the locked test |
| Controlled locality | Point-estimated collateral-regression rate <=1% on negative controls, with upper confidence bound reported |
| Sparse performance | <=25% of uncompressed candidate basis count, without materially worse verified accuracy |
| Continual stability | No significant negative trend in old-edit retention at 500 sequential events |
| Useful deployment | Benefit remains positive after counting extra candidate tokens, test executions, and p95 latency |

Note that strict <=1% locality *with statistical confidence* requires adequate holdout sample size; an observed 0% alone does not establish such a bound. Any target should be fixed before final testing, not selected after results are observed.

## 7. Validation and statistical safeguards

**No claim of a distribution-free guarantee from an RVM posterior.** Laplace covariance can be miscalibrated, especially under domain shift. A fixed policy evaluated on an independent exchangeable test set permits binomial one-sided bounds for collateral regression. After adaptive selection and repeated use of the same validation set, those bounds no longer apply naively; use fresh test batches, prespecified alpha spending, or an appropriate sequential-valid confidence sequence.

Avoid data leakage from online feedback: a fix at date t can be used only after its evidence becomes available. Compare correct behavior in the *specified library version*, not globally. Avoid learning from noisy outcomes (timeouts, flaky tests) without repeated adjudication. Preserve exact model and embedder versions; changes invalidate previously fitted feature spaces.

## 8. Production hardening before use with real customers

Security: tenant isolation, permissions to write/revoke memories, content sanitization, sandboxed code execution with network off by default, signed verifier outcomes, environment pinning, encrypted evidence and audit logs, per-tenant retention/deletion policy, secrets redaction, prevention of memory poisoning and prompt-injection-as-feedback.

Availability: request timeout and baseline fallback, concurrent immutable reads, rollback under load, crash-consistent snapshots, backup and recovery drills, alerting for increased regression rates, staged rollout. Observability: model and snapshot ID on every decision, rate of gate types, p95 latency, correction recommendation-to-test pass conversion, calibration drift, memory fill and refit frequency.

## 9. Development phases and decision gates

**Phase A — empirical feasibility.** Reproduce controlled synthetic result, implement independent code verifier, score 4 frozen-LLM candidates, create chronological positive/negative pairs, quantify candidate recall ceiling. If the correct answer is seldom in top-K candidates, reranking cannot solve the problem; shift to generation steering sooner.

**Phase B — competitive paper.** ARD + exact scope + posterior gate + blind update acceptance, comparisons against non-Bayesian selection and direct retrieval, studies of memory compression/transfer/locality, multiple seeds and uncertainty analysis, plus at least one additional domain.

**Phase C — deployable sidecar.** Authenticated API, model/embedding compatibility checks, cold-path updates, indexed and versioned storage, independent verifier service, canary routing, monitoring, and incident rollback. Only then describe the system as production ready.

## 10. Research risks

1. An RVM may lose to a versioned exact-match rules engine because the task is mostly symbolic. This is a meaningful negative result, not a defect to conceal.
2. The selected kernel may not represent the actual causal error region. Compare semantic-only, code AST + version, and late hidden-state embeddings; stress near-duplicates with different semantics.
3. Negative-control completeness is expensive. Candidate memories should abstain when evidence is sparse.
4. Gaussian/Laplace posterior uncertainty is approximate and may be overconfident. Calibration and external verifiers must remain separate requirements.
5. ARD can prune valid rare exceptions or keep redundant patterns, depending on hyperparameters and candidate screening. Study rare-event retention separately.
6. A candidate reranker cannot recover a correction absent from the generated candidate set. Measure oracle top-K success.
7. Third-party benchmark contamination and duplicate paraphrase leakage can exaggerate gains. Require temporal and group isolation.

**Overall:** The paper's competitive hypothesis is the accuracy/locality/sparsity frontier under verifier-constrained sequential edits. The product's competitive proposition is a reliable, auditable correction layer that decreases repeated failures without retraining the base LLM.
