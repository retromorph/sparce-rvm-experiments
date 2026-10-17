# SPaRCE PoC — final research report

**Author:** [agent]  
**Date:** [YYYY-MM-DD]  
**Repository commit:** [SHA]  
**MLflow experiment:** [URI/run IDs]  
**Overall decision:** GO / NO-GO / INCONCLUSIVE (explain which gate fails)

## 1. Executive summary

What was tried, strongest defensible empirical finding, biggest negative finding, and what cannot yet be concluded.

## 2. Reproducibility manifest

Frozen LLM + revision, embeddings + revision, GPU + driver + CUDA + torch, Python + uv.lock hash, benchmark upstream commit/license, split manifest sha, scope policy, seed list, hardware peak memory, setup commands, run IDs and full logs.

## 3. Methodology

Describe RVM likelihood, ARD inference, normalization, memory admission, scope and uncertainty gating, verification; any deviations from `docs/EXPERIMENT_PROTOCOL.md` with timestamps and rationale. Include dataset filtering, leakage audit, duplicate audit, budget controls, randomization, missing verifiers and failure handling.

## 4. E1 — Verified error repair

| Method | N | pass@1 (95% CI) | pass@K | paired delta vs base | tokens/task | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen base | | | | | | |
| Rule/doc RAG | | | | | | |
| Full kNN | | | | | | |
| SPaRCE | | | | | | |

Describe seen-family vs held-out-family transfer, case study with source trace, verifier evidence, failure modes, and GO/NO-GO/INCONCLUSIVE reasoning.

## 5. E2 — Accuracy versus sparse memory cost

Plot paired accuracy vs active **bytes**; active count; index bytes; posterior covariance bytes; cold evidence bytes; p50/p95 latency; update wall time; VRAM; storage. Include capacity-matched kNN, random, medoids and RVM ablations. Explicitly distinguish ARD-pruned vectors from screened candidate pool. State memory Pareto result and decision.

## 6. E3 — Regression, continual learning and rollback

Confusion changes with paired numerator/denominator; upper 95% collateral risk bound; wrong-version scope errors; 10/50/100/(500) event retention plot, Brier/ECE reliability, false-memory injection rejects, snapshots and rollback equality test. Include known limitations of posterior credibility and dependence among controls. State decision.

## 7. Statistics

Confidence interval method, cluster unit, bootstrap seeds, multiplicity/selection limitations, actual powers/N, raw results references. Mark unpowered result INCONCLUSIVE.

## 8. Failure analysis

At least 10 concrete failure cases with explanation, provenance and boundary. Distinguish generation candidate not in top K vs memory selection failure vs verifier failure vs scope failure.

## 9. Decisions and next steps

| Gate | Threshold | Observed | Confidence | Decision |
| --- | --- | --- | --- | --- |
| E1 | documented pre-data | | | |
| E2 | documented pre-data | | | |
| E3 | documented pre-data | | | |

Summarize technical debt, risk register, commercial viability, and prioritized next 3 experiments. End with a short **review packet**: report path, manifest path, MLflow experiment/run URLs (tunneled/local links are acceptable), and files the reviewer must inspect.
