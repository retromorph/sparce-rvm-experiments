"""Synthetic-only smoke test for RVM plumbing; not an evaluation result."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sparce.evaluation import evaluate_pairs
from sparce.ledger import SnapshotLedger
from sparce.policy import CorrectivePolicy
from sparce.rvm import RVMConfig, SparseBayesianPairRanker


def run_smoke(seed: int = 17) -> dict[str, float | int | bool | str]:
    """Train and evaluate on independent *synthetic* draws and test rollback."""
    rng = np.random.default_rng(seed)

    def examples(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
        a = np.array([[1.0, 0.05]] * (n // 2) + [[-1.0, 0.05]] * (n // 2))
        b = -a
        a += rng.normal(0, 0.08, size=a.shape)
        b += rng.normal(0, 0.08, size=b.shape)
        margin = np.full(n, -0.65)
        return a, b, margin, y

    a_tr, b_tr, d_tr, y_tr = examples(180)
    a_te, b_te, d_te, y_te = examples(180)
    prototypes = np.array([[1, 0], [0.98, 0.2], [0.95, -0.25], [0, 1]])
    model = SparseBayesianPairRanker(RVMConfig(max_basis=4)).fit(
        prototypes, a_tr, b_tr, d_tr, y_tr
    )
    scope = {"package": "fictional-lib", "version": "2.0"}
    policy = CorrectivePolicy(model, scope, min_support=0.6, z_score=1.0)
    report = evaluate_pairs(policy, a_te, b_te, d_te, [scope] * len(y_te), y_te)
    outside = policy.decide(a_te[0], b_te[0], d_te[0], {"package": "other", "version": "2"})
    assert outside.reason == "scope_mismatch"
    with tempfile.TemporaryDirectory() as directory:
        ledger = SnapshotLedger(Path(directory) / "ledger.sqlite")
        first = ledger.commit(policy, reason="synthetic_smoke", evidence_ids=["synthetic:001"])
        rolled = ledger.rollback(first)
        _, restored = ledger.load(rolled)
        assert restored.to_dict() == policy.to_dict()
        ledger.close()
    return {
        "synthetic": True,
        "train_n": len(y_tr),
        "test_n": len(y_te),
        "active_memory_count": model.active_count,
        "candidate_prototype_count": len(prototypes),
        "base_accuracy": report.baseline_correct / report.total,
        "policy_accuracy": report.corrected_correct / report.total,
        "fixes": report.fixes,
        "regressions": report.regressions,
        "scope_mismatch_blocked": True,
        "snapshot_rollback_equal": True,
        "warning": "Synthetic data only: no inference about real code repair.",
    }
