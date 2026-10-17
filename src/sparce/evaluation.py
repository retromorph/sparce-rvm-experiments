"""Paired, label-blind evaluation. Labels are only used after predictions."""
from dataclasses import dataclass
import numpy as np
from .policy import CorrectivePolicy


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    baseline_correct: int
    corrected_correct: int
    fixes: int
    regressions: int
    memory_activations: int

    @property
    def delta_accuracy(self):
        return (self.fixes - self.regressions) / self.total


def evaluate_pairs(policy, alternatives, incumbents, margins, contexts, labels):
    """Labels: 1 if alternative is correct, 0 if incumbent is correct.

    Assumes exactly one candidate is correct; task selection must make this true.
    Query labels MUST NOT be used to build contexts, fit, or select gates.
    """
    a, b = np.asarray(alternatives), np.asarray(incumbents)
    margins, labels = np.asarray(margins).reshape(-1), np.asarray(labels).reshape(-1)
    n = len(labels)
    if n == 0 or len(a) != n or len(b) != n or len(margins) != n or len(contexts) != n:
        raise ValueError("Holdout rows must align and be nonempty")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be verified binary outcomes")
    baseline_picks_alternative = margins > 0
    policy_picks_alternative = np.array([
        baseline_picks_alternative[i] or policy.decide(
            a[i], b[i], float(margins[i]), contexts[i]).recommend_alternative
        for i in range(n)], dtype=bool)
    correct_base = baseline_picks_alternative == labels.astype(bool)
    correct_policy = policy_picks_alternative == labels.astype(bool)
    return EvaluationResult(n, int(correct_base.sum()), int(correct_policy.sum()),
                            int((~correct_base & correct_policy).sum()),
                            int((correct_base & ~correct_policy).sum()),
                            int((~baseline_picks_alternative & policy_picks_alternative).sum()))


def gate_candidate_update(old_result, new_result, *, min_gain=0.0, max_regressions=0):
    """Operational gate on an isolated validation set, not a formal risk bound."""
    if old_result.total != new_result.total:
        raise ValueError("Comparable validation results must share a holdout")
    delta = (new_result.corrected_correct - old_result.corrected_correct) / old_result.total
    # Limits collateral regressions in the *new* candidate vs the baseline.
    return delta >= min_gain and new_result.regressions <= max_regressions
