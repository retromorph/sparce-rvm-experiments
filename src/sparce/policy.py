"""Conservative runtime activation policy around an RVM posterior."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from .rvm import SparseBayesianPairRanker


@dataclass(frozen=True)
class CorrectionDecision:
    recommend_alternative: bool
    auto_apply: bool
    reason: str
    base_margin: float
    corrected_margin: float
    correction_sd: float
    lower_margin: float
    prototype_support: float


class CorrectivePolicy:
    """Requires deterministic metadata scope + local support + credible margin.

    candidate_a is the proposed alternative; candidate_b is the incumbent.
    In production, ``auto_apply`` needs independent runtime verification of a.
    ``recommend_alternative`` is a ranker decision; evaluate it against *hidden*
    ground-truth labels without passing those labels into the policy.
    """
    def __init__(self, model: SparseBayesianPairRanker, scope: Mapping[str, str],
                 *, min_support=0.65, margin_threshold=0.05, z_score=1.64):
        self.model = model
        self.scope = dict(scope)
        self.min_support = float(min_support)
        self.margin_threshold = float(margin_threshold)
        self.z_score = float(z_score)
        if not 0 <= self.min_support <= 1 or self.z_score < 0:
            raise ValueError("Invalid gate thresholds")

    def decide(self, alternative, incumbent, base_margin, request_metadata,
               *, independent_verification=False):
        base = float(base_margin)
        # No update may be used outside an *exact* scope. Version conditions can
        # be extended later via a carefully validated explicit rules engine.
        if any(str(request_metadata.get(k)) != str(v) for k, v in self.scope.items()):
            return CorrectionDecision(False, False, "scope_mismatch", base, base, 0.0, base, 0.0)
        a = np.asarray(alternative, dtype=float).reshape(1, -1)
        b = np.asarray(incumbent, dtype=float).reshape(1, -1)
        support = float(self.model.support(a)[0])
        prediction = self.model.predict(a, b, [base])[0]
        lower = prediction.corrected_margin - self.z_score * prediction.correction_sd
        if support < self.min_support:
            reason = "low_support"
        elif lower <= self.margin_threshold:
            reason = "insufficient_credible_gain"
        else:
            reason = "posterior_supported"
        recommend = reason == "posterior_supported"
        return CorrectionDecision(recommend, recommend and independent_verification,
                                  reason, base, prediction.corrected_margin,
                                  prediction.correction_sd, float(lower), support)

    def to_dict(self):
        return {"model": self.model.to_dict(), "scope": self.scope,
                "min_support": self.min_support,
                "margin_threshold": self.margin_threshold, "z_score": self.z_score}

    @classmethod
    def from_dict(cls, state):
        return cls(SparseBayesianPairRanker.from_dict(state["model"]), state["scope"],
                   min_support=state["min_support"],
                   margin_threshold=state["margin_threshold"],
                   z_score=state["z_score"])
