"""Interfaces for agent-to-verifier research integration.

These contracts intentionally have no model downloads or untrusted code execution.
Use a version-pinned isolated executor for the concrete verifier implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CodeRequest:
    event_id: str
    prompt: str
    package: str
    package_version: str
    api_family: str
    event_timestamp: str
    environment_digest: str

    def scope(self) -> dict[str, str]:
        return {"package": self.package, "version": self.package_version}


@dataclass(frozen=True)
class Candidate:
    text: str
    # Mean token log likelihood, computed by the frozen LLM with a documented mask.
    normalized_log_likelihood: float


@dataclass(frozen=True)
class VerificationOutcome:
    # A failure is not always a provably incorrect solution (e.g. verifier timeout).
    status: str  # "pass", "fail", "timeout", "unverified"
    verifier_id: str
    evidence_hash: str
    execution_ms: float

    @property
    def passed(self) -> bool:
        return self.status == "pass"


class FrozenCodeLLM(Protocol):
    model_revision: str

    def generate_candidates(self, request: CodeRequest, k: int) -> list[Candidate]: ...

    def embed_joint(self, request: CodeRequest, candidate: Candidate) -> np.ndarray: ...


class IsolatedVerifier(Protocol):
    """Must run generated code in a resource-limited, network-disabled sandbox."""

    def verify(self, request: CodeRequest, candidate: Candidate) -> VerificationOutcome: ...


@dataclass(frozen=True)
class VerifiedPair:
    """Paired training record; NEVER infer correctness from the LLM's score.

    A pair may be doubly incorrect. Only admitted positive/negative comparisons
    with genuine independent verification may feed the binary RVM likelihood.
    """

    request: CodeRequest
    incumbent: Candidate
    alternative: Candidate
    incumbent_verification: VerificationOutcome
    alternative_verification: VerificationOutcome

    @property
    def pair_label(self) -> int | None:
        a = self.alternative_verification.status
        b = self.incumbent_verification.status
        if a not in ("pass", "fail") or b not in ("pass", "fail"):
            return None
        if a == b:
            return None  # ambiguous binary preference: do not fabricate a label
        return int(a == "pass")
