"""Corrective-memory methods sharing one interface and one evidence pool.

Every method decides, for a pair (alternative ``a``, incumbent ``b``), whether to
recommend ``a`` over the frozen model's incumbent pick. All methods see exactly
the same candidates, embeddings, base score margin ``d0 = s(a) - s(b)`` and
scope metadata. Hidden labels are used only to fit on the *training* stream and
to evaluate after decisions are made.

Methods
-------
base            frozen model only (never recommends)
rvm             SPaRCE: scoped Laplace-ARD RVM with support + credible-margin gate
rvm_noscope     ablation: RVM without the exact scope gate
rvm_nogate      ablation: scope gate, posterior mean margin only (no credible gate)
knn_full        scoped kNN over the full verified candidate pool
knn_cap         scoped kNN restricted to a capacity-matched, farthest-first screened pool
random          scoped kNN over randomly selected prototypes at matched capacity
medoids         scoped kNN over k-means medoids at matched capacity
rules           deterministic version-aware identifier substitution rules learned
                from verified training repairs (exact scope)
"""

from __future__ import annotations

import io
import json
import keyword
import time
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import expit

from sparce.rvm import RVMConfig, SparseBayesianPairRanker, _diverse_indices, _row_normalize


@dataclass
class PairExample:
    task_id: str
    cluster_id: str
    family_id: str
    kind: str
    split: str
    subset: str
    scope: str                 # "package==version"
    timestamp: str
    emb_a: np.ndarray          # joint embedding of (request, alternative)
    emb_b: np.ndarray          # joint embedding of (request, incumbent)
    d0: float                  # normalised base score margin s(a) - s(b) (<= 0 by construction)
    a_pass: bool | None        # hidden verified outcome of a (None = unverified/timeout)
    b_pass: bool | None
    request_tokens: frozenset[str] = field(default_factory=frozenset)
    tokens_a: frozenset[str] = field(default_factory=frozenset)
    tokens_b: frozenset[str] = field(default_factory=frozenset)
    evidence_bytes: int = 0    # raw evidence size (texts + verifier records), reported as cold storage
    verified: bool = True      # False for injected unverified submissions (must be rejected)
    evidence_hash: str = ""

    @property
    def label(self) -> int | None:
        if self.a_pass is None or self.b_pass is None or self.a_pass == self.b_pass:
            return None
        return int(self.a_pass)


@dataclass(frozen=True)
class Decision:
    recommend_alternative: bool
    reason: str
    margin: float          # mean corrected margin (logit scale) or method score
    lower: float
    sd: float
    support: float
    probability: float     # P(a better than b) used for calibration (RVM); logistic(margin) otherwise

    def to_dict(self) -> dict:
        return asdict(self)


def identifier_tokens(code: str) -> frozenset[str]:
    out: set[str] = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.NAME and not keyword.iskeyword(t.string):
                out.add(t.string)
    except (tokenize.TokenError, SyntaxError, IndexError):
        for w in code.replace("(", " ").replace(")", " ").replace(".", " ").split():
            if w.isidentifier():
                out.add(w)
    return frozenset(out)


# ----------------------------------------------------------------------------- evidence pool
@dataclass
class Prototype:
    embedding: np.ndarray
    label: int             # +1 verified pass, -1 verified fail
    scope: str
    source_task: str
    timestamp: str


def prototype_pool(events: Sequence[PairExample]) -> list[Prototype]:
    pool: list[Prototype] = []
    seen: set[tuple[str, bytes]] = set()
    for e in events:
        if not e.verified:
            continue
        for emb, ok in ((e.emb_a, e.a_pass), (e.emb_b, e.b_pass)):
            if ok is None:
                continue
            key = (e.scope, np.asarray(emb, dtype=np.float32).tobytes())
            if key in seen:
                continue
            seen.add(key)
            pool.append(Prototype(np.asarray(emb, dtype=np.float32), 1 if ok else -1, e.scope, e.task_id, e.timestamp))
    return pool


def screen_pool(pool: list[Prototype], capacity: int | None, strategy: str, seed: int) -> list[Prototype]:
    if capacity is None or len(pool) <= capacity:
        return list(pool)
    X = _row_normalize(np.stack([p.embedding for p in pool]))
    if strategy == "farthest_first":
        idx = _diverse_indices(X, capacity)
    elif strategy == "random":
        idx = np.random.default_rng(seed).choice(len(pool), size=capacity, replace=False)
    elif strategy == "medoids":
        idx = _kmedoids(X, capacity, seed)
    else:
        raise ValueError(strategy)
    return [pool[int(i)] for i in idx]


def _kmedoids(X: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ centers.T, axis=1)
        new = centers.copy()
        for j in range(k):
            members = X[assign == j]
            if len(members):
                c = members.mean(axis=0)
                n = np.linalg.norm(c)
                new[j] = c / n if n > 0 else centers[j]
        if np.allclose(new, centers):
            break
        centers = new
    sims = X @ centers.T
    medoids: list[int] = []
    for j in range(k):
        order = np.argsort(-sims[:, j])
        for i in order:
            if int(i) not in medoids:
                medoids.append(int(i))
                break
    return np.asarray(medoids, dtype=int)


def _bytes_of(arr: np.ndarray) -> int:
    return int(arr.nbytes)


# ----------------------------------------------------------------------------- base
class BaseOnly:
    name = "base"

    def __init__(self, **_: object):
        self.fit_seconds = 0.0

    def fit(self, events: Sequence[PairExample], seed: int = 0) -> None:
        return None

    def decide(self, ex: PairExample) -> Decision:
        return Decision(False, "base_only", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))

    @property
    def active_count(self) -> int:
        return 0

    def bytes_breakdown(self) -> dict[str, int]:
        return {"prototypes": 0, "weights": 0, "covariance": 0, "alpha": 0, "labels": 0, "metadata": 0, "total": 0}

    def snapshot(self) -> dict:
        return {"method": self.name}

    def restore(self, state: dict) -> None:
        return None


# ----------------------------------------------------------------------------- kNN family
class ScopedKNN:
    """Similarity-weighted vote of verified neighbours, per exact scope."""

    def __init__(self, *, capacity: int | None = None, strategy: str = "farthest_first", k: int = 5,
                 threshold: float = 0.2, min_support: float = 0.5, use_scope: bool = True, name: str = "knn_full"):
        self.capacity = capacity
        self.strategy = strategy
        self.k = int(k)
        self.threshold = float(threshold)
        self.min_support = float(min_support)
        self.use_scope = use_scope
        self.name = name
        self.prototypes: list[Prototype] = []
        self._X = np.zeros((0, 1), dtype=np.float32)
        self._y = np.zeros(0, dtype=np.int8)
        self._scopes: list[str] = []
        self.fit_seconds = 0.0

    def fit(self, events: Sequence[PairExample], seed: int = 0) -> None:
        start = time.perf_counter()
        pool = prototype_pool(events)
        self.prototypes = screen_pool(pool, self.capacity, self.strategy, seed)
        if self.prototypes:
            self._X = _row_normalize(np.stack([p.embedding for p in self.prototypes])).astype(np.float32)
        else:
            self._X = np.zeros((0, 1), dtype=np.float32)
        self._y = np.asarray([p.label for p in self.prototypes], dtype=np.int8)
        self._scopes = [p.scope for p in self.prototypes]
        self.fit_seconds = time.perf_counter() - start

    def _score(self, emb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        if not mask.any():
            return 0.0, 0.0
        q = _row_normalize(np.asarray(emb, dtype=np.float64).reshape(1, -1))[0]
        sims = self._X[mask] @ q
        ys = self._y[mask]
        top = np.argsort(-sims)[: self.k]
        w = np.clip(sims[top], 0.0, None)
        support = float(sims[top[0]])
        if w.sum() <= 0:
            return 0.0, support
        return float((w * ys[top]).sum() / w.sum()), support

    def decide(self, ex: PairExample) -> Decision:
        if len(self.prototypes) == 0:
            return Decision(False, "empty_memory", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))
        mask = np.asarray([s == ex.scope for s in self._scopes]) if self.use_scope else np.ones(len(self._scopes), bool)
        if not mask.any():
            return Decision(False, "scope_mismatch", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))
        va, sa = self._score(ex.emb_a, mask)
        vb, sb = self._score(ex.emb_b, mask)
        support = max(sa, sb)
        margin = va - vb
        if support < self.min_support:
            reason = "low_support"
        elif margin <= self.threshold:
            reason = "insufficient_vote_margin"
        else:
            reason = "memory_supported"
        return Decision(reason == "memory_supported", reason, margin, margin, 0.0, support, float(expit(ex.d0 + 4.0 * margin)))

    @property
    def active_count(self) -> int:
        return len(self.prototypes)

    def bytes_breakdown(self) -> dict[str, int]:
        meta = len(json.dumps(self._scopes).encode())
        b = {"prototypes": _bytes_of(self._X), "weights": 0, "covariance": 0, "alpha": 0,
             "labels": _bytes_of(self._y), "metadata": meta}
        b["total"] = sum(b.values())
        return b

    def snapshot(self) -> dict:
        return {"method": self.name, "X": self._X.tolist(), "y": self._y.tolist(), "scopes": self._scopes,
                "k": self.k, "threshold": self.threshold, "min_support": self.min_support}

    def restore(self, state: dict) -> None:
        self._X = np.asarray(state["X"], dtype=np.float32)
        self._y = np.asarray(state["y"], dtype=np.int8)
        self._scopes = list(state["scopes"])
        self.prototypes = [Prototype(self._X[i], int(self._y[i]), self._scopes[i], "", "") for i in range(len(self._y))]


# ----------------------------------------------------------------------------- SPaRCE RVM
class ScopedRVM:
    """One Laplace-ARD pairwise RVM per exact scope; capacity shared proportionally."""

    def __init__(self, *, capacity: int | None = 64, length_scale: float = 0.6, min_support: float = 0.65,
                 margin_threshold: float = 0.05, z_score: float = 1.64, use_scope: bool = True,
                 credible_gate: bool = True, min_pairs: int = 4, name: str = "rvm"):
        self.capacity = capacity
        self.length_scale = float(length_scale)
        self.min_support = float(min_support)
        self.margin_threshold = float(margin_threshold)
        self.z_score = float(z_score)
        self.use_scope = use_scope
        self.credible_gate = credible_gate
        self.min_pairs = int(min_pairs)
        self.name = name
        self.models: dict[str, SparseBayesianPairRanker] = {}
        self.pool_sizes: dict[str, int] = {}
        self.fit_seconds = 0.0
        self.fit_log: dict[str, dict] = {}

    def _partition(self, events: Sequence[PairExample]) -> dict[str, list[PairExample]]:
        parts: dict[str, list[PairExample]] = {}
        for e in events:
            if not e.verified:
                continue
            key = e.scope if self.use_scope else "*"
            parts.setdefault(key, []).append(e)
        return parts

    def fit(self, events: Sequence[PairExample], seed: int = 0) -> None:
        start = time.perf_counter()
        self.models, self.fit_log, self.pool_sizes = {}, {}, {}
        parts = self._partition(events)
        pools = {k: prototype_pool(v) for k, v in parts.items()}
        total_pool = sum(len(p) for p in pools.values()) or 1
        for key, evs in parts.items():
            pairs = [e for e in evs if e.label is not None]
            pool = pools[key]
            self.pool_sizes[key] = len(pool)
            labels = {e.label for e in pairs}
            if len(pairs) < self.min_pairs or len(labels) < 2 or not pool:
                self.fit_log[key] = {"status": "skipped", "n_pairs": len(pairs), "n_pool": len(pool)}
                continue
            cap = None if self.capacity is None else max(2, int(round(self.capacity * len(pool) / total_pool)))
            cap = len(pool) if cap is None else min(cap, len(pool))
            model = SparseBayesianPairRanker(RVMConfig(length_scale=self.length_scale, max_basis=cap))
            Z = np.stack([p.embedding for p in pool])
            A = np.stack([e.emb_a for e in pairs])
            B = np.stack([e.emb_b for e in pairs])
            d0 = np.asarray([e.d0 for e in pairs], dtype=float)
            y = np.asarray([e.label for e in pairs], dtype=float)
            try:
                model.fit(Z, A, B, d0, y)
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.fit_log[key] = {"status": "failed", "error": str(exc)[:200], "n_pairs": len(pairs)}
                continue
            self.models[key] = model
            self.fit_log[key] = {"status": "fit", "n_pairs": len(pairs), "n_pool": len(pool),
                                 "screened": cap, "active": model.active_count}
        self.fit_seconds = time.perf_counter() - start

    def decide(self, ex: PairExample) -> Decision:
        key = ex.scope if self.use_scope else "*"
        model = self.models.get(key)
        if model is None:
            return Decision(False, "scope_mismatch" if self.use_scope else "empty_memory", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))
        a = np.asarray(ex.emb_a, dtype=float).reshape(1, -1)
        b = np.asarray(ex.emb_b, dtype=float).reshape(1, -1)
        support = float(max(model.support(a)[0], model.support(b)[0]))
        pred = model.predict(a, b, [ex.d0])[0]
        lower = pred.corrected_margin - (self.z_score * pred.correction_sd if self.credible_gate else 0.0)
        if support < self.min_support:
            reason = "low_support"
        elif lower <= self.margin_threshold:
            reason = "insufficient_credible_gain"
        else:
            reason = "posterior_supported"
        return Decision(reason == "posterior_supported", reason, pred.corrected_margin, float(lower),
                        pred.correction_sd, support, pred.adjusted_probability)

    @property
    def active_count(self) -> int:
        return sum(m.active_count for m in self.models.values())

    def bytes_breakdown(self) -> dict[str, int]:
        b = {"prototypes": 0, "weights": 0, "covariance": 0, "alpha": 0, "labels": 0, "metadata": 0}
        for key, m in self.models.items():
            b["prototypes"] += _bytes_of(m.prototypes.astype(np.float32))
            b["weights"] += _bytes_of(m.weights)
            b["covariance"] += _bytes_of(m.covariance)
            b["alpha"] += _bytes_of(m.alpha)
            b["metadata"] += len(json.dumps({"scope": key, "config": vars(m.config), "gates": [self.min_support, self.margin_threshold, self.z_score]}).encode())
        b["total"] = sum(b.values())
        return b

    def snapshot(self) -> dict:
        return {"method": self.name, "models": {k: m.to_dict() for k, m in self.models.items()},
                "min_support": self.min_support, "margin_threshold": self.margin_threshold,
                "z_score": self.z_score, "use_scope": self.use_scope, "credible_gate": self.credible_gate}

    def restore(self, state: dict) -> None:
        self.models = {k: SparseBayesianPairRanker.from_dict(v) for k, v in state["models"].items()}
        self.min_support = float(state["min_support"])
        self.margin_threshold = float(state["margin_threshold"])
        self.z_score = float(state["z_score"])


# ----------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class SubstitutionRule:
    scope: str
    removed: frozenset[str]
    added: frozenset[str]


class VersionRules:
    """Exact-scope identifier substitution rules mined from verified training repairs."""

    name = "rules"

    def __init__(self, *, max_rule_size: int = 3, min_count: int = 1, **_: object):
        self.max_rule_size = int(max_rule_size)
        self.min_count = int(min_count)
        self.rules: dict[SubstitutionRule, int] = {}
        self.fit_seconds = 0.0

    def fit(self, events: Sequence[PairExample], seed: int = 0) -> None:
        start = time.perf_counter()
        counts: dict[SubstitutionRule, int] = {}
        for e in events:
            if not e.verified or e.a_pass is None or e.b_pass is None or e.a_pass == e.b_pass:
                continue
            good, bad = (e.tokens_a, e.tokens_b) if e.a_pass else (e.tokens_b, e.tokens_a)
            removed = frozenset(bad - good)
            added = frozenset(good - bad)
            if not removed or not added or len(removed) > self.max_rule_size or len(added) > self.max_rule_size:
                continue
            rule = SubstitutionRule(e.scope, removed, added)
            counts[rule] = counts.get(rule, 0) + 1
        self.rules = {r: c for r, c in counts.items() if c >= self.min_count}
        self.fit_seconds = time.perf_counter() - start

    def _consistent(self, ex: PairExample, tokens: frozenset[str]) -> bool:
        for rule in self.rules:
            if rule.scope != ex.scope or not rule.removed <= ex.request_tokens:
                continue
            if rule.added <= tokens and not (rule.removed & tokens):
                return True
        return False

    def decide(self, ex: PairExample) -> Decision:
        if not any(r.scope == ex.scope for r in self.rules):
            return Decision(False, "scope_mismatch", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))
        ca, cb = self._consistent(ex, ex.tokens_a), self._consistent(ex, ex.tokens_b)
        if ca and not cb:
            return Decision(True, "rule_match", 1.0, 1.0, 0.0, 1.0, 1.0)
        return Decision(False, "no_rule_preference", ex.d0, ex.d0, 0.0, 0.0, float(expit(ex.d0)))

    @property
    def active_count(self) -> int:
        return len(self.rules)

    def bytes_breakdown(self) -> dict[str, int]:
        meta = len(json.dumps([[r.scope, sorted(r.removed), sorted(r.added)] for r in self.rules]).encode())
        return {"prototypes": 0, "weights": 0, "covariance": 0, "alpha": 0, "labels": 0, "metadata": meta, "total": meta}

    def snapshot(self) -> dict:
        return {"method": self.name, "rules": [[r.scope, sorted(r.removed), sorted(r.added), c] for r, c in self.rules.items()]}

    def restore(self, state: dict) -> None:
        self.rules = {SubstitutionRule(s, frozenset(rm), frozenset(ad)): c for s, rm, ad, c in state["rules"]}


# ----------------------------------------------------------------------------- factory
def make_method(name: str, *, capacity: int | None = None, seed: int = 0, params: dict | None = None):
    p = dict(params or {})
    if name == "base":
        return BaseOnly()
    if name == "rvm":
        return ScopedRVM(capacity=capacity, name="rvm", **p)
    if name == "rvm_noscope":
        return ScopedRVM(capacity=capacity, use_scope=False, name="rvm_noscope", **p)
    if name == "rvm_nogate":
        return ScopedRVM(capacity=capacity, credible_gate=False, name="rvm_nogate", **p)
    if name == "knn_full":
        return ScopedKNN(capacity=None, name="knn_full", **p)
    if name == "knn_cap":
        return ScopedKNN(capacity=capacity, strategy="farthest_first", name="knn_cap", **p)
    if name == "random":
        return ScopedKNN(capacity=capacity, strategy="random", name="random", **p)
    if name == "medoids":
        return ScopedKNN(capacity=capacity, strategy="medoids", name="medoids", **p)
    if name == "rules":
        return VersionRules(**p)
    raise ValueError(f"unknown method {name}")


def cold_evidence_bytes(events: Sequence[PairExample]) -> int:
    return int(sum(e.evidence_bytes for e in events if e.verified))
