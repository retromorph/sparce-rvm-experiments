"""Minimal paired/statistical evaluation utilities, not a full power analysis."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np


def wilson_upper_onesided_95(errors: int, total: int) -> float:
    """One-sided 95% Wilson interval upper bound for independent Bernoulli trials.

    For correlated API samples, replace by clustered inference; do NOT treat
    this as a distribution-free guarantee under adaptive validation.
    """
    if total <= 0 or errors < 0 or errors > total:
        raise ValueError("Require 0 <= errors <= total and total > 0")
    z = 1.6448536269514722
    p = errors / total
    q = z * z / total
    midpoint = p + q / 2
    correction = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return min(1.0, (midpoint + correction) / (1 + q))


def paired_accuracy_delta_interval(
    base_correct: Sequence[bool],
    treated_correct: Sequence[bool],
    *,
    cluster_ids: Sequence[str] | None = None,
    seed: int = 42,
    n_boot: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired percentile bootstrap. Optional cluster resampling by API family.

    Designed for preregistered analysis, not for adaptive threshold tuning on test.
    """
    base = np.asarray(base_correct, dtype=bool)
    treated = np.asarray(treated_correct, dtype=bool)
    if base.ndim != 1 or treated.shape != base.shape or len(base) < 2:
        raise ValueError("Require two aligned, nontrivial boolean arrays")
    if n_boot < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    if cluster_ids is None:
        groups = [np.array([i]) for i in range(len(base))]
    else:
        if len(cluster_ids) != len(base):
            raise ValueError("Cluster IDs must align with paired outcomes")
        clusters: dict[str, list[int]] = {}
        for idx, cluster in enumerate(cluster_ids):
            clusters.setdefault(str(cluster), []).append(idx)
        groups = [np.asarray(indices, dtype=int) for indices in clusters.values()]
        if len(groups) < 2:
            raise ValueError("Need at least two clusters")
    diffs = treated.astype(float) - base.astype(float)
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for idx in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = np.concatenate([groups[j] for j in chosen])
        out[idx] = diffs[sampled].mean()
    alpha = (1 - confidence) / 2
    low, high = np.quantile(out, [alpha, 1 - alpha])
    return point, float(low), float(high)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape or len(p) == 0:
        raise ValueError("Need aligned, nonempty probabilities and labels")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> dict:
    """ECE with equal-width bins; returns the reliability table as well."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape or len(p) == 0:
        raise ValueError("Need aligned, nonempty probabilities and labels")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf, acc, n = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += n / len(p) * abs(acc - conf)
        table.append({"bin_low": float(lo), "bin_high": float(hi), "n": n, "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "n": int(len(p)), "bins": table}


def cluster_bootstrap_proportion_upper(events: Sequence[bool], cluster_ids: Sequence[str], *, seed: int = 42,
                                       n_boot: int = 2000, confidence: float = 0.95) -> float:
    """One-sided upper bound for a proportion under cluster dependence (percentile bootstrap)."""
    e = np.asarray(events, dtype=float)
    if len(e) == 0 or len(cluster_ids) != len(e):
        raise ValueError("Need aligned nonempty events and cluster ids")
    clusters: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_ids):
        clusters.setdefault(str(c), []).append(i)
    groups = [np.asarray(v, dtype=int) for v in clusters.values()]
    if len(groups) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in chosen])
        out[b] = e[idx].mean()
    return float(min(1.0, np.quantile(out, confidence)))


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, q)) if len(arr) else float("nan")
