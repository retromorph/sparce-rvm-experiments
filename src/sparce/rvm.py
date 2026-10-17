"""Laplace-ARD relevance-vector model for *pairwise corrective scores*.

For verified pair (a,b), y=1 iff a is preferable/correct compared with b.
P(y=1) = sigmoid(base_log_odds(a,b) + (k(a,Z)-k(b,Z)) @ w)
w_j ~ Normal(0, alpha_j^{-1}).  This is an approximate, empirical-Bayes
classification RVM: posterior covariance uses a Laplace approximation.

Importantly, embeddings here represent (request, candidate response) jointly,
NOT just the request. No user-visible logits are altered in this reference.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit


@dataclass(frozen=True)
class RVMConfig:
    length_scale: float = 0.60
    max_basis: int = 64
    max_ard_iterations: int = 80
    max_newton_iterations: int = 70
    initial_alpha: float = 1.0
    prune_alpha: float = 1e6
    min_alpha: float = 1e-6
    max_alpha: float = 1e12
    damping: float = 0.5
    tolerance: float = 5e-3


@dataclass(frozen=True)
class PairPrediction:
    base_margin: float
    correction_mean: float
    correction_sd: float
    corrected_margin: float
    adjusted_probability: float


def _row_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional array of embeddings")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("A candidate embedding has zero length")
    if not np.all(np.isfinite(x)):
        raise ValueError("Candidate embeddings must be finite")
    return x / norm


def _diverse_indices(vectors, budget):
    """Deterministic farthest-first coverage before sparse Bayesian selection.
    This candidate screen is not the Bayesian/ARD selection itself.
    """
    if len(vectors) <= budget:
        return np.arange(len(vectors))
    chosen = [0]
    nearest = np.sum((vectors - vectors[0]) ** 2, axis=1)
    for _ in range(1, budget):
        nearest[chosen] = -1
        i = int(np.argmax(nearest))
        chosen.append(i)
        d = np.sum((vectors - vectors[i]) ** 2, axis=1)
        nearest = np.minimum(nearest, d)
    return np.asarray(chosen, dtype=int)


class SparseBayesianPairRanker:
    def __init__(self, config=None):
        self.config = config if config is not None else RVMConfig()
        if self.config.length_scale <= 0 or self.config.max_basis <= 0:
            raise ValueError("length_scale and max_basis must be positive")
        self.prototypes = None
        self.weights = None
        self.covariance = None
        self.alpha = None
        self.n_train = 0

    @property
    def active_count(self):
        return 0 if self.prototypes is None else len(self.prototypes)

    def _kernel(self, x, z):
        dots = np.clip(x @ z.T, -1.0, 1.0)
        dist_sq = np.maximum(0.0, 2.0 - 2.0 * dots)
        return np.exp(-0.5 * dist_sq / self.config.length_scale**2)

    def _design(self, x_a, x_b, prototypes):
        return self._kernel(x_a, prototypes) - self._kernel(x_b, prototypes)

    @staticmethod
    def _objective(phi, offset, labels, alpha, weights):
        s = offset + phi @ weights
        return float(np.sum(np.logaddexp(0.0, s) - labels * s) + 0.5 * np.dot(alpha, weights * weights))

    def _map_and_covariance(self, phi, offset, labels, alpha, init):
        weights = np.array(init, dtype=np.float64, copy=True)
        identity = np.eye(phi.shape[1])
        for _ in range(self.config.max_newton_iterations):
            s = offset + phi @ weights
            probabilities = expit(s)
            variance = np.maximum(1e-9, probabilities * (1.0 - probabilities))
            gradient = phi.T @ (probabilities - labels) + alpha * weights
            hessian = phi.T @ (variance[:, None] * phi) + np.diag(alpha) + 1e-10 * identity
            factor = cho_factor(hessian, lower=True, check_finite=False)
            step = cho_solve(factor, gradient, check_finite=False)
            old = self._objective(phi, offset, labels, alpha, weights)
            scale = 1.0
            # Monotone damped Newton: important for nearly separable pairs.
            while scale >= 1.0 / 1024:
                candidate = weights - scale * step
                if self._objective(phi, offset, labels, alpha, candidate) <= old + 1e-10:
                    break
                scale *= 0.5
            weights = candidate
            if np.max(np.abs(scale * step)) < 1e-7:
                break
        p = expit(offset + phi @ weights)
        curvature = np.maximum(p * (1 - p), 1e-9)
        hessian = phi.T @ (curvature[:, None] * phi) + np.diag(alpha) + 1e-10 * identity
        factor = cho_factor(hessian, lower=True, check_finite=False)
        covariance = cho_solve(factor, identity, check_finite=False)
        return weights, covariance

    def fit(self, candidate_prototypes, candidate_a, candidate_b, base_log_odds, verified_a_better):
        z = _row_normalize(candidate_prototypes)
        a = _row_normalize(candidate_a)
        b = _row_normalize(candidate_b)
        offset = np.asarray(base_log_odds, dtype=np.float64).reshape(-1)
        labels = np.asarray(verified_a_better, dtype=np.float64).reshape(-1)
        if len(z) == 0 or len(a) == 0:
            raise ValueError("Need candidate memory prototypes and verified training pairs")
        if len(a) != len(b) or len(a) != len(offset) or len(a) != len(labels):
            raise ValueError("Training pairs, labels and base margins must align")
        if a.shape[1] != b.shape[1] or a.shape[1] != z.shape[1]:
            raise ValueError("All embeddings must have the same dimension")
        if not np.all(np.isfinite(offset)) or not np.all(np.isin(labels, [0.0, 1.0])):
            raise ValueError("Offsets must be finite and verified labels must be binary")
        if np.unique(labels).size < 2:
            raise ValueError("Include both positive corrections and negative/control pairs")
        screened = _diverse_indices(z, self.config.max_basis)
        z = z[screened]
        phi = self._design(a, b, z)
        alpha = np.full(len(z), self.config.initial_alpha, dtype=float)
        weights = np.zeros(len(z))
        for _ in range(self.config.max_ard_iterations):
            weights, covariance = self._map_and_covariance(phi, offset, labels, alpha, weights)
            gamma = np.clip(1.0 - alpha * np.diag(covariance), 0.0, 1.0)
            proposed = np.where(gamma < 1e-7, self.config.max_alpha,
                                gamma / np.maximum(weights**2, 1e-12))
            proposed = np.clip(proposed, self.config.min_alpha, self.config.max_alpha)
            old_log = np.log(alpha)
            next_log = (1 - self.config.damping) * old_log + self.config.damping * np.log(proposed)
            alpha = np.exp(next_log)
            retain = alpha < self.config.prune_alpha
            if not retain.any():
                retain[np.argmin(alpha)] = True
            removed = bool(np.any(~retain))
            if removed:
                phi, z, alpha, weights = phi[:, retain], z[retain], alpha[retain], weights[retain]
            if not removed and float(np.max(np.abs(next_log - old_log))) < self.config.tolerance:
                break
        weights, covariance = self._map_and_covariance(phi, offset, labels, alpha, weights)
        self.prototypes = z
        self.weights = weights
        self.covariance = covariance
        self.alpha = alpha
        self.n_train = len(a)
        return self

    def predict(self, candidate_a, candidate_b, base_log_odds):
        if self.active_count == 0:
            raise RuntimeError("Fit a model before calling predict")
        a, b = _row_normalize(candidate_a), _row_normalize(candidate_b)
        offset = np.asarray(base_log_odds, dtype=np.float64).reshape(-1)
        if len(a) != len(b) or len(a) != len(offset):
            raise ValueError("Pair batch and base margins must align")
        if a.shape[1] != self.prototypes.shape[1]:
            raise ValueError("Embedding model/dimension differs from memory snapshot")
        design = self._design(a, b, self.prototypes)
        mean = design @ self.weights
        variances = np.einsum("ij,jk,ik->i", design, self.covariance, design)
        sd = np.sqrt(np.maximum(0.0, variances))
        margin = offset + mean
        # Logistic-normal moment approximation: NOT a calibration guarantee.
        probability = expit(margin / np.sqrt(1 + np.pi * sd**2 / 8))
        return [PairPrediction(float(o), float(m), float(v), float(s), float(p))
                for o, m, v, s, p in zip(offset, mean, sd, margin, probability)]

    def support(self, candidate_a):
        a = _row_normalize(candidate_a)
        if self.active_count == 0:
            return np.zeros(len(a))
        return np.max(self._kernel(a, self.prototypes), axis=1)

    def to_dict(self):
        if self.active_count == 0:
            raise RuntimeError("Cannot snapshot an unfit model")
        return {"config": vars(self.config), "prototypes": self.prototypes.tolist(),
                "weights": self.weights.tolist(), "covariance": self.covariance.tolist(),
                "alpha": self.alpha.tolist(), "n_train": self.n_train}

    @classmethod
    def from_dict(cls, state):
        instance = cls(RVMConfig(**state["config"]))
        instance.prototypes = np.asarray(state["prototypes"], dtype=float)
        instance.weights = np.asarray(state["weights"], dtype=float)
        instance.covariance = np.asarray(state["covariance"], dtype=float)
        instance.alpha = np.asarray(state["alpha"], dtype=float)
        instance.n_train = int(state["n_train"])
        return instance
