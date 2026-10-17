"""Joint (request, candidate) embeddings with a pinned sentence-transformer.

The joint text is an *edit-focused* representation: the target scope, the
first line of the observed error, and the unified diff between the request
code and the candidate. The identical embedder and text construction is used by
every embedding-based method (RVM, kNN, random, medoids), so none is favoured.
Hidden tests and reference fixes never enter the text.
"""

from __future__ import annotations

import difflib
import hashlib
from typing import Sequence

import numpy as np

MAX_DIFF_LINES = 60


def error_headline(stderr_tail: str) -> str:
    lines = [ln.strip() for ln in (stderr_tail or "").strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ":" in ln and not ln.startswith("File ") and not ln.startswith("^"):
            return ln[:200]
    return lines[-1][:200] if lines else ""


def joint_text(package: str, version: str, request_code: str, error_tail: str, candidate_code: str) -> str:
    diff = list(difflib.unified_diff(request_code.splitlines(), candidate_code.splitlines(), lineterm="", n=0))
    diff = [d for d in diff if not d.startswith(("---", "+++", "@@"))]
    if not diff:
        diff_text = "(no change)"
    else:
        diff_text = "\n".join(diff[:MAX_DIFF_LINES])
    return f"scope: {package}=={version}\nerror: {error_headline(error_tail)}\nedit:\n{diff_text}"


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()[:16]


class JointEmbedder:
    def __init__(self, model_id: str, revision: str, device: str | None = None, batch_size: int = 32,
                 local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = revision
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(model_id, revision=revision, device=device, local_files_only=local_files_only)
        self.model.max_seq_length = 256
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def metadata(self) -> dict:
        return {"embedding_id": self.model_id, "embedding_revision": self.revision,
                "dimension": self.dimension, "max_seq_length": self.model.max_seq_length}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vec = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)


class HashEmbedder:
    """Deterministic CPU stand-in for tests (bag-of-tokens hashing). Not for results."""

    model_id = "hash-embedder"
    revision = "n/a"

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def metadata(self) -> dict:
        return {"embedding_id": self.model_id, "embedding_revision": self.revision, "dimension": self.dimension}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.replace("\n", " ").split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dimension] += 1.0 if (h >> 8) % 2 else -1.0
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n > 0 else out[i] + 1.0 / np.sqrt(self.dimension)
        return out
