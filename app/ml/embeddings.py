"""Semantic similarity with sentence-transformers/all-MiniLM-L6-v2 (pretrained, no training).

Loaded once, lazily, from a local folder (backend/models/all-MiniLM-L6-v2) with Hugging Face
offline mode on, so a scan never touches the network. If the model or library is missing,
`available` is False and every caller falls back to string matching (RapidFuzz / shingles).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings

log = logging.getLogger("dataguardian.embeddings")


class SemanticModel:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, path: Path):
        self.path = path
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}
        self.error = ""

    @property
    def installed(self) -> bool:
        return all((self.path / f).exists() for f in ("config.json", "modules.json", "model.safetensors", "tokenizer.json"))

    @property
    def available(self) -> bool:
        return get_settings().semantic_enabled and self.installed and self._load() is not None

    def _load(self):
        if self._model is not None or self.error:
            return self._model
        with self._lock:
            if self._model is None and not self.error:
                try:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(str(self.path), device="cpu")
                    log.info("loaded %s from %s", self.name, self.path)
                except Exception as e:  # missing torch, corrupt files...
                    self.error = f"{type(e).__name__}: {e}"
                    log.warning("semantic model unavailable: %s", self.error)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings (cached per text)."""
        model = self._load()
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = model.encode(missing, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.vstack([self._cache[t] for t in texts]) if texts else np.zeros((0, 384))

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        return float(np.dot(va, vb))

    def matrix(self, left: list[str], right: list[str]) -> np.ndarray:
        if not left or not right:
            return np.zeros((len(left), len(right)))
        return self.encode(left) @ self.encode(right).T

    def document_vector(self, text: str, chunk_words: int = 180, max_chunks: int = 24) -> np.ndarray | None:
        """Mean of chunk embeddings: MiniLM reads 256 tokens at a time, documents are longer."""
        words = re.findall(r"\S+", text)
        if len(words) < 20:
            return None
        chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)][:max_chunks]
        v = self.encode(chunks).mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else None

    def status(self) -> dict:
        return {"name": self.name, "installed": self.installed, "loaded": self._model is not None,
                "path": str(self.path), "error": self.error, "dimensions": 384}


def humanize(col: str) -> str:
    """'cust_mobile_no' -> 'cust mobile no' ; 'SignupDate' -> 'signup date'."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(col))
    return re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()


@lru_cache
def get_semantic() -> SemanticModel:
    return SemanticModel(get_settings().models_dir / "all-MiniLM-L6-v2")
