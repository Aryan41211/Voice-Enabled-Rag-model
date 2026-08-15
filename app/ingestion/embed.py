"""Embedding model wrapper.

Lazy-loads a SentenceTransformer model (once per process), batches encodes,
and applies the ``query:``/``passage:`` instruction prefixes that
e5/gte-style models require for retrieval. Model and device come from
configuration (``EMBEDDING_MODEL``, ``DEVICE``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import get_settings


def _requires_prefix(model_name: str) -> bool:
    lowered = model_name.lower()
    return "e5" in lowered or "gte" in lowered


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device_resolved
        self.batch_size = batch_size or settings.embedding_batch_size
        self._prefix = "query: " if _requires_prefix(self.model_name) else ""
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        model = self._load()
        emb = model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode corpus passages with the ``passage:`` prefix if required."""
        prefixed = [f"passage: {t}" for t in texts] if self._prefix else list(texts)
        return self.encode(prefixed, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with the ``query:`` prefix if required."""
        q = f"query: {query}" if self._prefix else query
        return self.encode([q])[0]

    def clear(self) -> None:
        self._model = None
