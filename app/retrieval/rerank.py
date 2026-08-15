"""Optional cross-encoder reranker.

Reranks a small candidate list from the fast ANN/BM25 stage using a
cross-encoder, which attends over the (query, chunk) pair directly. This is
the only stage in the hot path that calls a transformer per candidate, so it
is toggleable and its latency is reported separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import get_settings
from app.harness.schemas import RetrievedChunk

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or DEFAULT_RERANKER
        self.device = device or settings.device_resolved
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=512
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        model = self._load()
        pairs = [(query, text) for text in texts]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype="float32")

    def rerank(
        self,
        query: str,
        hits: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        if not hits:
            return hits
        texts = [h.text for h in hits]
        scores = self.score(query, texts)
        order = np.argsort(scores)[::-1]
        reranked = []
        for rank, idx in enumerate(order):
            hit = hits[int(idx)].model_copy(update={"score": float(scores[idx])})
            reranked.append(hit)
        if top_n is not None:
            reranked = reranked[:top_n]
        return reranked

    def clear(self) -> None:
        self._model = None
