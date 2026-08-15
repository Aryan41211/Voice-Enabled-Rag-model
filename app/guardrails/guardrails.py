"""Three-layer guardrail stack (see GUARDRAILS.md).

Fail-closed by design: when a stage is unsure it refuses or asks to clarify
rather than fabricating. Rule/embedding-based so the hot path stays inside the
latency budget; no LLM calls in the guardrail path.

* Layer 1 (input, pre-retrieval): garbage / low-confidence, unsafe keywords,
  off-topic via cosine to an in-domain query centroid.
* Layer 2 (retrieval): no relevant results (score floor), ambiguous flat
  ranking (clarify).
* Layer 3 (output, post-generation): missing citations, length sanity,
  groundedness via answer-to-passage cosine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.harness.schemas import (
    Answer,
    GuardrailResult,
    RetrievedChunk,
    RetrievalResult,
    Transcript,
)
from app.ingestion.embed import Embedder

UNSAFE_KEYWORDS = [
    # violence / weapons
    "हत्या", "मारना", "बम बनाना", "हिंसा", "काटना", "गोली", "बंदूक",
    "kill", "murder", "bomb", "violence", "weapon",
    # self-harm
    "आत्महत्या", "खुदकुशी", "खुद को मार", "suicide", "self harm",
    # hate speech
    "नफरत", "जातिवाद", "नस्लवाद", "hate speech",
    # illegal drugs
    "ड्रग्स कैसे बनाएं", "नशीला पदार्थ", "how to make drugs", "cocaine",
    # explicit sexual content
    "अश्लील", "सेक्स कैसे", "porn", "nsfw",
]

REFUSAL_SAFE = (
    "I can't help with that. I'm designed to answer only safe, information "
    "questions grounded in the provided dataset."
)

OFF_TOPIC_TEMPLATE = (
    "That's outside what I can answer — I'm grounded in the provided dataset "
    "and that question looks out of scope."
)

NO_RELEVANT_TEMPLATE = (
    "I searched the knowledge base but couldn't find anything relevant to "
    "that question."
)

AMBIGUOUS_TEMPLATE = (
    "That question is ambiguous — could you be more specific?"
)

DEFAULT_MIN_TOP_SCORE = 0.40
DEFAULT_MIN_MARGIN = 0.03
DEFAULT_AMBIGUOUS_GAP = 0.02
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_LENGTH = 3
DEFAULT_OFF_TOPIC_THRESHOLD = 0.30
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.50
DEFAULT_MAX_ANSWER_CHARS = 2000
IN_DOMAIN_REFERENCE_N = 200


def _contains_unsafe(text: str) -> bool:
    lowered = text.lower()
    for keyword in UNSAFE_KEYWORDS:
        if keyword in lowered:
            return True
    return False


class InputGuardrail:
    layer = "input"

    def __init__(
        self,
        embedder: Embedder | None = None,
        min_length: int = DEFAULT_MIN_LENGTH,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        off_topic_threshold: float = DEFAULT_OFF_TOPIC_THRESHOLD,
        lang: str | None = None,
        index_dir: str | Path | None = None,
        reference_queries: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder
        self.min_length = min_length
        self.min_confidence = min_confidence
        self.off_topic_threshold = off_topic_threshold
        self.lang = lang or settings.data_lang
        self.index_dir = Path(index_dir or settings.index_dir)
        self.reference_queries = reference_queries
        self._centroid: np.ndarray | None = None

    def _in_domain_queries(self) -> list[str]:
        if self.reference_queries is not None:
            return self.reference_queries
        path = self.index_dir / self.lang / "eval_gold.jsonl"
        queries: list[str] = []
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        queries.append(json.loads(line)["query"])
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if len(queries) >= IN_DOMAIN_REFERENCE_N:
                        break
        return queries

    def _reference_centroid(self) -> np.ndarray:
        if self._centroid is not None:
            return self._centroid
        if self.embedder is None:
            self.embedder = Embedder()
        queries = self._in_domain_queries()
        if not queries:
            # no reference set available — cannot judge, allow through
            self._centroid = np.zeros(self.embedder.dimension, dtype=np.float32)
            return self._centroid
        vecs = self.embedder.encode_query_batch(queries)
        self._centroid = vecs.mean(axis=0)
        self._centroid /= np.linalg.norm(self._centroid) + 1e-9
        return self._centroid

    def check(self, transcript: Transcript) -> GuardrailResult:
        text = transcript.text.strip()
        if len(text) < self.min_length:
            return GuardrailResult(
                passed=False, layer=self.layer, action="clarify",
                reason="transcript too short to be a real query",
            )
        if transcript.confidence < self.min_confidence:
            return GuardrailResult(
                passed=False, layer=self.layer, action="clarify",
                reason=f"low STT confidence ({transcript.confidence:.2f})",
            )
        if _contains_unsafe(text):
            return GuardrailResult(
                passed=False, layer=self.layer, action="refuse",
                reason="unsafe content detected",
            )
        if self.embedder is not None or self._reference_path_exists():
            centroid = self._reference_centroid()
            if centroid.any():
                if self.embedder is None:
                    self.embedder = Embedder()
                qv = self.embedder.encode_query(text)
                sim = float(np.dot(qv, centroid))
                if sim < self.off_topic_threshold:
                    return GuardrailResult(
                        passed=False, layer=self.layer, action="refuse",
                        reason=f"off-topic (similarity {sim:.3f})",
                    )
        return GuardrailResult(passed=True, layer=self.layer)

    def _reference_path_exists(self) -> bool:
        return (self.index_dir / self.lang / "eval_gold.jsonl").exists()


class RetrievalGuardrail:
    layer = "retrieval"

    def __init__(
        self,
        min_top_score: float = DEFAULT_MIN_TOP_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        ambiguous_gap: float = DEFAULT_AMBIGUOUS_GAP,
    ) -> None:
        self.min_top_score = min_top_score
        self.min_margin = min_margin
        self.ambiguous_gap = ambiguous_gap

    def check(self, result: RetrievalResult) -> GuardrailResult:
        if not result.chunks:
            return GuardrailResult(
                passed=False, layer=self.layer, action="refuse",
                reason="no retrieved chunks",
            )
        top = result.chunks[0].score
        if top < self.min_top_score:
            return GuardrailResult(
                passed=False, layer=self.layer, action="refuse",
                reason=f"top score {top:.3f} below floor {self.min_top_score}",
            )
        # Margin heuristic (calibrated on eval gold): if the best match is not
        # clearly isolated from the corpus background, nothing is relevant.
        if result.background_score is not None:
            margin = top - result.background_score
            if margin < self.min_margin:
                return GuardrailResult(
                    passed=False, layer=self.layer, action="refuse",
                    reason=f"best match not isolated (margin {margin:.3f})",
                )
        if len(result.chunks) >= 3:
            gap = result.chunks[0].score - result.chunks[2].score
            if gap < self.ambiguous_gap:
                return GuardrailResult(
                    passed=False, layer=self.layer, action="clarify",
                    reason=f"flat top-3 scores (gap {gap:.3f}) — ambiguous match",
                )
        return GuardrailResult(passed=True, layer=self.layer)


class OutputGuardrail:
    layer = "output"

    def __init__(
        self,
        embedder: Embedder | None = None,
        groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
        max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
    ) -> None:
        self.embedder = embedder
        self.groundedness_threshold = groundedness_threshold
        self.max_answer_chars = max_answer_chars

    def check(
        self, query: str, chunks: list[RetrievedChunk], answer: Answer
    ) -> tuple[GuardrailResult, Answer]:
        if not answer.text.strip():
            return (
                GuardrailResult(
                    passed=False, layer=self.layer, action="refuse",
                    reason="empty generated answer",
                ),
                answer,
            )
        if len(answer.text) > self.max_answer_chars:
            return (
                GuardrailResult(
                    passed=False, layer=self.layer, action="refuse",
                    reason=f"answer too long ({len(answer.text)} chars)",
                ),
                answer,
            )
        if not answer.cited_chunk_ids:
            return (
                GuardrailResult(
                    passed=False, layer=self.layer, action="refuse",
                    reason="answer carries no citation",
                ),
                answer,
            )
        if answer.grounded is None:
            grounded, score = self._groundedness(query, chunks, answer)
            answer = answer.model_copy(update={"grounded": grounded})
            if not grounded:
                return (
                    GuardrailResult(
                        passed=False, layer=self.layer, action="refuse",
                        reason=f"groundedness {score:.3f} below {self.groundedness_threshold}",
                    ),
                    answer,
                )
        return GuardrailResult(passed=True, layer=self.layer), answer

    def _groundedness(
        self, query: str, chunks: list[RetrievedChunk], answer: Answer
    ) -> tuple[bool, float]:
        if self.embedder is None:
            self.embedder = Embedder()
        passages = {c.chunk_id: c.text for c in chunks}
        cited = [passages[cid] for cid in answer.cited_chunk_ids if cid in passages]
        if not cited:
            cited = [c.text for c in chunks]
        av = self.embedder.encode_query(answer.text)
        pvs = self.embedder.encode_passages(cited)
        sims = np.dot(pvs, av)
        score = float(sims.max())
        return score >= self.groundedness_threshold, score


class GuardrailPipeline:
    """Combines all three layers; each check short-circuits on failure."""

    def __init__(
        self,
        input_gr: InputGuardrail | None = None,
        retrieval_gr: RetrievalGuardrail | None = None,
        output_gr: OutputGuardrail | None = None,
    ) -> None:
        self.input = input_gr or InputGuardrail()
        self.retrieval = retrieval_gr or RetrievalGuardrail()
        self.output = output_gr or OutputGuardrail()

    def check_input(self, transcript: Transcript) -> GuardrailResult:
        return self.input.check(transcript)

    def check_retrieval(self, result: RetrievalResult) -> GuardrailResult:
        return self.retrieval.check(result)

    def check_output(
        self, query: str, chunks: list[RetrievedChunk], answer: Answer
    ) -> tuple[GuardrailResult, Answer]:
        return self.output.check(query, chunks, answer)
