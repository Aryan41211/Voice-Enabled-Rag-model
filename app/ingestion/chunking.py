"""Chunking strategies.

MS MARCO passages are short, self-contained snippets (~300 chars), so the
strategies here are designed to *organize* short passages meaningfully rather
than aggressively split long ones (see CHUNKING_STRATEGY.md).

Every strategy consumes the passages of one MSMARCO-XI example and emits
:class:`Chunk` objects with a consistent schema:

* ``text``    — the searchable unit (embedded / BM25-indexed)
* ``context`` — the text handed to the LLM when this chunk is retrieved
  (equals ``text`` for most strategies; the *window* / *parent* for
  sentence-window and hierarchical chunking).

Strategies:
1. fixed          — fixed-size token windows with overlap (baseline)
2. semantic       — sentence groups merged while cosine similarity holds
3. sentence_window— index sentences, return surrounding-sentence windows
4. metadata       — passage-level chunks carrying full metadata (filterable)
5. hierarchical   — index sentences (children), return the full passage (parent)
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

STRATEGIES = ("fixed", "semantic", "sentence_window", "metadata", "hierarchical")


class Chunk(BaseModel):
    chunk_id: str
    text: str
    context: str
    source_query_id: int
    passage_index: int
    language: str
    strategy: str
    position: int
    passage_is_selected: int = 0
    parent_chunk_id: str | None = None


class Embedder(Protocol):
    """Minimal interface for any sentence embedder used by chunking."""

    def encode(self, texts: list[str]) -> "object": ...


# --------------------------------------------------------------------------
# text utilities
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation (incl. Devanagari danda)."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), used for window sizing only."""
    return max(1, len(text) // 4)


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _cosine(a: "object", b: "object") -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def group_by_similarity(
    sentences: list[str],
    vectors: "object",
    threshold: float,
    max_sentences: int,
) -> list[list[str]]:
    """Merge adjacent sentences while consecutive cosine sim >= threshold."""
    if not sentences:
        return []
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine(vectors[i - 1], vectors[i])
        if sim >= threshold and len(groups[-1]) < max_sentences:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])
    return groups


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------


class FixedSizeChunker:
    """Strategy 1 — fixed-size windows with overlap (baseline)."""

    name = "fixed"

    def __init__(self, chunk_size: int = 150, overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        tokens = _words(text)
        if len(tokens) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(" ".join(tokens[start:end]))
            if end == len(tokens):
                break
            start = max(start + self.chunk_size - self.overlap, start + 1)
        return chunks


class SemanticChunker:
    """Strategy 2 — merge sentences while consecutive similarity holds."""

    name = "semantic"

    def __init__(
        self,
        threshold: float = 0.75,
        max_sentences: int = 8,
        embedder: Embedder | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_sentences = max_sentences
        self.embedder = embedder

    def group_sentences(self, sentences: list[str]) -> list[list[str]]:
        if not sentences:
            return []
        if self.embedder is None:
            return [list(sentences)]
        vecs = self.embedder.encode(sentences)
        return group_by_similarity(sentences, vecs, self.threshold, self.max_sentences)

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        groups = self.group_sentences(sentences)
        return [" ".join(g) for g in groups if g]


class SentenceWindowChunker:
    """Strategy 3 — index sentences, retrieve with a surrounding window."""

    name = "sentence_window"

    def __init__(self, window_size: int = 2) -> None:
        self.window_size = window_size

    def windows(self, sentences: list[str]) -> list[dict]:
        out: list[dict] = []
        for i, s in enumerate(sentences):
            lo = max(0, i - self.window_size)
            hi = min(len(sentences), i + self.window_size + 1)
            window = sentences[lo:hi]
            out.append({"sentence": s, "window": " ".join(window)})
        return out


class MetadataChunker:
    """Strategy 4 — passage-level chunks with full metadata (filterable)."""

    name = "metadata"

    def split_text(self, text: str) -> list[str]:
        return [text.strip()] if text.strip() else []


class HierarchicalChunker:
    """Strategy 5 — sentence children searched, full passage parent returned."""

    name = "hierarchical"

    def children(self, text: str) -> list[str]:
        return [s for s in split_sentences(text) if s.strip()]


# --------------------------------------------------------------------------
# example → chunks
# --------------------------------------------------------------------------


def chunk_example(
    example: dict,
    strategy: str,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Convert one MSMARCO-XI example's passages into chunks for a strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}', choose from {STRATEGIES}")

    passages = example["passages"]["Translated_passages"]
    selected = example["passages"]["is_selected"]
    lang = example["target_lang"]
    qid = example["query_id"]

    chunks: list[Chunk] = []
    pos = 0

    for pidx, passage in enumerate(passages):
        is_sel = int(selected[pidx]) if pidx < len(selected) else 0
        prefix = f"{lang}:{strategy}:{qid}:{pidx}"

        if strategy == "fixed":
            unit = FixedSizeChunker()
            for piece in unit.split_text(passage):
                chunks.append(
                    Chunk(
                        chunk_id=f"{prefix}:{pos}",
                        text=piece,
                        context=piece,
                        source_query_id=qid,
                        passage_index=pidx,
                        language=lang,
                        strategy=strategy,
                        position=pos,
                        passage_is_selected=is_sel,
                    )
                )
                pos += 1

        elif strategy == "semantic":
            unit = SemanticChunker(embedder=embedder)
            for piece in unit.split_text(passage):
                chunks.append(
                    Chunk(
                        chunk_id=f"{prefix}:{pos}",
                        text=piece,
                        context=piece,
                        source_query_id=qid,
                        passage_index=pidx,
                        language=lang,
                        strategy=strategy,
                        position=pos,
                        passage_is_selected=is_sel,
                    )
                )
                pos += 1

        elif strategy == "sentence_window":
            unit = SentenceWindowChunker()
            for win in unit.windows(split_sentences(passage)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{prefix}:{pos}",
                        text=win["sentence"],
                        context=win["window"],
                        source_query_id=qid,
                        passage_index=pidx,
                        language=lang,
                        strategy=strategy,
                        position=pos,
                        passage_is_selected=is_sel,
                    )
                )
                pos += 1

        elif strategy == "metadata":
            unit = MetadataChunker()
            for piece in unit.split_text(passage):
                chunks.append(
                    Chunk(
                        chunk_id=f"{prefix}:{pos}",
                        text=piece,
                        context=piece,
                        source_query_id=qid,
                        passage_index=pidx,
                        language=lang,
                        strategy=strategy,
                        position=pos,
                        passage_is_selected=is_sel,
                    )
                )
                pos += 1

        elif strategy == "hierarchical":
            unit = HierarchicalChunker()
            parent_text = passage.strip()
            parent_id = f"{prefix}:parent"
            children = unit.children(passage)
            if not children:
                continue
            for cidx, child in enumerate(children):
                chunks.append(
                    Chunk(
                        chunk_id=f"{prefix}:{pos}",
                        text=child,
                        context=parent_text,
                        source_query_id=qid,
                        passage_index=pidx,
                        language=lang,
                        strategy=strategy,
                        position=pos,
                        passage_is_selected=is_sel,
                        parent_chunk_id=parent_id,
                    )
                )
                pos += 1

    return chunks


def chunk_examples(
    examples: list[dict],
    strategy: str,
    embedder: Embedder | None = None,
    max_examples: int | None = None,
    semantic_threshold: float = 0.75,
    semantic_max_sentences: int = 8,
) -> list[Chunk]:
    """Apply ``chunk_example`` across examples, optionally capping the count.

    The ``semantic`` strategy uses a single batched embedding call across all
    sentences (much faster than one ``encode`` call per passage).
    """
    if strategy == "semantic" and embedder is not None:
        return _chunk_examples_semantic_batched(
            examples,
            embedder,
            max_examples=max_examples,
            threshold=semantic_threshold,
            max_sentences=semantic_max_sentences,
        )
    out: list[Chunk] = []
    for i, ex in enumerate(examples):
        if max_examples is not None and i >= max_examples:
            break
        out.extend(chunk_example(ex, strategy, embedder=embedder))
    return out


def _chunk_examples_semantic_batched(
    examples: list[dict],
    embedder: Embedder,
    max_examples: int | None = None,
    threshold: float = 0.75,
    max_sentences: int = 8,
) -> list[Chunk]:
    """Semantic chunking with one batched encode for all sentences."""
    items: list[tuple[int, int, str]] = []
    for i, ex in enumerate(examples):
        if max_examples is not None and i >= max_examples:
            break
        passages = ex["passages"]["Translated_passages"]
        for pidx, passage in enumerate(passages):
            for sent in split_sentences(passage):
                items.append((i, pidx, sent))
    if not items:
        return []

    texts = [t for _, _, t in items]
    vecs = embedder.encode(texts)

    out: list[Chunk] = []
    pos = 0
    cursor = 0
    n = len(items)
    while cursor < n:
        i, pidx, _ = items[cursor]
        ex = examples[i]
        selected = ex["passages"]["is_selected"]
        is_sel = int(selected[pidx]) if pidx < len(selected) else 0
        lang = ex["target_lang"]
        qid = ex["query_id"]
        prefix = f"{lang}:semantic:{qid}:{pidx}"

        pass_sents: list[str] = []
        pass_vecs: list[object] = []
        while cursor < n and items[cursor][0] == i and items[cursor][1] == pidx:
            pass_sents.append(items[cursor][2])
            pass_vecs.append(vecs[cursor])
            cursor += 1

        for group in group_by_similarity(
            pass_sents, pass_vecs, threshold, max_sentences
        ):
            text = " ".join(group)
            out.append(
                Chunk(
                    chunk_id=f"{prefix}:{pos}",
                    text=text,
                    context=text,
                    source_query_id=qid,
                    passage_index=pidx,
                    language=lang,
                    strategy="semantic",
                    position=pos,
                    passage_is_selected=is_sel,
                )
            )
            pos += 1
    return out
