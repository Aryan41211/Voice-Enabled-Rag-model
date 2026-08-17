import numpy as np

from app.guardrails.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    RetrievalGuardrail,
    _contains_unsafe,
)
from app.harness.schemas import (
    Answer,
    RetrievedChunk,
    RetrievalResult,
    Transcript,
)


def _transcript(text, confidence=1.0):
    return Transcript(text=text, confidence=confidence)


def _chunk(cid, text, score):
    return RetrievedChunk(chunk_id=cid, text=text, score=score, source="dense")


class FakeEmbedder:
    def __init__(self, in_domain_vecs=None, answer_sims=None):
        self._dim = 4
        self._in_domain = np.asarray(in_domain_vecs or [[1, 0, 0, 0]], dtype="float32")
        self._answer_sims = answer_sims or {}
        self.query_calls = 0

    @property
    def dimension(self):
        return self._dim

    def encode_query(self, text):
        self.query_calls += 1
        # fake: in-domain queries are near (1,0,0,0)
        if "बेटिंग" in text or "गलत" in text:
            return np.asarray([-1, 0, 0, 0], dtype="float32")
        return np.asarray([1, 0, 0, 0], dtype="float32")

    def encode_query_batch(self, queries):
        return np.asarray([[1, 0, 0, 0]] * len(queries), dtype="float32")

    def encode_passages(self, texts):
        return np.asarray([[1, 0, 0, 0]] * len(texts), dtype="float32")


def test_unsafe_keyword_detection():
    assert _contains_unsafe("मुझे हत्या कैसे करूँ")
    assert _contains_unsafe("how to make a bomb")
    assert _contains_unsafe("खुद को नुकसान पहुंचाने का तरीका")
    assert not _contains_unsafe("दिल्ली की राजधानी क्या है")


def test_input_empty_short_clarifies(tmp_path):
    gr = InputGuardrail(embedder=FakeEmbedder(), index_dir=tmp_path)
    assert gr.check(_transcript("ह")).action == "clarify"


def test_input_single_token_clarifies(tmp_path):
    gr = InputGuardrail(embedder=FakeEmbedder(), index_dir=tmp_path)
    assert gr.check(_transcript("xyz")).action == "clarify"


def test_input_low_confidence_clarifies(tmp_path):
    gr = InputGuardrail(embedder=FakeEmbedder(), index_dir=tmp_path)
    assert gr.check(_transcript("सवाल", confidence=0.2)).action == "clarify"


def test_input_unsafe_refuses(tmp_path):
    gr = InputGuardrail(embedder=FakeEmbedder(), index_dir=tmp_path)
    assert gr.check(_transcript("मुझे हत्या कैसे करूँ")).action == "refuse"


def test_input_off_topic_refuses():
    gr = InputGuardrail()
    assert gr.check(_transcript("बेटिंग के नियम क्या हैं")).action == "refuse"


def test_input_in_domain_proceeds():
    gr = InputGuardrail()
    assert gr.check(_transcript("दिल्ली कहाँ है")).action == "proceed"


def test_retrieval_no_chunks_refuses():
    gr = RetrievalGuardrail()
    assert gr.check(RetrievalResult(query="q")).action == "refuse"


def test_retrieval_low_score_refuses():
    gr = RetrievalGuardrail(min_top_score=0.6)
    result = RetrievalResult(query="q", chunks=[_chunk("c1", "t", 0.2)])
    assert gr.check(result).action == "refuse"


def test_retrieval_ambiguous_clarifies():
    gr = RetrievalGuardrail(min_top_score=0.0, ambiguous_gap=0.05)
    result = RetrievalResult(
        query="q",
        chunks=[
            _chunk("c1", "a", 0.70),
            _chunk("c2", "b", 0.68),
            _chunk("c3", "c", 0.67),
        ],
    )
    assert gr.check(result).action == "clarify"


def test_retrieval_not_isolated_refuses():
    gr = RetrievalGuardrail(min_top_score=0.0, min_margin=0.05, ambiguous_gap=0.0)
    result = RetrievalResult(
        query="q",
        chunks=[_chunk("c1", "a", 0.70)],
        background_score=0.68,
    )
    assert gr.check(result).action == "refuse"


def test_retrieval_isolated_proceeds():
    gr = RetrievalGuardrail(min_top_score=0.0, min_margin=0.05, ambiguous_gap=0.0)
    result = RetrievalResult(
        query="q",
        chunks=[_chunk("c1", "a", 0.70)],
        background_score=0.40,
    )
    assert gr.check(result).passed is True


def test_retrieval_clear_proceeds():
    gr = RetrievalGuardrail(min_top_score=0.5, ambiguous_gap=0.05)
    result = RetrievalResult(
        query="q",
        chunks=[
            _chunk("c1", "a", 0.90),
            _chunk("c2", "b", 0.60),
            _chunk("c3", "c", 0.55),
        ],
    )
    assert gr.check(result).passed is True


def test_output_empty_refuses():
    gr = OutputGuardrail(embedder=FakeEmbedder())
    result, ans = gr.check("q", [], Answer(text="  "))
    assert result.action == "refuse"


def test_output_missing_citation_refuses():
    gr = OutputGuardrail(embedder=FakeEmbedder())
    result, ans = gr.check(
        "q", [_chunk("c1", "t", 1.0)], Answer(text="उत्तर", grounded=True)
    )
    assert result.action == "refuse"


def test_output_grounded_answer_passes():
    gr = OutputGuardrail(embedder=FakeEmbedder(), groundedness_threshold=0.0)
    chunks = [_chunk("c1", "दिल्ली राजधानी है", 1.0)]
    result, ans = gr.check("q", chunks, Answer(text="दिल्ली", cited_chunk_ids=["c1"]))
    assert result.passed is True
    assert ans.grounded is True


def test_input_embedding_offtopic_refuses():
    gr = InputGuardrail(
        embedder=FakeEmbedder(),
        reference_queries=["प्रश्न एक", "प्रश्न दो"],
        use_embedding_offtopic=True,
        off_topic_threshold=0.5,
    )
    assert gr.check(_transcript("गलत सवाल")).action == "refuse"


def test_input_embedding_offtopic_in_domain_proceeds():
    gr = InputGuardrail(
        embedder=FakeEmbedder(),
        reference_queries=["प्रश्न एक", "प्रश्न दो"],
        use_embedding_offtopic=True,
        off_topic_threshold=0.5,
    )
    assert gr.check(_transcript("दिल्ली कहाँ है")).passed is True


def test_input_embedding_offtopic_no_reference_allows():
    gr = InputGuardrail(
        embedder=FakeEmbedder(),
        reference_queries=[],
        use_embedding_offtopic=True,
        off_topic_threshold=0.5,
    )
    assert gr.check(_transcript("गलत सवाल")).passed is True


def test_output_too_long_refuses():
    gr = OutputGuardrail(embedder=FakeEmbedder(), max_answer_chars=10)
    result, _ = gr.check(
        "q", [_chunk("c1", "t", 1.0)], Answer(text="x" * 50, cited_chunk_ids=["c1"])
    )
    assert result.action == "refuse"
    assert "too long" in result.reason


def test_output_groundedness_falls_back_to_all_chunks():
    gr = OutputGuardrail(embedder=FakeEmbedder(), groundedness_threshold=0.0)
    chunks = [_chunk("c1", "दिल्ली राजधानी है", 1.0)]
    result, ans = gr.check(
        "q", chunks, Answer(text="दिल्ली", cited_chunk_ids=["missing"])
    )
    assert result.passed is True
    assert ans.grounded is True


def test_input_short_in_context_proceeds(tmp_path):
    """Short follow-up proceeds when conversation context has prior turns."""
    gr = InputGuardrail(embedder=FakeEmbedder(), index_dir=tmp_path)
    context = [
        {"role": "user", "text": "दिल्ली के बारे में बताओ"},
        {"role": "assistant", "text": "दिल्ली भारत की राजधानी है।"},
    ]
    result = gr.check(_transcript("यह कैसे काम करता है"), conversation_context=context)
    assert result.action == "proceed"
