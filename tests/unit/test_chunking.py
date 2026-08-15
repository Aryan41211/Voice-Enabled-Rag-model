import numpy as np

from app.ingestion.chunking import (
    STRATEGIES,
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    MetadataChunker,
    SemanticChunker,
    SentenceWindowChunker,
    approx_tokens,
    chunk_example,
    split_sentences,
)


def _example(text="परीक्षण पाठ।", n_passages=2, selected=(1, 0)):
    return {
        "query_id": 42,
        "target_lang": "hin_Deva",
        "query": "प्रश्न",
        "passages": {
            "Translated_passages": [text] * n_passages,
            "English_passages": ["test"] * n_passages,
            "is_selected": list(selected),
        },
    }


class FakeEmbedder:
    def encode(self, texts):
        # deterministic fake vectors: identity-ish so equal sentences are close
        rng = np.random.RandomState(0)
        return np.array([rng.rand(8) for _ in texts])


def test_split_sentences_danda():
    s = "पहला वाक्य। दूसरा वाक्य! तीसरा? चौथा ॥"
    assert len(split_sentences(s)) == 4


def test_split_sentences_english():
    s = "One sentence. Two sentence! Three?"
    assert split_sentences(s) == ["One sentence.", "Two sentence!", "Three?"]


def test_approx_tokens_positive():
    assert approx_tokens("") == 1
    assert approx_tokens("abcd") == 1
    assert approx_tokens("abcdefgh") == 2


def test_fixed_short_text_single_chunk():
    assert FixedSizeChunker().split_text("छोटा पाठ।") == ["छोटा पाठ।"]


def test_fixed_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(400)]
    text = " ".join(words)
    pieces = FixedSizeChunker(chunk_size=150, overlap=20).split_text(text)
    assert len(pieces) > 1
    assert all(0 < len(p.split()) <= 150 for p in pieces)
    assert pieces[1].startswith("w130")  # 150 - 20 = overlap window start


def test_semantic_no_embedder_returns_single_group():
    s = "एक। दो। तीन।"
    assert SemanticChunker().split_text(s) == ["एक। दो। तीन।"]


def test_semantic_groups_by_similarity():
    chunker = SemanticChunker(threshold=0.99, max_sentences=3, embedder=FakeEmbedder())
    sentences = ["एक वाक्य।", "दूसरा वाक्य।", "तीसरा वाक्य।"]
    groups = chunker.group_sentences(sentences)
    assert groups == [[sentences[0]], [sentences[1]], [sentences[2]]]


def test_sentence_window_builds_windows():
    s = ["s0", "s1", "s2", "s3"]
    wins = SentenceWindowChunker(window_size=1).windows(s)
    assert wins[0]["window"] == "s0 s1"
    assert wins[1]["window"] == "s0 s1 s2"
    assert wins[3]["window"] == "s2 s3"


def test_metadata_single_chunk_per_passage():
    assert MetadataChunker().split_text("कुछ पाठ।") == ["कुछ पाठ।"]


def test_hierarchical_children():
    text = "पहला वाक्य। दूसरा वाक्य।"
    assert HierarchicalChunker().children(text) == ["पहला वाक्य।", "दूसरा वाक्य।"]


def test_chunk_example_metadata_schema():
    chunks = chunk_example(_example(), "metadata")
    assert len(chunks) == 2
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.source_query_id == 42
    assert c.passage_index == 0
    assert c.passage_is_selected == 1
    assert c.strategy == "metadata"
    assert c.chunk_id.startswith("hin_Deva:metadata:42:0:")


def test_chunk_example_hierarchical_parent_context():
    chunks = chunk_example(_example("पहला। दूसरा।", n_passages=1), "hierarchical")
    assert len(chunks) == 2
    assert chunks[0].parent_chunk_id
    assert chunks[0].context == "पहला। दूसरा।"
    assert chunks[0].text == "पहला।"


def test_chunk_example_fixed_multiple():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_example(_example(text, n_passages=1), "fixed")
    assert len(chunks) > 1


def test_chunk_example_all_strategies_emit_chunks():
    for strat in STRATEGIES:
        chunks = chunk_example(_example(), strat)
        assert len(chunks) >= 2, f"{strat} produced no chunks"


def test_chunk_example_unknown_strategy_raises():
    try:
        chunk_example(_example(), "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
