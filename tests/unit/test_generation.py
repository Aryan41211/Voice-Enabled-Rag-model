import asyncio

from app.generation.generator import (
    ExtractiveGenerator,
    LLMGenerator,
    make_generator,
)
from app.harness.schemas import GenerationError, RetrievedChunk


def _hit(cid, text, score=0.9):
    return RetrievedChunk(chunk_id=cid, text=text, score=score, source="dense")


def test_extractive_returns_top_passage():
    gen = ExtractiveGenerator()
    chunks = [
        _hit("c1", "दिल्ली भारत की राजधानी है।", score=0.9),
        _hit("c2", "अन्य पाठ", score=0.5),
    ]
    ans = gen.generate("राजधानी", chunks)
    assert "राजधानी" in ans.text
    assert ans.cited_chunk_ids == ["c1"]
    assert ans.grounded is True
    assert ans.ttft_ms >= 0


def test_extractive_picks_query_relevant_sentence_not_top_chunk():
    # A lower-scored chunk holds the sentence that answers the query; the
    # top chunk is only loosely related. The extractor must prefer the
    # sentence with query-token overlap over the top chunk verbatim.
    gen = ExtractiveGenerator()
    chunks = [
        _hit(
            "c1",
            "भारत में अनेक राज्य हैं। हिमालय उत्तर में स्थित है।",
            score=0.9,
        ),
        _hit(
            "c2",
            "मोर भारत का राष्ट्रीय पक्षी है। यह अपने नृत्य के लिए प्रसिद्ध है।",
            score=0.7,
        ),
    ]
    ans = gen.generate("भारत का राष्ट्रीय पक्षी कौन सा है", chunks)
    assert "मोर" in ans.text
    assert "राष्ट्रीय पक्षी" in ans.text
    assert ans.cited_chunk_ids == ["c2"]


def test_extractive_appends_context_when_sentence_is_short():
    gen = ExtractiveGenerator()
    chunks = [
        _hit("c1", "पहला वाक्य कुछ और। मोर राष्ट्रीय पक्षी है। अगला संदर्भ वाक्य।", score=0.8),
    ]
    ans = gen.generate("राष्ट्रीय पक्षी", chunks)
    assert "मोर" in ans.text
    assert ans.cited_chunk_ids == ["c1"]


def test_extractive_falls_back_to_top_chunk_when_no_overlap():
    gen = ExtractiveGenerator()
    chunks = [
        _hit("c1", "पूरी तरह अलग विषय का पाठ। दूसरा वाक्य।", score=0.9),
        _hit("c2", "कुछ नहीं", score=0.5),
    ]
    # No query token appears in any passage -> verbatim top-chunk opening.
    ans = gen.generate("क्वांटम कंप्यूटिंग", chunks)
    assert ans.cited_chunk_ids == ["c1"]


def test_extractive_empty_chunks_raises():
    gen = ExtractiveGenerator()
    try:
        gen.generate("q", [])
        assert False, "should have raised"
    except GenerationError:
        pass


def test_extractive_truncates_long_passages():
    gen = ExtractiveGenerator(max_chars=20)
    chunks = [_hit("c1", "क" * 100)]
    ans = gen.generate("q", chunks)
    assert len(ans.text) <= 25


def test_llm_missing_key_raises():
    gen = LLMGenerator(api_key="")
    try:
        asyncio.run(gen.generate("q", [_hit("c1", "पाठ")]))
        assert False, "should have raised"
    except GenerationError:
        pass


class FakeResponse:
    def __init__(self, lines):
        self.status_code = 200
        self._lines = lines

    async def aread(self):
        return b""

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line

        return gen()


class FakeStream:
    def __init__(self, lines):
        self._resp = FakeResponse(lines)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, **kwargs):
        return FakeStream(self._lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _run_llm(events, api_key="k", cited_line=None):
    lines = list(events)
    if cited_line:
        lines.append(
            "data: " + '{"choices":[{"delta":{"content":"' + cited_line + '"}}]}'
        )
    lines.append("data: [DONE]")

    import app.generation.generator as genmod

    orig = genmod.httpx.AsyncClient
    genmod.httpx.AsyncClient = lambda timeout=None: FakeClient(lines)
    try:
        gen = LLMGenerator(api_key=api_key, model="m", base_url="http://fake")
        return asyncio.run(
            gen.generate("राजधानी क्या है?", [_hit("c1", "पहला"), _hit("c2", "दूसरा")])
        )
    finally:
        genmod.httpx.AsyncClient = orig


def test_llm_streams_and_measures_ttft():
    events = [
        "data: " + '{"choices":[{"delta":{"content":"दिल्ली राजधानी"}}]}',
        "data: " + '{"choices":[{"delta":{"content":" है।"}}]}',
        'data: {"choices":[{"delta":{}}]}',
    ]
    ans = _run_llm(events)
    assert "दिल्ली" in ans.text
    assert ans.cited_chunk_ids  # no Cited line → falls back to top chunk
    assert ans.grounded is None
    assert ans.ttft_ms >= 0


def test_llm_parses_cited_indices():
    events = [
        "data: " + '{"choices":[{"delta":{"content":"जवाब।"}}]}',
    ]
    ans = _run_llm(events, cited_line="Cited: [2]")
    assert ans.cited_chunk_ids == ["c2"]


def test_make_generator_defaults_to_extractive(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: Settings(_env_file=None, llm_provider="extractive"),
    )
    assert make_generator().name == "extractive"
