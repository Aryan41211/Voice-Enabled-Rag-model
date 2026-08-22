"""Tests for pipeline query-expansion integration."""

from unittest.mock import MagicMock

import pytest

from app.harness.pipeline import Pipeline


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.encode_query.return_value = [0.1] * 768
    return embedder


def _make_mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = []
    return retriever


def _make_settings(**overrides):
    from app.config import Settings

    defaults = dict(
        query_expansion_enabled=False,
        expansion_k=15,
        max_paraphrases=2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.asyncio
async def test_expansion_off_uses_cache_and_single_query():
    embedder = _make_mock_embedder()
    retriever = _make_mock_retriever()
    settings = _make_settings(query_expansion_enabled=False)

    pipeline = Pipeline(
        embedder=embedder,
        retriever=retriever,
        guardrails=MagicMock(),
        dense_retriever=MagicMock(),
        settings=settings,
    )
    await pipeline._retrieve("test query")

    # encode_query called once (cache miss first time)
    embedder.encode_query.assert_called_once_with("test query")
    # retriever.search called with the encoded vector
    retriever.search.assert_called_once()


@pytest.mark.asyncio
async def test_expansion_on_calls_expand_and_retrieve():
    embedder = _make_mock_embedder()
    retriever = _make_mock_retriever()
    dense_mock = MagicMock()
    dense_mock.search.return_value = []
    settings = _make_settings(query_expansion_enabled=True)

    pipeline = Pipeline(
        embedder=embedder,
        retriever=retriever,
        guardrails=MagicMock(),
        dense_retriever=dense_mock,
        settings=settings,
    )
    result = await pipeline._retrieve("test query")

    # expand_and_retrieve was called (via asyncio.to_thread),
    # so embedder.encode_query is called inside expand_and_retrieve
    # for the original + paraphrases — at least 1 call
    assert embedder.encode_query.call_count >= 1
    # dense_mock.search called for each query variant
    assert dense_mock.search.call_count >= 1
    # result is a list
    assert isinstance(result, list)
