"""Low-STT-confidence detection tests."""

import asyncio

from app.harness.schemas import Transcript
from tests.integration.test_pipeline import _pipeline


def test_low_confidence_triggers_clarification():
    p = _pipeline()
    t = Transcript(text="दिल्ली कहाँ है", confidence=0.3, language="hi")
    resp = asyncio.run(p.query_async(t, request_id="lowconf"))
    assert resp.refused is True
    assert "low_stt_confidence" in (resp.refusal_reason or "")


def test_high_confidence_passes_through():
    p = _pipeline()
    t = Transcript(text="दिल्ली कहाँ है", confidence=0.9, language="hi")
    resp = asyncio.run(p.query_async(t, request_id="highconf"))
    assert resp.refused is False
    assert resp.answer is not None


def test_default_confidence_passes_through():
    """Transcript without explicit confidence (defaults to 1.0) should pass."""
    p = _pipeline()
    t = Transcript(text="दिल्ली कहाँ है")
    resp = asyncio.run(p.query_async(t, request_id="defaultconf"))
    assert resp.refused is False
