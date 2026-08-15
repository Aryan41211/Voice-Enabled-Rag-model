"""Sarvam STT client tests (WS failure → REST fallback, parsing, WAV→PCM)."""

import asyncio
import wave

import httpx
import pytest

import app.stt.client as sttmod
from app.harness.schemas import STTError
from app.stt.client import SarvamSTT, _to_pcm


class FakeWSResult:
    def __init__(self, messages):
        self._messages = messages

    async def send(self, message):
        pass

    def aiter(self):
        async def gen():
            for m in self._messages:
                yield m

        return gen()

    def __aiter__(self):
        return self.aiter()


def _rest_client(handler):
    import app.generation.generator as genmod  # reuse pattern via httpx patch

    orig = sttmod.httpx.AsyncClient
    sttmod.httpx.AsyncClient = lambda timeout=None: FakeAsyncClient(handler)
    return orig


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    async def post(self, url, headers=None, data=None, files=None):
        return self._handler(url, headers, data, files)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeWebsockets:
    def __init__(self, messages):
        self._messages = messages

    async def __aenter__(self):
        return FakeWSResult(self._messages)

    async def __aexit__(self, *exc):
        return False


def _patch_ws(monkeypatch, messages=None, exc=None):
    def fake_connect(url, **kwargs):
        if exc is not None:
            raise exc
        return FakeWebsockets(messages or [])

    import types

    fake_mod = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setattr(sttmod, "_get_websockets", lambda: fake_mod)


def _make_wav(path, frames=b"\x00\x00" * 1600, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def test_to_pcm_converts_wav(tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)
    pcm = _to_pcm(p)
    assert len(pcm) == 3200


def test_no_key_raises(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: _settings(sarvam_api_key=""))
    stt = SarvamSTT(api_key="")
    try:
        asyncio.run(stt.transcribe("x.wav"))
        assert False, "should have raised"
    except STTError:
        pass


def _boom(*a, **k):
    raise ConnectionError("ws down")


def _patch_boom(monkeypatch):
    import types

    monkeypatch.setattr(
        sttmod, "_get_websockets", lambda: types.SimpleNamespace(connect=_boom)
    )


def test_rest_transcribe_parses_transcript(monkeypatch, tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)

    def handler(url, headers, data, files):
        return FakeResponse(200, {"transcript": "दिल्ली की राजधानी"})

    orig = _rest_client(handler)
    _patch_boom(monkeypatch)
    try:
        stt = SarvamSTT(api_key="k")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "दिल्ली की राजधानी"
        assert result.stt_latency_ms >= 0
    finally:
        sttmod.httpx.AsyncClient = orig


def test_ws_transcript_used_first(monkeypatch, tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)
    _patch_ws(
        monkeypatch,
        messages=[
            '{"event":"session.begin"}',
            '{"event":"transcript.final","text":"हैलो"}',
            '{"event":"session.end"}',
        ],
    )
    stt = SarvamSTT(api_key="k")
    result = asyncio.run(stt.transcribe(str(p)))
    assert result.text == "हैलो"


def test_ws_error_is_fatal(monkeypatch, tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)
    _patch_ws(
        monkeypatch,
        messages=['{"event":"error","is_fatal":true,"message":"quota"}'],
    )
    stt = SarvamSTT(api_key="k")
    try:
        asyncio.run(stt.transcribe(str(p)))
        assert False, "should have raised"
    except STTError:
        pass


def _settings(**overrides):
    from app.config import Settings

    return Settings(_env_file=None, **overrides)
