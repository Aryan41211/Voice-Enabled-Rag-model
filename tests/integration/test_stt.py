"""Sarvam STT client tests (WS failure → REST fallback, parsing, WAV→PCM)."""

import asyncio
import wave

import numpy as np

import app.stt.client as sttmod
from app.harness.schemas import STTError
from app.stt.client import DEFAULT_REST_URL, SarvamSTT, _to_pcm


def test_rest_url_matches_documented_endpoint():
    # Verified against Sarvam docs (2026): the synchronous REST endpoint is
    # POST https://api.sarvam.ai/speech-to-text (no /v1 prefix). The old
    # /v1/speech-to-text path returns 404 and silently kills the fallback.
    assert DEFAULT_REST_URL == "https://api.sarvam.ai/speech-to-text"


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
    orig = sttmod.httpx.AsyncClient
    sttmod.httpx.AsyncClient = lambda timeout=None: FakeAsyncClient(handler)
    return orig


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = (
            payload if isinstance(payload, str) else __import__("json").dumps(payload)
        )

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    async def post(self, url, headers=None, data=None, files=None, json=None):
        return self._handler(url, headers, data, files)

    async def put(self, url, headers=None, content=None):
        return self._handler(url, headers, content, None)

    async def get(self, url, headers=None):
        return self._handler(url, headers, None, None)

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
    """Fake the websockets module; returns the list of connect URLs seen."""
    urls = []

    def fake_connect(url, **kwargs):
        urls.append(url)
        if exc is not None:
            raise exc
        return FakeWebsockets(messages or [])

    import types

    fake_mod = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setattr(sttmod, "_get_websockets", lambda: fake_mod)
    return urls


def _make_wav(path, frames=b"\x00\x00" * 1600, rate=16000, channels=1, duration_s=None):
    if duration_s is not None:
        frames = b"\x00\x00" * int(rate * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def test_to_pcm_converts_wav(tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)
    pcm = _to_pcm(p)
    assert len(pcm) == 3200


def test_to_pcm_keeps_16k_mono_untouched(tmp_path):
    p = tmp_path / "a.wav"
    frames = (np.arange(1600) % 256).astype("<i2").tobytes()
    _make_wav(p, frames=frames)
    assert _to_pcm(p) == frames


def test_to_pcm_downmixes_stereo(tmp_path):
    p = tmp_path / "s.wav"
    lch = (np.arange(800) % 100).astype("<i2")
    rch = (np.arange(800, 1600) % 100).astype("<i2")
    interleaved = np.stack([lch, rch], axis=1).flatten().tobytes()
    _make_wav(p, frames=interleaved, channels=2)
    pcm = np.frombuffer(_to_pcm(p), dtype="<i2")
    assert len(pcm) == 800
    expected = ((lch.astype(np.float32) + rch.astype(np.float32)) / 2).astype("<i2")
    assert np.array_equal(pcm, expected)


def test_to_pcm_resamples_8khz_to_16khz(tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p, frames=b"\x00\x00" * 800, rate=8000)
    assert len(_to_pcm(p)) == 3200  # 800 -> 1600 frames


def test_to_pcm_resamples_44k1_to_16khz(tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p, frames=b"\x00\x00" * 4410, rate=44100)
    assert len(_to_pcm(p)) == 3200  # 4410 -> 1600 frames


def test_to_pcm_rejects_unsupported_width(tmp_path):
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 16)
    try:
        _to_pcm(p)
        assert False, "should have raised"
    except STTError:
        pass


def test_no_key_raises(monkeypatch):
    # patch the client module's bound reference, not app.config, so the test is
    # hermetic even when a real .env (with a key) is present locally
    monkeypatch.setattr(
        "app.stt.client.get_settings", lambda: _settings(sarvam_api_key="")
    )
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
    # Fixed-language deployments use the realtime WS first; when the WS
    # yields a transcript it must win with NO REST round-trip (previously
    # this test silently hit the live REST API when routing changed).
    p = tmp_path / "a.wav"
    _make_wav(p)

    def no_rest(url, headers, data, files):
        raise AssertionError("REST must not be called when WS succeeds")

    orig = _rest_client(no_rest)
    urls = _patch_ws(
        monkeypatch,
        messages=[
            '{"event":"session.begin"}',
            '{"event":"transcript.final","text":"हैलो"}',
            '{"event":"session.end"}',
        ],
    )
    try:
        stt = SarvamSTT(api_key="k", language_code="hi")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "हैलो"
        # The requested fixed language is what the WS was opened with and
        # what the Transcript reports back.
        assert len(urls) == 1 and "language_code=hi-IN" in urls[0]
        assert result.language == "hi-IN"
    finally:
        sttmod.httpx.AsyncClient = orig


class HangingWSResult:
    async def send(self, message):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()


def _patch_hanging_ws(monkeypatch):
    import types

    class HangingConn:
        async def __aenter__(self):
            return HangingWSResult()

        async def __aexit__(self, *exc):
            return False

    fake_mod = types.SimpleNamespace(connect=lambda url, **kwargs: HangingConn())
    monkeypatch.setattr(sttmod, "_get_websockets", lambda: fake_mod)


def test_ws_timeout_falls_back_to_rest(monkeypatch, tmp_path):
    # A WS that never answers must not eat the whole stt_timeout_s budget:
    # transcribe() gives the realtime path half the budget, then falls back to
    # REST (previously the asyncio.TimeoutError escaped and the whole voice
    # request failed on long / slow audio).
    p = tmp_path / "a.wav"
    _make_wav(p)
    _patch_hanging_ws(monkeypatch)

    def handler(url, headers, data, files):
        return FakeResponse(200, {"transcript": "लंबा ऑडियो ठीक रहा"})

    orig = _rest_client(handler)
    try:
        # Fixed language keeps the WS-first flow active (auto-detect skips
        # WS entirely), so this genuinely exercises timeout -> REST fallback.
        stt = SarvamSTT(api_key="k", language_code="hi", timeout_s=1.0)
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "लंबा ऑडियो ठीक रहा"
    finally:
        sttmod.httpx.AsyncClient = orig


def test_ws_joins_multiple_finals(monkeypatch, tmp_path):
    # Server VAD splits multi-utterance audio (pause >1s) into several
    # transcript.final events. The client must keep all of them, not truncate
    # to the first (previously it broke on the first final).
    p = tmp_path / "a.wav"
    _make_wav(p)

    def no_rest(url, headers, data, files):
        raise AssertionError("REST must not be called when WS succeeds")

    orig = _rest_client(no_rest)
    urls = _patch_ws(
        monkeypatch,
        messages=[
            '{"event":"session.begin"}',
            '{"event":"transcript.final","text":"पहला वाक्य।"}',
            '{"event":"transcript.final","text":"दूसरा वाक्य?"}',
            '{"event":"session.end"}',
        ],
    )
    try:
        stt = SarvamSTT(api_key="k", language_code="hi")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "पहला वाक्य। दूसरा वाक्य?"
        assert len(urls) == 1 and "language_code=hi-IN" in urls[0]
        assert result.language == "hi-IN"
    finally:
        sttmod.httpx.AsyncClient = orig


def test_ws_empty_finals_fall_back_to_rest(monkeypatch, tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p)
    _patch_ws(
        monkeypatch, messages=['{"event":"session.begin"}', '{"event":"session.end"}']
    )

    def handler(url, headers, data, files):
        return FakeResponse(200, {"transcript": "rest तो ठीक है"})

    orig = _rest_client(handler)
    try:
        # Fixed language so the WS-empty-finals -> REST fallback is actually
        # exercised (auto-detect would skip the WS entirely).
        stt = SarvamSTT(api_key="k", language_code="hi")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "rest तो ठीक है"
    finally:
        sttmod.httpx.AsyncClient = orig


def test_ws_error_is_fatal(monkeypatch, tmp_path):
    # Fixed language so the fatal error genuinely comes from the WS handler
    # (auto-detect would skip WS and fail via REST instead — or worse, hit
    # the live API when no REST fake is installed).
    p = tmp_path / "a.wav"
    _make_wav(p)
    urls = _patch_ws(
        monkeypatch,
        messages=['{"event":"error","is_fatal":true,"message":"quota"}'],
    )
    stt = SarvamSTT(api_key="k", language_code="hi")
    try:
        asyncio.run(stt.transcribe(str(p)))
        assert False, "should have raised"
    except STTError:
        pass
    assert len(urls) == 1 and "language_code=hi-IN" in urls[0]


def test_auto_detect_routes_to_rest_and_reports_language(monkeypatch, tmp_path):
    # STT_LANGUAGE=auto (the default) must route short clips straight to REST
    # — never WS, whose "auto" implies translate-to-English. The request must
    # carry language_code=unknown + mode=transcribe, and the API-reported
    # language/probability must surface on the Transcript.
    p = tmp_path / "a.wav"
    _make_wav(p)
    seen = {}

    def handler(url, headers, data, files):
        seen["url"] = url
        seen["data"] = data
        return FakeResponse(
            200,
            {
                "transcript": "ভারতের জাতীয় পাখি কি?",
                "language_code": "bn-IN",
                "language_probability": 0.94,
            },
        )

    orig = _rest_client(handler)

    def never_ws(url, **kwargs):
        raise AssertionError("WS must not be attempted in auto-detect mode")

    import types

    monkeypatch.setattr(
        sttmod, "_get_websockets", lambda: types.SimpleNamespace(connect=never_ws)
    )
    try:
        stt = SarvamSTT(api_key="k", language_code="auto")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "ভারতের জাতীয় পাখি কি?"
        assert seen["url"] == DEFAULT_REST_URL
        assert seen["data"]["language_code"] == "unknown"
        assert seen["data"]["mode"] == "transcribe"
        assert result.language == "bn-IN"
        assert abs(result.confidence - 0.94) < 1e-9
    finally:
        sttmod.httpx.AsyncClient = orig


def _batch_client(job_id="j1", transcript="लंबा ऑडियो ठीक", state="Completed"):
    """Fake Sarvam batch job API; dispatch is by URL (mirrors live flow)."""

    def handler(url, headers, data, files):
        base = sttmod.BATCH_BASE_URL
        if url == base:
            return FakeResponse(202, {"job_id": job_id, "job_state": "Accepted"})
        if url == f"{base}/upload-files":
            return FakeResponse(
                200, {"upload_urls": {"a.wav": {"file_url": "https://blob/in/a.wav"}}}
            )
        if url == "https://blob/in/a.wav":
            return FakeResponse(201, "")
        if url == f"{base}/{job_id}/start":
            return FakeResponse(200, {})
        if url == f"{base}/{job_id}/status":
            return FakeResponse(
                200,
                {
                    "job_state": state,
                    "job_details": [{"outputs": [{"file_name": "0.json"}]}],
                },
            )
        if url == f"{base}/download-files":
            return FakeResponse(
                200,
                {"download_urls": {"0.json": {"file_url": "https://blob/out/0.json"}}},
            )
        if url == "https://blob/out/0.json":
            return FakeResponse(200, {"transcript": transcript})
        raise AssertionError(f"unexpected url: {url}")

    return handler


def test_long_audio_uses_batch_directly(monkeypatch, tmp_path):
    # A WAV longer than Sarvam's 30 s sync cap must go straight to the batch
    # job API — the realtime WS path must never be attempted.
    p = tmp_path / "a.wav"
    _make_wav(p, duration_s=35.0)

    def never_ws(url, **kwargs):
        raise AssertionError("WS must not be attempted for long audio")

    import types

    monkeypatch.setattr(
        sttmod, "_get_websockets", lambda: types.SimpleNamespace(connect=never_ws)
    )
    monkeypatch.setattr(sttmod, "BATCH_POLL_SECONDS", 0.01)
    orig = _rest_client(_batch_client())
    try:
        stt = SarvamSTT(api_key="k")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "लंबा ऑडियो ठीक"
    finally:
        sttmod.httpx.AsyncClient = orig


def test_rest_duration_error_falls_back_to_batch(monkeypatch, tmp_path):
    # Short-enough-to-try-WS audio that still gets a 400 from the sync REST
    # endpoint (duration over the 30 s limit) must reroute to the batch API
    # instead of surfacing a hard error.
    p = tmp_path / "a.wav"
    _make_wav(p)
    _patch_boom(monkeypatch)
    monkeypatch.setattr(sttmod, "BATCH_POLL_SECONDS", 0.01)
    calls = {"rest": 0}

    def handler(url, headers, data, files):
        if url == DEFAULT_REST_URL:
            calls["rest"] += 1
            return FakeResponse(
                400,
                {"error": "Audio duration exceeds the maximum limit of 30 seconds."},
            )
        return _batch_client()(url, headers, data, files)

    orig = _rest_client(handler)
    try:
        stt = SarvamSTT(api_key="k")
        result = asyncio.run(stt.transcribe(str(p)))
        assert result.text == "लंबा ऑडियो ठीक"
        assert calls["rest"] == 1
    finally:
        sttmod.httpx.AsyncClient = orig


def test_batch_failed_state_raises(monkeypatch, tmp_path):
    p = tmp_path / "a.wav"
    _make_wav(p, duration_s=35.0)
    monkeypatch.setattr(sttmod, "BATCH_POLL_SECONDS", 0.01)
    orig = _rest_client(_batch_client(state="Failed"))
    try:
        stt = SarvamSTT(api_key="k")
        try:
            asyncio.run(stt.transcribe(str(p)))
            assert False, "should have raised"
        except STTError:
            pass
    finally:
        sttmod.httpx.AsyncClient = orig


def _settings(**overrides):
    from app.config import Settings

    return Settings(_env_file=None, **overrides)
