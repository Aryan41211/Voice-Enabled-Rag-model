"""Sarvam AI speech-to-text client.

Primary path is the realtime WebSocket API (``saaras:v3-realtime``): streams
PCM audio frames as base64 ``audio_input`` messages and reads interim /
final transcripts. On WebSocket failure it falls back to the synchronous REST
endpoint (files < 30 s), per ROADMAP Day 5.

Endpoints (Sarvam docs):
* WS:  ``wss://api.sarvam.ai/speech-to-text-realtime/ws``
* REST: ``https://api.sarvam.ai/speech-to-text``
Auth header for both: ``api-subscription-key``.
"""

from __future__ import annotations

import base64
import json
import time
import wave
from pathlib import Path

import httpx

from app.config import get_settings
from app.harness.schemas import STTError, Transcript

DEFAULT_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
DEFAULT_REST_URL = "https://api.sarvam.ai/speech-to-text"
LANGUAGE_CODES = {
    "hi": "hi-IN",
    "en": "en-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "pa": "pa-IN",
    "or": "or-IN",
    "as": "as-IN",
    "ne": "ne-IN",
    "sa": "sa-IN",
}
AUDIO_CHUNK_BYTES = 2048


def _to_pcm(audio_path: str | Path) -> bytes:
    """Return raw linear16 PCM bytes from a WAV file or raw PCM file."""
    path = Path(audio_path)
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise STTError(
                    f"unsupported sample width {wav.getsampwidth()} (need 16-bit)"
                )
            return wav.readframes(wav.getnframes())
    except wave.Error:
        return path.read_bytes()


def _get_websockets():
    """Lazy import so the package works without the dependency installed."""
    import websockets

    return websockets


class SarvamSTT:
    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        ws_url: str = DEFAULT_WS_URL,
        rest_url: str = DEFAULT_REST_URL,
        timeout_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.sarvam_api_key
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.timeout_s = timeout_s or settings.stt_timeout_s
        lang = language_code or settings.data_lang
        self.language_code = LANGUAGE_CODES.get(lang, f"{lang}-IN")

    async def transcribe(self, audio_path: str | Path) -> Transcript:
        if not self.api_key:
            raise STTError("Sarvam STT selected but no API key configured")
        try:
            return await self._transcribe_ws(audio_path)
        except STTError as exc:
            if not exc.retryable:
                raise
            # WS path is best-effort; fall back to REST batch.
            return await self._transcribe_rest(audio_path)

    async def _transcribe_ws(self, audio_path: str | Path) -> Transcript:
        websockets = _get_websockets()

        pcm = _to_pcm(audio_path)
        params = {
            "language_code": self.language_code,
            "model": "saaras:v3-realtime",
            "stream_type": "balanced",
            "mode": "transcribe",
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": 16000,
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.ws_url}?{query}"

        t0 = time.perf_counter()
        try:
            async with websockets.connect(
                url,
                additional_headers={"api-subscription-key": self.api_key},
                open_timeout=self.timeout_s,
            ) as ws:
                for i in range(0, len(pcm), AUDIO_CHUNK_BYTES):
                    await ws.send(
                        json.dumps(
                            {
                                "event": "audio_input",
                                "audio": base64.b64encode(
                                    pcm[i : i + AUDIO_CHUNK_BYTES]
                                ).decode(),
                            }
                        )
                    )
                await ws.send(json.dumps({"event": "end"}))

                final_text: str | None = None
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    event = data.get("event")
                    if event == "transcript.final":
                        final_text = (data.get("text") or "").strip()
                        if final_text:
                            break
                    elif event == "error":
                        fatal = bool(data.get("is_fatal", True))
                        raise STTError(
                            data.get("message", "Sarvam realtime error"),
                            retryable=not fatal,
                        )
                    elif event == "session.end":
                        break
        except STTError:
            raise
        except Exception as exc:  # network / handshake / send failures
            raise STTError(f"Sarvam websocket failed: {exc}", retryable=True)

        if not final_text:
            raise STTError("Sarvam returned an empty transcript", retryable=True)

        latency_ms = (time.perf_counter() - t0) * 1000
        return Transcript(
            text=final_text,
            language=self.language_code,
            is_final=True,
            stt_latency_ms=latency_ms,
        )

    async def _transcribe_rest(self, audio_path: str | Path) -> Transcript:
        headers = {"api-subscription-key": self.api_key}
        files = {
            "file": (Path(audio_path).name, Path(audio_path).read_bytes(), "audio/wav")
        }
        data = {
            "model": "saaras:v3",
            "language_code": self.language_code,
            "with_timestamps": "false",
        }
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    self.rest_url, headers=headers, data=data, files=files
                )
        except httpx.HTTPError as exc:
            raise STTError(f"Sarvam REST failed: {exc}", retryable=True)

        if resp.status_code >= 400:
            raise STTError(
                f"Sarvam REST HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=resp.status_code >= 500,
            )
        try:
            body = resp.json()
        except ValueError:
            raise STTError("Sarvam REST returned non-JSON response", retryable=True)

        text = (
            body.get("transcript") or (body.get("output") or {}).get("transcript") or ""
        ).strip()
        if not text:
            raise STTError("Sarvam REST returned an empty transcript", retryable=True)

        latency_ms = (time.perf_counter() - t0) * 1000
        return Transcript(
            text=text,
            language=self.language_code,
            is_final=True,
            stt_latency_ms=latency_ms,
        )


class FakeSTT:
    """Deterministic STT stub for tests and keyless local runs."""

    def __init__(self, transcripts: dict[str, str] | None = None) -> None:
        self.transcripts = transcripts or {}

    async def transcribe(self, audio_path: str | Path) -> Transcript:
        name = Path(audio_path).stem
        text = self.transcripts.get(name, "दिल्ली की राजधानी क्या है")
        return Transcript(
            text=text, language="hi-IN", is_final=True, stt_latency_ms=1.0
        )
