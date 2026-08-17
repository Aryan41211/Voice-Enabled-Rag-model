"""Full diagnostic of the STT pipeline against the real Sarvam API.

Tests WebSocket and REST paths, logs ALL API responses, checks format
matches, and compares ground-truth sentences.
"""

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.stt.client import (  # noqa: E402
    AUDIO_CHUNK_BYTES,
    DEFAULT_REST_URL,
    DEFAULT_WS_URL,
    SarvamSTT,
    _to_pcm,
)

settings = get_settings()

GROUND_TRUTHS = [
    ("gt_01", "भारत का राष्ट्रीय पक्षी कौन सा है"),
    ("gt_02", "चंद्रयान तीन का प्रक्षेपण कब हुआ था"),
    ("gt_03", "क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है"),
    ("gt_04", "जब भारत को आज़ादी मिली, तब देश का पहला प्रधानमंत्री कौन बना"),
    ("gt_05", "मुझे भारत की national bird के बारे में information चाहिए"),
]

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


async def test_diagnostic_websocket():
    """Send a known WAV via WebSocket, log EVERY response event."""
    import websockets

    stt = SarvamSTT(language_code="hi")
    pcm = _to_pcm(FIXTURES / "gt_01.wav")

    params = {
        "language_code": stt.language_code,
        "model": "saaras:v3-realtime",
        "stream_type": "balanced",
        "mode": "transcribe",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": 16000,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{stt.ws_url}?{query}"

    print(f"\n{'='*60}")
    print("DIAGNOSTIC: WebSocket STT")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Language code: {stt.language_code}")
    print(f"PCM bytes: {len(pcm)} ({len(pcm)/32000:.2f}s)")
    print(
        f"Chunk size: {AUDIO_CHUNK_BYTES} bytes ({AUDIO_CHUNK_BYTES/2/16000*1000:.1f}ms)"
    )
    print(f"Chunks: {(len(pcm)+AUDIO_CHUNK_BYTES-1)//AUDIO_CHUNK_BYTES}")

    all_events = []
    t0 = time.perf_counter()
    try:
        async with websockets.connect(
            url,
            additional_headers={"api-subscription-key": settings.sarvam_api_key},
            open_timeout=10,
        ) as ws:
            n_chunks = 0
            for i in range(0, len(pcm), AUDIO_CHUNK_BYTES):
                n_chunks += 1
                chunk = pcm[i : i + AUDIO_CHUNK_BYTES]
                await ws.send(
                    json.dumps(
                        {
                            "event": "audio_input",
                            "audio": base64.b64encode(chunk).decode(),
                        }
                    )
                )
            await ws.send(json.dumps({"event": "end"}))
            print(f"Sent {n_chunks} chunks, waiting for events...")

            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"  [non-JSON] {message[:200]}")
                    continue
                event = data.get("event") or data.get("type") or "unknown"
                all_events.append(data)
                elapsed = (time.perf_counter() - t0) * 1000

                if event in ("transcript.partial", "transcript.interim"):
                    print(f"  [{elapsed:.0f}ms] PARTIAL: {data.get('text', '')[:100]}")
                elif event == "transcript.final":
                    print(f"  [{elapsed:.0f}ms] FINAL: {data.get('text', '')[:200]}")
                elif event == "session.begin":
                    print(f"  [{elapsed:.0f}ms] SESSION.BEGIN")
                elif event == "session.end":
                    print(f"  [{elapsed:.0f}ms] SESSION.END")
                    break
                elif event == "error":
                    print(f"  [{elapsed:.0f}ms] ERROR: {data}")
                    break
                else:
                    print(
                        f"  [{elapsed:.0f}ms] {event}: {json.dumps(data, ensure_ascii=False)[:200]}"
                    )
    except Exception as exc:
        print(f"CONNECTION ERROR: {exc}")
        return

    total_ms = (time.perf_counter() - t0) * 1000
    finals = [
        e.get("text", "") for e in all_events if e.get("event") == "transcript.final"
    ]
    print(f"\nTotal time: {total_ms:.0f}ms")
    print(f"Final transcripts: {len(finals)}")
    for i, f in enumerate(finals):
        print(f"  final[{i}]: {f}")
    print(f"Joined: {' '.join(finals)}")
    print(f"All events: {[e.get('event', e.get('type', '?')) for e in all_events]}")


async def test_diagnostic_rest():
    """Send a known WAV via REST, log the FULL response."""
    stt = SarvamSTT(language_code="hi")
    wav_path = FIXTURES / "gt_01.wav"
    wav_data = wav_path.read_bytes()

    headers = {"api-subscription-key": settings.sarvam_api_key}
    files = {"file": ("gt_01.wav", wav_data, "audio/wav")}
    data = {
        "model": "saaras:v3",
        "language_code": stt.language_code,
        "with_timestamps": "false",
    }

    print(f"\n{'='*60}")
    print("DIAGNOSTIC: REST STT")
    print(f"{'='*60}")
    print(f"URL: {DEFAULT_REST_URL}")
    print(f"Language code: {stt.language_code}")
    print(f"File size: {len(wav_data)} bytes")

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                DEFAULT_REST_URL, headers=headers, data=data, files=files
            )
    except httpx.HTTPError as exc:
        print(f"HTTP ERROR: {exc}")
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Status: {resp.status_code}")
    print(f"Time: {elapsed_ms:.0f}ms")
    print(f"Response body: {resp.text[:500]}")
    try:
        body = resp.json()
        print(f"Transcript: {body.get('transcript', '(missing)')}")
    except Exception:
        print("Non-JSON response")


async def test_diagnostic_format():
    """Verify the exact audio format being sent to the API."""
    wav_path = FIXTURES / "gt_01.wav"

    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Audio Format")
    print(f"{'='*60}")

    # Check WAV header
    with wave.open(str(wav_path), "rb") as wav:
        print(f"WAV file: {wav_path.name}")
        print(f"  Sample rate: {wav.getframerate()} Hz")
        print(f"  Channels: {wav.getnchannels()}")
        print(
            f"  Sample width: {wav.getsampwidth()} bytes ({wav.getsampwidth()*8}-bit)"
        )
        print(f"  Frames: {wav.getnframes()}")
        print(f"  Duration: {wav.getnframes()/wav.getframerate():.3f}s")
        print(
            f"  Encoding: PCM {'signed' if wav.getsampwidth() == 2 else 'unsigned'}-int16 LE"
        )

    # Check PCM after normalization
    pcm = _to_pcm(wav_path)
    print("\nAfter _to_pcm:")
    print(f"  Bytes: {len(pcm)}")
    print(f"  Duration: {len(pcm)/32000:.3f}s (at 16kHz mono 16-bit)")
    print(f"  First 20 bytes (hex): {pcm[:20].hex()}")
    samples = np.frombuffer(pcm, dtype="<i2")
    print(f"  Sample range: [{samples.min()}, {samples.max()}]")
    print(f"  RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")

    # Check what WebSocket audio_input message format
    chunk = pcm[:AUDIO_CHUNK_BYTES]
    msg = json.dumps(
        {
            "event": "audio_input",
            "audio": base64.b64encode(chunk).decode(),
        }
    )
    print("\nWebSocket message (first chunk):")
    print("  Event: audio_input")
    print("  Audio encoding: base64 PCM linear16")
    print(f"  Chunk bytes: {len(chunk)}")
    print(f"  Message size: {len(msg)} bytes")
    print(
        "  NOTE: API docs expect AudioData format with data,sample_rate,encoding fields"
    )


async def test_all_ground_truths():
    """Run all ground truths through WS and REST, comparing results."""
    print(f"\n{'='*60}")
    print("GROUND TRUTH COMPARISON (WS vs REST)")
    print(f"{'='*60}")

    stt = SarvamSTT(language_code="hi")
    results = []

    for name, expected in GROUND_TRUTHS:
        wav = FIXTURES / f"{name}.wav"
        if not wav.exists():
            print(f"  SKIP {name}: fixture not found")
            continue

        print(f"\n--- {name} ---")
        print(f"  Expected: {expected}")

        # WS path
        try:
            ws_result = await asyncio.wait_for(stt.transcribe(str(wav)), timeout=15)
            print(f"  WS result: {ws_result.text}")
            print(f"  WS latency: {ws_result.stt_latency_ms:.0f}ms")
        except Exception as exc:
            ws_result = None
            print(f"  WS error: {exc}")

        # REST path (force by sending raw WAV)
        try:
            headers = {"api-subscription-key": settings.sarvam_api_key}
            files = {"file": (wav.name, wav.read_bytes(), "audio/wav")}
            data = {
                "model": "saaras:v3",
                "language_code": stt.language_code,
                "with_timestamps": "false",
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    DEFAULT_REST_URL, headers=headers, data=data, files=files
                )
            if resp.status_code == 200:
                rest_text = resp.json().get("transcript", "")
                print(f"  REST result: {rest_text}")
            else:
                print(f"  REST HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            print(f"  REST error: {exc}")

        results.append((name, expected, ws_result.text if ws_result else "ERROR"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, expected, actual in results:
        match = "✓" if expected in actual or actual in expected else "✗"
        print(f"  {match} {name}: expected='{expected[:40]}' actual='{actual[:40]}'")


async def main():
    print("=" * 60)
    print("STT PIPELINE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Language: {settings.data_lang}")
    print(f"STT provider: {settings.stt_provider}")
    print(f"Sarvam API key: {settings.sarvam_api_key[:8]}...")
    print(f"WS URL: {DEFAULT_WS_URL}")
    print(f"REST URL: {DEFAULT_REST_URL}")

    await test_diagnostic_format()
    await test_diagnostic_rest()
    await test_diagnostic_websocket()
    await test_all_ground_truths()


if __name__ == "__main__":
    asyncio.run(main())
