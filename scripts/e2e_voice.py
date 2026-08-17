"""End-to-end voice round-trip on eval_gold queries.

For each selected query the script:
  1. Synthesises spoken Hindi via edge-tts (hi-IN-SwaraNeural, 16 kHz mono WAV).
  2. Transcribes via Sarvam STT (WS → REST → batch fallback).
  3. Runs the full pipeline (retrieval + extractive generation).
  4. Prints per-stage timings and the answer.

Requires: ``SARVAM_API_KEY``, ``edge-tts`` (pip), ``miniaudio`` (pip).
Run with ``--count N`` (default 10).
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLD = Path(__file__).resolve().parent.parent / "data" / "index" / "hi" / "eval_gold.jsonl"


async def _synthesize_wav(text: str) -> Path:
    """Synthesise text to a 16 kHz mono WAV (via edge-tts + miniaudio)."""
    import edge_tts
    import miniaudio

    buf = io.BytesIO()
    async for msg in edge_tts.Communicate(text, "hi-IN-SwaraNeural").stream():
        if msg["type"] == "audio":
            buf.write(msg["data"])
    buf.seek(0)
    decoded = miniaudio.decode(
        buf.read(),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=16000,
    )
    pcm = bytes(decoded.samples)
    out = Path(f"e2e_temp_{abs(hash(text)) & 0xFFFFFFFF:08x}.wav")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return out


async def run_one(query: str, idx: int, pipe, stt) -> dict:
    """Full voice round-trip on a single query; returns a timing dict."""
    wav = await _synthesize_wav(query)
    try:
        t0 = time.perf_counter()
        transcript = await stt.transcribe(str(wav))
        t_stt = (time.perf_counter() - t0) * 1000

        from app.harness.schemas import Transcript

        t1 = time.perf_counter()
        resp = await pipe.query_async(Transcript(text=transcript.text))
        t_total = (time.perf_counter() - t0) * 1000

        return {
            "idx": idx,
            "query": query,
            "transcript": transcript.text,
            "answer": (resp.answer or resp.refusal_reason or "")[:140],
            "refused": resp.refused,
            "stt_ms": t_stt,
            "retrieval_ms": resp.timings_ms.get("retrieval_ms", 0),
            "generation_ms": resp.timings_ms.get("generation_ms", 0),
            "total_ms": t_total,
            "sources": len(resp.sources),
        }
    finally:
        wav.unlink(missing_ok=True)


async def main(count: int = 10) -> None:
    gold = [json.loads(line) for line in open(GOLD, encoding="utf-8")]
    queries = [d["query"] for d in gold[:count]]

    from app.stt.client import SarvamSTT

    stt = SarvamSTT(language_code="hi")
    t0 = time.perf_counter()
    from app.harness.pipeline import Pipeline

    pipe = Pipeline.from_index()
    pipe.warmup()
    print(f"pipeline loaded in {time.perf_counter() - t0:.1f}s\n")

    results: list[dict] = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{count}] {q}")
        r = await run_one(q, i, pipe, stt)
        results.append(r)
        tag = "REFUSED" if r["refused"] else "OK"
        print(f"  [{tag}] stt={r['stt_ms']:.0f}ms "
              f"retr={r['retrieval_ms']:.1f}ms "
              f"gen={r['generation_ms']:.1f}ms "
              f"total={r['total_ms']:.0f}ms "
              f"src={r['sources']}")
        print(f"  A: {r['answer']}\n")

    refused = sum(1 for r in results if r["refused"])
    if results:
        t_med = sorted(r["total_ms"] for r in results)[len(results) // 2]
        t_retr_med = sorted(r["retrieval_ms"] for r in results)[len(results) // 2]
        print(f"summary: {len(results)} queries, {refused} refused, "
              f"median total={t_med:.0f}ms  median retrieval={t_retr_med:.1f}ms")
    else:
        print("summary: no queries")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    asyncio.run(main(ap.parse_args().count))
