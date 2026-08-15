"""CI real-API integration: transcribe a Hindi WAV via Sarvam, fail if empty."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stt.client import SarvamSTT


async def _run(audio_path: str) -> int:
    transcript = await SarvamSTT().transcribe(Path(audio_path))
    print(
        json.dumps(
            {
                "text": transcript.text,
                "language": transcript.language,
                "stt_latency_ms": transcript.stt_latency_ms,
            },
            ensure_ascii=False,
        )
    )
    if not transcript.text.strip():
        print("[ci_stt] FAIL: empty transcript", file=sys.stderr)
        return 1
    print("[ci_stt] OK: non-empty transcript received")
    return 0


def main() -> int:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample_hi.wav"
    return asyncio.run(_run(audio_path))


if __name__ == "__main__":
    raise SystemExit(main())
