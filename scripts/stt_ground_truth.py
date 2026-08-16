"""Ground-truth STT regression check against the live Sarvam API.

Transcribes the 5 committed Hindi fixtures through the app's SarvamSTT
client (WebSocket realtime -> REST fallback) and compares each result to
its expected transcript. These fixtures were synthesized from the exact
spoken text (edge-tts, hi-IN-SwaraNeural, 16 kHz mono) and verified to
transcribe exactly.

Usage:
    python scripts/stt_ground_truth.py            # use committed fixtures
    python scripts/stt_ground_truth.py --synth    # (re)synthesize fixtures

Exit code: 0 all pass, 1 any fail, 2 missing fixture.
Requires SARVAM_API_KEY (reads app config / .env). ``edge_tts`` is only
needed for --synth.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
except AttributeError:
    pass

from app.stt.client import SarvamSTT

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# (fixture, spoken text fed to TTS, expected ASR, check mode)
# mode "exact": normalized equality; mode "tokens": every required token present.
# Sarvam appends sentence punctuation and transliterates English loanwords to
# Devanagari (GT-05 code-switched case) — both are expected behavior.
GROUND_TRUTHS = [
    (
        "gt_01.wav",
        "भारत का राष्ट्रीय पक्षी कौन सा है",
        "भारत का राष्ट्रीय पक्षी कौन सा है?",
        "exact",
        [],
    ),
    (
        "gt_02.wav",
        "चंद्रयान तीन का प्रक्षेपण कब हुआ था",
        "चंद्रयान तीन का प्रक्षेपण कब हुआ था?",
        "exact",
        [],
    ),
    (
        "gt_03.wav",
        "क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है",
        "क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है?",
        "exact",
        [],
    ),
    (
        "gt_04.wav",
        "जब भारत को आज़ादी मिली, तब देश का पहला प्रधानमंत्री कौन बना",
        "जब भारत को आजादी मिली, तब देश का पहला प्रधानमंत्री कौन बना?",
        "exact",
        [],
    ),
    (
        "gt_05.wav",
        "मुझे भारत की national bird के बारे में information चाहिए",
        "मुझे भारत की नेशनल बर्ड के बारे में इंफॉर्मेशन चाहिए।",
        "tokens",
        ["भारत", "बारे", "में"],
    ),
]

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Keep letters/digits only so punctuation and script differences vanish."""
    return _NON_WORD.sub("", text)


async def _synth(force: bool) -> None:
    import edge_tts

    FIXTURES.mkdir(parents=True, exist_ok=True)
    voice = "hi-IN-SwaraNeural"
    for name, spoken, _expected, _mode, _tokens in GROUND_TRUTHS:
        out = FIXTURES / name
        if out.exists() and not force:
            continue
        await edge_tts.Communicate(spoken, voice).save(str(out))
        print(f"[synth] {name}")


async def _check() -> int:
    failures = 0
    for name, _spoken, expected, mode, tokens in GROUND_TRUTHS:
        path = FIXTURES / name
        if not path.exists():
            print(f"[ground-truth] missing fixture {path} (run with --synth)")
            return 2
        transcript = await SarvamSTT().transcribe(path)
        actual = transcript.text
        norm_actual = normalize(actual)
        if mode == "tokens":
            ok = all(normalize(t) in norm_actual for t in tokens)
        else:
            ok = norm_actual == normalize(expected)
        status = "PASS" if ok else "FAIL"
        print(f"[ground-truth] {status} {name}: {actual}")
        if not ok:
            failures += 1
            print(f"  expected: {expected}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", action="store_true", help="(re)synthesize fixtures")
    parser.add_argument("--force", action="store_true", help="overwrite fixtures")
    args = parser.parse_args()
    if args.synth:
        asyncio.run(_synth(args.force))
    return asyncio.run(_check())


if __name__ == "__main__":
    raise SystemExit(main())
