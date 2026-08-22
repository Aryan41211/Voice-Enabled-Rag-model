"""Synthesize ground-truth audio clips for STT evaluation.

Generates 20 clips across 3 languages (Hindi, Bengali, Tamil) covering:
- Clear speech at native pace
- Fast/casual speech
- Background noise (artificially added)
- Domain-specific terms from MSMARCO-XI queries

Each clip is paired with its exact expected transcript. A manifest.json
file records all metadata.

Usage:
    python tests/stt_ground_truth/synthesize.py          # generate clips
    python tests/stt_ground_truth/synthesize.py --force   # regenerate all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import edge_tts

FIXTURES = Path(__file__).resolve().parent / "clips"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

# (id, language, voice, text, expected_transcript, rate_modifier, add_noise, category)
TEST_CASES = [
    # ── Hindi: clear speech ──────────────────────────────────────────
    (
        "hi_01",
        "hi",
        "hi-IN-SwaraNeural",
        "भारत का राष्ट्रीय पक्षी कौन सा है",
        "भारत का राष्ट्रीय पक्षी कौन सा है",
        "+0%",
        False,
        "clear",
    ),
    (
        "hi_02",
        "hi",
        "hi-IN-SwaraNeural",
        "चंद्रयान तीन का प्रक्षेपण कब हुआ था",
        "चंद्रयान तीन का प्रक्षेपण कब हुआ था",
        "+0%",
        False,
        "clear",
    ),
    (
        "hi_03",
        "hi",
        "hi-IN-SwaraNeural",
        "जलवायु परिवर्तन का पर्यावरण पर क्या प्रभाव पड़ता है",
        "जलवायु परिवर्तन का पर्यावरण पर क्या प्रभाव पड़ता है",
        "+0%",
        False,
        "clear",
    ),
    # ── Hindi: domain-specific terms ─────────────────────────────────
    (
        "hi_04",
        "hi",
        "hi-IN-SwaraNeural",
        "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग में क्या अंतर है",
        "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग में क्या अंतर है",
        "+0%",
        False,
        "domain",
    ),
    (
        "hi_05",
        "hi",
        "hi-IN-SwaraNeural",
        "भारतीय अंतरिक्ष अनुसंधान संगठन ने कितने उपग्रह प्रक्षेपित किए हैं",
        "भारतीय अंतरिक्ष अनुसंधान संगठन ने कितने उपग्रह प्रक्षेपित किए हैं",
        "+0%",
        False,
        "domain",
    ),
    (
        "hi_06",
        "hi",
        "hi-IN-SwaraNeural",
        "क्वांटम कंप्यूटिंग की मूलभूत सिद्धांत क्या है",
        "क्वांटम कंप्यूटिंग की मूलभूत सिद्धांत क्या है",
        "+0%",
        False,
        "domain",
    ),
    # ── Hindi: fast/casual speech ────────────────────────────────────
    (
        "hi_07",
        "hi",
        "hi-IN-SwaraNeural",
        "मुझे भारत की राजधानी के बारे में बताओ",
        "मुझे भारत की राजधानी के बारे में बताओ",
        "+30%",
        False,
        "fast",
    ),
    (
        "hi_08",
        "hi",
        "hi-IN-SwaraNeural",
        "सूर्य और पृथ्वी के बीच की औसत दूरी कितनी है",
        "सूर्य और पृथ्वी के बीच की औसत दूरी कितनी है",
        "+30%",
        False,
        "fast",
    ),
    (
        "hi_09",
        "hi",
        "hi-IN-SwaraNeural",
        "डीएनए की संरचना किसने खोजी थी",
        "डीएनए की संरचना किसने खोजी थी",
        "+30%",
        False,
        "fast",
    ),
    # ── Hindi: noise-augmented ───────────────────────────────────────
    (
        "hi_10",
        "hi",
        "hi-IN-SwaraNeural",
        "भारत का राष्ट्रीय प्रतीक क्या है",
        "भारत का राष्ट्रीय प्रतीक क्या है",
        "+0%",
        True,
        "noisy",
    ),
    (
        "hi_11",
        "hi",
        "hi-IN-SwaraNeural",
        "महात्मा गांधी ने असहयोग आंदोलन कब शुरू किया",
        "महात्मा गांधी ने असहयोग आंदोलन कब शुरू किया",
        "+0%",
        True,
        "noisy",
    ),
    (
        "hi_12",
        "hi",
        "hi-IN-MadhurNeural",
        "गणित में पाइथागोरस प्रमेय क्या कहता है",
        "गणित में पाइथागोरस प्रमेय क्या कहता है",
        "+0%",
        True,
        "noisy",
    ),
    # ── Bengali: clear speech ────────────────────────────────────────
    (
        "bn_01",
        "bn",
        "bn-IN-TanishaaNeural",
        "ভারতের জাতীয় পাখি কী",
        "ভারতের জাতীয় পাখি কী",
        "+0%",
        False,
        "clear",
    ),
    (
        "bn_02",
        "bn",
        "bn-IN-TanishaaNeural",
        "কৃত্রিম বুদ্ধিমত্তা কীভাবে কাজ করে",
        "কৃত্রিম বুদ্ধিমত্তা কীভাবে কাজ করে",
        "+0%",
        False,
        "clear",
    ),
    (
        "bn_03",
        "bn",
        "bn-IN-BashkarNeural",
        "বাংলাদেশের স্বাধীনতা কবে হয়েছিল",
        "বাংলাদেশের স্বাধীনতা কবে হয়েছিল",
        "+0%",
        False,
        "clear",
    ),
    # ── Bengali: fast speech ─────────────────────────────────────────
    (
        "bn_04",
        "bn",
        "bn-IN-TanishaaNeural",
        "সূর্য থেকে পৃথ্বীর দূরত্ব কত",
        "সূর্য থেকে পৃথ্বীর দূরত্ব কত",
        "+25%",
        False,
        "fast",
    ),
    # ── Tamil: clear speech ──────────────────────────────────────────
    (
        "ta_01",
        "ta",
        "ta-IN-PallaviNeural",
        "இந்தியாவின் தேசிய பறவை எது",
        "இந்தியாவின் தேசிய பறவை எது",
        "+0%",
        False,
        "clear",
    ),
    (
        "ta_02",
        "ta",
        "ta-IN-PallaviNeural",
        "செயற்கை நுண்ணறிவு என்றால் என்ன",
        "செயற்கை நுண்ணறிவு என்றால் என்ன",
        "+0%",
        False,
        "clear",
    ),
    # ── Tamil: domain terms ──────────────────────────────────────────
    (
        "ta_03",
        "ta",
        "ta-IN-ValluvarNeural",
        "விண்வெளி ஆராய்ச்சி நிலையம் எங்குள்ளது",
        "விண்வெளி ஆராய்ச்சி நிலையம் எங்குள்ளது",
        "+0%",
        False,
        "domain",
    ),
    # ── Tamil: fast speech ───────────────────────────────────────────
    (
        "ta_04",
        "ta",
        "ta-IN-PallaviNeural",
        "கதிர்வீச்சு உயிரணுக்களை எவ்வாறு பாதிக்கிறது",
        "கதிர்வீச்சு உயிரணுக்களை எவ்வாறு பாதிக்கிறது",
        "+25%",
        False,
        "fast",
    ),
]


def add_white_noise(pcm_bytes: bytes, noise_level: int = 500) -> bytes:
    """Add low-amplitude white noise to PCM audio."""
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype="<i2").copy()
    noise = np.random.RandomState(42).randint(-noise_level, noise_level + 1, size=len(samples), dtype=np.int16)
    samples = np.clip(samples.astype(np.int32) + noise, -32768, 32767).astype("<i2")
    return samples.tobytes()


async def synthesize_one(case: tuple, force: bool) -> dict:
    cid, lang, voice, text, expected, rate, add_noise, category = case
    out_dir = FIXTURES / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{cid}.wav"

    if wav_path.exists() and not force:
        return {"id": cid, "path": str(wav_path), "text": text, "expected": expected,
                "language": lang, "voice": voice, "rate": rate, "noise": add_noise,
                "category": category, "status": "exists"}

    # edge-tts outputs mp3; we need WAV 16kHz mono
    mp3_path = out_dir / f"{cid}.mp3"
    comm = edge_tts.Communicate(text, voice, rate=rate)
    await comm.save(str(mp3_path))

    # Convert mp3 to WAV using bundled ffmpeg from imageio-ffmpeg
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    mp3_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return {"id": cid, "error": proc.stderr.decode()[:200], "status": "ffmpeg_fail"}

    # Optionally add noise
    if add_noise:
        with wave.open(str(wav_path), "rb") as w:
            pcm = w.readframes(w.getnframes())
        noisy = add_white_noise(pcm)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(noisy)

    return {
        "id": cid,
        "path": str(wav_path),
        "text": text,
        "expected": expected,
        "language": lang,
        "voice": voice,
        "rate": rate,
        "noise": add_noise,
        "category": category,
        "status": "synthesized",
    }


async def main(force: bool = False) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    results = []
    for case in TEST_CASES:
        r = await synthesize_one(case, force)
        results.append(r)
        status = r["status"]
        cid = r["id"]
        print(f"[{status}] {cid}: {r.get('expected', r.get('error', ''))[:50]}")

    # Write manifest
    manifest = {
        "version": 1,
        "total_clips": len(results),
        "languages": sorted(set(c[1] for c in TEST_CASES)),
        "categories": sorted(set(c[7] for c in TEST_CASES)),
        "clips": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote manifest with {len(results)} clips to {MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate all clips")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
