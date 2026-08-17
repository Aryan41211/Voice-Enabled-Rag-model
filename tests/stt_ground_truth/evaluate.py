"""Evaluate STT accuracy against ground-truth clips.

Runs each clip in ``clips/`` through the live Sarvam STT pipeline,
computes word error rate (WER) per clip and aggregate, and reports
which words/phrases fail most often.

Usage:
    python tests/stt_ground_truth/evaluate.py

Requires SARVAM_API_KEY in .env.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from app.stt.client import SarvamSTT

MANIFEST = Path(__file__).resolve().parent / "manifest.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

# Language code mapping for Sarvam
LANG_MAP = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
}


def _normalize(text: str) -> str:
    """Normalize text for WER comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Remove punctuation but keep Devanagari/Bengali/Tamil diacritics
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> tuple[float, list[str], list[str]]:
    """Compute Word Error Rate using Levenshtein distance at word level.

    Returns (wer_score, reference_words, hypothesis_words).
    """
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0, ref_words, hyp_words

    # Levenshtein DP
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[n][m] / n, ref_words, hyp_words


def find_errors(ref_words: list[str], hyp_words: list[str]) -> list[dict]:
    """Find specific word-level errors between reference and hypothesis."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
        back[i][0] = ("del", i - 1, 0)
    for j in range(m + 1):
        d[0][j] = j
        back[0][j] = ("ins", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            ops = [
                (d[i - 1][j] + 1, "del", i - 1, j),
                (d[i][j - 1] + 1, "ins", i, j - 1),
                (d[i - 1][j - 1] + cost, "sub" if cost else "ok", i - 1, j - 1),
            ]
            best = min(ops, key=lambda x: x[0])
            d[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3])

    errors = []
    i, j = n, m
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]
        if op == "del":
            errors.append({"type": "delete", "ref": ref_words[i - 1], "hyp": ""})
            i -= 1
        elif op == "ins":
            errors.append({"type": "insert", "ref": "", "hyp": hyp_words[j - 1]})
            j -= 1
        elif op == "sub":
            errors.append(
                {"type": "substitute", "ref": ref_words[i - 1], "hyp": hyp_words[j - 1]}
            )
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1

    errors.reverse()
    return errors


async def evaluate_clip(stt: SarvamSTT, clip: dict) -> dict:
    """Transcribe one clip and compute WER."""
    clip_path = Path(clip["path"])
    if not clip_path.exists():
        return {"id": clip["id"], "error": f"missing file: {clip_path}", "wer": None}

    lang = clip["language"]
    stt.language_code = LANG_MAP.get(lang, f"{lang}-IN")

    try:
        t0 = time.perf_counter()
        result = await stt.transcribe(str(clip_path))
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {
            "id": clip["id"],
            "error": f"{type(e).__name__}: {e}",
            "wer": None,
        }

    actual = result.text
    expected = clip["expected"]
    w, ref_words, hyp_words = wer(expected, actual)
    errors = find_errors(ref_words, hyp_words)

    return {
        "id": clip["id"],
        "language": lang,
        "category": clip["category"],
        "expected": expected,
        "actual": actual,
        "wer": round(w, 4),
        "word_errors": len(errors),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "errors": errors,
        "latency_ms": round(latency_ms),
    }


async def main() -> None:
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}")
        print("Run synthesize.py first to generate clips.")
        return

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    print(f"Loaded {len(clips)} clips from manifest\n")

    stt = SarvamSTT()
    print(f"Sarvam API key: {stt.api_key[:10]}...{stt.api_key[-4:]}")
    print(f"Language: {stt.language_code}\n")

    results = []
    for clip in clips:
        r = await evaluate_clip(stt, clip)
        results.append(r)
        if r["wer"] is not None:
            status = "PASS" if r["wer"] < 0.10 else "FAIL"
            print(
                f"  [{status}] {r['id']:6s} WER={r['wer']:.1%} "
                f"({r['word_errors']}/{r['ref_words']} words) "
                f"latency={r['latency_ms']}ms"
            )
            if r["errors"]:
                err_words = [
                    f"{e['ref'] or '∅'}→{e['hyp'] or '∅'}"
                    for e in r["errors"]
                    if e["type"] != "ok"
                ]
                if err_words:
                    print(f"         errors: {', '.join(err_words[:5])}")
        else:
            print(f"  [ERR ] {r['id']:6s} {r.get('error', 'unknown')}")

    # Aggregate
    valid = [r for r in results if r["wer"] is not None]
    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        max_wer = max(r["wer"] for r in valid)
        by_lang = {}
        by_cat = {}
        for r in valid:
            by_lang.setdefault(r["language"], []).append(r["wer"])
            by_cat.setdefault(r["category"], []).append(r["wer"])

        print(f"\n{'='*60}")
        print(f"AGGREGATE: {len(valid)}/{len(clips)} clips evaluated")
        print(f"  Average WER: {avg_wer:.1%}")
        print(f"  Max WER:     {max_wer:.1%}")
        print(f"  Target:      <10%")
        print(f"  Status:      {'PASS' if avg_wer < 0.10 else 'FAIL'}")

        print(f"\n  By language:")
        for lang, wers in sorted(by_lang.items()):
            print(f"    {lang}: avg={sum(wers)/len(wers):.1%} (n={len(wers)})")

        print(f"\n  By category:")
        for cat, wers in sorted(by_cat.items()):
            print(f"    {cat}: avg={sum(wers)/len(wers):.1%} (n={len(wers)})")

        # Most common error words
        all_errors = []
        for r in valid:
            for e in r["errors"]:
                if e["type"] in ("substitute", "delete"):
                    all_errors.append((e["ref"], e["hyp"]))
        if all_errors:
            from collections import Counter

            freq = Counter(all_errors).most_common(10)
            print(f"\n  Most common word errors:")
            for (ref, hyp), count in freq:
                print(f"    {ref or '∅':>20s} → {hyp or '∅':<20s}  (×{count})")

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_clips": len(clips),
        "evaluated": len(valid),
        "average_wer": round(avg_wer, 4) if valid else None,
        "target_wer": 0.10,
        "passed": avg_wer < 0.10 if valid else False,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
