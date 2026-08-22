"""Run the STT ground-truth suite through the ACTUAL browser recording path.

For every clip in tests/stt_ground_truth/manifest.json this script:
  1. launches Chromium with --use-file-for-fake-audio-capture=<clip.wav>, so
     the fake mic feeds real speech into the page,
  2. drives the UI: click Record -> wait one playthrough -> click Stop,
  3. waits for the transcript produced by the full pipeline
     (MediaRecorder -> blob -> webmToWav -> POST /v1/voice -> Sarvam STT),
  4. computes WER against the manifest's expected text using the exact same
     WER implementation as the direct-API suite (tests/stt_ground_truth/evaluate.py),
  5. compares aggregate WER against a direct-API results file.

Usage:
    .venv/Scripts/python tests/browser_e2e/run_browser_suite.py \
        [--base-url http://127.0.0.1:8014] \
        [--direct-results tests/stt_ground_truth/results.json] \
        [--limit N] [--out tests/stt_ground_truth/results_browser.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "stt_ground_truth"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from evaluate import wer, find_errors  # noqa: E402  (same WER as direct-API suite)

HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "tests" / "stt_ground_truth" / "manifest.json"


def clip_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8014")
    ap.add_argument(
        "--direct-results",
        default=str(ROOT / "tests" / "stt_ground_truth" / "results.json"),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(ROOT / "tests" / "stt_ground_truth" / "results_browser.json"),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel browser cycles (fake mic is per-launch, safe to parallelize)",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    if args.limit:
        clips = clips[: args.limit]

    results = []
    failures = []

    # Resume support: keep prior per-clip results from a previous partial run
    # so an interrupted suite can be continued without re-paying Sarvam calls.
    out_path = Path(args.out)
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in prev.get("results", [])}
        results.extend(prev.get("results", []))
        failures.extend(prev.get("failures", []))
        done_ids |= {f["id"] for f in prev.get("failures", [])}
        clips = [c for c in clips if c["id"] not in done_ids]
        if clips:
            print(f"resuming: {len(clips)} clips left", flush=True)

    def run_clip(clip: dict):
        cid = clip["id"]
        wav = Path(clip["path"])
        dur = clip_duration(wav)
        out_json = HERE / f"_cycle_{cid}.json"
        cmd = [
            "node",
            str(HERE / "record_cycle.mjs"),
            f"--url={args.base_url}",
            f"--clip={wav}",
            f"--duration={dur:.2f}",
            f"--out={out_json}",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8"
        )
        wall_s = time.perf_counter() - t0

        if not out_json.exists():
            return None, {
                "id": cid,
                "stage": "launch",
                "detail": (proc.stdout + proc.stderr)[-800:],
            }

        rep = json.loads(out_json.read_text(encoding="utf-8"))
        r = rep["results"][0]

        problems = []
        if not rep["allOk"]:
            problems.append(f"no transcript / error: {r['error']}")
        if not rep["zeroConsoleErrors"]:
            problems.append("console errors: " + "; ".join(r["consoleErrorsThisCycle"]))
        if not rep["validBlobs"]:
            problems.append(
                f"invalid blob: size={r['rawBlobSize']} mime={r['rawBlobMime']}"
            )
        if problems:
            return None, {"id": cid, "stage": "pipeline", "detail": "; ".join(problems)}

        hyp, ref = r["transcript"], clip["expected"]
        w, ref_words, hyp_words = wer(ref, hyp)
        entry = {
            "id": cid,
            "language": clip["language"],
            "category": clip["category"],
            "expected": ref,
            "actual": hyp,
            "wer": round(w, 4),
            "word_errors": len(find_errors(ref_words, hyp_words)),
            "ref_words": len(ref_words),
            "hyp_words": len(hyp_words),
            "raw_blob_mime": r["rawBlobMime"],
            "chosen_mime": r["chosenMime"],
            "detected_lang": r.get("detectedLang"),
            "raw_blob_size": r["rawBlobSize"],
            "console_errors": 0,
            "wall_s": round(wall_s, 1),
        }
        out_json.unlink(missing_ok=True)
        return entry, None

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        pairs = [(c, ex.submit(run_clip, c)) for c in clips]
        for clip, fut in pairs:
            cid = clip["id"]
            entry, failure = fut.result()
            if failure:
                failures.append(failure)
                print(f"  [FAIL] {cid}: {failure['detail'][:140]}", flush=True)
                continue
            results.append(entry)
            w = entry["wer"]
            status = "PASS" if w < 0.10 else "WARN"
            print(
                f"  [{status}] {cid:6s} WER={w:.1%} "
                f"({entry['word_errors']}/{entry['ref_words']}) mime={entry['raw_blob_mime']} "
                f"det={entry.get('detected_lang', '?')} {entry['wall_s']}s",
                flush=True,
            )

    valid = [r for r in results if r["wer"] is not None]
    summary: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "path": "browser-recorder",
        "total_clips": len(clips),
        "evaluated": len(valid),
        "failures": failures,
        "results": results,
    }

    if valid:
        avg = sum(r["wer"] for r in valid) / len(valid)
        summary["average_wer"] = round(avg, 4)
        by_lang: dict[str, list[float]] = {}
        for r in valid:
            by_lang.setdefault(r["language"], []).append(r["wer"])
        summary["by_language"] = {
            k: {"avg_wer": round(sum(v) / len(v), 4), "n": len(v)}
            for k, v in sorted(by_lang.items())
        }
        hi = [r for r in valid if r["language"] == "hi"]
        if hi:
            summary["hi_avg_wer"] = round(sum(r["wer"] for r in hi) / len(hi), 4)

        print(
            f"\nBROWSER PATH: {len(valid)}/{len(clips)} clips OK, "
            f"{len(failures)} failures"
        )
        print(f"  Average WER: {avg:.1%}")
        for lang, st in summary["by_language"].items():
            print(f"    {lang}: avg={st['avg_wer']:.1%} (n={st['n']})")

    direct_path = Path(args.direct_results)
    if direct_path.exists() and valid:
        d = json.loads(direct_path.read_text(encoding="utf-8"))
        dval = [r for r in d["results"] if r.get("wer") is not None]
        # Apples-to-apples: same clip IDs both paths + hi-only subset
        # (browser always uses the app default language hi-IN).
        dmap = {r["id"]: r["wer"] for r in dval}
        common = [r for r in valid if r["id"] in dmap]
        if common:
            bavg = sum(r["wer"] for r in common) / len(common)
            davgsame = sum(dmap[r["id"]] for r in common) / len(common)
            hi_common = [r for r in common if r["language"] == "hi"]
            line = (
                f"\nWER COMPARISON on {len(common)} shared clips:\n"
                f"  direct API : {davgsame:.1%}\n"
                f"  browser    : {bavg:.1%}\n"
                f"  delta      : {(bavg - davgsame):+.1%}"
            )
            if hi_common:
                bh = sum(r["wer"] for r in hi_common) / len(hi_common)
                dh = sum(dmap[r["id"]] for r in hi_common) / len(hi_common)
                line += (
                    f"\n  hi-only ({len(hi_common)} clips): direct={dh:.1%} "
                    f"browser={bh:.1%} delta={(bh - dh):+.1%}"
                )
            print(line)
            summary["comparison"] = {
                "shared_clips": len(common),
                "direct_avg_wer": round(davgsame, 4),
                "browser_avg_wer": round(bavg, 4),
                "delta": round(bavg - davgsame, 4),
            }

    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
