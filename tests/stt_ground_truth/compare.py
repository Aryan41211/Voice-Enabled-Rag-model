"""Compare WER across TTS, human, and real-world STT evaluation sets."""
import json
import sys
from pathlib import Path


def compare(results_files: list[str]) -> None:
    datasets = []
    for f in results_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        valid = [r for r in data["results"] if r.get("wer") is not None]
        avg_wer = sum(r["wer"] for r in valid) / len(valid) if valid else 0
        datasets.append({
            "name": Path(f).stem.replace("results_", ""),
            "clips": len(valid),
            "avg_wer": avg_wer,
            "max_wer": max((r["wer"] for r in valid), default=0),
        })
    print(f"\n{'Dataset':<25} {'Clips':>6} {'Avg WER':>10} {'Max WER':>10}")
    print("-" * 55)
    for d in datasets:
        print(f"{d['name']:<25} {d['clips']:>6} {d['avg_wer']:>9.1%} {d['max_wer']:>9.1%}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/stt_ground_truth/compare.py results1.json results2.json ...")
        sys.exit(1)
    compare(sys.argv[1:])
