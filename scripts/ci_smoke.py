"""CI end-to-end smoke: load the built index, answer gold queries, fail if none."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    gold_path = Path("data/index/hi/eval_gold.jsonl")
    if not gold_path.exists():
        print(
            "[ci_smoke] eval_gold.jsonl missing - build the index first",
            file=sys.stderr,
        )
        return 1

    from app.harness.pipeline import Pipeline

    gold = [
        json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()
    ]
    if not gold:
        print("[ci_smoke] eval_gold.jsonl is empty", file=sys.stderr)
        return 1

    pipeline = Pipeline.from_index(lang="hi", strategy="metadata")
    pipeline.warmup()

    answered = 0
    for ex in gold:
        resp = pipeline.query(ex["query"])
        ok = not resp.refused and bool(resp.answer)
        answered += int(ok)
        print(
            json.dumps(
                {
                    "query": ex["query"],
                    "refused": resp.refused,
                    "answer_len": len(resp.answer or ""),
                    "total_ms": resp.timings_ms.get("total_ms"),
                },
                ensure_ascii=False,
            )
        )

    if answered == 0:
        print("[ci_smoke] FAIL: no gold query produced an answer", file=sys.stderr)
        return 1
    print(f"[ci_smoke] OK: {answered}/{len(gold)} gold queries answered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
