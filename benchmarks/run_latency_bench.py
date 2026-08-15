"""Full-pipeline latency benchmark (P50/P70/P100) over real corpus queries.

Methodology follows LATENCY_BENCHMARK.md:
* queries are sampled with a fixed seed from ``data/index/{lang}/eval_gold.jsonl``
  (real MSMARCO-XI Hindi queries, not hand-picked);
* the system is warmed up first (discarded requests);
* runs are sequential; per-stage timings come from the harness tracing in
  ``QueryResponse.timings_ms``;
* generation is the default **extractive** stage, so no network is involved.

Reports three totals per query (see docs for how each is defined) and saves
the JSON to ``benchmarks/results/latency_bench.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.harness.pipeline import Pipeline


def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def _load_queries(lang: str, index_dir: Path, n: int, seed: int) -> list[str]:
    path = index_dir / lang / "eval_gold.jsonl"
    with open(path, encoding="utf-8") as fh:
        queries = [json.loads(line)["query"] for line in fh]
    rng = random.Random(seed)
    rng.shuffle(queries)
    return queries[:n]


def main(
    lang: str = "hi",
    strategy: str = "metadata",
    n: int = 110,
    warmup: int = 10,
    seed: int = 42,
    index_dir: str | Path | None = None,
) -> dict:
    base = Path(index_dir or "./data/index")
    queries = _load_queries(lang, base, n + warmup, seed)

    pipeline = Pipeline.from_index(lang=lang, strategy=strategy, index_dir=base)
    pipeline.warmup()

    retrieval_only: list[float] = []
    retrieval_ttft: list[float] = []
    retrieval_full_gen: list[float] = []
    refused = 0

    for i, q in enumerate(queries):
        resp = pipeline.query(q)
        if i < warmup:
            continue
        t = resp.timings_ms
        retrieval_only.append(
            t.get("retrieval_ms", 0.0) + t.get("retrieval_guardrail_ms", 0.0)
        )
        retrieval_ttft.append(
            t.get("input_guardrail_ms", 0.0)
            + t.get("retrieval_ms", 0.0)
            + t.get("retrieval_guardrail_ms", 0.0)
            + t.get("ttft_ms", 0.0)
        )
        retrieval_full_gen.append(t.get("total_ms", 0.0))
        if resp.refused:
            refused += 1

    report = {
        "benchmark_run": "pipeline latency",
        "queries": len(retrieval_only),
        "sampled_from": f"MSMARCO-XI hi/validation (eval_gold seed={seed})",
        "strategy": strategy,
        "generation": pipeline.generator.name,
        "refused": refused,
        "stages": {
            name: {
                "p50": round(percentile(v, 50), 2),
                "p70": round(percentile(v, 70), 2),
                "p100": round(percentile(v, 100), 2),
            }
            for name, v in (
                ("retrieval_only", retrieval_only),
                ("retrieval_ttft", retrieval_ttft),
                ("retrieval_full_gen", retrieval_full_gen),
            )
        },
    }
    out = Path("./benchmarks/results/latency_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-pipeline latency benchmark")
    p.add_argument("--lang", default="hi")
    p.add_argument("--strategy", default="metadata")
    p.add_argument("--n", type=int, default=110)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index-dir", default=None)
    args = p.parse_args()
    main(
        lang=args.lang,
        strategy=args.strategy,
        n=args.n,
        warmup=args.warmup,
        seed=args.seed,
        index_dir=args.index_dir,
    )
