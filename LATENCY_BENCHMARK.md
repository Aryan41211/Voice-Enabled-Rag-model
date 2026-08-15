# Latency Benchmarking

## Methodology
1. Sample **≥100 real queries** from `MSMARCO-XI` (not hand-picked, not a single "lucky run" — brief explicitly calls this out). Use `datasets`' shuffle + a fixed seed for reproducibility.
2. For each query, run the full pipeline and record **per-stage** timestamps via the harness's tracing (STT excluded/separate — see ARCHITECTURE.md §2):
   - `t_query_embed`
   - `t_ann_search`
   - `t_bm25_search`
   - `t_fusion`
   - `t_rerank` (if enabled)
   - `t_guardrail_pre`
   - `t_ttft` (generation time-to-first-token)
   - `t_full_generation` (report separately, don't hide it)
   - `t_guardrail_post`
   - `t_total` = sum of everything counted in your stated "pipeline" definition
3. Warm up the system first (discard the first ~10 requests — cold caches/model loading shouldn't count against you, but say so explicitly).
4. Run sequentially, not in parallel, unless you're also benchmarking concurrency (call that out separately if you do it).
5. Compute **P50, P70, P100 (max)** — not average, per the brief.

## Reporting Template

```
Benchmark run: <date>
Queries: <N> sampled from MSMARCO-XI (<language>), seed=<seed>
Environment: <local / cloud, hardware spec>

Retrieval-only pipeline (embed → search → fusion → rerank → guardrails):
  P50: __ ms   P70: __ ms   P100: __ ms

Retrieval + generation TTFT:
  P50: __ ms   P70: __ ms   P100: __ ms

Retrieval + full generation (for transparency, not claimed as "the 200ms number"):
  P50: __ ms   P70: __ ms   P100: __ ms

STT (separate stage):
  P50: __ ms   P70: __ ms   P100: __ ms
```

## Script Skeleton

```python
# benchmarks/run_latency_bench.py
import time, json
from datasets import load_dataset
from app.harness import run_pipeline  # your orchestrator entrypoint

def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)

def main(n=100, warmup=10, lang="hi"):
    ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="train").shuffle(seed=42).select(range(n + warmup))
    stage_timings = {"retrieval_only": [], "retrieval_ttft": [], "retrieval_full_gen": []}

    for i, ex in enumerate(ds):
        trace = run_pipeline(ex["query"], return_trace=True)
        if i < warmup:
            continue
        stage_timings["retrieval_only"].append(trace["t_guardrail_pre"] - trace["t_query_embed_start"])
        stage_timings["retrieval_ttft"].append(trace["t_ttft"] - trace["t_query_embed_start"])
        stage_timings["retrieval_full_gen"].append(trace["t_full_generation"] - trace["t_query_embed_start"])

    report = {
        name: {"p50": percentile(v, 50), "p70": percentile(v, 70), "p100": percentile(v, 100)}
        for name, v in stage_timings.items()
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open("benchmarks/results/latest.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

## Common Pitfalls (avoid these)
- Reporting only average latency — brief asks for P50/P70/P100 explicitly.
- Benchmarking on the same 3 queries you used to build the demo — use a real sample.
- Including index-build time in per-query latency — that's offline, exclude it.
- Silently excluding the generation stage from your "200ms" claim without disclosing it — disclose it (see README §2).

## Where results go
Final numbers go in both this file's template (fill it in, don't leave placeholders) and summarized in `EVALUATION.md` alongside retrieval-quality results, so judges see latency and accuracy together.
