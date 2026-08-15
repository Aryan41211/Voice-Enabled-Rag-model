# Guardrails

The brief: *"Show that your system knows when not to answer, not just how to answer."* This is scored — build it as a first-class pipeline stage, not an afterthought prompt instruction.

## Layer 1 — Input Guardrails (pre-retrieval, cheap & fast)

| Check | Method | Action on fail |
|---|---|---|
| Empty/garbage transcript | Min length + token count + STT confidence threshold | Ask user to repeat, don't call retrieval |
| Off-topic query | Domain keyword gate (gambling/crypto/dating/etc.) — measured: embedding similarity to an in-domain centroid does **not** separate off-topic queries (all score 0.83–0.89), see EVALUATION.md §6 | Respond: "That's outside what I can answer from this dataset" |
| Unsafe/inappropriate input | Keyword + moderation-classifier check | Refuse with a fixed safe response, log the event |
| Wrong/unsupported language | Language-ID on transcript vs. supported index languages | Ask user to switch language or auto-route to the matching language index |

## Layer 2 — Retrieval Guardrails

| Check | Method | Action on fail |
|---|---|---|
| No relevant results | Isolation margin (top-1 vs rank-20 cosine, `min_margin=0.03`) calibrated on eval gold | Skip generation entirely → "I couldn't find relevant information for that" |
| Low retrieval confidence | Score gap between top-1 and top-k too flat — measured too noisy, **disabled by default** (see EVALUATION.md §6) | Ask a clarifying question instead of guessing |

## Layer 3 — Output Guardrails (post-generation, before returning to user)

| Check | Method | Action on fail |
|---|---|---|
| Groundedness / hallucination | Embedding similarity or NLI entailment between generated answer sentences and retrieved passages; every claim must be supported | Suppress the answer, fall back to returning the top passage verbatim (extractive) with a note that it couldn't synthesize confidently |
| Answer contains no citation | Require the prompt/output schema to include passage IDs used; reject if missing | Regenerate once with a stricter prompt, then fall back |
| Length/format sanity | Refuse absurdly long or malformed outputs | Truncate/regenerate |

## Design Principles
1. **Fail closed, not open.** When uncertain, refuse or hedge — don't generate a confident-sounding but ungrounded answer.
2. **Guardrails are cheap and fast, not another full LLM call where avoidable.** Use small classifiers, embedding similarity, or rule-based checks in the hot path to stay inside the latency budget. Save any LLM-based guardrail (e.g. LLM-as-judge groundedness) for an *offline* eval pass, not the live request.
3. **Every refusal must be explainable in the demo.** Prepare 2–3 canned queries in your demo video that show refusal working live: one off-topic question, one where retrieval has nothing relevant, one where you deliberately ask something the dataset can't ground.
4. **Log everything.** Every guardrail trigger should be logged with the reason — this becomes evidence in your video that the system "knows when not to answer" rather than a claim you can't back up.

## Suggested Refusal Templates
- Off-topic: *"I can only answer questions grounded in the provided dataset — that looks outside its scope."*
- No relevant retrieval: *"I searched the knowledge base but couldn't find anything relevant to that question."*
- Ungrounded generation: *"I found related information but I'm not confident enough in a synthesized answer — here's the closest passage I found instead: [passage]."*

## Testing Guardrails
Adversarial test cases belong in `TESTING.md` — every guardrail row above needs at least one automated test that intentionally triggers it, not just manual spot-checks before the demo.
