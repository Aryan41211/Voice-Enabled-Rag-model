# Demo & Video 2 Script

Purpose: a repeatable, honest end-to-end demo for the judges and for recording
**Video 2 (demo)**. Run every line against the real live link before recording.

## Pre-flight (do this first, every time)

1. Start the live link:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
   ```
   Copy the printed `https://<random>.trycloudflare.com` URL.
2. Open it in an **incognito window** and in a **phone on cellular data** (not your Wi-Fi).
   Both must load `/` and answer a text query.
3. Text check: `POST /query` with `{"text":"भारत का राष्ट्रीय पक्षी कौन सा है","language":"hi"}` → expect `refused: false`, Hindi answer with a source.
4. Mic check: click the mic button, speak the same sentence. If STT is keyless, use text (the refusal + text flow alone still proves the RAG).

## 60–90 second demo flow (Video 2)

| Time | Shot | Action on screen | Expected result |
|---|---|---|---|
| 0:00 | Open | Type tunnel URL, page loads | Hindi voice UI at `/`, API banner visible |
| 0:05 | Voice #1 | Speak: **"भारत का राष्ट्रीय पक्षी कौन सा है?"** | STT → retrieval → answer appears with source badge |
| 0:20 | Voice #2 | Speak: **"दिल्ली की राजधानी क्या है?"** | Answer + source |
| 0:35 | Text | Type: **"मोर की ऊँचाई कितनी होती है?"** (or any retrieval-grounded question) | Answer, not a hallucination |
| 0:50 | Refusal | Speak/type: **"कल के क्रिकेट मैच का स्कोर क्या है?"** | System **refuses** honestly (guardrail fires) |
| 1:05 | Close | Zoom on latency (if overlay used) | Retrieval P50 ~15 ms, full gen < 200 ms; STT is network-bound (~2 s), disclose honestly |

## Guardrail refusal — do it live, don't fake it

Use a question with **no support in the corpus**: future events, scores, prices,
opinions, or anything the passage store can't answer. The system must say it
can't answer rather than invent. Two safe examples:
- "कल का क्रिकेट मैच का स्कोर क्या है?"
- "आज दिल्ली में मौसम कैसा है?"

If a refusal is needed for the story arc, deliberately query something *almost*
in-corpus but out of scope (e.g., a passage-adjacent follow-up) so the guardrail
catches it naturally.

## Pro tips for the video

- Pre-verify every spoken query via the text box *before* hitting record — one
  query that gets refused is the point, but surprises on camera are not.
- Show the refusal **once, explicitly**: narrate "the system refused instead of
  hallucinating" — that's a judging criterion.
- Overlay the endpoint timing if easy; if not, say "retrieval is ~15 ms" and show
  it in the README/latency doc. Never claim 200 ms for the voice round-trip —
  STT is network-bound.
- Record with the laptop on AC power and a stable connection; the tunnel dies if
  the laptop sleeps. Keep the window open and disable sleep for the demo window.
- Have a **text-query fallback** ready in case the mic is flaky on the judge's
  machine (or cellular network).

## After recording

- [ ] Both incognito and phone loaded `/` and answered a query
- [ ] Video 2 includes one live refusal
- [ ] Repo link, live link, and both videos pasted in the submission form
- [ ] Posted with `#RAGInGoa` by every member on every platform
