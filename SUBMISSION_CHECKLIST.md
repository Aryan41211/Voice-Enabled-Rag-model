# Submission Checklist

⚠️ **No resubmissions allowed.** Verify every item below before hitting submit on the form.

## Core Deliverables
- [ ] Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58
- [x] GitHub repo link — public, README complete, real (non-placeholder) benchmark numbers
- [ ] Live working link — tested from a fresh browser/network, not just localhost
      (ready: `scripts/start_demo.ps1` serves the app + a public https tunnel;
      verified /health + /query through trycloudflare — still needs an
      incognito/different-device check)
- [ ] Video 1 uploaded (Team/process, 90 seconds, shows process not product)
- [ ] Video 2 uploaded (Demo, full end-to-end working demo)

## Technical Requirement Verification
- [x] STT via Sarvam **or** ElevenLabs (only one, clearly stated in README) — Sarvam `saaras:v3-realtime`, verified live in CI
- [x] Multiple chunking strategies implemented and documented with comparison data in EVALUATION.md (not just claimed)
- [x] Latency numbers reported as P50/P70/P100 across a real query sample (not one lucky run) — LATENCY_BENCHMARK.md, 110 real queries
- [x] 200ms claim is scoped and disclosed honestly (retrieval vs. TTFT vs. full generation — see README §2)
- [x] Harness: retries, structured I/O, error recovery all demonstrably present in code (not just described)
- [ ] Guardrails: at least one live demonstration in Video 2 of the system refusing to answer

## Promotion Requirement (mandatory, per-member)
For **every** team member, on **every** platform:

| Member | Instagram post + #RAGInGoa | X post + #RAGInGoa | LinkedIn post + #RAGInGoa | Account public? |
|---|---|---|---|---|
| | ☐ | ☐ | ☐ | ☐ (≥1 IG account must be public) |
| | ☐ | ☐ | ☐ | |
| | ☐ | ☐ | ☐ | |
| | ☐ | ☐ | ☐ | |

- [ ] Both videos posted (not just one) by each member
- [ ] `#RAGInGoa` present on **every single post**, every platform, every member — no exceptions
- [ ] At least one Instagram account among the team is set to public before posting

## Final Pre-Submit Sanity Check
- [ ] Fresh `git clone` + follow README instructions exactly → does it actually run?
- [ ] Live link accessed from an incognito window / different device
- [ ] All links in the form (repo, live link, videos) open correctly, no typos
- [ ] Deadline confirmed: **August 22, 2026, 11:59 PM IST** — submit with buffer, not at 11:58 PM
