# Deployment

The brief requires a **live working link**. Latency matters, so hosting choices aren't just "get it online" — they affect whether your P50/P70/P100 numbers hold up when judges actually click the link.

## Recommended Setup

| Component | Recommendation | Why |
|---|---|---|
| Backend (API/harness) | Single VM/container (e.g. a Fly.io/Railway/Render instance, or a cloud VM) — colocate with your vector index | Avoid network hops between retrieval and generation |
| Vector index | Loaded in-process (FAISS in memory) on the same host as the backend | A remote hosted vector DB adds a network round-trip that can blow your latency budget |
| Frontend | Static site (Vercel/Netlify) calling the backend over HTTPS/WSS | Simple, fast, free tier is enough for a hackathon demo |
| STT | Sarvam API (external, unavoidable network hop) | This is why STT latency is reported separately — it's inherently network-bound |
| Generation | Fast-inference API (e.g. Groq) — also external | Same reasoning as STT: report its latency honestly rather than pretending it's free |

## Environment Variables (`.env.example`)

```bash
SARVAM_API_KEY=
ELEVENLABS_API_KEY=
LLM_API_KEY=
LLM_PROVIDER=groq   # or your chosen fast-inference provider
VECTOR_INDEX_PATH=./data/index
EMBEDDING_MODEL=BAAI/bge-m3
LOG_LEVEL=INFO
```

## Steps
1. Build the offline index (`app/ingestion/build_index.py`) and commit the built index artifact (or a script that rebuilds it on deploy — index build time is not part of your latency budget, but startup time matters for demo reliability).
2. Containerize the backend (`Dockerfile` + `docker-compose.yml` for local dev parity).
3. Deploy backend to your chosen host; confirm WebSocket support if you're streaming STT/generation (not all free-tier PaaS hosts support long-lived WebSockets by default — verify this early, not on Day 8).
4. Deploy frontend, point it at the backend's public URL.
5. Smoke-test the live link from a network other than your dev machine before recording Video 2.

## Reliability for Demo Day
- Set conservative timeouts with graceful fallbacks (see `ARCHITECTURE.md` §3) so a flaky external API (Sarvam/LLM provider) doesn't hard-crash the live demo mid-recording.
- Keep a local/offline fallback demo (screen recording of a working run) as insurance in case the live link goes down right before judging — not a replacement for the live link, just a backup.
- Monitor basic uptime (even a free UptimeRobot check) for the submission window so you know immediately if the live link goes down.
