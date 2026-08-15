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
STT_PROVIDER=sarvam          # sarvam | fake (keyless dev)
LLM_PROVIDER=extractive      # extractive | groq | openai
GROQ_API_KEY=
OPENAI_API_KEY=
DATA_STRATEGY=metadata       # measured best chunking strategy
DATA_LANG=hi
RETRIEVAL_TIMEOUT_S=30
PORT=8000
```

## Containerized Deploy
The repo ships `Dockerfile` + `docker-compose.yml`. `scripts/entrypoint.py` checks for the built index and rebuilds it on first boot (only the selected `DATA_STRATEGY`), so `docker compose up --build` is the fastest path to a live link. Pre-build the index locally and bake it into the image for faster startup — index build time is not part of your latency budget, but startup time matters for demo reliability.

## Steps
1. Build the offline index (`python -m app.ingestion.build_index --lang hi --strategies metadata`), optionally bake into the image.
2. `docker compose up --build`, then `curl localhost:8000/health`.
3. Deploy the container to your chosen host (Fly.io/Railway/Render/cloud VM); confirm WebSocket support if you stream STT/generation — not all free-tier PaaS hosts support long-lived WebSockets by default. Verify this early, not on Day 8.
4. Deploy a frontend (static site on Vercel/Netlify) that posts WAV audio to `POST /v1/voice`.
5. Smoke-test the live link from a network other than your dev machine before recording Video 2.

## Reliability for Demo Day
- Set conservative timeouts with graceful fallbacks (see `ARCHITECTURE.md` §3) so a flaky external API (Sarvam/LLM provider) doesn't hard-crash the live demo mid-recording.
- Keep a local/offline fallback demo (screen recording of a working run) as insurance in case the live link goes down right before judging — not a replacement for the live link, just a backup.
- Monitor basic uptime (even a free UptimeRobot check) for the submission window so you know immediately if the live link goes down.
