---
title: Voice RAG Demo — Hindi
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Voice RAG Demo (Hindi)

Voice → transcript → retrieval → guardrailed answer. Backed by
`ai4bharat/MSMARCO-XI` (hi) with metadata-aware passage chunking and
`intfloat/multilingual-e5-small` embeddings. Full pipeline (retrieval +
extractive generation) ~15 ms P50 on a laptop GPU.

- `GET /health` — liveness + config
- `POST /query` — text in / answer out
- `POST /v1/voice` — WAV audio in / answer out (Sarvam STT)
- `/` — browser voice UI (record with the mic button, or type below it)

Set the `SARVAM_API_KEY` Space secret to enable voice; text queries work
without it.
