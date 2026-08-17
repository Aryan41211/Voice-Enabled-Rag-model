# STT Diagnostic Report (pre-fix)

Date: 2026-08-17
Scope: voice-to-text pipeline (browser capture -> `POST /v1/voice` -> Sarvam
`saaras:v3-realtime` WS, REST fallback). All findings below were verified by
live calls to the real Sarvam API using the configured key, with full
wire-level logging. No implementation changes were made before this report.

## AUDIO FORMAT

| Item | Value |
|---|---|
| Actual sample rate | 16000 Hz (verified from synthesized WAV headers) |
| Configured sample rate | 16000 (hardcoded WS `sample_rate=16000`; frontend `AudioContext({sampleRate:16000})`) |
| Actual channels | 1 (mono; frontend uses `getChannelData(0)`) |
| Actual bit depth | 16-bit signed little-endian PCM |
| Actual encoding | WAV container -> raw linear16 PCM (`_to_pcm`), sent base64 |
| Actual bytes/chunk | 2048 B = 64 ms @16 kHz mono |
| Browser transforms | WebM/Opus -> `decodeAudioData` on `AudioContext({sampleRate:16000})` -> nearest-neighbor `downsample()` -> WAV. Browser sampleRate-option support NOT verified. |
| STT expected format | linear16, **mono only**, 8000 or 16000 Hz only (docs: other rates close connection 4000) |
| MATCH / MISMATCH | MATCH for bundled frontend. MISMATCH RISK: `_to_pcm` validates only `sampwidth==2`; a 44.1 kHz or stereo WAV is sent mislabeled as 16 kHz mono. |

## LANGUAGE / MODEL

| Item | Value |
|---|---|
| Language configured | `data_lang=hi` -> `LANGUAGE_CODES["hi"]` = `hi-IN` |
| Language actually sent | `hi-IN` (verified in `session.begin` echo) |
| Model configured / used | `saaras:v3-realtime` WS / `saaras:v3` REST. WS model echoed by server. |
| SDK / default overrides | None. ElevenLabs is NOT in the pipeline (only an unused env placeholder). No per-request language hint. |
| Code-switching | `mode=transcribe` transliterates English loanwords to Devanagari (GT-05: "national bird" -> "नेशनल बर्ड"). App is Hindi-only by design. |

## VAD / STREAMING

| Item | Value |
|---|---|
| VAD | Server-side (`endpointing=vad`). Negotiated: threshold 0.3, prefix_padding_ms 300, **silence_duration_ms 1000**, min_speech_duration_ms 250 |
| Buffer strategy | WHOLE-FILE BLAST: 3.53 s clip = 56 chunks sent in 9 ms; 6.17 s = 97 chunks in 33 ms. Not realtime. |
| Truncation | **CONFIRMED**: client breaks on the FIRST `transcript.final`; a 2-utterance file (1.5 s gap) produced 2 server finals, client returned only the first |
| Dropped chunks | No sequence numbering / ack / ordering checks. Not observed dropping in tests; unobservable in production. |
| Partials | Ignored (correct) but never logged. |

## NETWORK

| Item | Value |
|---|---|
| Transport | `wss://api.sarvam.ai/speech-to-text-realtime/ws` (WS) + REST fallback |
| Reconnect | None; single-shot WS, one-shot REST fallback |
| REST fallback | **CONFIRMED BROKEN**: `https://api.sarvam.ai/v1/speech-to-text` -> HTTP 404. Correct endpoint `https://api.sarvam.ai/speech-to-text` verified 200 + exact transcripts. |
| Long audio | **CONFIRMED**: ~50 s file exceeded 10 s `stt_timeout_s` (asyncio.TimeoutError); TimeoutError bypasses WS->REST fallback, whole request fails |

## GROUND TRUTH (TTS-generated 16 kHz mono WAV, hi-IN, real Sarvam API)

| # | Ground truth | WS final (exact app behavior) |
|---|---|---|
| 1 | भारत का राष्ट्रीय पक्षी कौन सा है | भारत का राष्ट्रीय पक्षी कौन सा है? (exact) |
| 2 | चंद्रयान तीन का प्रक्षेपण कब हुआ था | चंद्रयान तीन का प्रक्षेपण कब हुआ था? (exact) |
| 3 | क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है | ... में क्या संबंध है? (exact) |
| 4 | जब भारत को आज़ादी मिली, तब देश का पहला प्रधानमंत्री कौन बना | exact (nuqta normalized आजादी) |
| 5 | मुझे भारत की national bird के बारे में information चाहिए | ... नेशनल बर्ड ... इंफॉर्मेशन चाहिए (English -> Devanagari) |

## ROOT CAUSE

- **Primary**: multi-utterance truncation (`_transcribe_ws` returns after the
  first `transcript.final`; server VAD splits turns at ~1 s silence).
- **Secondary**: stale REST fallback URL (404); no audio-format validation
  before the API call; 10 s timeout kills long clips before fallback can run;
  no per-chunk/sequence/segment logging.

## POST-FIX VERIFICATION (2026-08-17)

All fixes committed to `main` and verified live against the real Sarvam API:

| Fix | Commit | Verified |
|---|---|---|
| REST URL -> `https://api.sarvam.ai/speech-to-text` | `3f1c839` | 200 + exact transcripts; multi-utterance join |
| WS collects ALL `transcript.final` (no first-final truncation) | `a8a7f74` | 2-utterance file returns both sentences |
| Audio normalized to 16 kHz mono before the API call (numpy) | `3a5fa62` | 8 kHz, 16 kHz and 48 kHz-stereo WAVs all exact |
| WS attempt capped at half the STT budget -> REST fallback on timeout | `6233b30` | hanging-WS test; `stt_timeout_s` 10->20 |
| `/v1/voice` temp path sanitized (traversal + dead-code precedence) | `b2b3d5d` | upload with `../../evil.wav` stays in tmp dir |
| Buffer spec / WS stream / REST status logging | `4ba8f8b` | buffer specs at INFO, transcripts DEBUG-only |
| Frontend 16 kHz decode via OfflineAudioContext (+ linear downsample) | `38c931e` | server-side format validation covers mismatch (NOT browser-verified) |
| Ground-truth fixtures + regression script | `b320812` | `scripts/stt_ground_truth.py`: 5/5 PASS live |

Ground-truth after fixes (through `SarvamSTT().transcribe`, WS primary):

| # | Result |
|---|---|
| 1 | भारत का राष्ट्रीय पक्षी कौन सा है? (exact) |
| 2 | चंद्रयान तीन का प्रक्षेपण कब हुआ था? (exact) |
| 3 | क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है? (exact) |
| 4 | जब भारत को आजादी मिली, तब देश का पहला प्रधानमंत्री कौन बना? (exact) |
| 5 | मुझे भारत की नेशनल बर्ड के बारे में इंफॉर्मेशन चाहिए। (code-switched Devanagari) |

Full test suite: 126 passed (was 116). `ruff check` and `ruff format --check` clean.

## IMPROVEMENT ROUND (2026-08-17)

Four commits pushed after the initial fix round, all to `main`:

| Area | Commit | What changed |
|---|---|---|
| Answer quality | `cc51a8d` | `ExtractiveGenerator` now sentence-splits top chunks, scores by query-token overlap (stopwords stripped), and returns a crisp fact instead of a verbatim passage dump |
| Long-audio batch | `ff296be` | `SarvamSTT._transcribe_batch()` — the 7-step async batch job API (initiate → upload-files → PUT → start → poll → download-files → GET transcript). Routed up front from WAV-header duration check (>30 s) and via REST 400 duration-limit error fallback |
| E2e voice + latency | `7060ee8` | `scripts/e2e_voice.py` — full voice round-trip (edge-tts → Sarvam WS STT → pipeline → answer) on N eval_gold queries with per-stage timings |
| Frontend + docs | `26982c3` | Enter-to-submit, collapsible `<details>` sources, mic permission error hints; `data/NOTES.md` with corpus expansion steps + latency benchmarks |

### Post-improvement latency (real API)

| Stage | P50 |
|---|---|
| Retrieval (hybrid dense+sparse) | ~34 ms |
| Generation (sentence-extractive) | <1 ms |
| STT (WS path, short audio) | ~2-3 s |
| **End-to-end total** | **~3-3.5 s** |
| Pipeline cold load | ~13 s (SentenceTransformer + FAISS mmap) |

Full test suite: **132 passed** (was 126 after STT fixes). `ruff check` / `ruff format --check` clean.
