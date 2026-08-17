# STT Pipeline Diagnostic Report

Date: 2026-08-17
Engineer: Senior Audio/STT Debugging
Scope: Voice-to-text pipeline (Sarvam saaras:v3-realtime WebSocket, REST fallback, browser capture)

---

## DIAGNOSTIC FINDINGS

### AUDIO FORMAT
- Actual sample rate: **16000 Hz** (verified via `_to_pcm` + WAV header)
- Configured sample rate: **16000 Hz** (both browser WAV encoding and WS params)
- Actual channels: **1** (mono, after OfflineAudioContext(1,1,16000) + _normalize_pcm)
- Actual bit depth: **16-bit signed LE** (PCM s16le)
- Actual encoding: **WAV PCM s16le** (browser sends WAV; server reads via wave module)
- Actual bytes/chunk (WS): **2048 bytes** = 1024 samples = 64ms at 16kHz
- Actual duration/chunk: **64ms** (2048 bytes / 2 bytes per sample / 16000 Hz)
- Browser/client transformations: WebM/Opus (MediaRecorder) → OfflineAudioContext(1,1,16000) decode+resample → WAV s16le 16kHz mono → POST /v1/voice
- STT expected format (realtime WS): `{"event":"audio_input","audio":"<base64>"}` with `encoding=linear16` connection param
- MATCH: ✓ (format is correct; Sarvam realtime endpoint accepts linear16 PCM in audio_input events)

### LANGUAGE / MODEL
- Language configured: `hi` (data_lang from .env)
- Language actually sent: `hi-IN` (via LANGUAGE_CODES mapping) — ✓ correct
- Model configured: `saaras:v3-realtime` (hardcoded in _transcribe_ws)
- Model actually used: `saaras:v3-realtime` — ✓ confirmed in WS URL params
- SDK/default overrides: None; config reads from .env, STT client uses settings directly
- MATCH: ✓ (language and model are correct for the realtime endpoint)

### VAD / STREAMING
- VAD implementation: **Server-side VAD** (endpointing=vad)
- Speech start threshold: **default 0.3** (not explicitly configured)
- Speech end threshold: **default** (silence_duration_ms=500ms, not explicitly configured)
- Silence duration: **500ms** (default; not explicitly configured)
- Min speech duration: **250ms** (default; not explicitly configured)
- Chunk duration: **64ms** (2048 bytes; Sarvam recommends ~100ms = 3200 bytes)
- Buffer strategy: Entire audio sent in burst (no pacing/delay between chunks)
- **POTENTIAL ISSUE 1**: Chunk size (64ms) is smaller than recommended (100ms). May cause suboptimal VAD frame processing.
- **POTENTIAL ISSUE 2**: Audio is sent in burst, not paced. The realtime API example paces with `asyncio.sleep(0.1)`. Burst delivery may cause VAD to process frames differently than real-time.
- **POTENTIAL ISSUE 3**: No `high_vad_sensitivity` configured. For short utterances, default sensitivity may miss speech boundaries.
- **POTENTIAL ISSUE 4**: `min_speech_duration_ms=250` (default) may cut off short words/syllables.
- Potential truncation: VAD detected speech_start at 1.3s for a 3.5s clip (burst delivery delays VAD). Not a truncation issue but latency issue.
- Potential dropped chunks: None observed (all 56 chunks sent and received).

### NETWORK
- Transport: WebSocket (wss://api.sarvam.ai/speech-to-text-realtime/ws)
- WebSocket connection behavior: Connects, receives session.begin, streams all events, closes on session.end
- Reconnect behavior: No reconnect logic (single attempt, fallback to REST on failure)
- Dropped frames/chunks: None observed
- Ordering issues: None observed
- **NOTE**: WebSocket path produces results in ~1.5-2.5s for 3.5s audio. REST path produces results in ~4.8s. Both produce identical transcripts.

### BROWSER CAPTURE PATH
- MediaRecorder format: WebM/Opus (lossy compression, ~32kbps typical)
- Audio constraints: `getUserMedia({ audio: true })` — NO constraints specified
- **POTENTIAL ISSUE 5**: No audio constraints → browser picks default device settings. May capture stereo, wrong sample rate, or include noise.
- **POTENTIAL ISSUE 6**: WebM/Opus is lossy. 32kbps Opus compresses 16kHz mono PCM (256kbps) by 8:1. Quality loss for speech is usually acceptable but may degrade accuracy for quiet/ambient speech.
- Conversion: OfflineAudioContext(1, 1, 16000) forces 16kHz decode + mono extraction → WAV. Conversion is correct but adds processing overhead.

---

## GROUND TRUTH TEST RESULTS (5 sentences, pre-recorded WAV)

| # | Expected | WS Result | REST Result | WS Match | REST Match |
|---|----------|-----------|-------------|----------|------------|
| 1 | भारत का राष्ट्रीय पक्षी कौन सा है | भारत का राष्ट्रीय पक्षी कौन सा है? | same | ✓ exact | ✓ exact |
| 2 | चंद्रयान तीन का प्रक्षेपण कब हुआ था | चंद्रयान तीन का प्रक्षेपण कब हुआ था? | same | ✓ exact | ✓ exact |
| 3 | क्वांटम कंप्यूटिंग और आर्टिफिशियल इंटेलिजेंस में क्या संबंध है | same + ? | same | ✓ exact | ✓ exact |
| 4 | जब भारत को आज़ादी मिली, तब देश का पहला प्रधानमंत्री कौन बना | जब भारत को आजादी मिली, ... (nuqta normalized) | same | ✓ nuqta | ✓ nuqta |
| 5 | मुझे भारत की national bird के बारे में information चाहिए | ... नेशनल बर्ड ... इंफॉर्मेशन चाहिए। | same | ✓ translit | ✓ translit |

**Pre-recorded WAV accuracy: 5/5 = 100%**

The Sarvam STT engine itself produces correct transcripts for all test sentences.
The "nuqta normalization" (आज़ादी → आजादी) and "code-switch transliteration" (national bird → नेशनल बर्ड) are expected Sarvam behavior in `transcribe` mode.

---

## ROOT CAUSE ANALYSIS

Since pre-recorded WAV transcription is 100% accurate, the "poor accuracy" issue is likely caused by factors in the **live microphone capture path** that are NOT present when sending pre-recorded WAVs:

### Primary: Browser audio capture quality
- `getUserMedia({ audio: true })` with no constraints → browser picks default settings
- MediaRecorder captures WebM/Opus lossy (~32kbps) → quality loss
- No noise suppression, echo cancellation, or auto-gain control configured
- Different browsers/devices produce different audio quality

### Secondary: VAD tuning for pre-recorded audio
- Audio is sent in burst (not paced) → VAD processes all frames at once
- Chunk size (64ms) is smaller than recommended (100ms)
- Default VAD parameters may not be optimal for all speech patterns
- `min_speech_duration_ms=250` may cut off short utterances

### Tertiary: No diagnostic logging for live path
- No logging of actual browser audio format
- No logging of MediaRecorder settings
- No logging of conversion quality metrics

---

## RECOMMENDED FIXES

1. **Add browser audio constraints** (noiseSuppression, echoCancellation, autoGainControl)
2. **Increase WS chunk size** from 2048 to 3200 bytes (~100ms)
3. **Add VAD tuning** (high_vad_sensitivity=true, lower threshold)
4. **Add diagnostic logging** for browser audio capture format
5. **Pace audio delivery** for WS streaming (optional, for consistency)
6. **Add audio quality checks** before STT API call
