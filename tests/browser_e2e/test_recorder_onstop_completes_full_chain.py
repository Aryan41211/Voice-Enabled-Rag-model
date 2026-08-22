"""Regression test: recorder.onstop must run the FULL capture chain.

Guards against the mimeType crash class: `recorder` used to be set to null in
the onstop handler *before* `recorder.mimeType` was read, which threw
"Cannot read properties of null (reading 'mimeType')" and aborted onstop
BEFORE webmToWav()/postVoice() ever ran - so recordings silently produced no
transcript at all.

This test drives a real Chromium against the real served page with a fake
microphone and asserts the complete chain actually EXECUTES end to end:
    record click -> MediaRecorder start -> stop click -> onstop completes ->
    webmToWav() emits "[stt] converted WAV:" console line ->
    POST /v1/voice succeeds (route intercepted, no backend needed) ->
    showResult() renders the transcript into #transcript.

It fails if any link is skipped - not merely if nothing throws.

Run:  pytest tests/browser_e2e/test_recorder_onstop_completes_full_chain.py
Requires: playwright (pip install playwright && playwright install chromium)
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

playwright = pytest.importorskip(
    "playwright", reason="pip install playwright && playwright install chromium"
)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "api" / "static"
SAMPLE_CLIP = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "stt_ground_truth"
    / "clips"
    / "hi"
    / "hi_01.wav"
)

MOCK_VOICE_RESPONSE = {
    "request_id": "test-000000",
    "transcript": "mock transcript ok",
    "transcript_language": "hi-IN",
    "answer": "mock answer ok",
    "refused": False,
    "sources": [],
    "timings_ms": {"total_ms": 1.0},
}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture()
def static_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_recorder_onstop_completes_full_chain(static_server):
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    page_errors: list[str] = []
    stt_logs: list[str] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-audio-capture={SAMPLE_CLIP}",
                ],
            )
        except Exception as exc:  # browser binary not installed
            pytest.skip(f"chromium unavailable: {exc}")

        try:
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()
            page.on(
                "console",
                lambda m: (
                    console_errors.append(m.text)
                    if m.type == "error"
                    else stt_logs.append(m.text)
                    if "[stt]" in m.text
                    else None
                ),
            )
            page.on("pageerror", lambda e: page_errors.append(str(e)))

            # Serve the real page; intercept the API so the test is hermetic.
            page.route(
                "**/v1/voice",
                lambda route: route.fulfill(status=200, json=MOCK_VOICE_RESPONSE),
            )
            page.route(
                "**/query",
                lambda route: route.fulfill(status=200, json=MOCK_VOICE_RESPONSE),
            )
            page.goto(static_server, wait_until="domcontentloaded")

            # --- record ---------------------------------------------------
            page.click("#recBtn")
            page.locator("#status", has_text="Listening…").wait_for(timeout=5000)
            page.wait_for_timeout(2000)
            page.click("#recBtn")  # stop

            # --- chain completion assertions ------------------------------
            # 1. webmToWav executed and returned audio bytes (not just "no throw").
            #    Console events arrive async over CDP - poll for the log line.
            import time

            def _converted_line():
                return next(
                    (line for line in stt_logs if "[stt] converted WAV:" in line), None
                )

            deadline = time.time() + 15
            converted_line = _converted_line()
            while converted_line is None and time.time() < deadline:
                page.wait_for_timeout(100)
                converted_line = _converted_line()
            assert (
                converted_line is not None
            ), "onstop never reached webmToWav() output - chain aborted early"
            wav_bytes = int(converted_line.split(":")[1].strip().split()[0])
            assert wav_bytes > 44, f"WAV suspiciously small: {wav_bytes}B"

            # 2. postVoice succeeded and showResult rendered the transcript -
            #    proves onstop ran past the old crash point to completion.
            page.wait_for_function(
                "() => document.getElementById('transcript')"
                ".textContent.trim() === 'mock transcript ok'",
                timeout=15000,
            )
            # 3. Recorder state fully reset for the next cycle.
            assert page.locator("#status").inner_text().strip() == "Ready"
            assert page.locator("#recBtn").inner_text().strip() == "🎤 Record"

            # 4. No JS errors of any kind during the cycle.
            assert not page_errors, f"pageerrors: {page_errors}"
            assert not console_errors, f"console errors: {console_errors}"

            # 5. A valid mime was chosen by the fallback chain.
            mime_lines = [
                line for line in stt_logs if "MediaRecorder using mimeType" in line
            ]
            assert mime_lines, "mimeType fallback chain did not log its choice"
        finally:
            browser.close()
