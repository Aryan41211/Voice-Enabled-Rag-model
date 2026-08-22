/**
 * Browser record→transcribe cycle runner for the Voice RAG frontend.
 *
 * Launches Chromium with a fake microphone fed by a real speech WAV file,
 * drives the actual UI recording path (getUserMedia → MediaRecorder with the
 * mimeType fallback chain → blob → webmToWav → POST /v1/voice), and reports
 * console errors + transcript + captured mime/blob diagnostics as JSON.
 *
 * Usage:
 *   node record_cycle.mjs --url=http://127.0.0.1:8014 --clip=path.wav \
 *        --duration=4.0 --out=result.json [--cycles=10]
 *
 * --cycles=N performs N consecutive record→transcribe cycles on the SAME page
 * (no reload) using the looping fake mic, verifying recorder re-initialization.
 */

import { chromium } from "playwright";
import fs from "node:fs";

function arg(name, dflt) {
  const hit = process.argv.find((a) => a.startsWith("--" + name + "="));
  return hit ? hit.split("=").slice(1).join("=") : dflt;
}

const url = arg("url", "http://127.0.0.1:8014");
const clip = arg("clip", "");
const duration = parseFloat(arg("duration", "4"));
const cycles = parseInt(arg("cycles", "1"), 10);
const outPath = arg("out", "");
const saveWav = arg("save-wav", "");

const results = [];
let uiTexts = {};
const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    ...(clip ? [`--use-file-for-fake-audio-capture=${clip}`] : []),
    "--autoplay-policy=no-user-gesture-required",
  ],
});

try {
  const context = await browser.newContext({ permissions: ["microphone"] });
  const page = await context.newPage();

  // Always intercept /v1/voice so cycles can report what the API returned
  // (transcript_language etc.); file capture additionally needs saveWav.
  await page.addInitScript(() => {
      window.__lastWavB64 = null;
      window.__lastVoiceResp = null;
      const origFetch = window.fetch;
      window.fetch = async function (...args) {
        try {
          const urlStr = typeof args[0] === "string" || args[0] instanceof URL
            ? String(args[0])
            : args[0].url;
          if (urlStr.includes("/v1/voice")) {
            const resp = await origFetch.apply(window, args);
            try {
              window.__lastVoiceResp =
                (resp.clone ? await resp.clone().json() : null) || null;
            } catch (_) { /* non-JSON */ }
            const req = args[0] instanceof Request
              ? args[0].clone()
              : new Request(urlStr, args[1]);
            const buf = await req.arrayBuffer();
            let bin = "";
            const bytes = new Uint8Array(buf);
            for (let i = 0; i < bytes.length; i += 0x8000)
              bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
            window.__lastWavB64 = btoa(bin);
            return resp;
          }
        } catch (_) { /* diagnostic only */ }
        return origFetch.apply(window, args);
      };
    });

  const consoleErrors = [];
  const sttLogs = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      consoleErrors.push(`[console.${msg.type()}] ${msg.text()}`);
    }
    const text = msg.text();
    if (text.includes("[stt]")) sttLogs.push(text);
  });
  page.on("pageerror", (err) => consoleErrors.push(`[pageerror] ${err.message}`));

  await page.goto(url, { waitUntil: "domcontentloaded" });

  // UI chrome must be English
  uiTexts = {
    recBtn: await page.locator("#recBtn").innerText(),
    status: await page.locator("#status").innerText(),
    placeholder: await page.locator("#textInput").getAttribute("placeholder"),
    askBtn: await page.locator("#askBtn").innerText(),
  };

  for (let i = 0; i < cycles; i++) {
    const errsBefore = consoleErrors.length;
    const logsBefore = sttLogs.length;

    await page.locator("#recBtn").click();
    await page.locator("#status", { hasText: "Listening…" }).waitFor({ timeout: 5000 });

    // Record for one playthrough of the clip plus a small margin (the fake
    // mic loops the file; too much margin appends the loop's opening words).
    await page.waitForTimeout(Math.round(duration * 1000) + 400);

    await page.locator("#recBtn").click(); // stop

    // Wait for transcript to populate; surface any error text immediately.
    const outcome = await page.waitForFunction(
      () => {
        const t = document.getElementById("transcript").textContent.trim();
        const e = document.getElementById("err").textContent.trim();
        if (e) return { state: "error", error: e };
        if (t) return { state: "ok", transcript: t };
        return null;
      },
      null,
      { timeout: 45000, polling: 250 }
    ).then((h) => h.jsonValue());

    // Console messages reach Node asynchronously over CDP; wait until this
    // cycle's "raw capture" diagnostic has actually been delivered.
    const logDeadline = Date.now() + 4000;
    while (
      !sttLogs.slice(logsBefore).some((l) => l.includes("raw capture:")) &&
      Date.now() < logDeadline
    ) {
      await page.waitForTimeout(100);
    }

    const cycleLogs = sttLogs.slice(logsBefore);
    let voiceResp = null;
    try { voiceResp = await page.evaluate(() => window.__lastVoiceResp); } catch (_) {}
    const rawLine = cycleLogs.find((l) => l.includes("raw capture:")) || "";
    const mimeLine =
      cycleLogs.find((l) => l.includes("MediaRecorder using mimeType:")) || "";
    const sizeMatch = rawLine.match(/raw capture:\s*(\d+)\s*bytes\s*(\S*)/);
    const mimeMatch = mimeLine.match(/mimeType:\s*(.+)$/);

    results.push({
      cycle: i + 1,
      ok: outcome.state === "ok",
      error: outcome.error || null,
      transcript: outcome.transcript || null,
      rawBlobSize: sizeMatch ? parseInt(sizeMatch[1], 10) : null,
      rawBlobMime: sizeMatch ? sizeMatch[2] || null : null,
      chosenMime: mimeMatch ? mimeMatch[1].trim() : null,
      detectedLang: voiceResp ? voiceResp.transcript_language || null : null,
      consoleErrorsThisCycle: consoleErrors.slice(errsBefore),
      logs: cycleLogs,
    });
  }

  if (saveWav && results.length && results[results.length - 1].ok) {
    const b64 = await page.evaluate(() => window.__lastWavB64);
    if (b64) fs.writeFileSync(saveWav, Buffer.from(b64, "base64"));
  }

  await context.close();
} finally {
  await browser.close();
}

// Global assertions
const allOk = results.every((r) => r.ok);
const zeroConsoleErrors = results.every((r) => r.consoleErrorsThisCycle.length === 0);
const validBlobs = results.every(
  (r) => r.rawBlobSize > 1000 && r.rawBlobMime && /^audio\//.test(r.rawBlobMime)
);

const report = {
  allOk,
  zeroConsoleErrors,
  validBlobs,
  uiTexts,
  results,
};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf-8");
console.log(JSON.stringify(report, null, 2));
process.exit(allOk && zeroConsoleErrors && validBlobs ? 0 : 1);
