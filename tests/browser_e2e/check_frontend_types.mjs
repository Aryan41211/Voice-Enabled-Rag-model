/**
 * Frontend type-check guard for the recorder lifecycle.
 *
 * Extracts the inline <script> from app/api/static/index.html and runs
 * tsc --strict --checkJs over it. With strict null checks, reading a property
 * off `recorder` after the handler assigns `recorder = null` (the exact bug
 * that caused "Cannot read properties of null (reading 'mimeType')") is a
 * compile error, so this ordering mistake cannot silently reappear.
 *
 * Usage:  node tests/browser_e2e/check_frontend_types.mjs
 * Exit 0 = clean; non-zero = type errors (printed).
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");
const htmlPath = path.join(repoRoot, "app", "api", "static", "index.html");
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "frontend-tscheck-"));

const html = fs.readFileSync(htmlPath, "utf-8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) {
  console.error("No <script> block found in", htmlPath);
  process.exit(2);
}

// Header gives tsc the ambient types the inline script assumes at runtime
// (it is served after the DOM is parsed) without weakening strictness.
const header = `// @ts-check
"use strict";
`;

const srcPath = path.join(tmpDir, "extracted_script.js");
fs.writeFileSync(srcPath, header + m[1], "utf-8");

const cfgPath = path.join(tmpDir, "tsconfig.json");
fs.writeFileSync(
  cfgPath,
  JSON.stringify({
    compilerOptions: {
      allowJs: true,
      checkJs: true,
      strict: true,
      // Catch blocks receive unknown under full strict; that's orthogonal to
      // the null-ordering bug class this guard exists for.
      useUnknownInCatchVariables: false,
      noEmit: true,
      target: "es2020",
      lib: ["es2020", "dom"],
      types: [],
    },
    include: [srcPath],
  }),
  "utf-8"
);

const tscJs = path.join(here, "node_modules", "typescript", "bin", "tsc");
const res = spawnSync(process.execPath, [tscJs, "-p", cfgPath], {
  encoding: "utf-8",
});
if (res.status !== 0) {
  console.error("frontend type-check FAILED:");
  console.error(res.stdout || "");
  console.error(res.stderr || "");
  process.exit(1);
}
console.log((res.stdout || "").trim() || "frontend type-check: clean");
console.log(`checked ${path.relative(repoRoot, htmlPath)} (${m[1].length} chars of JS)`);
