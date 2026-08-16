# Starts the FastAPI server (if needed) and opens a public cloudflared tunnel
# so the demo is reachable over HTTPS from anywhere. The laptop must stay on.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
#
# Prints the public URL (https://<random>.trycloudflare.com). Open it, click
# the mic, and speak Hindi. Text input works too (no STT key needed).
# For live voice STT, set SARVAM_API_KEY in .env before starting.

param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$cloudflared = Join-Path $env:LOCALAPPDATA 'cloudflared\cloudflared.exe'
if (-not (Test-Path $cloudflared)) {
    Write-Error "cloudflared not found at $cloudflared - run winget install Cloudflare.cloudflared"
    exit 1
}

function Test-OurHealth {
    param([int]$p)
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 3
        return ($r.status -eq 'ok' -and $r.version -eq '1.0')
    } catch { return $false }
}

function Test-PortAvailable {
    param([int]$p)
    return $null -eq (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

# Pick the port: reuse an already-running copy of OUR API, else the first free
# port at or above $Port (probe upward past any foreign service).
$selected = -1
for ($try = 0; $try -lt 20; $try++) {
    $candidate = $Port + $try
    if (Test-OurHealth $candidate) { $selected = $candidate; break }
    if (Test-PortAvailable $candidate) { $selected = $candidate; break }
}
if ($selected -lt 0) {
    Write-Error "no free port found near :$Port"
    exit 1
}

if (Test-OurHealth $selected) {
    Write-Host "[demo] API server already running on :$selected"
} else {
    Write-Host "[demo] starting API server on :$selected ..."
    $env:PYTHONIOENCODING = 'utf-8'
    Start-Process -FilePath python -ArgumentList '-m','uvicorn','app.api.server:app','--host','127.0.0.1','--port',"$selected" -WorkingDirectory (Get-Location) -WindowStyle Minimized | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        if (Test-OurHealth $selected) { $ready = $true; break }
    }
    if (-not $ready) { Write-Error "API server failed to become healthy" ; exit 1 }
    Write-Host "[demo] API server ready on :$selected"
}

Write-Host "[demo] opening public tunnel - keep this window open. Copy the https:// URL below."
& $cloudflared tunnel --url "http://127.0.0.1:$selected"
