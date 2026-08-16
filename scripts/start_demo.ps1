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

function Test-Health {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if (Test-Health) {
    Write-Host "[demo] server already running on :$Port"
} else {
    Write-Host "[demo] starting API server on :$Port ..."
    $env:PYTHONIOENCODING = 'utf-8'
    Start-Process -FilePath python -ArgumentList '-m','uvicorn','app.api.server:app','--host','127.0.0.1','--port',"$Port" -WorkingDirectory (Get-Location) -WindowStyle Minimized | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Health) { $ready = $true; break }
    }
    if (-not $ready) { Write-Error "API server failed to become healthy" ; exit 1 }
    Write-Host "[demo] API server ready"
}

Write-Host "[demo] opening public tunnel - keep this window open. Copy the https:// URL below."
& $cloudflared tunnel --url "http://127.0.0.1:$Port"
