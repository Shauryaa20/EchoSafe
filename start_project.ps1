$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' was not found. Install Python 3.10+ and retry." }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm was not found. Install Node.js LTS and retry." }

$venv = Join-Path $root "backend\.venv"
if (-not (Test-Path $venv)) { py -3 -m venv $venv }
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install -r (Join-Path $root "backend\requirements.txt")
Push-Location (Join-Path $root "frontend")
if (-not (Test-Path "node_modules")) { npm install }
Pop-Location

Write-Host "Starting EchoSafe API at http://127.0.0.1:8001 and UI at http://localhost:5173" -ForegroundColor Cyan
$apiRunning = $false
try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/health" -TimeoutSec 2
  if ($health.status -eq "ok") { $apiRunning = $true }
} catch {}

$api = $null
if (-not $apiRunning) {
  $api = Start-Process -FilePath $python -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 8001" -WorkingDirectory (Join-Path $root "backend") -PassThru
  Start-Sleep -Seconds 3
}
$ui = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory (Join-Path $root "frontend") -PassThru
try {
  Write-Host "Press Ctrl+C to stop both processes."
  if ($api) { Wait-Process -Id $api.Id } else { Wait-Process -Id $ui.Id }
} finally {
  if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id }
  if ($ui -and -not $ui.HasExited) { Stop-Process -Id $ui.Id }
}
