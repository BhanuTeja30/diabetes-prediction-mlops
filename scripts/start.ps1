<#
.SYNOPSIS
    Start the Diabetes Prediction MLOps stack (API, frontend, MLflow, Prometheus, Grafana).

.DESCRIPTION
    1. Starts Docker Desktop if the Docker engine is not running, and waits for it.
    2. Runs `docker compose up -d` in the project folder.
    3. Waits until the API reports healthy, then (optionally) opens the app in the browser.
    Progress is written to logs\autostart.log. Used by Start-App.cmd and the Windows login trigger.

.PARAMETER OpenBrowser
    Open the Streamlit app in the default browser once everything is up.

.PARAMETER TimeoutSeconds
    How long to wait for Docker and for the services (default 300 s each).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -OpenBrowser
#>
param(
    [switch]$OpenBrowser,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "autostart.log"

function Write-Log([string]$Message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

function Get-EnvValue([string]$Name, [string]$Default) {
    # Read a non-secret setting (ports) from .env without loading anything else.
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        $match = Select-String -Path $envFile -Pattern "^\s*$Name\s*=\s*(.+?)\s*$" | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value }
    }
    return $Default
}

function Test-DockerEngine {
    docker info --format "{{.ServerVersion}}" *> $null
    return ($LASTEXITCODE -eq 0)
}

Write-Log "=== Starting Diabetes Prediction MLOps stack ($ProjectDir) ==="

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Log "ERROR: the 'docker' command was not found. Install Docker Desktop first."
    exit 1
}
if (-not (Test-Path (Join-Path $ProjectDir ".env"))) {
    Write-Log "ERROR: .env is missing. Copy .env.example to .env and fill in the secrets (see README)."
    exit 1
}

# --- 1. Docker engine ---------------------------------------------------------------
if (-not (Test-DockerEngine)) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe")
    )
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) {
        Write-Log "ERROR: Docker engine is not running and Docker Desktop.exe was not found."
        exit 1
    }
    Write-Log "Docker engine not running - launching Docker Desktop..."
    Start-Process -FilePath $exe
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while (-not (Test-DockerEngine)) {
        if ((Get-Date) -gt $deadline) {
            Write-Log "ERROR: Docker engine did not start within $TimeoutSeconds s."
            exit 1
        }
        Start-Sleep -Seconds 5
    }
}
Write-Log "Docker engine is running."

# --- 2. Compose stack ---------------------------------------------------------------
Push-Location $ProjectDir
try {
    $output = & docker compose up -d 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $output | ForEach-Object { Add-Content -Path $LogFile -Value "    $_" -Encoding utf8 }
    if ($code -ne 0) {
        Write-Log "ERROR: 'docker compose up -d' failed (exit $code). See the lines above in $LogFile."
        exit 1
    }
} finally {
    Pop-Location
}

# --- 3. Wait for the API ------------------------------------------------------------
$apiPort = Get-EnvValue "API_PORT" "8000"
$frontendPort = Get-EnvValue "FRONTEND_PORT" "8501"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$healthy = $false
while (-not $healthy -and (Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$apiPort/health" -UseBasicParsing -TimeoutSec 5
        $healthy = ($resp.StatusCode -eq 200)
    } catch {
        Start-Sleep -Seconds 3
    }
}
if (-not $healthy) {
    Write-Log "WARNING: API did not report healthy within $TimeoutSeconds s (is a model trained and promoted?)."
    Write-Log "         Check: docker compose logs api"
    exit 2
}

$grafanaPort = Get-EnvValue "GRAFANA_PORT" "3000"
$mlflowPort = Get-EnvValue "MLFLOW_PORT" "5000"
Write-Log "All services started. App: http://localhost:$frontendPort  API docs: http://localhost:$apiPort/docs  MLflow: http://localhost:$mlflowPort  Grafana: http://localhost:$grafanaPort"

if ($OpenBrowser) {
    Start-Process "http://localhost:$frontendPort"
}
exit 0
