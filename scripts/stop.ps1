<#
.SYNOPSIS
    Stop the Diabetes Prediction MLOps stack. Data volumes (MLflow, Prometheus, Grafana) are kept.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stop.ps1
#>
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

docker info --format "{{.ServerVersion}}" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Docker engine is not running - nothing to stop."
    exit 0
}

Push-Location $ProjectDir
try {
    & docker compose down 2>&1 | ForEach-Object { Add-Content -Path $LogFile -Value "    $_" -Encoding utf8 }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: 'docker compose down' failed (exit $LASTEXITCODE)."
        exit 1
    }
} finally {
    Pop-Location
}
Write-Log "Diabetes Prediction MLOps stack stopped."
exit 0
