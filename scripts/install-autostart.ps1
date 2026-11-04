<#
.SYNOPSIS
    Start the Diabetes Prediction app automatically every time you log in to Windows.

.DESCRIPTION
    Creates a shortcut named "Diabetes" in your personal Startup folder
    (shell:startup). At login it runs scripts\start.ps1 in a hidden window, which starts
    Docker Desktop if needed and brings up the whole stack. No administrator rights needed.
    Remove it with scripts\uninstall-autostart.ps1, or disable it in
    Task Manager -> Startup apps -> "Diabetes".

.PARAMETER OpenBrowser
    Also open the app in the browser after login once it is ready.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1
#>
param([switch]$OpenBrowser)

$ErrorActionPreference = "Stop"
$ShortcutName = "Diabetes"
$StartScript = Join-Path $PSScriptRoot "start.ps1"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "$ShortcutName.lnk"

if (-not (Test-Path $StartScript)) {
    throw "start.ps1 not found next to this script ($StartScript)."
}

$arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`""
if ($OpenBrowser) { $arguments += " -OpenBrowser" }

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $ProjectDir
$shortcut.WindowStyle = 7  # minimized
$shortcut.Description = "Start the Diabetes Prediction MLOps app at login"
$shortcut.Save()

Write-Host "Installed login trigger '$ShortcutName': $ShortcutPath"
Write-Host "It runs: powershell $arguments"
Write-Host "Log file: $(Join-Path $ProjectDir 'logs\autostart.log')"
