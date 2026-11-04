<#
.SYNOPSIS
    Remove the "Diabetes" login trigger created by install-autostart.ps1.
    The app itself is not stopped; use Stop-App.cmd for that.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1
#>
$ShortcutPath = Join-Path ([Environment]::GetFolderPath("Startup")) "Diabetes.lnk"
if (Test-Path $ShortcutPath) {
    Remove-Item -LiteralPath $ShortcutPath
    Write-Host "Removed login trigger 'Diabetes' ($ShortcutPath)."
} else {
    Write-Host "Login trigger 'Diabetes' is not installed."
}
