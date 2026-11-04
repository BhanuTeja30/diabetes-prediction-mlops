@echo off
REM Double-click to start the Diabetes Prediction app and open it in your browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" -OpenBrowser
if errorlevel 1 pause
