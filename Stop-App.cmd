@echo off
REM Double-click to stop the Diabetes Prediction app (data is kept).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1"
pause
