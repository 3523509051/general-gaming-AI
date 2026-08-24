@echo off
chcp 65001 >nul
title NitroGen Web Console
echo ============================================
echo   NitroGen Evaluation Workbench
echo   Starting backend, please wait...
echo ============================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1"
