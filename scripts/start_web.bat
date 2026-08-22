@echo off
chcp 65001 >nul
title NitroGen 评估工作台 - 一键启动
echo ============================================
echo   NitroGen 评估工作台
echo   正在启动，请稍候（首次推理会加载模型）
echo ============================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1"
