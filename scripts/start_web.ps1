# NitroGen 评估工作台 一键启动
# 推荐双击 scripts\start_web.bat（此 ps1 由 bat 调用，报错不会闪退）
# 也可手动: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_web.ps1

$ErrorActionPreference = "Stop"
# PS 7.3+ 默认会在原生命令（taskkill 等）返回非零退出码时抛异常，
# 导致 finally 里 taskkill "进程不存在"时输出红字并中断清理。置 false 消除。
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "NitroGen\.venv\Scripts\python.exe"
$dataDir = Join-Path $root "data"
$logOut = Join-Path $dataDir "app.log"
$logErr = Join-Path $dataDir "app.err.log"
$startedThisRun = $false   # 本次脚本是否实际启动了 Flask（决定 finally 是否清理）

function Test-Port($port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $port); return $true }
    catch { return $false }
    finally { $c.Dispose() }
}

try {
    if (-not (Test-Path $py)) { throw "venv 不存在: $py（请确认已按 README 安装 NitroGen 依赖）" }

    # 0) 检测端口 5000 是否已有服务在跑：有则提示，随后按"先停旧、再启新"重启后端
    if (Test-Port 5000) {
        $portPid = (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty OwningProcess)
        Write-Host "检测到已有服务在运行 (PID=$portPid)，将停止并重启后端..." -ForegroundColor Yellow
    }

    # 1) 停掉旧的 app.py 进程（无论 venv 还是系统 Python，命令行含 app.py 的都停）
    Write-Host "1/4 停止旧实例..." -ForegroundColor Cyan
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*app.py*" } |
        ForEach-Object {
            Write-Host "   已停止 PID=$($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Start-Sleep -Seconds 1
    # 1.5) 兜底：若端口 5000 仍被占用（残留进程），强制释放
    $portPid = (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess)
    if ($portPid) {
        Write-Host "   端口 5000 被 PID=$portPid 占用，强制释放..."
        Stop-Process -Id $portPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    # 2) 清理旧日志（防止文件被旧进程占用导致重定向失败）
    Write-Host "2/4 清理日志..." -ForegroundColor Cyan
    Remove-Item $logOut, $logErr -Force -ErrorAction SilentlyContinue

    # 3) 后台启动 Flask（隐藏窗口）；关闭本窗口时由 finally 停止服务
    Write-Host "3/4 启动 Flask 后端（首次访问 /api/frame 时加载模型）..." -ForegroundColor Cyan
    $flaskProc = Start-Process -FilePath $py -ArgumentList "scripts\app.py" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
    $startedThisRun = $true

    # 4) 等待端口 5000 就绪（最多 15s）
    Write-Host "4/4 等待端口 5000 就绪..." -ForegroundColor Cyan
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port 5000) { $ready = $true; break }
        if ($flaskProc.HasExited) { break }
    }
    if (-not $ready) {
        $tail = ""
        if (Test-Path $logErr) { $tail = (Get-Content $logErr -Tail 15 -ErrorAction SilentlyContinue) -join "`n" }
        throw "服务未就绪。`n--- app.err.log 末尾 ---`n$tail"
    }
    Write-Host ""
    Write-Host "启动成功: http://localhost:5000" -ForegroundColor Green
    Write-Host "日志: data\app.log / data\app.err.log"
    Start-Process "http://localhost:5000"
    Write-Host ""
    Write-Host "服务正在后台运行。按回车键停止服务并关闭窗口（直接关窗口也会停止）。" -ForegroundColor DarkGray
    Read-Host | Out-Null
}
catch {
    Write-Host ""
    Write-Host "启动失败: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "（可查看 data\app.err.log 定位具体原因）"
    Write-Host ""
    Write-Host "按回车键关闭此窗口..." -ForegroundColor Cyan
    Read-Host | Out-Null
}
finally {
    # 仅当本次脚本实际启动了 Flask 才清理（避免误杀预先存在的服务实例）
    if ($startedThisRun) {
        # 停止本脚本启动的 Flask 及其全部子进程（taskkill /T 连带杀子进程）
        if ($flaskProc) {
            try { taskkill /PID $flaskProc.Id /T /F 2>$null | Out-Null } catch { }
        }
        # 释放端口 5000（兜底，防止残留进程占用）
        try {
            $portPid = (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty OwningProcess)
            if ($portPid) {
                Stop-Process -Id $portPid -Force -ErrorAction SilentlyContinue
            }
        } catch { }
    }
}
