# 通用 PyTorch 安装脚本（显卡自适应）
# 自动检测本机 NVIDIA 显卡架构，选择对应的 PyTorch + CUDA 索引版本。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1          # 检测并提示，不自动装
#   powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1 -Install  # 检测后自动安装
#   powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1 -ShowTable # 只打印兼容表
#
# 说明：
# - 官方 NitroGen 推理代码硬编码 CUDA（model.to("cuda")），无 NVIDIA GPU 的机器无法直接复现，
#   脚本会明确报错并提示原因（这不是"换 CPU torch"能解决的）。
# - RTX 50 系（Blackwell, sm_120）必须 cu128；RTX 40 系（Ada）建议 cu126/cu128；
#   RTX 30 系及更早建议 cu121/cu118。下表给出推荐。

param(
    [switch]$Install,       # 检测后直接执行 pip 安装
    [switch]$ShowTable,     # 只打印兼容表不检测
    [string]$Python = ""    # 指定 python（默认取 NitroGen\.venv\Scripts\python.exe，缺则用 python）
)

$ErrorActionPreference = "Stop"

# ---- 显卡架构 -> PyTorch CUDA 索引版本 映射表 ----
# 依据：PyTorch 官方 wheel 索引（download.pytorch.org/whl/<cuda>），
# Blackwell 需 cu128（torch>=2.7）；Ada/Ampere 兼容 cu121/cu126；老架构 cu118。
$TorchTable = @(
    @{ Pattern = "RTX 50"; Cuda = "cu128";  Note = "RTX 50 系 (Blackwell, sm_120)，必须 cu128" }
    @{ Pattern = "RTX 40"; Cuda = "cu126";  Note = "RTX 40 系 (Ada Lovelace, sm_89)，推荐 cu126" }
    @{ Pattern = "RTX 30"; Cuda = "cu121";  Note = "RTX 30 系 (Ampere, sm_86)，推荐 cu121" }
    @{ Pattern = "RTX 20"; Cuda = "cu118";  Note = "RTX 20 系 (Turing, sm_75)，推荐 cu118" }
    @{ Pattern = "GTX 16"; Cuda = "cu118";  Note = "GTX 16 系 (Turing, sm_75)，推荐 cu118" }
    @{ Pattern = "GTX 10"; Cuda = "cu118";  Note = "GTX 10 系 (Pascal, sm_61)，cu118 或更老" }
)

if ($ShowTable) {
    Write-Host ""
    Write-Host "=== PyTorch + CUDA 索引 兼容表 ===" -ForegroundColor Cyan
    Write-Host ("{0,-14}{1,-10}{2}" -f "显卡", "CUDA索引", "说明")
    foreach ($row in $TorchTable) {
        Write-Host ("{0,-14}{1,-10}{2}" -f $row.Pattern, $row.Cuda, $row.Note)
    }
    Write-Host ""
    Write-Host "安装示例（RTX 40 系）："
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
    exit 0
}

# ---- 1. 定位 python ----
if (-not $Python) {
    $venvPy = Join-Path $PSScriptRoot "..\NitroGen\.venv\Scripts\python.exe"
    if (Test-Path $venvPy) { $Python = $venvPy } else { $Python = "python" }
}

# ---- 2. 检测 NVIDIA 显卡 ----
$gpuName = $null
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
} catch { }
if (-not $gpuName) {
    Write-Host ""
    Write-Host "!! 未检测到 NVIDIA 显卡（或 nvidia-smi 不可用）。" -ForegroundColor Red
    Write-Host "   官方 NitroGen 推理硬编码 CUDA（model.to(\"cuda\")），无 NVIDIA GPU 的机器无法直接复现。"
    Write-Host "   这不是换 CPU 版 PyTorch 能解决的；如需运行请使用带 NVIDIA GPU 的机器。"
    exit 1
}
Write-Host ""
Write-Host ("检测到显卡: {0}" -f $gpuName) -ForegroundColor Green

# ---- 3. 匹配架构 ----
$matched = $null
foreach ($row in $TorchTable) {
    if ($gpuName -match $row.Pattern) { $matched = $row; break }
}
if (-not $matched) {
    # 未知/专业卡：提示手动选择
    Write-Host "显卡型号未在兼容表中（$gpuName），请参照下表手动选择 CUDA 索引："
    $ShowTable = $true
    & $PSScriptRoot\install_torch.ps1 -ShowTable
    exit 1
}

$cmd = "$Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$($matched.Cuda)"
Write-Host ""
Write-Host ("推荐安装命令（{0}）：" -f $matched.Note) -ForegroundColor Cyan
Write-Host "  $cmd"

if ($Install) {
    Write-Host ""
    Write-Host "正在安装...（大文件，请耐心等待）" -ForegroundColor Yellow
    Invoke-Expression $cmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host ("安装完成。验证：{0} -c `"import torch; print(torch.__version__, torch.cuda.is_available())`"" -f $Python) -ForegroundColor Green
    } else {
        Write-Host "安装失败（退出码 $LASTEXITCODE）。" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "（未加 -Install 参数，仅检测提示。确认无误后加 -Install 自动安装。）"
}
