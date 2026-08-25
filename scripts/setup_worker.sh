#!/usr/bin/env bash
# =============================================================================
# 远程 GPU Worker 一键部署脚本（Linux + NVIDIA GPU，如 A100）
# 用法：
#   bash scripts/setup_worker.sh [/root/workspace/nitrogen_worker] [cu121|cu130]
# 说明：
#   - 目录参数默认 /root/workspace/nitrogen_worker（服务器规范：数据须在 /root/workspace）
#   - CUDA 参数：实测 A100(Ampere) = cu130（torch 2.13.0+cu130），H100(Hopper)=cu130，
#     较老驱动可退 cu121，默认 cu130
#   - 若目录内缺 scripts/gpu_worker.py，会自动 git clone 本仓库；服务器无法访问
#     GitHub 时请手动上传（git clone 会失败并给出提示）
#   - 产物：.venv（torch+依赖）、start_worker.sh（启动脚本，监听 56272，内置 HF_HUB_OFFLINE=1）
# =============================================================================
set -euo pipefail

BASE="${1:-/root/workspace/nitrogen_worker}"
CUDA_TAG="${2:-cu130}"
REPO="https://github.com/3523509051/general-gaming-AI.git"
NITROGEN_REPO="https://github.com/MineDojo/NitroGen.git"
PYTHON_BIN="python3"

echo "==> [1/8] 系统依赖（ffmpeg 抽帧必需，实测评估用系统 /usr/bin/ffmpeg）"
apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null
ffmpeg -version | head -1

echo "==> [2/8] 目录准备: $BASE（CUDA=$CUDA_TAG）"
mkdir -p "$BASE"
cd "$BASE"

echo "==> [3/8] 获取本仓库脚本（缺 scripts/gpu_worker.py 时 clone）"
if [ ! -f scripts/gpu_worker.py ]; then
  git clone --depth 1 "$REPO" . || { echo "clone 失败（服务器可能无法访问 GitHub），请手动上传 scripts/ 后重试"; exit 1; }
else
  echo "    已存在 scripts/，跳过 clone（更新请手动覆盖）"
fi

echo "==> [4/8] 获取官方 nitrogen（缺 NitroGen/nitrogen 时 clone）"
if [ ! -d NitroGen/nitrogen ]; then
  git clone --depth 1 "$NITROGEN_REPO" NitroGen || { echo "clone 失败，请手动上传 NitroGen/ 后重试"; exit 1; }
else
  echo "    已存在 NitroGen/，跳过 clone"
fi

echo "==> [5/8] 创建 venv 并装 torch（阿里云 pytorch-wheels 源，实测 A100 用 cu130）"
"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install torch torchvision \
  --index-url "https://mirrors.aliyun.com/pytorch-wheels/$CUDA_TAG/" -q

echo "==> [6/8] 安装其余依赖（清华源）"
python -m pip install -r scripts/requirements-worker.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple -q
python -m pip install -e "NitroGen/.[serve]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo "==> [7/8] 资产校验（缺一不可）"
MISS=0
for f in NitroGen/ng.pt scripts/finetune.py scripts/evaluate.py scripts/gpu_worker.py; do
  if [ -f "$f" ]; then echo "  OK  $f"; else echo "  MISS  $f（请上传）"; MISS=1; fi
done
# HF 离线缓存（A100 访问不了 HF，模型加载必须离线缓存；实测需 siglip2 图像处理器 + Qwen 语言模型 tokenizer）
HF_CACHE="$(python - <<'PYEOF'
from huggingface_hub import constants
print(constants.HF_HUB_CACHE)
PYEOF
)"
for MOD in models--google--siglip2-large-patch16-256 models--Qwen--Qwen3-1.7B; do
  if [ -d "$HF_CACHE/$MOD" ]; then
    echo "  OK  $MOD"
  else
    echo "  MISS $MOD"
    echo "      请从本机 ~/.cache/huggingface/hub/ 复制整个 $MOD 到 $HF_CACHE/"
    MISS=1
  fi
done
[ "$MISS" = "1" ] && { echo "==> 存在缺失资产，请补齐后重启服务"; exit 1; }

echo "==> [8/8] 生成启动脚本 start_worker.sh（内置 HF_HUB_OFFLINE=1）"
cat > start_worker.sh <<EOF
#!/bin/bash
cd "$BASE"
export HF_HUB_OFFLINE=1
exec .venv/bin/python scripts/gpu_worker.py --port 56272
EOF
chmod +x start_worker.sh

echo ""
echo "======================================================================"
echo "  安装完成。启动服务："
echo "    cd $BASE && setsid ./start_worker.sh > worker.log 2>&1 &"
echo "    验证：curl http://127.0.0.1:56272/health   （应返回 GPU 型号+显存）"
echo "======================================================================"
