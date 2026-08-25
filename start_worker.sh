#!/bin/bash
# 远程 GPU Worker 启动脚本（部署到 /root/workspace/nitrogen_worker/ 根目录）
# 由 scripts/setup_worker.sh 自动生成，也可手动放置。
# 用法：
#   cd /root/workspace/nitrogen_worker && chmod +x start_worker.sh
#   setsid ./start_worker.sh > worker.log 2>&1 &
#   curl http://127.0.0.1:56272/health   # 期望返回 A100 型号 + 显存
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
exec .venv/bin/python scripts/gpu_worker.py --port 56272
