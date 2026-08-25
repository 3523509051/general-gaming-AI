# 通用游戏智能体（课题七 · 腾讯 IEG 课题）

基于 NVIDIA **NitroGen** 开源视觉-动作基础模型（500M 参数 DiT，flow-matching 行为克隆，训练于 4 万小时 / 1000+ 游戏的手柄标注）的 zero-shot 推理、单游戏数据统计与定量评估实践。

- 论文：<https://arxiv.org/abs/2601.02427>
- 官网：<https://nitrogen.minedojo.org/>
- 官方代码：<https://github.com/MineDojo/NitroGen>
- 模型权重（HF Hub）：<https://huggingface.co/nvidia/NitroGen>（含 `ng.pt` 预训练权重，CC BY-NC 4.0 非商业用途）
- 数据集（HF Hub）：<https://huggingface.co/datasets/nvidia/NitroGen>（100 个分片，约 165 GB；本课题按需下载少量分片）
- 本课题使用的切片：<https://huggingface.co/datasets/nvidia/NitroGen/blob/main/actions/SHARD_0034.tar.gz>（直接下载地址）

> **⚠ 复现边界（先读）**
> 本仓库**不含实验数据**（`data/` 整体在 `.gitignore`，含切片/视频/标注/结果库）。
> **复现验证仅需证明代码可运行**，请按以下边界操作，**不要**擅自下载数据分片或视频：
> - **必须做**：按「环境复现步骤」装依赖 → 下载 ng.pt → 跑通冒烟测试（第 6 节）→ 启动 Web 平台 → 确认页面/接口正常返回。
> - **不必做**：下载 `SHARD_*.tar.gz` 切片、下载视频、提取标注、运行评估（这些是**可选项**，仅本课题作者按需执行，非复现验证的必要步骤）。
> - 若需查看平台三 Tab 效果而本地无数据，请以"临时假实现/跳过"处理并在差距清单中注明，**不要**为补齐数据而下载约 3 GB 的分片与视频。
> - 遇到 YouTube 下载需登录等外部依赖问题时，如实记录即可，不要绕路强行下载。

## 课题范围（MVP 摘要）

| # | 必做内容 | 状态 |
| --- | --- | --- |
| 1 | 跑通官方 ng.pt 推理，README 可复现 | ✅ 环境就绪，冒烟测试通过 |
| 2 | 同一游戏标注 ≥ 500 帧：按键/摇杆分布统计 + ≥ 10 条序列可视化 | ✅ Hades（53.8 万帧）/ lies_of_p / star_fox_64 已提取并出图 |
| 3 | 测试集（默认 200 帧，前端可调 10~5000）：按键一致率、摇杆 MSE / 相关系数 | ✅ hades / lies_of_p 已评估（B 口径，纯随机抽样，指标见指标条） |
| 4 | zero-shot 基线对比 | ✅ 已对比（按键达标 ~0.5，摇杆相关 ~0 未达 0.4，分析见项目备忘）；并实现扩展 A 小样本微调对照（本机/远程双后端，见下文） |
| 5 | 第 5 天演示：模型输出 vs 标注对比 | 🔶 前后端 Web 平台已完成，待演示走查 |
| 6 | 归档代码与指标表，实验说明写入结课大报告 | 待做 |

扩展方向：**可视化工具**——前后端 Web 平台，批量导出 ≥ 20 段动作曲线，标出差异最大的 5 帧（差异定义见项目备忘）。完整范围见 `立项书.md`（仅本地保留）。

## 目录结构

```
general-gaming-AI/
├── README.md             # 本文件（可复现说明 + 前后端工具用法）
├── 项目备忘.md            # 技术约定、实验结果、待决问题（仅本地保留，不入库）
├── 网页可视化平台实施文档.md # Web 平台设计（12 节，仅本地保留，不入库）
├── scripts/              # 本课题自有脚本（数据流水线 + 平台后端）
│   ├── app.py            # Flask 后端（Web 平台，10+ 接口）
│   ├── scan_shard.py     # 一键识别切片：游戏/视频/链接清单（--merge 合并多切片）
│   ├── extract_game.py   # 通用标注提取（--game/--video/--limit/--shard，自动发现全部切片）
│   ├── evaluate.py       # 通用 zero-shot 评估（建测试集/推理/shift 扫描/写库；--ckpt 可评估微调权重）
│   ├── finetune.py       # 单游戏小样本微调（行为克隆，输出 ng_ft_<N>.pt，供 --ckpt 评估）
│   ├── gpu_worker.py     # 远程 GPU Worker（部署到 A100，HTTP 接口跑微调+评估，CSV 回传本地）
│   ├── stats_viz.py      # 统计可视化（按键/摇杆分布/10 条序列，中文）
│   ├── plot_shift_scan.py      # shift 扫描曲线（中文，含双摇杆）
│   ├── init_db.py        # 建 eval_results.db 结果库 + 种子数据
│   ├── probe_videos.py   # 视频链接可用性探测（yt-dlp）
│   ├── install_torch.ps1 # PyTorch 显卡自适应安装（检测显卡→匹配 CUDA 索引→安装）
│   ├── smoke_test.py     # ng.pt 端到端推理冒烟测试
│   ├── start_web.ps1     # Web 平台一键启动（停旧实例→起 Flask→等端口→开浏览器）
│   └── start_web.bat     # start_web.ps1 的壳（双击可用，报错不闪退，推荐）
├── web/                  # 前端（原生 HTML/JS + ECharts，无构建）
│   ├── index.html        # 三 Tab 工作台入口
│   └── static/           # app.js / style.css / js/echarts.min.js
├── NitroGen/             # 官方代码仓库克隆（不入库，见 .gitignore）
│   ├── ng.pt             # 预训练权重 1.84 GB（不入库）
│   └── .venv/            # Python 虚拟环境（不入库）
└── data/                 # 实验数据（不入库）
    ├── games_scan.json   # 切片扫描合并清单（Web 下拉用，171 游戏）
    ├── shards/           # 切片 tar.gz + 各切片独立清单（SHARD_XXXX.games.json）
    ├── videos/           # 已下载游戏视频（mp4）
    ├── hades|lies_of_p|star_fox_64/  # 标注/测试集/评估产物
    └── eval_results.db   # SQLite 结果库
```

## 环境复现步骤

验证环境：Windows 11 + NVIDIA RTX 5060（8GB，Blackwell）。以下步骤已在 2026-08-19 实测通过。

### 1. 克隆本仓库与官方代码

```powershell
git clone https://github.com/3523509051/general-gaming-AI.git
cd general-gaming-AI
git clone https://github.com/MineDojo/NitroGen.git
```

### 2. 创建虚拟环境

```powershell
python -m venv NitroGen\.venv
NitroGen\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3. 安装 PyTorch（显卡自适应）

`scripts/install_torch.ps1`：**PyTorch 显卡自适应安装脚本**。原理：调用 `nvidia-smi` 检测本机显卡型号 → 匹配兼容表 → 输出/执行对应 CUDA 索引的 `pip install` 命令（不写死架构，任何 NVIDIA 显卡都能装）。

官方 NitroGen 推理硬编码 CUDA（`model.to("cuda")`），**必须有 NVIDIA GPU**。不同显卡架构对应不同 PyTorch + CUDA 索引版本，用自适应脚本自动匹配：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1          # 先检测，提示推荐命令
powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1 -ShowTable # 查看兼容表
powershell -ExecutionPolicy Bypass -File scripts\install_torch.ps1 -Install   # 确认后自动安装
```

**PyTorch + CUDA 索引兼容表**（脚本内置，也可手动装）：

| 显卡系列 | 架构 | CUDA 索引 | 说明 |
| --- | --- | --- | --- |
| RTX 50 系 | Blackwell (sm_120) | `cu128` | 必须 cu128（torch≥2.7） |
| RTX 40 系 | Ada Lovelace (sm_89) | `cu126` | 推荐 cu126 |
| RTX 30 系 | Ampere (sm_86) | `cu121` | 推荐 cu121 |
| RTX 20 / GTX 16 系 | Turing (sm_75) | `cu118` | 推荐 cu118 |
| GTX 10 系及更早 | Pascal (sm_61) | `cu118` | cu118 或更老 |

手动安装示例（以 RTX 40 系为例）：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`。

> **无 NVIDIA GPU 的机器**（macOS / AMD / 纯 CPU）：无法复现官方推理（`model.to("cuda")` 硬编码），`install_torch.ps1` 会明确报错；这不是换 CPU 版 PyTorch 能解决的，请使用带 NVIDIA GPU 的环境。

### 4. 安装其余依赖

```powershell
# ① 官方 nitrogen 运行时：必须先进入 NitroGen/ 目录（pyproject.toml 在其中）
cd NitroGen
NitroGen\.venv\Scripts\python.exe -m pip install -e ".[serve]"
cd ..
# ② 顶层依赖清单（分析 + Web）
NitroGen\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **重要**：官方代码不锁定 transformers 版本，必须锁定 **transformers == 4.57.1**（5.x 有 `SiglipVisionModel.vision_model` 破坏性变更，会导致加载 ng.pt 报 `AttributeError`）：
>
> ```powershell
> NitroGen\.venv\Scripts\python.exe -m pip install "transformers==4.57.1" -i https://pypi.org/simple
> ```

> 国内网络若 `pip install` 慢，可在命令后加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（清华镜像）。

### 5. 下载预训练权重

> **国内网络先设 HF 镜像**（直连 huggingface.co 慢/经常失败）：
> ```powershell
> $env:HF_ENDPOINT = "https://hf-mirror.com"   # 每个新终端执行一次
> ```

```powershell
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','ng.pt',local_dir=r'NitroGen')"
```

首次推理会自动下载 SigLIP2-large 视觉编码器（约 3.4 GB，缓存在 `%USERPROFILE%\.cache\huggingface\hub`）——**同样需要先设上面的 HF 镜像**，否则编码器下载会卡住。

### 6. 验证：跑通 ng.pt 推理（冒烟测试）

加载模型后**立即**做一次等价调用，确认安装正确：

```powershell
NitroGen\.venv\Scripts\python.exe scripts\smoke_test.py
```

预期输出（实测值，2026-08-19）：

```
[1/3] loading model from ng.pt ...
      model loaded in 11.0s
[2/3] running predict on a dummy 1280x720 frame ...
      inference done in 1.0s
[3/3] outputs:
      buttons shape: (18, 21)
      j_left  shape: (18, 2)
      j_right shape: (18, 2)
SMOKE TEST PASSED
```

`smoke_test.py` 内部即官方 `InferenceSession` 加载 ng.pt + 单帧推理（等价调用）：输入一帧 RGB 图像，输出 18 步动作块（每步 21 维按键概率 + 左/右摇杆 x、y）。**出现 `SMOKE TEST PASSED` 即安装与加载正确**，可继续下一步。

### 7. 下载数据分片（可选，评估新游戏用）

课程约束：不得下载数据集全库（约 165 GB），按需下载**少量分片**即可。每个分片约 1~2 GB，含若干游戏的手柄标注：

```powershell
# 例：下载 SHARD_0034（本课题已测的 hades / lies_of_p / star_fox_64 所在）
# 国内网络先设 HF 镜像：$env:HF_ENDPOINT = "https://hf-mirror.com"
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','actions/SHARD_0034.tar.gz',repo_type='dataset')"
# 下载会落在 HF 缓存，必须复制到 data/shards/（scan_shard/extract_game 只在该目录找切片）：
New-Item -ItemType Directory -Path data\shards -Force | Out-Null
Copy-Item "$env:USERPROFILE\.cache\huggingface\hub\datasets--nvidia--NitroGen\snapshots\*\actions\SHARD_0034.tar.gz" data\shards\ -Force
```

**导入新切片（多切片支持）**——切片放在项目 **`data/shards/`** 目录（各自独立，不分片合并）：

- **方式 A（Web 端拖拽）**：启动平台 → 顶栏「导入切片 📦」→ 把 `SHARD_*.tar.gz` **直接拖进弹窗**（或点击选择文件）→ 自动识别并复制到 `data/shards/` → 勾选切片「扫描选中切片」→ 生成独立清单 `data/shards/<SHARD>.games.json`，弹窗内**分开显示各切片内容**（游戏/视频数）→ 选游戏点「读取本地分片」。
- **方式 B（命令行）**：
  ```powershell
  # 1) 扫描切片，生成独立清单（默认写到 data/shards/<SHARD>.games.json）
  python scripts\scan_shard.py --shard data\shards\SHARD_XXXX.tar.gz
  # 2) 提取目标游戏标注（自动发现 data/shards/ 全部切片，游戏可能跨切片）
  NitroGen\.venv\Scripts\python.exe scripts\extract_game.py --game <game>
  ```
- 已验证：`data/shards/` 中 SHARD_0000（86 游戏）、SHARD_0026（89 游戏）、SHARD_0034（89 游戏）独立清单分开显示；合并清单（games_scan.json，171 游戏）供 Web 下拉用，切片独立清单各存各的。

## 启动 / 停止

### Web 可视化平台（主演示入口）

**一键启动**（自动停旧实例 → 启动后端 → 打开浏览器）：

- **方式 1（推荐，双击即可）**：直接双击 `scripts\start_web.bat`（它是 start_web.ps1 的壳，报错不会闪退，可查看错误后按回车关闭）
- **方式 2（命令行）**：
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\start_web.ps1
  ```

> `start_web.ps1` 内部完成：停掉旧 app.py → 清日志 → 后台启动 Flask → 等端口 5000 就绪（最多 15s）→ 自动打开浏览器。日志在 `data\app.log` / `data\app.err.log`。

**直接运行**（不用一键脚本，适合已在调试场景）：

```powershell
NitroGen\.venv\Scripts\python.exe scripts\app.py   # 监听 http://localhost:5000
```

> 与一键启动的区别：直接运行**不会**自动停旧实例 / 自动打开浏览器；停止需 `Ctrl+C`。一般情况用上面的一键启动即可。

**停止**：

- **一键启动（bat/ps1）**：窗口按回车即关闭（脚本内已停旧进程）；
- **直接运行（app.py）**：`Ctrl+C`；若端口被占，强制停：
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*app.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`

### 数据 / 评估命令行

> **评估/微调抽帧依赖 ffmpeg**：本机由 `requirements.txt` 的 `imageio-ffmpeg` 提供（首次自动下载 ffmpeg 二进制，走 GitHub releases；国内网络若失败，请安装系统 ffmpeg 并加入 PATH，`evaluate.py` 会自动探测到）。

```powershell
# 识别切片（生成 data/games_scan.json；默认自动定位 data/shards/ 中的切片）
python scripts\scan_shard.py
python scripts\scan_shard.py --shard data\shards\SHARD_XXXX.tar.gz --merge   # 扫描新切片并合并（多切片导入）

# 提取指定游戏标注（自动发现 data/shards/ 全部 SHARD_*.tar.gz，游戏可能跨切片）
# hades / lies_of_p / star_fox_64 已提取；指定切片用 --shard <路径>
NitroGen\.venv\Scripts\python.exe scripts\extract_game.py --game <game>

# zero-shot 评估（自动建测试集 → 模型推理 → shift 扫描 → 写入 eval_results.db）
# 默认 200 帧纯随机抽样（B 口径）；复用旧测试集时帧数不符会自动重建
NitroGen\.venv\Scripts\python.exe scripts\evaluate.py --game <game> --video <video> --fps <fps>
# 可选：--test-size 200（帧数，默认 200，Web 端指标条可调） / --sample-mode random|stratified / --seq-mode（连续片段序列集）
#       --ckpt <权重>  评估微调权重（默认 ng.pt；如 NitroGen/ng_ft_1000.pt）
#       --tag <标签>   额外写 metrics/predictions 带标签副本（供零样本 vs 微调对照）
#       --no-plots     跳过统计图生成（远程 worker 评估用）

# 统计图 / shift 扫描图（matplotlib，中文）
NitroGen\.venv\Scripts\python.exe scripts\stats_viz.py --game <game>
NitroGen\.venv\Scripts\python.exe scripts\plot_shift_scan.py --game <game> --video <video>
```

## 扩展 A：小样本微调 + 零样本对照（可选，需 NVIDIA GPU）

**做什么**：在官方 ng.pt 基础上，用该游戏若干标注帧做一轮行为克隆微调（flow-matching loss，视觉编码器冻结），再用 `evaluate.py --ckpt` 评估微调权重，与零样本基线逐指标对照（按键一致率 B/A 口径、摇杆相关、MSE），回答"微调是否有效"。

### 两种执行方式（Web 面板二选一）

| | 方案一 · 本机 GPU | 方案二 · 远程 GPU（A100 等） |
| --- | --- | --- |
| 适用场景 | 8GB 以上独显、小样本（≤5k 帧） | 服务器大显存、大样本 / 全帧微调 |
| 训练速度 | 较慢（8GB 本机自动 batch=2~4） | 快（80GB 自动 batch=8） |
| 部署成本 | 零（本机即用） | 首次一次性部署约 30~60 分钟 |
| 数据 | 直接用本地 `data/` | 缺数据时本地自动上传（约 10~20 分钟） |
| 权重 | 直接存本地 | 留远程，Δ>0 时按需下载 |
| 断点恢复 | 关页面即中断（任务在本地） | 本地关闭不影响，重开自动接管 |

### 方案一：本机 GPU（开箱即用）

1. 打开 Web（http://localhost:5000）→ Tab③「扩展 A · 小样本微调对照闭环」面板；
2. 顶部后端选 **本机 GPU**；
3. 填样本帧数（默认 1000；填超过总标注帧数即用全量）/ epochs（默认 1）/ batch（留空自动按显存，8GB→2~4）；
4. 点「启动微调并评估」→ 自动执行：补零样本基线副本（首次）→ 本机微调 → 本机评估；
5. 完成后对照表自动出现；「运行日志」终端窗口实时显示训练 step/loss。

命令行等价：

```powershell
# 微调（输出 NitroGen/ng_ft_<N>.pt；--samples 超过总标注帧数即全量）
NitroGen\.venv\Scripts\python.exe scripts\finetune.py --game <game> --video <video> --fps <fps> --samples 500 --epochs 1 --batch 4
# 评估微调权重并与零样本基线对照（--ckpt + --tag 副本）
NitroGen\.venv\Scripts\python.exe scripts\evaluate.py --game <game> --video <video> --fps <fps> --ckpt NitroGen/ng_ft_500.pt --tag ng_ft_500
```

### 方案二：远程 GPU（A100 服务器）

**原理**：`scripts/gpu_worker.py` 是微调专用远程服务（HTTP 接口，监听内网端口 56272），只做"微调 + 评估"，产物权重留远程、评估 CSV 回传本地（避免 2GB 权重来回传卡链路）。

**一次性部署（A100 上，约 30~60 分钟，已提供自动化脚本）**：
1. **拉仓库 + 自动安装**（`scripts/setup_worker.sh` 一步完成：建 venv、装 torch（阿里云源，cu121/cu130 可选）、装 `scripts/requirements-worker.txt` 依赖、生成 `start_worker.sh`）：
   ```bash
   git clone https://github.com/3523509051/general-gaming-AI.git /root/workspace/nitrogen_worker
   cd /root/workspace/nitrogen_worker
   bash scripts/setup_worker.sh /root/workspace/nitrogen_worker cu121   # A100=cu121，H100=cu130
   ```
2. **补齐仓库外资产**（setup 脚本会校验缺失并提示）：`NitroGen/ng.pt`（1.97GB，本机复制）→ `NitroGen/`（官方包，setup 自动 clone，**服务器访问不了 GitHub 时手动上传 `NitroGen/` 与 `scripts/` 即可，setup 会跳过 clone**）→ 目标游戏视频 mp4 与 `data/<game>/` 标注（本地自动上传或手动）→ **HF 离线缓存**（A100 访问不了 HuggingFace，需把本机 `~/.cache/huggingface/hub/` 下的 `models--google--siglip2-large-patch16-256` 与 `models--Qwen--Qwen3-1.7B` 两个目录整个复制到服务器同路径；实测模型加载需要这两个缓存）。

   **本机 → 服务器传输示例**（本地 PowerShell，`<服务器IP>` 换成实际地址）：
   ```powershell
   # ng.pt 权重
   scp -P 56271 NitroGen\ng.pt root@<服务器IP>:/root/workspace/nitrogen_worker/NitroGen/
   # HF 离线缓存（两个目录各 1~3 GB，传一次即可）
   scp -r -P 56271 "$env:USERPROFILE\.cache\huggingface\hub\models--google--siglip2-large-patch16-256" root@<服务器IP>:/root/.cache/huggingface/hub/
   scp -r -P 56271 "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-1.7B" root@<服务器IP>:/root/.cache/huggingface/hub/
   ```
3. 启动服务并验证：
   ```bash
   cd /root/workspace/nitrogen_worker && chmod +x start_worker.sh
   setsid ./start_worker.sh > worker.log 2>&1 &
   curl http://127.0.0.1:56272/health   # 期望返回 A100 型号 + 显存
   ```

**本地配置（一次性，30 秒）**：Tab③ 面板顶部后端选 **远程服务器** → 填 `http://<服务器IP>:56272` + SSH 密码（缺数据自动上传用，凭据存 `data/remote_worker.json`，本地文件不入库）→ 点「连接」，提示远程 GPU 在线。

**日常使用**：
1. 选游戏/视频（自动恢复上次选择）、填样本帧数/epochs/batch（留空自动按 A100 显存=8）；
2. 点「启动微调并评估」→ 缺数据时自动上传 → 远程微调 → 远程评估 → CSV 回传本地；
3. 终端窗口实时看进度（3s 刷新）；完成后自动渲染对照表（best shift / acc_B 主口径 / acc_A / corr / mse / Δ）；
4. Δ>0（微调有效）时出现「⬇ 下载该优质权重」按钮（后台异步拉取 2GB，不阻塞）；
5. 中途关本地 → 远程任务照跑 → 重开 Web 打开页面自动接管并补拉结果。

**远程注意事项**：数据须放 `/root/workspace`（服务器规范）；端口只用 56272/56273、不暴露公网；换新服务器重走部署步骤即可。

> 指标口径提醒：论文报告微调 +10%~52% 相对提升是**任务完成率**口径（Gymnasium 模拟器），本表是**离线按键一致率 / 摇杆相关**口径，两者不等价，微调是否提升按键一致率以本表实测为准。



## 前后端工具：Web 可视化平台（扩展 C）

启动后浏览器打开 `http://localhost:5000`，顶部选择"游戏 + 视频"，即锁定数据血缘 `(game, video, absolute_frame)`（杜绝拿 B 视频比对 A 数据）。

### 三个 Tab

| Tab | 功能 |
| --- | --- |
| ① 模型识别 | 帧滑条选帧 → 模型实时推理（约 0.3s/帧）→ SVG 手柄 17 键可视化 + 双摇杆轨迹（18 步动作块）；绿=命中/蓝=漏报/橙=误报 |
| ② 统计分布 | 全量标注帧实时计算：按键触发率柱状图、左右摇杆位置分布（同等地位）、统计摘要、matplotlib 静态图（可一键生成） |
| ③ 序列对比 | 综合差异分 D 曲线 + 分段手柄动作曲线（每段 10 帧，测试集帧数/10 段）+ 差异最大 Top-5 帧表（含左右摇杆 L2、差异明细，可跳转 Tab① 详情） |

### 顶栏按钮

| 按钮 | 作用 |
| --- | --- |
| `⬇ 下载视频` | 后台下载当前视频到 `data/videos/`，进度条实时显示 |
| `探测链接 ↻` | 探测当前游戏视频链接可用性（未选游戏时全量探测，需确认） |
| `导入切片 📦` | 把 `SHARD_*.tar.gz` 拖进弹窗 → 自动复制到 `data/shards/` → 扫描生成独立清单（多切片导入） |
| `读取本地分片` | 未提取游戏时出现：读取本地 SHARD 分片并提取标注（纯本地不联网，先确认预计耗时） |
| `生成静态图` | Tab② 统计图中无图时：后台生成 matplotlib PNG |
| `运行评估` | Tab③ 无评估产物时：后台跑 evaluate.py（建测试集+推理+shift 扫描+写库），自动刷新序列对比 |

> 游戏下拉框按切片分组显示（optgroup）；指标条带「测试集帧数 + 重新评估」入口（默认 200，可调 10~5000）。

> 后台任务（提取/评估/探测/下载）互斥，同一时间只运行一个；所有任务都有进度横幅 + 可停止。

### 后端接口

除 `/api/games`、`/api/shaders` 外，其余接口均需查询参数 `game=<游戏>&video=<视频>`；`/api/frame` 另需 `frame=<绝对帧号>`；`/api/metrics` 可选 `shift=k`（不传则用最优 shift）。

| 接口 | 必填参数 | 说明 |
| --- | --- | --- |
| `/api/games` | 无 | 游戏列表（含已测标记、按切片分组） |
| `/api/games/<game>/videos` | 路径参数 game | 视频列表（含下载/探测状态） |
| `/api/frame` | game, video, frame | 单帧识别：血缘断言 + shift + 18 步动作块 |
| `/api/stats` | game, video | 统计分布（按键/摇杆，全量标注实时算） |
| `/api/sequences` | game, video | 序列 + Top-5 + 差异定义 |
| `/api/testset` | game, video | 测试集帧列表 |
| `/api/metrics` | game, video（shift 可选） | 核心指标对比条 |
| `/api/rescan` | game（可选，不传全量） | 探测视频链接 |
| `/api/evaluate` | game, video（test_size 可选） | 触发评估 |
| `/api/download` | game, video | 触发视频下载 |
| `/api/genplots` | game, video | 生成静态图 |
| `/api/shaders` | 无 | 本地切片列表 |
| `/api/scan_shard` | shard | 扫描合并切片 |
| `/api/upload_shard` | 文件 | 拖拽上传切片 |
| `/api/finetune` | game, video, samples, epochs, batch | 启动微调+评估链路（按后端分发：本机/远程） |
| `/api/finetune/backend` | mode, url, ssh_* | 设置/查询微调后端（local/remote + SSH 凭据，凭据存本地不入库） |
| `/api/finetune/status` | 无 | 微调链路状态 + 日志尾部 |
| `/api/finetune/recover` | 无 | 本地重启后接管远程运行中任务/补拉已完成结果 |
| `/api/finetune/logtail` | out, lines, game, video | 微调/评估日志尾部（前端终端窗口） |
| `/api/finetune/compare` | game, video, samples | 零样本基线 vs 微调对照（两行指标 + Δ） |
| `/api/finetune/pull_weight` | game, video, samples | 后台异步下载远程微调权重（2GB，不阻塞） |
| `/api/prefs` | game, video, samples, epochs, batch | 保存/恢复上次选择的游戏视频与微调参数 |

## 必要环境变量

| 变量 | 作用 | 说明 |
| --- | --- | --- |
| `HF_HUB_OFFLINE=1` | 跳过 HF Hub 联网检查 | 权重/编码器已缓存时秒级加载（评估脚本已内置） |
| `HF_ENDPOINT` | HF 镜像 | 网络受限时指向镜像站 |
| `HTTPS_PROXY` | 视频探测/下载代理 | `probe_videos.py`/`download` 访问 Twitch/YouTube 需要时设置 |

### YouTube 下载（bot 验证规避）

部分 IP 段访问 YouTube 会触发 "Sign in to confirm you're not a bot"，此时下载需要登录态 cookies：

1. 在浏览器（Edge/Chrome）登录 YouTube，安装 **Get cookies.txt LOCALLY** 扩展；
2. 打开目标视频页 → 点扩展 → **Export**，得到 `cookies.txt`；
3. 放入 `data/cookies.txt`（不入库，`.gitignore` 已排除）——下载流程检测到即自动加 `--cookies`；
4. 重新点"⬇ 下载视频"即可。

探测侧（`probe_videos.py`）已内置 **oEmbed 二次确认**：yt-dlp 被 bot 验证拦下时，自动用 YouTube oEmbed 接口（无需登录）确认真实性，避免把有效链接误标为"未知/失效"。

### Twitch 下载（长视频限流与规避）

本课题切片中大量视频源为 **Twitch**（`source: twitch`），实测发现：

**现象**：Twitch VOD 下载到**约 44%（≈50 分钟视频量 / ~800MB）后断流或极速限流**（从几 MB/s 降到几百 KB/s），yt-dlp 重试无效，退出码 1。位置固定（同一视频每次都卡在同一进度），已排除分片损坏（该处片段单独下载正常）。

**原因**：Twitch 对**非订阅/匿名下载**的长 VOD 有累计时长限制（约 50 分钟视频量后触发），属平台侧限制，非 yt-dlp 或本仓库问题。

**有效解法（已实测验证）：cookie 登录态**

Twitch 对**已登录账户**的长 VOD 下载**无此限流**（实测：112 分钟视频用登录态下载 50x 速度完整通过；匿名则固定卡在约 44%）。获取方式（浏览器插件导出）：

1. Edge/Chrome 打开 `https://www.twitch.tv` 并**登录**；
2. 安装浏览器扩展 **Cookie-Editor**（或 Get cookies.txt LOCALLY），点扩展图标 → **Export** → 格式选 **Netscape**（cookies.txt）；
3. 把导出内容粘贴保存为 `data/cookies.txt`（UTF-8 编码；`data/` 已整体在 `.gitignore`，登录凭据不会入库）；
4. 下载流程检测到该文件即自动加 `--cookies`（无需改代码/重启），重新点「⬇ 下载视频」即可突破限流；
5. 登录态会**过期**（几天~几周），过期后重新导出覆盖即可。

**已否决的路径**：
1. `--fragment-retries 10 / --retries 10`：能减少偶发断流，但**无法突破累计时长限制**（重试仍被掐）；
2. `--cookies-from-browser edge`：新版 Chromium（≥127）的 cookie 采用 **app-bound encryption**，yt-dlp 报 `Could not copy Chrome cookie database`（[issue #7271](https://github.com/yt-dlp/yt-dlp/issues/7271)），关闭浏览器也无法读取；
3. **分段下载**（`--download-sections` 切 <40 分钟段再 concat）：切出的片段时间戳/分片边界与**原视频标注的绝对帧无法对齐**，抽帧会错位，已否决。

## 首跑验证（预期输出）

1. 启动 Web 平台 → 浏览器打开 `http://localhost:5000` → 游戏下拉按切片分组显示，`hades`/`lies_of_p` 标"已测"。（游戏数取决于已导入切片：仅 SHARD_0034 时 89 个，导入 SHARD_0000/0026/0034 三个分片时为 171 个）
2. 选 `lies_of_p / v2276819038` → Tab① 选帧 4963 → 手柄可视化 + 指标卡（首次推理需模型加载约 10s）。
3. 命令行验证：
   ```powershell
   curl http://localhost:5000/api/games   # -> {"ok": true, "data": {...}}
   ```
4. `stats_viz.py --game hades` 应在 `data/hades/stats/` 生成中文 PNG。

## 关键依赖版本

| 包 | 版本 | 备注 |
| --- | --- | --- |
| Python | ≥ 3.12（本机实测 3.13.5） | 官方 NitroGen 要求 ≥ 3.12 |
| torch | 按显卡自适应（见第 3 步兼容表） | 本机实测 2.11.0+cu128（RTX 50 系） |
| transformers | **4.57.1** | 必须锁 4.x，5.x 不兼容 |
| diffusers | 0.39.0 | |
| numpy | 2.2.6 | |
| flask | 3.x | Web 平台后端 |
| polars / pandas / matplotlib / pyarrow | 最新 | 统计与评估 |

## 许可说明

- NitroGen 权重与数据集：CC BY-NC 4.0（非商业用途）
- 本仓库自有代码与文档：课程作业用途
