# 通用游戏智能体（课题七 · 腾讯 IEG 课题）

基于 NVIDIA **NitroGen** 开源视觉-动作基础模型（500M 参数 DiT，flow-matching 行为克隆，训练于 4 万小时 / 1000+ 游戏的手柄标注）的 zero-shot 推理、单游戏数据统计与定量评估实践。

- 论文：<https://arxiv.org/abs/2601.02427>
- 官网：<https://nitrogen.minedojo.org/>
- 官方代码：<https://github.com/MineDojo/NitroGen>
- 模型权重（HF Hub）：<https://huggingface.co/nvidia/NitroGen>（含 `ng.pt` 预训练权重，CC BY-NC 4.0 非商业用途）
- 数据集（HF Hub）：<https://huggingface.co/datasets/nvidia/NitroGen>（100 个分片，约 165 GB；本课题仅下载 1 个分片）
- 本课题使用的切片：<https://huggingface.co/datasets/nvidia/NitroGen/blob/main/actions/SHARD_0034.tar.gz>（直接下载地址）

## 课题范围（MVP 摘要）

| # | 必做内容 | 状态 |
| --- | --- | --- |
| 1 | 跑通官方 ng.pt 推理，README 可复现 | ✅ 环境就绪，冒烟测试通过 |
| 2 | 同一游戏标注 ≥ 500 帧：按键/摇杆分布统计 + ≥ 10 条序列可视化 | ✅ Hades（53.8 万帧）/ lies_of_p / star_fox_64 已提取并出图 |
| 3 | 约 500 帧量级测试集（B 口径）：按键一致率、摇杆 MSE / 相关系数 | ✅ hades 493 帧 acc=0.41；lies_of_p 198 帧 acc=0.42（B 口径，纯随机抽样） |
| 4 | zero-shot 基线对比 | ✅ 已对比（按键达标 ~0.5，摇杆相关 ~0 未达 0.4，分析见项目备忘） |
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
│   ├── scan_shard.py     # 一键识别切片：89 游戏/311 视频/链接清单
│   ├── extract_game.py   # 通用标注提取（--game/--video/--limit）
│   ├── evaluate.py       # 通用 zero-shot 评估（建测试集/推理/shift 扫描/写库）
│   ├── build_hades_testset.py  # Hades 200 帧测试集构建
│   ├── stats_viz.py      # 统计可视化（按键/摇杆分布/10 条序列，中文）
│   ├── plot_shift_scan.py      # shift 扫描曲线（中文，含双摇杆）
│   ├── init_db.py        # 建 eval_results.db 结果库 + 种子数据
│   ├── probe_videos.py   # 视频链接可用性探测（yt-dlp）
│   ├── smoke_test.py     # ng.pt 端到端推理冒烟测试
│   └── start_web.ps1 / start_web.bat  # Web 平台一键启动
├── web/                  # 前端（原生 HTML/JS + ECharts，无构建）
│   ├── index.html        # 三 Tab 工作台入口
│   └── static/           # app.js / style.css / js/echarts.min.js
├── NitroGen/             # 官方代码仓库克隆（不入库，见 .gitignore）
│   ├── ng.pt             # 预训练权重 1.84 GB（不入库）
│   └── .venv/            # Python 虚拟环境（不入库）
└── data/                 # 实验数据（不入库）
    ├── games_scan.json   # 89 游戏 311 视频清单
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
NitroGen\.venv\Scripts\python.exe -m pip install -e ".[serve]"          # 官方 nitrogen 运行时
NitroGen\.venv\Scripts\python.exe -m pip install -r requirements.txt    # 顶层依赖清单（分析 + Web）
```

> **重要**：官方代码不锁定 transformers 版本，必须锁定 **transformers == 4.57.1**（5.x 有 `SiglipVisionModel.vision_model` 破坏性变更，会导致加载 ng.pt 报 `AttributeError`）：
>
> ```powershell
> NitroGen\.venv\Scripts\python.exe -m pip install "transformers==4.57.1" -i https://pypi.org/simple
> ```

> 国内网络若 `pip install` 慢，可在命令后加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（清华镜像）。

### 5. 下载预训练权重

```powershell
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','ng.pt',local_dir=r'NitroGen')"
```

首次推理会自动下载 SigLIP2-large 视觉编码器（约 3.4 GB，缓存在 `%USERPROFILE%\.cache\huggingface\hub`）。

### 6. 下载数据分片（可选，评估新游戏用）

课程约束：不得下载数据集全库（约 165 GB），本课题只下载 1 个分片：

```powershell
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','actions/SHARD_0034.tar.gz',repo_type='dataset')"
```

## 启动 / 停止

### Web 可视化平台（主演示入口）

**一键启动**（自动停旧实例 → 启动后端 → 打开浏览器）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_web.ps1
```

**分步启动**（同效）：

```powershell
NitroGen\.venv\Scripts\python.exe scripts\app.py   # 监听 http://localhost:5000
```

**停止**：

- 一键脚本：窗口按回车即关闭（脚本内已停旧进程）；
- 手动分步启动：`Ctrl+C`；若端口被占：
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*app.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`

### 推理冒烟测试

```powershell
NitroGen\.venv\Scripts\python.exe scripts\smoke_test.py
# 预期最后一行：SMOKE TEST PASSED
```

### 数据 / 评估命令行

```powershell
# 识别切片（生成 data/games_scan.json，首次或更新时用）
python scripts\scan_shard.py

# 提取指定游戏标注（hades / lies_of_p / star_fox_64 已提取，新游戏需先下载对应分片）
NitroGen\.venv\Scripts\python.exe scripts\extract_game.py --game <game>

# zero-shot 评估（自动建测试集 → 模型推理 → shift 扫描 → 写入 eval_results.db）
# 默认 500 帧纯随机抽样（B 口径）；复用旧测试集时帧数不符会自动重建
NitroGen\.venv\Scripts\python.exe scripts\evaluate.py --game <game> --video <video> --fps <fps>
# 可选：--test-size 500（帧数，默认 500） / --sample-mode random|stratified / --seq-mode（连续片段序列集 10x50 帧）

# 统计图 / shift 扫描图（matplotlib，中文）
NitroGen\.venv\Scripts\python.exe scripts\stats_viz.py --game <game>
NitroGen\.venv\Scripts\python.exe scripts\plot_shift_scan.py --game <game> --video <video>
```

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
| `读取本地分片` | 未提取游戏时出现：读取本地 SHARD 分片并提取标注（纯本地不联网，先确认预计耗时） |
| `生成静态图` | Tab② 统计图中无图时：后台生成 matplotlib PNG |
| `运行评估` | Tab③ 无评估产物时：后台跑 evaluate.py（建测试集+推理+shift 扫描+写库），自动刷新序列对比 |

> 后台任务（提取/评估/探测/下载）互斥，同一时间只运行一个；所有任务都有进度横幅 + 可停止。

### 后端接口（GET）

`/api/games`（游戏列表含已测标记）、`/api/games/<game>/videos`（视频列表含状态）、`/api/frame`（单帧识别，血缘断言 + shift）、`/api/stats`（统计分布）、`/api/sequences`（序列 + Top-5 + 差异定义）、`/api/testset`（测试集帧列表）、`/api/rescan`（探测）、`/api/evaluate`（评估）、`/api/download`（下载）、`/api/genplots`（静态图）。

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

## 首跑验证（预期输出）

1. 启动 Web 平台 → 浏览器打开 `http://localhost:5000` → 游戏下拉出现 89 个游戏，`hades`/`lies_of_p` 标"已测"。
2. 选 `lies_of_p / v2276819038` → Tab① 选帧 4963 → 手柄可视化 + 指标卡（首次推理需模型加载约 10s）。
3. 命令行验证：
   ```powershell
   curl http://localhost:5000/api/games   # -> {"ok": true, "data": {...}}
   ```
4. `stats_viz.py --game hades` 应在 `data/hades/stats/` 生成中文 PNG。

## 关键依赖版本

| 包 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.10.11 | |
| torch | 按显卡自适应（见第 3 步兼容表） | 本机实测 2.11.0+cu128（RTX 50 系） |
| transformers | **4.57.1** | 必须锁 4.x，5.x 不兼容 |
| diffusers | 0.39.0 | |
| numpy | 2.2.6 | |
| flask | 3.x | Web 平台后端 |
| polars / pandas / matplotlib / pyarrow | 最新 | 统计与评估 |

## 许可说明

- NitroGen 权重与数据集：CC BY-NC 4.0（非商业用途）
- 本仓库自有代码与文档：课程作业用途
