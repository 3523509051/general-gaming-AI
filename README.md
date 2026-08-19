# 通用游戏智能体（课题七 · 腾讯 IEG 课题）

基于 NVIDIA **NitroGen** 开源视觉-动作基础模型（500M 参数 DiT，flow-matching 行为克隆，训练于 4 万小时 / 1000+ 游戏的手柄标注数据）的 zero-shot 推理、单游戏数据统计与定量评估实践。

- 论文：<https://arxiv.org/abs/2601.02427>
- 官网：<https://nitrogen.minedojo.org/>
- 官方代码：<https://github.com/MineDojo/NitroGen>
- 数据集：<https://huggingface.co/datasets/nvidia/NitroGen>（本课题仅下载单个分片，不下全库）
- 预训练权重：`ng.pt`（CC BY-NC 4.0，仅限非商业研究用途）

## 课题范围（MVP 摘要）

| # | 必做内容 | 状态 |
| --- | --- | --- |
| 1 | 跑通官方 ng.pt 推理，README 可复现 | ✅ 环境就绪，冒烟测试通过 |
| 2 | 单游戏（Hollow Knight）≥ 500 帧标注：按键/摇杆分布统计 + ≥ 10 条序列可视化 | 进行中 |
| 3 | ≥ 200 帧测试集：按键准确率、摇杆 MSE / 相关系数 | 待做 |
| 4 | zero-shot 基线：按键准确率 ≥ 50%，摇杆相关系数 ≥ 0.4 | 待做 |
| 5 | 第 5 天演示：模型输出 vs 标注对比 | 待做 |
| 6 | 归档代码与指标表，实验说明写入结课大报告 | 待做 |

扩展方向：**可视化工具**（批量导出 ≥ 20 段动作曲线，标出差异最大的 5 帧）。
完整范围、进度计划与验收标准见 [`立项书.md`](./立项书.md)。

## 目录结构

```
general-gaming-AI/
├── README.md            # 本文件（可复现说明）
├── 立项书.md             # 第 1 天立项书
├── scripts/             # 本课题自有脚本
│   └── smoke_test.py    # ng.pt 端到端推理冒烟测试
├── NitroGen/            # 官方代码仓库克隆（不入库，见 .gitignore）
│   ├── ng.pt            # 预训练权重 1.84 GB（不入库）
│   └── .venv/           # Python 虚拟环境（不入库）
└── data/                # 提取后的标注数据（不入库）
```

## 环境复现步骤

硬件：Windows 11 + NVIDIA RTX 5060（8GB，Blackwell）。以下步骤已在 2026-08-19 实测通过。

### 1. 克隆本仓库与官方代码

```powershell
git clone https://github.com/3523509051/general-gaming-AI.git
cd general-gaming-AI
git clone https://github.com/MineDojo/NitroGen.git
```

### 2. 创建虚拟环境并安装依赖

RTX 50 系列（Blackwell，sm_120）需要 PyTorch ≥ 2.7 + CUDA 12.8：

```powershell
python -m venv NitroGen\.venv
NitroGen\.venv\Scripts\python.exe -m pip install --upgrade pip
NitroGen\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
NitroGen\.venv\Scripts\python.exe -m pip install -e ".[serve]"
```

> **重要**：官方代码不锁定 transformers 版本，transformers 5.x 存在 `SiglipVisionModel.vision_model` 破坏性变更，会导致加载 ng.pt 报 `AttributeError`。必须锁定 **transformers == 4.57.1**（若安装时被解析到 5.x，需手动降级）：
>
> ```powershell
> NitroGen\.venv\Scripts\python.exe -m pip install "transformers==4.57.1" -i https://pypi.org/simple
> ```

### 3. 下载预训练权重

```powershell
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','ng.pt',local_dir=r'NitroGen')"
```

首次推理时会自动从 HF Hub 下载 SigLIP2-large 视觉编码器（约 3.4 GB，缓存在 `%USERPROFILE%\.cache\huggingface\hub`）。

### 4. 运行推理冒烟测试（可复现验证）

```powershell
NitroGen\.venv\Scripts\python.exe scripts\smoke_test.py
```

预期输出（实测值）：

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

即：输入一帧 RGB 图像，模型输出 18 步动作块（每步 21 维按键概率 + 左/右摇杆 x、y）。

### 5. 官方推理服务（可选，接 Windows 游戏实时控制用）

```powershell
NitroGen\.venv\Scripts\python.exe NitroGen\scripts\serve.py NitroGen\ng.pt
# 另开终端：
NitroGen\.venv\Scripts\python.exe NitroGen\scripts\play.py --process '<游戏进程名>.exe'
```

本课题评估离线进行，不依赖此步骤。

## 数据获取（仅单分片，不下全库）

课程约束：**不得下载数据集全库**（100 个分片共约 165 GB）。本课题只下载 1 个分片：

```powershell
NitroGen\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/NitroGen','actions/SHARD_0034.tar.gz',repo_type='dataset')"
```

从解压后的分片中按 `metadata.json` 的 `game == "hollow_knight"` 过滤提取标注（每个 chunk 含 `actions_processed.parquet` / `actions_raw.parquet` / `metadata.json`）。

选定游戏统计（实测，2026-08-19）：Hollow Knight 共 1,444,200 帧 / 1439 chunks / 6 个视频；抽样实测按键触发率 26.0%、左摇杆移动率 39.7%、IDLE 帧占比 49.2%。

## 关键依赖版本

| 包 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.10.11 | |
| torch | 2.11.0+cu128 | RTX 5060 需要 cu128 |
| transformers | **4.57.1** | 必须锁 4.x，5.x 不兼容 |
| diffusers | 0.39.0 | |
| numpy | 2.2.6 | |

## 许可说明

- NitroGen 权重与数据集：CC BY-NC 4.0（非商业用途）
- 本仓库自有代码与文档：课程作业用途
