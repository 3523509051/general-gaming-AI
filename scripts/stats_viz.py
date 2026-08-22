# -*- coding: utf-8 -*-
"""统计 Hades 标注的按键/摇杆分布，并可视化 10 条连续动作序列曲线。

对应指导书"同一游戏标注约 500 帧量级：统计按键/摇杆分布，可视化约 10 条序列"。

用法（仓库根目录）：
    python scripts\\stats_viz.py            # 默认统计 Hades，输出 data/hades/stats/
    python scripts\\stats_viz.py --game hollow_knight

输出（data/<game>/stats/）：
    1. button_press_dist.png      17 键触发频率分布柱状图
    2. joystick_dist.png          左/右摇杆取值分布（2D 直方图 + 边缘分布）
    3. sequences.png              10 条连续动作序列曲线（按键时序 + 摇杆轨迹，两图并排）
    4. stats_summary.csv          按键频率/摇杆统计汇总表

说明：
- 数据来自 data/<game>/annotations.parquet（17 键 + j_left/j_right）
- 10 条序列从全量数据中均匀抽样（固定种子 42），每条 120 帧（4 秒）
- 摇杆分布对全量 53 万帧计算（满足 >=500 帧要求，超额）
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境，保存 PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体（Windows：微软雅黑），避免中文标签乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# 17 键（字母序，与数据 schema 一致）
BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
SEED = 42
SEQ_LEN = 120  # 每条序列 120 帧 = 4 秒
N_SEQ = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hades")
    ap.add_argument("--annotations", default=None, help="直接指定 parquet 路径（覆盖 --game）")
    args = ap.parse_args()

    if args.annotations:
        parquet = Path(args.annotations)
        game = parquet.parent.name
    else:
        # 基于仓库根（scripts/ 的上一级）定位 data/，避免写死本机绝对路径
        repo_root = Path(__file__).resolve().parent.parent
        parquet = repo_root / "data" / args.game / "annotations.parquet"
        game = args.game

    out_dir = parquet.parent / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {parquet}")
    df = pd.read_parquet(parquet)
    print(f"rows: {df.shape[0]}, cols: {df.shape[1]}")

    missing = [c for c in BUTTON_COLS if c not in df.columns]
    assert not missing, f"缺失按键列: {missing}"

    # ---------- 1. 按键触发频率分布 ----------
    btn = df[BUTTON_COLS].astype(int)
    press_rate = btn.mean().sort_values(ascending=False)  # 触发率（该键被按下的帧占比）

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(press_rate)), press_rate.values * 100, color="#4C72B0")
    ax.set_xticks(range(len(press_rate)))
    ax.set_xticklabels(press_rate.index, rotation=45, ha="right")
    ax.set_xlabel("按键")
    ax.set_ylabel("触发率 (%)")
    ax.set_title(f"{game}：按键触发率分布（全量 {df.shape[0]:,} 帧）")
    for b, v in zip(bars, press_rate.values * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "button_press_dist.png", dpi=150)
    plt.close(fig)
    print("saved button_press_dist.png")

    # ---------- 2. 摇杆取值分布 ----------
    jl = np.array(df["j_left"].to_list())   # (N, 2)
    jr = np.array(df["j_right"].to_list())

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, data, name in [(axes[0][0], jl, "left"), (axes[0][1], jr, "right")]:
        h = ax.hist2d(data[:, 0], data[:, 1], bins=60, range=[[-1, 1], [-1, 1]], cmap="viridis")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{'左' if name == 'left' else '右'}摇杆位置密度（对数色标）")
        ax.set_facecolor("#fafafa")
        fig.colorbar(h[3], ax=ax, shrink=0.8)
    # 边缘分布：x/y 分量直方图
    for idx, (data, name) in enumerate([(jl, "left"), (jr, "right")]):
        row = 1
        ax = axes[row][idx]
        ax.hist(data[:, 0], bins=60, alpha=0.6, label="x", color="#4C72B0")
        ax.hist(data[:, 1], bins=60, alpha=0.6, label="y", color="#DD8452")
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{'左' if name == 'left' else '右'}摇杆 x/y 边缘分布")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "joystick_dist.png", dpi=150)
    plt.close(fig)
    print("saved joystick_dist.png")

    # ---------- 3. 10 条连续动作序列曲线 ----------
    # 均匀抽样 10 个起点（不重叠）
    rng = np.random.RandomState(SEED)
    max_start = df.shape[0] - SEQ_LEN
    starts = np.sort(rng.choice(max_start, size=N_SEQ, replace=False))

    # 按键时序热力图 + 摇杆轨迹，两图并排
    n_cols = 2
    fig, axes = plt.subplots(N_SEQ, n_cols, figsize=(14, N_SEQ * 3))
    for i, start in enumerate(starts):
        seg = df.iloc[start:start + SEQ_LEN]

        # 子图 A：按键时序（17 键 0/1 阶梯，只画有触发的键）
        ax = axes[i][0]
        seg_btn = seg[BUTTON_COLS].astype(int)
        active_keys = [c for c in BUTTON_COLS if seg_btn[c].sum() > 0]
        if active_keys:
            y_pos = np.arange(len(active_keys))
            for j, key in enumerate(active_keys):
                vals = seg_btn[key].values
                ax.step(range(SEQ_LEN), vals + j * 1.2, where="post", lw=1.2)
            ax.set_yticks([j * 1.2 + 0.5 for j in range(len(active_keys))])
            ax.set_yticklabels(active_keys, fontsize=7)
        ax.set_xlim(0, SEQ_LEN)
        ax.set_xlabel("帧")
        ax.set_title(f"序列 {i+1}：按键时序（起点帧 {start}）", fontsize=9)

        # 子图 B：摇杆轨迹（左/右）
        ax = axes[i][1]
        seg_jl = np.array(seg["j_left"].to_list())
        seg_jr = np.array(seg["j_right"].to_list())
        ax.plot(seg_jl[:, 0], seg_jl[:, 1], "-o", ms=2, lw=0.8, label="L", color="#4C72B0")
        ax.plot(seg_jr[:, 0], seg_jr[:, 1], "-s", ms=2, lw=0.8, label="R", color="#DD8452")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"序列 {i+1}：摇杆轨迹", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "sequences.png", dpi=150)
    plt.close(fig)
    print("saved sequences.png")

    # ---------- 4. 汇总 CSV ----------
    jl_min, jl_max = jl.min(axis=0), jl.max(axis=0)
    jr_min, jr_max = jr.min(axis=0), jr.max(axis=0)
    idle = (
        (np.abs(jl).max(axis=1) <= 0.1)
        & (np.abs(jr).max(axis=1) <= 0.1)
        & (btn.sum(axis=1) == 0)
    )
    summary = pd.DataFrame({
        "button": press_rate.index,
        "press_rate": press_rate.values,
        "press_count": btn.sum().loc[press_rate.index].values,
    })
    summary.to_csv(out_dir / "stats_summary.csv", index=False)

    print(f"\n--- 汇总 ---")
    print(f"总帧数    : {df.shape[0]:,}")
    print(f"IDLE 占比 : {idle.mean()*100:.1f}%  (17 键全 0 且 |摇杆|<=0.1)")
    print(f"有按键帧  : {(btn.sum(axis=1) > 0).mean()*100:.1f}%")
    print(f"左摇杆活动: {(np.abs(jl).max(axis=1) > 0.1).mean()*100:.1f}%")
    print(f"右摇杆活动: {(np.abs(jr).max(axis=1) > 0.1).mean()*100:.1f}%")
    print(f"j_left  range: {jl_min} ~ {jl_max}")
    print(f"j_right range: {jr_min} ~ {jr_max}")
    print(f"\noutputs in {out_dir}")


if __name__ == "__main__":
    main()
