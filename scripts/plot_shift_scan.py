# -*- coding: utf-8 -*-
"""绘制 shift 扫描结果曲线：吻合度指标随动作块步偏移的变化。

用法（venv python，需 matplotlib）：
    python scripts/plot_shift_scan.py --game hades --video v1805686899

读 data/<game>/eval/metrics_<video>.csv（回退 metrics.csv），
输出 data/<game>/eval/shift_scan_<video>.png（回退 shift_scan.png）。
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows：微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"


def main():
    ap = argparse.ArgumentParser(description="绘制 shift 扫描曲线")
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    eval_dir = DATA_ROOT / args.game / "eval"
    if args.video:
        metrics_csv = eval_dir / f"metrics_{args.video}.csv"
        out_png = eval_dir / f"shift_scan_{args.video}.png"
    else:
        metrics_csv = eval_dir / "metrics.csv"
        out_png = eval_dir / "shift_scan.png"
    if not metrics_csv.exists():
        raise SystemExit(f"metrics 文件不存在: {metrics_csv}")

    metrics = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    shifts = [int(m["shift"]) for m in metrics]
    acc17 = [float(m["acc_17keys_all"]) for m in metrics]
    acc17_nidle = [float(m.get("acc_17keys_nonidle") or "nan") for m in metrics]
    recall = [float(m["btn_recall"]) for m in metrics]
    prec = [float(m["btn_precision"]) for m in metrics]
    corr_x = [float(m["corr_jl_x"]) for m in metrics]
    corr_y = [float(m["corr_jl_y"]) for m in metrics]
    corr_rx = [float(m["corr_jr_x"]) if m.get("corr_jr_x") not in (None, "") else float("nan") for m in metrics]
    corr_ry = [float(m["corr_jr_y"]) if m.get("corr_jr_y") not in (None, "") else float("nan") for m in metrics]
    has_right = any(not (x != x or y != y) for x, y in zip(corr_rx, corr_ry))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # 上：按键指标
    ax = axes[0]
    ax.plot(shifts, acc17, "-o", label="按键一致率 17键（全帧）", color="#4C72B0")
    ax.plot(shifts, acc17_nidle, "-s", label="按键一致率（过滤IDLE）", color="#55A868")
    ax.plot(shifts, recall, "-^", label="按键召回率", color="#C44E52")
    ax.plot(shifts, prec, "-d", label="按键精确率", color="#DD8452")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="参考水平 0.50")
    ax.set_xlabel("shift（动作块步偏移）")
    ax.set_ylabel("比率")
    ax.set_title(f"shift 扫描：按键一致率 vs 动作块偏移（{args.game}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 下：摇杆相关（左 + 右，右摇杆有数据才画）
    ax = axes[1]
    ax.plot(shifts, corr_x, "-o", label="左摇杆相关 x", color="#4C72B0")
    ax.plot(shifts, corr_y, "-s", label="左摇杆相关 y", color="#DD8452")
    if has_right:
        ax.plot(shifts, corr_rx, "-^", label="右摇杆相关 x", color="#55A868")
        ax.plot(shifts, corr_ry, "-d", label="右摇杆相关 y", color="#8172B3")
    ax.axhline(0.4, ls="--", color="gray", lw=0.8, label="参考水平 0.40")
    ax.axhline(0.0, ls=":", color="black", lw=0.8)
    ax.set_xlabel("shift（动作块步偏移）")
    ax.set_ylabel("皮尔逊 r")
    ax.set_title("shift 扫描：摇杆相关系数 vs 动作块偏移")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("saved", out_png)

    # 打印最优 shift
    best_acc = max(zip(shifts, acc17), key=lambda x: x[1])
    best_recall = max(zip(shifts, recall), key=lambda x: x[1])
    best_corrx = max(zip(shifts, corr_x), key=lambda x: x[1])
    print(f"best shift by acc17  : k={best_acc[0]} (acc={best_acc[1]:.3f})")
    print(f"best shift by recall : k={best_recall[0]} (recall={best_recall[1]:.3f})")
    print(f"best shift by corr_x : k={best_corrx[0]} (corr={best_corrx[1]:.3f})")


if __name__ == "__main__":
    main()
