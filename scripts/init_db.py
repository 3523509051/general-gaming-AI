# -*- coding: utf-8 -*-
"""建 SQLite 结果库（eval_results.db）并写入已测游戏的种子数据。

表结构见《网页可视化平台实施文档.md》4.2 节。
种子数据来自已完成的两次评估（Hades / lies_of_p，见 项目备忘.md）。
evaluate.py 跑完新评估后会 INSERT OR REPLACE 更新本库。

用法（任意 Python，内置 sqlite3）：
    python scripts/init_db.py
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "eval_results.db"

DDL = """
CREATE TABLE IF NOT EXISTS eval_results (
  game           TEXT NOT NULL,
  video          TEXT NOT NULL,
  stats_frames   INTEGER,
  test_frames    INTEGER,
  acc_17keys     REAL,
  recall         REAL,
  precision      REAL,
  corr_jl_x      REAL,
  corr_jl_y      REAL,
  mse_jl         REAL,
  best_shift     INTEGER,
  stats_plot     TEXT,
  seq_plot       TEXT,
  shift_scan     TEXT,
  test_set_csv   TEXT,
  evaluated_at   TEXT DEFAULT (datetime('now','localtime')),
  PRIMARY KEY (game, video)
);
"""

# 种子数据：已完成的 zero-shot 评估（B 口径：逐帧全对 acc_17keys；详见 项目备忘.md）
# stats_frames 为该视频实际标注帧数（非游戏全量合计）；后续 evaluate.py 会 INSERT OR REPLACE 更新。
SEEDS = [
    # hades v1805686899（该视频 29,400 帧，200 帧测试集，B 口径 acc=0.480）
    ("hades", "v1805686899", 29400, 200, 0.480, 0.672, 0.467,
     0.043, 0.210, 0.723, 15,
     "data/hades/stats/button_press_dist.png", "data/hades/stats/sequences.png",
     "data/hades/eval/shift_scan.png", "data/hades/test_set_v1805686899.csv"),
    # lies_of_p v2276819038（该视频 21,600 帧，488 帧测试集，B 口径 acc=0.447）
    ("lies_of_p", "v2276819038", 21600, 488, 0.447, 0.411, 0.326,
     -0.046, -0.031, 0.309, 0,
     "data/lies_of_p/stats/button_press_dist.png", "data/lies_of_p/stats/sequences.png",
     "data/lies_of_p/eval/shift_scan.png", "data/lies_of_p/test_set_v2276819038.csv"),
]


def upsert(conn: sqlite3.Connection, row: tuple) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO eval_results
          (game, video, stats_frames, test_frames, acc_17keys, recall, precision,
           corr_jl_x, corr_jl_y, mse_jl, best_shift,
           stats_plot, seq_plot, shift_scan, test_set_csv)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        row,
    )


def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute(DDL)
    for row in SEEDS:
        upsert(conn, row)
    conn.commit()
    # 打印确认
    for r in conn.execute("SELECT game, video, acc_17keys, best_shift, evaluated_at FROM eval_results"):
        print(f"  {r[0]}/{r[1]}: acc17={r[2]}, best_shift={r[3]}, at={r[4]}")
    conn.close()
    print(f"DB ready -> {DB}")


if __name__ == "__main__":
    main()
