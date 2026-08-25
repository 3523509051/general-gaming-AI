# -*- coding: utf-8 -*-
"""通用 zero-shot 评估脚本（原 evaluate_hades.py 的通用化版本）。

支持任意已提取标注 + 已下载视频的 (game, video)：
1. 若无 test_set.csv：抽样构建测试集（默认纯随机，论文口径约 500 帧量级；--seq-mode 抽连续片段序列集）+ ffmpeg 抽帧
2. 加载 ng.pt 逐帧推理 -> 18 步动作块
3. shift 0~17 全扫描 -> metrics.csv
4. 最优 shift 逐帧对照 -> predictions.csv（含预测按键明细 pred_* 列）
5. INSERT OR REPLACE 写入 data/eval_results.db（Web 平台结果库）
6. 若无 stats/ 则调用 stats_viz.py 生成统计图

用法（仓库根目录，venv python）：
    NitroGen\\.venv\\Scripts\\python.exe scripts\\evaluate.py --game hades --video v1805686899 --fps 30
    NitroGen\\.venv\\Scripts\\python.exe scripts\\evaluate.py --game lies_of_p --video v2276819038 --fps 60

参数：
    --game      游戏名（data/<game>/ 必须已有 manifest.json + annotations.parquet）
    --video     视频 ID（data/videos/<game>_<video>.mp4 必须已存在；默认取该游戏第一个已下载视频）
    --fps       视频帧率（构建测试集时必填；已有 test_set.csv 时可省略）
    --test-size 测试集帧数（默认 200）
    --seed      抽样种子（默认 42）
    --skip-existing-testset  已有 test_set.csv 时直接复用（默认行为）

输出：
    data/<game>/test_set.csv
    data/<game>/test_frames/*.jpg
    data/<game>/eval/metrics.csv        shift 扫描指标表
    data/<game>/eval/predictions.csv    最优 shift 逐帧预测 vs 真值（含 pred 明细）
    data/eval_results.db                指标入库（Web 平台读取）
"""
import builtins
import csv
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

builtins.input = lambda *a: ""  # 无条件模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 权重已缓存，跳过 HF 联网检查

import numpy as np
import polars as pl
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "NitroGen") not in sys.path:
    sys.path.insert(0, str(REPO / "NitroGen"))  # nitrogen 包所在目录（本地/远程均可靠）

from nitrogen.inference_session import InferenceSession

DATA_ROOT = REPO / "data"
CKPT = REPO / "NitroGen" / "ng.pt"
DB_PATH = DATA_ROOT / "eval_results.db"

BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
MODEL_BUTTONS = sorted(BUTTON_COLS + ["right_bottom", "right_left", "right_right", "right_up"])
MODEL17_IDX = [MODEL_BUTTONS.index(b) for b in BUTTON_COLS]
IDLE_THRESH = 0.1


def find_ffmpeg() -> str:
    """定位 ffmpeg：优先 imageio_ffmpeg 自带，其次搜索常见位置（跨机器可移植，不写死用户名）。"""
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass
    # PATH 中的 ffmpeg
    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path
    # 常见安装位置
    for pat in (Path.home() / "AppData/Local/Python", Path("C:/Program Files"),
                Path("C:/Program Files (x86)")):
        if pat.exists():
            for hit in pat.rglob("imageio_ffmpeg/binaries/ffmpeg*.exe"):
                return str(hit)
    raise RuntimeError("ffmpeg not found（请安装 imageio-ffmpeg 或将其加入 PATH）")


def locate_video(game: str, video: str | None) -> tuple[str, Path]:
    """定位该游戏已下载的视频文件。"""
    videos_dir = DATA_ROOT / "videos"
    if video:
        p = videos_dir / f"{game}_{video}.mp4"
        if p.exists():
            return video, p
        raise FileNotFoundError(f"视频未下载: {p}")
    # 默认取该游戏第一个已下载视频
    for p in sorted(videos_dir.glob(f"{game}_*.mp4")):
        return p.stem[len(game) + 1:], p
    raise FileNotFoundError(f"data/videos/ 下没有 {game} 的视频，请先下载")


def build_testset(game: str, video: str, video_file: Path, fps: int,
                  test_size: int, seed: int, csv_path: Path | None = None,
                  sample_mode: str = "random",
                  seq_segments: int = 0, seq_len: int = 50) -> list[dict]:
    """抽样 + ffmpeg 抽帧，生成测试集 CSV。

    sample_mode：
      - "random"     （默认，论文口径）纯随机均匀抽样：从视频全部可抽帧范围随机抽
                     test_size 帧（默认 500，约 500 帧量级），代表整体分布，
                     用于报告按键一致率 + 摇杆 MSE/相关。
      - "stratified" （旧行为）按 chunk 均摊抽样，每 chunk 抽 test_size/总chunk 帧。
    seq_segments/seq_len：>0 时改为"连续片段"抽样（论文口径的序列集），
                     在视频内随机抽 seq_segments 段 × 每段 seq_len 连续帧，
                     总帧数 ≈ seq_segments*seq_len（默认 10×50=500 帧），
                     用于 Tab③ 序列可视化 + Tab② 统计分布，序列具时序连续性。

    默认写 test_set_<video>.csv（按视频隔离，多视频游戏互不覆盖）。
    """
    manifest = json_load(DATA_ROOT / game / "manifest.json")
    annotations = pl.read_parquet(DATA_ROOT / game / "annotations.parquet")
    chunks = sorted([c for c in manifest["chunks"] if c["video"] == video], key=lambda c: c["chunk"])
    assert chunks, f"manifest 中未找到 {video} 的 chunk"

    out_dir = DATA_ROOT / game / "test_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    rng = random.Random(seed)

    if seq_segments > 0:
        # 连续片段抽样：随机抽 seq_segments 段起点，每段 seq_len 连续帧（不跨 chunk）
        samples = []
        seg = 0
        while seg < seq_segments:
            c = rng.choice(chunks)
            max_start = max(0, c["rows"] - seq_len)
            start = rng.randrange(max_start + 1)
            for fid in range(seq_len):
                absolute_frame = c["start_frame"] + start + fid
                samples.append({
                    "video": video, "chunk": c["chunk"], "frame_idx": start + fid,
                    "absolute_frame": absolute_frame, "second": absolute_frame / fps,
                    "seq_id": seg,
                })
            seg += 1
        print(f"[build_testset] 连续片段抽样: {seq_segments}段 x {seq_len}帧 = {len(samples)} 帧")
    elif sample_mode == "stratified":
        # 旧行为：按 chunk 均摊
        per_chunk = (test_size + len(chunks) - 1) // len(chunks)
        samples = []
        for c in chunks:
            for fid in sorted(rng.sample(range(c["rows"]), k=min(per_chunk, c["rows"]))):
                absolute_frame = c["start_frame"] + fid
                samples.append({
                    "video": video, "chunk": c["chunk"], "frame_idx": fid,
                    "absolute_frame": absolute_frame, "second": absolute_frame / fps,
                })
        samples = samples[:test_size]
    else:
        # 纯随机均匀抽样（论文口径）：全视频可抽帧范围随机抽 test_size 帧
        total = sum(c["rows"] for c in chunks)
        # 全局帧号 -> chunk 映射：off[i] 为 chunk i 的全局起始偏移
        off = [0]
        for c in chunks[:-1]:
            off.append(off[-1] + c["rows"])
        picked = sorted(rng.sample(range(total), k=test_size))
        samples = []
        for g in picked:
            # 二分找 g 落在哪个 chunk 区间
            lo, hi = 0, len(chunks) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if off[mid] <= g:
                    lo = mid
                else:
                    hi = mid - 1
            c = chunks[lo]
            fid = g - off[lo]
            absolute_frame = c["start_frame"] + fid
            samples.append({
                "video": video, "chunk": c["chunk"], "frame_idx": fid,
                "absolute_frame": absolute_frame, "second": absolute_frame / fps,
            })
        print(f"[build_testset] 纯随机抽样: {len(samples)} 帧（全视频 {total} 帧范围）")

    ann_lookup = {
        (r["video"], r["chunk"], r["frame_idx"]): r
        for r in annotations.filter(pl.col("video") == video).to_dicts()
    }

    rows = []
    for i, s in enumerate(samples):
        out_path = out_dir / f"{s['video']}_f{s['absolute_frame']:05d}.jpg"
        if not (out_path.exists() and out_path.stat().st_size > 0):
            for attempt in range(2):
                subprocess.run(
                    [ffmpeg, "-y", "-ss", f"{s['second']:.3f}", "-i", str(video_file),
                     "-frames:v", "1", str(out_path)],
                    capture_output=True, timeout=30,
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    break
                out_path.unlink(missing_ok=True)
        if not (out_path.exists() and out_path.stat().st_size > 0):
            print(f"WARN: 抽帧失败 frame={s['absolute_frame']}", flush=True)
            continue
        ann = ann_lookup.get((s["video"], s["chunk"], s["frame_idx"]), {})
        jl = ann.get("j_left", [0.0, 0.0])
        jr = ann.get("j_right", [0.0, 0.0])
        n_btn = sum(int(ann.get(b, 0)) for b in BUTTON_COLS)
        rows.append({
            "frame_path": f"{game}/test_frames/{out_path.name}",  # 跨平台正斜杠（Windows/Linux 均可）
            "video": s["video"], "chunk": s["chunk"], "frame_idx": s["frame_idx"],
            "absolute_frame": s["absolute_frame"], "second": round(s["second"], 3),
            "n_button_presses": n_btn,
            "j_left_x": jl[0], "j_left_y": jl[1], "j_right_x": jr[0], "j_right_y": jr[1],
        })
        if (i + 1) % 50 == 0:
            print(f"  extracted {i+1}/{len(samples)}", flush=True)

    if csv_path is None:
        csv_path = DATA_ROOT / game / f"test_set_{video}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([r[k] for k in rows[0].keys()])
    print(f"test set built: {len(rows)} frames -> {csv_path}")
    return rows


def json_load(p: Path):
    import json
    return json.load(open(p, encoding="utf-8"))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def save_to_db(game, video, stats_frames, metrics, best_k, test_csv: Path):
    """评估结果写入 SQLite（Web 平台结果库）。"""
    best = metrics[best_k]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
          game TEXT NOT NULL, video TEXT NOT NULL,
          stats_frames INTEGER, test_frames INTEGER,
          acc_17keys REAL, recall REAL, precision REAL,
          corr_jl_x REAL, corr_jl_y REAL, mse_jl REAL, best_shift INTEGER,
          stats_plot TEXT, seq_plot TEXT, shift_scan TEXT, test_set_csv TEXT,
          PRIMARY KEY (game, video)
        )
        """
    )  # 幂等建表：远程 worker 评估无 DB 时自动创建（本地已有表不受影响）
    conn.execute(
        """
        INSERT OR REPLACE INTO eval_results
          (game, video, stats_frames, test_frames, acc_17keys, recall, precision,
           corr_jl_x, corr_jl_y, mse_jl, best_shift,
           stats_plot, seq_plot, shift_scan, test_set_csv)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (game, video, stats_frames, len(metrics) and best.get("n_frames", 200),
         best["acc_17keys_all"], best["btn_recall"], best["btn_precision"],
         best["corr_jl_x"], best["corr_jl_y"], best["mse_jl_x"],
         best_k,
         f"data/{game}/stats/button_press_dist_{video}.png",
         f"data/{game}/stats/sequences_{video}.png",
         f"data/{game}/eval/shift_scan_{video}.png",
         f"data/{game}/{test_csv.name}"),
    )
    conn.commit()
    conn.close()
    print(f"DB updated: {game}/{video} (acc17={best['acc_17keys_all']:.3f}, shift={best_k})")


def ensure_stats(game: str, video: str | None = None):
    """若无统计图，调用 stats_viz.py 生成（带 video 则按视频生成）。"""
    stats_dir = DATA_ROOT / game / "stats"
    suffix = f"_{video}" if video else ""
    need = [f"button_press_dist{suffix}.png", f"joystick_dist{suffix}.png",
            f"sequences{suffix}.png", f"stats_summary{suffix}.csv"]
    if all((stats_dir / n).exists() for n in need):
        return
    print("generating stats plots ...")
    cmd = [sys.executable, str(REPO / "scripts" / "stats_viz.py"), "--game", game]
    if video:
        cmd += ["--video", video]
    subprocess.run(cmd, check=True)


def plot_shift_scan(game: str, video: str):
    """评估后自动绘制 shift 扫描曲线（读 metrics_<video>.csv）。"""
    eval_dir = DATA_ROOT / game / "eval"
    out = eval_dir / f"shift_scan_{video}.png"
    if out.exists():
        return
    try:
        subprocess.run([sys.executable, str(REPO / "scripts" / "plot_shift_scan.py"),
                        "--game", game, "--video", video], check=True)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: shift_scan 图生成失败: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="通用 zero-shot 评估")
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=None, help="视频帧率（构建测试集时必填）")
    ap.add_argument("--test-size", type=int, default=200,
                    help="测试集帧数（默认 200，可在 Web 平台评估引导处调整）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infer-seed", type=int, default=None,
                    help="推理采样种子（不传=不固定；传值则推理前 torch.manual_seed，同 seed 结果可复现；"
                         "多组重复评估取均值时传不同值）")
    ap.add_argument("--rebuild-testset", action="store_true", help="强制重建测试集")
    ap.add_argument("--sample-mode", choices=["random", "stratified"], default="random",
                    help="抽样方式：random=纯随机（论文口径，默认）/ stratified=按chunk均摊（旧行为）")
    ap.add_argument("--seq-mode", action="store_true",
                    help="序列集模式：抽 seq_segments 段 × seq_len 连续帧（论文口径，默认 10×50=500 帧）")
    ap.add_argument("--seq-segments", type=int, default=10, help="连续片段段数（--seq-mode 下）")
    ap.add_argument("--seq-len", type=int, default=50, help="每段连续帧数（--seq-mode 下）")
    ap.add_argument("--ckpt", default=None,
                    help="模型权重路径（默认 NitroGen/ng.pt；微调权重如 NitroGen/ng_finetuned.pt）")
    ap.add_argument("--no-plots", action="store_true",
                    help="跳过统计图与 shift 扫描图生成（远程 worker 评估用，避免依赖 stats_viz/plot_shift_scan）")
    ap.add_argument("--tag", default=None,
                    help="输出文件标签（默认取 ckpt 文件名去扩展名，如 ng_finetuned；"
                         "用于写 metrics/predictions 带 tag 副本，供 zero-shot vs 微调对照）")
    args = ap.parse_args()

    # 模型权重：--ckpt 可选（默认 ng.pt）；tag 用于带副本输出的文件名后缀
    ckpt = Path(args.ckpt) if args.ckpt else CKPT
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        raise SystemExit(f"权重不存在: {ckpt}")
    tag = args.tag if args.tag is not None else ("" if ckpt == CKPT else ckpt.stem)

    game_dir = DATA_ROOT / args.game
    assert (game_dir / "manifest.json").exists(), f"缺少 {game_dir}/manifest.json，请先跑 extract_game.py"
    assert (game_dir / "annotations.parquet").exists(), "缺少 annotations.parquet"

    video, video_file = locate_video(args.game, args.video)
    print(f"[setup] game={args.game} video={video} file={video_file.name}")

    # 1) 测试集（按视频隔离：test_set_<video>.csv 优先；兼容旧版单文件 test_set.csv）
    # 帧数校验：复用前检查帧数是否与 --test-size 一致，不一致（旧 200 帧 → 新 500 帧）自动删除重建，
    # 保证前端点一次"运行评估"即自动切换到当前口径，无需手动删旧测试集。
    test_csv = game_dir / f"test_set_{video}.csv"
    if args.rebuild_testset:
        test_csv.unlink(missing_ok=True)
    if test_csv.exists():
        rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
        rows = [r for r in rows if r["video"] == video]
        if len(rows) == args.test_size:
            print(f"[setup] reuse test set: {len(rows)} frames -> {test_csv.name}")
        else:
            print(f"[setup] test set 帧数 {len(rows)} != --test-size {args.test_size}，重建（新口径）")
            test_csv.unlink(missing_ok=True)
            rows = []
    else:
        # 旧版单文件里可能有该视频的帧（如 v1805686899 在 test_set.csv）
        legacy = game_dir / "test_set.csv"
        if legacy.exists() and not args.rebuild_testset:
            rows = [r for r in csv.DictReader(open(legacy, encoding="utf-8")) if r["video"] == video]
            if len(rows) == args.test_size:
                print(f"[setup] reuse legacy test set: {len(rows)} frames -> {legacy.name}")
                test_csv = legacy
            else:
                print(f"[setup] legacy test set 帧数 {len(rows)} != --test-size {args.test_size}，重建（新口径）")
                rows = []
        else:
            rows = []
    if not rows:
        if not args.fps:
            raise SystemExit("构建测试集需要 --fps（如 hades v1805686899 为 30，lies_of_p 为 60）")
        if args.seq_mode:
            # 序列集模式：抽连续片段，写入 seq_set_<video>.csv
            test_csv = game_dir / f"seq_set_{video}.csv"
            rows = build_testset(args.game, video, video_file, args.fps, args.test_size, args.seed,
                                 csv_path=test_csv,
                                 seq_segments=args.seq_segments, seq_len=args.seq_len)
        else:
            rows = build_testset(args.game, video, video_file, args.fps, args.test_size, args.seed,
                                 csv_path=test_csv, sample_mode=args.sample_mode)
    assert rows, "测试集为空"

    # 2) 标注真值关联（17 键明细）
    ann = pl.read_parquet(game_dir / "annotations.parquet")
    lookup = {}
    for row in ann.select(["video", "chunk", "frame_idx"] + BUTTON_COLS).to_dicts():
        lookup[(row["video"], row["chunk"], row["frame_idx"])] = row

    gt_btn = np.zeros((len(rows), len(BUTTON_COLS)), dtype=int)
    gt_jl = np.zeros((len(rows), 2))
    gt_jr = np.zeros((len(rows), 2))
    for i, r in enumerate(rows):
        gt_jl[i] = [float(r["j_left_x"]), float(r["j_left_y"])]
        gt_jr[i] = [float(r["j_right_x"]), float(r["j_right_y"])]
        arow = lookup.get((r["video"], r["chunk"], int(r["frame_idx"])))
        if arow is not None:
            for j, b in enumerate(BUTTON_COLS):
                gt_btn[i, j] = int(arow[b])
    print(f"[setup] linked {(gt_btn.sum(axis=1) >= 0).sum()}/{len(rows)} frames to annotations")

    # 3) 推理
    print(f"[1/3] loading model: {ckpt.name} (tag={tag or '-'}) ...", flush=True)
    t0 = time.time()
    session = InferenceSession.from_ckpt(str(ckpt))
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)
    if args.infer_seed is not None:
        import torch
        torch.manual_seed(args.infer_seed)
        print(f"      torch.manual_seed({args.infer_seed})（推理采样可复现）", flush=True)
    preds = []
    t0 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(DATA_ROOT / r["frame_path"])
        preds.append(session.predict(img))
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(rows)} frames ({time.time()-t0:.1f}s)", flush=True)
    buttons_all = np.stack([p["buttons"] for p in preds])
    jl_all = np.stack([p["j_left"] for p in preds])
    jr_all = np.stack([p["j_right"] for p in preds])
    print(f"[2/3] inference done ({(time.time()-t0)/len(rows):.2f}s/frame)")

    idle_mask = (
        (np.abs(gt_jl).max(axis=1) <= IDLE_THRESH)
        & (np.abs(gt_jr).max(axis=1) <= IDLE_THRESH)
        & (gt_btn.sum(axis=1) == 0)
    )

    # 4) shift 扫描
    # 按键一致率主口径 = B（逐帧全对：17 键全部一致才算该帧对），
    # 与课程参考水平"约 50%"对齐（口径敏感性分析见项目备忘 八·五 补充口径 3）。
    # 同时保留 A 口径（逐键逐帧）acc_17keys_bits 作对照。
    metrics = []
    for k in range(18):
        pred_btn17 = buttons_all[:, k, :][:, MODEL17_IDX] >= 0.5
        pred_jl, pred_jr = jl_all[:, k, :], jr_all[:, k, :]
        gt_any = gt_btn.sum(axis=1) > 0
        pred_any = pred_btn17.sum(axis=1) > 0
        gt_none = ~gt_any              # 真值无按键
        fp_mask = pred_any & gt_none   # 误触发：预测有按键但真值无按键
        fp_nonidle = fp_mask & ~idle_mask  # 非 IDLE 帧中的误触发（IDLE/无效帧过滤口径）
        gt_none_nonidle = gt_none & ~idle_mask
        n_gp = int((gt_any & pred_any).sum())
        m = {
            "shift": k,
            "acc_17keys_all": float((pred_btn17 == gt_btn).all(axis=1).mean()),  # B: 逐帧全对
            "acc_17keys_bits": float((pred_btn17 == gt_btn).mean()),            # A: 逐键逐帧（对照）
            "btn_recall": n_gp / int(gt_any.sum()) if gt_any.sum() else float("nan"),
            "btn_precision": n_gp / int(pred_any.sum()) if pred_any.sum() else float("nan"),
            # 误触发按键率（扩展 B：IDLE/无效帧过滤，目标一：相对过滤前约 -20%）
            "btn_fpr": float(fp_mask.sum()) / int(gt_none.sum()) if gt_none.sum() else float("nan"),      # 全帧口径（过滤前）
            "btn_fpr_nonidle": float(fp_nonidle.sum()) / int(gt_none_nonidle.sum()) if gt_none_nonidle.sum() else float("nan"),  # 非IDLE口径（过滤后）
            "mse_jl_x": float(((pred_jl[:, 0] - gt_jl[:, 0]) ** 2).mean()),
            "mse_jl_y": float(((pred_jl[:, 1] - gt_jl[:, 1]) ** 2).mean()),
            "corr_jl_x": pearson(pred_jl[:, 0], gt_jl[:, 0]),
            "corr_jl_y": pearson(pred_jl[:, 1], gt_jl[:, 1]),
            "n_frames": len(rows),
            # 过滤 IDLE 口径：仅非 IDLE 帧（is_idle=双摇杆幅度≤0.1 且 17 键全 0）的逐帧全对
            "acc_17keys_all_nonidle": float(
                (pred_btn17 == gt_btn)[~idle_mask].all(axis=1).mean()
            ) if (~idle_mask).any() else float("nan"),
            "n_frames_nonidle": int((~idle_mask).sum()),
        }
        metrics.append(m)

    eval_dir = game_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    metrics_csv = eval_dir / f"metrics_{video}.csv"   # 按视频隔离，避免多视频互相覆盖
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics[0].keys())
        for m in metrics:
            w.writerow(m.values())
    if tag:  # 带 tag 副本：供 zero-shot vs 微调对照（主文件保持 Web 兼容）
        shutil.copy(metrics_csv, eval_dir / f"metrics_{video}_{tag}.csv")

    best_k = int(max(metrics, key=lambda m: m["acc_17keys_all"])["shift"])
    print(f"[3/3] best shift k={best_k}, acc17={metrics[best_k]['acc_17keys_all']:.3f}")

    # 5) predictions_<video>.csv（含预测明细 pred_* 列；按视频隔离）
    pred_btn17 = buttons_all[:, best_k, :][:, MODEL17_IDX] >= 0.5
    pred_jl, pred_jr = jl_all[:, best_k, :], jr_all[:, best_k, :]
    pred_csv = eval_dir / f"predictions_{video}.csv"
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "video", "chunk", "frame_idx", "absolute_frame", "is_idle",
                    "gt_n_press", "pred_n_press", "gt_jl_x", "gt_jl_y",
                    "pred_jl_x", "pred_jl_y", "gt_jr_x", "gt_jr_y", "pred_jr_x", "pred_jr_y",
                    "n_mismatch_17"]
                   + [f"gt_{b}" for b in BUTTON_COLS]
                   + [f"pred_{b}" for b in BUTTON_COLS])
        for i, r in enumerate(rows):
            n_mismatch = int((pred_btn17[i] != gt_btn[i]).sum())
            w.writerow([
                r["frame_path"], r["video"], r["chunk"], r["frame_idx"],
                r.get("absolute_frame", ""), int(idle_mask[i]),
                int(gt_btn[i].sum()), int(pred_btn17[i].sum()),
                f"{gt_jl[i,0]:.4f}", f"{gt_jl[i,1]:.4f}",
                f"{pred_jl[i,0]:.4f}", f"{pred_jl[i,1]:.4f}",
                f"{gt_jr[i,0]:.4f}", f"{gt_jr[i,1]:.4f}",
                f"{pred_jr[i,0]:.4f}", f"{pred_jr[i,1]:.4f}",
                n_mismatch,
            ] + [int(v) for v in gt_btn[i]] + [int(v) for v in pred_btn17[i]])
    if tag:  # 带 tag 副本：供 zero-shot vs 微调逐帧对照
        shutil.copy(pred_csv, eval_dir / f"predictions_{video}_{tag}.csv")

    # 6) 入库 + 统计图 + shift 扫描图
    stats_frames = int(ann.filter(pl.col("video") == video).height)
    save_to_db(args.game, video, stats_frames, metrics, best_k, test_csv)
    if not args.no_plots:  # 远程 worker 评估跳过统计图（不依赖 stats_viz/plot_shift_scan）
        ensure_stats(args.game, video)
        plot_shift_scan(args.game, video)
    print(f"DONE -> {eval_dir}")


if __name__ == "__main__":
    main()
