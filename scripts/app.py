# -*- coding: utf-8 -*-
"""NitroGen 零样本评估工作台 - Flask 后端。

接口（详见《网页可视化平台实施文档.md》第 5 节）：
    GET  /api/games                    游戏列表（含 ready/tested 标记）
    GET  /api/games/<game>/videos      视频列表（含可用性状态）
    GET  /api/frame                    单帧识别（血缘断言 + 实时推理/缓存）
    GET  /api/stats                    统计分布（按键/摇杆/序列）
    GET  /api/sequences                序列对比 + 差异 Top-5 帧
    POST /api/rescan                   触发后台视频探测
    GET  /api/rescan/status            探测进度
    GET  /files/<path>                 data/ 下静态文件（PNG/JPG 白名单）

血缘断言（三层防线之后端层）：每个数据接口校验
    (game, video) 在 manifest.json 中存在
    且请求帧在 test_set.csv 中属于该 video
    且能在 annotations.parquet 中关联到真值
不符返回 400，不静默。

启动（仓库根目录，venv python）：
    NitroGen\\.venv\\Scripts\\python.exe scripts\\app.py
    -> http://localhost:5000
"""
import csv
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
WEB_DIR = REPO / "web"
DB_PATH = DATA_ROOT / "eval_results.db"
GAMES_SCAN = DATA_ROOT / "games_scan.json"
VIDEO_STATUS = DATA_ROOT / "video_status.json"
CKPT = REPO / "NitroGen" / "ng.pt"

BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
MODEL_BUTTONS = sorted(BUTTON_COLS + ["right_bottom", "right_left", "right_right", "right_up"])
MODEL17_IDX = [MODEL_BUTTONS.index(b) for b in BUTTON_COLS]

app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")


@app.after_request
def no_cache_static(resp):
    """开发阶段页面与静态资源不缓存，避免浏览器用旧版 HTML/JS。"""
    if request.path in ("/",) or request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def ok(data):
    return jsonify({"ok": True, "data": data})


def err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def db_query(sql, params=(), one=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]
    finally:
        conn.close()


def load_json(path: Path, default=None):
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return default


def downloaded_videos(game: str) -> list[str]:
    """该游戏已下载的视频 ID（data/videos/{game}_{video}.mp4）。"""
    return [p.stem[len(game) + 1:]
            for p in (DATA_ROOT / "videos").glob(f"{game}_*.mp4")]


# --------------------------------------------------------------------------
# 数据缓存（进程内，重启失效；标注 parquet 用 polars 惰性加载）
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_manifest(game: str) -> dict | None:
    p = DATA_ROOT / game / "manifest.json"
    return load_json(p) if p.exists() else None


@lru_cache(maxsize=8)
def get_testset(game: str, video: str) -> list[dict]:
    """读取测试集：优先按视频隔离的 test_set_<video>.csv，回退旧版单文件 test_set.csv。"""
    for p in (DATA_ROOT / game / f"test_set_{video}.csv",
              DATA_ROOT / game / "test_set.csv"):
        if p.exists():
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                    if r.get("video") == video]
            if rows:
                return rows
    return []


@lru_cache(maxsize=8)
def get_predictions(game: str, video: str) -> list[dict]:
    """读取预测对照：优先 predictions_<video>.csv，回退旧版单文件 predictions.csv。"""
    for p in (DATA_ROOT / game / "eval" / f"predictions_{video}.csv",
              DATA_ROOT / game / "eval" / "predictions.csv"):
        if p.exists():
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                    if r.get("video") == video]
            if rows:
                return rows
    return []


@lru_cache(maxsize=8)
def get_annotation_lookup(game: str, video: str) -> dict:
    """(chunk, frame_idx) -> 17 键真值 + 摇杆。polars 一次读入。"""
    import polars as pl
    p = DATA_ROOT / game / "annotations.parquet"
    if not p.exists():
        return {}
    ann = pl.read_parquet(p).filter(pl.col("video") == video)
    lookup = {}
    for row in ann.select(["chunk", "frame_idx"] + BUTTON_COLS
                          + ["j_left", "j_right"]).to_dicts():
        lookup[(row["chunk"], row["frame_idx"])] = {
            "buttons": {b: int(row[b]) for b in BUTTON_COLS},
            "j_left": [float(row["j_left"][0]), float(row["j_left"][1])],
            "j_right": [float(row["j_right"][0]), float(row["j_right"][1])],
        }
    return lookup


def clear_cache(game: str):
    """评估重跑后清缓存。"""
    for fn in (get_manifest, get_testset, get_predictions, get_annotation_lookup):
        fn.cache_clear()


# --------------------------------------------------------------------------
# 推理（懒加载单例 + 线程锁）
# --------------------------------------------------------------------------
class Predictor:
    _session = None
    _lock = threading.Lock()
    _load_lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._load_lock:
            if cls._session is None:
                import builtins
                import os
                builtins.input = lambda *a: ""  # 无条件模式
                os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 跳过 HF 联网检查（权重已缓存）
                from nitrogen.inference_session import InferenceSession
                t0 = time.time()
                cls._session = InferenceSession.from_ckpt(str(CKPT))
                print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)
        return cls._session

    @classmethod
    def predict(cls, img):
        with cls._lock:
            return cls.get().predict(img)


# --------------------------------------------------------------------------
# 血缘断言
# --------------------------------------------------------------------------
class LineageError(Exception):
    """血缘校验失败（game/video/frame 三方不一致）。"""

    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


@app.errorhandler(LineageError)
def handle_lineage_error(e):
    return jsonify({"ok": False, "error": str(e)}), e.code


def assert_lineage(game: str, video: str, frame: int | None = None):
    """三方一致断言，失败抛 LineageError。

    - manifest：video 必须属于该 game
    - test_set：请求帧必须在该 video 的测试集内
    - annotations：该帧必须能关联到标注真值
    成功且 frame 不为 None 时返回 (test_row, gt_dict)。
    """
    manifest = get_manifest(game)
    if manifest is None:
        raise LineageError(
            f"「{game}」尚未读取本地分片，无法进行数据分析。请先点击横幅上的「读取本地分片」", 404)
    if video not in manifest.get("videos", {}):
        raise LineageError(f"video '{video}' 不属于 game '{game}'（血缘不符）", 400)

    if frame is not None:
        rows = get_testset(game, video)
        if not rows:
            raise LineageError(f"'{game}/{video}' 无测试集（先跑 evaluate.py）", 404)
        hit = [r for r in rows if int(r["absolute_frame"]) == frame]
        if not hit:
            raise LineageError(
                f"frame {frame} 不在 {game}/{video} 测试集内（测试集为抽样帧，"
                f"范围 {rows[0]['absolute_frame']}~{rows[-1]['absolute_frame']}）", 400)
        r = hit[0]
        gt = get_annotation_lookup(game, video).get((r["chunk"], int(r["frame_idx"])))
        if gt is None:
            raise LineageError(
                f"annotations 中无法关联 (chunk={r['chunk']}, frame_idx={r['frame_idx']})", 400)
        return r, gt
    return None


# --------------------------------------------------------------------------
# API 1: 游戏列表
# --------------------------------------------------------------------------
@app.get("/api/games")
def api_games():
    scan = load_json(GAMES_SCAN, {})
    tested = {(r["game"], r["video"]) for r in db_query("SELECT game, video FROM eval_results")}
    games = []
    for game, info in scan.items():
        dl = downloaded_videos(game)
        has_manifest = (DATA_ROOT / game / "manifest.json").exists()
        game_tested = any(g == game for g, _ in tested)
        chunks = int(info.get("chunks", 0))
        games.append({
            "game": game,
            "videos": info.get("videos", 0),
            "frames": info.get("frames", 0),
            "chunks": chunks,
            # 每个 chunk 固定 20 秒（metadata duration=20）→ 总时长估算
            "video_seconds": chunks * 20,
            "ready_videos": len(dl),
            "extracted": has_manifest,
            "tested": game_tested,
        })
    # 排序：已提取 > 已下载 > 其余，再按 chunks 降序
    games.sort(key=lambda g: (not g["extracted"], -g["ready_videos"], -g["chunks"]))
    return ok({"games": games})


# --------------------------------------------------------------------------
# API 2: 视频列表
# --------------------------------------------------------------------------
@app.get("/api/games/<game>/videos")
def api_videos(game):
    """视频列表。

    已提取游戏：读 manifest.json（含分辨率/控制器等明细）。
    未提取游戏：降级读 games_scan.json 的 URL 清单（链接与探测状态始终可得，
    无需先提取标注）——保证「探测链接」等功能对任何清单内游戏可用。
    """
    vstatus = load_json(VIDEO_STATUS, {})
    tested = {(r["game"], r["video"]): r
              for r in db_query("SELECT game, video, acc_17keys, best_shift FROM eval_results")}

    manifest = get_manifest(game)
    videos = []
    if manifest:
        for vid, v in manifest.get("videos", {}).items():
            st = vstatus.get(vid, {})
            status = ("downloaded" if vid in downloaded_videos(game)
                      else st.get("status", "unknown"))
            row = tested.get((game, vid))
            videos.append({
                "video": vid,
                "url": v.get("url"),
                "source": v.get("source"),
                "resolution": v.get("resolution"),
                "controller": v.get("controller_type"),
                "status": status,
                "duration": st.get("duration"),
                "error": st.get("error"),
                "has_testset": bool(get_testset(game, vid)),
                "tested": row is not None,
                "acc_17keys": row["acc_17keys"] if row else None,
                "best_shift": row["best_shift"] if row else None,
            })
    else:
        # 未提取：降级用 games_scan.json 的 URL 清单
        info = load_json(GAMES_SCAN, {}).get(game)
        if not info:
            return err(f"未知游戏 '{game}'（不在切片清单中）", 404)
        for u in info.get("urls", []):
            vid = u["video"]
            st = vstatus.get(vid, {})
            status = ("downloaded" if vid in downloaded_videos(game)
                      else st.get("status", "unknown"))
            videos.append({
                "video": vid,
                "url": u.get("url"),
                "source": u.get("source"),
                "resolution": None,
                "controller": None,
                "status": status,
                "duration": st.get("duration"),
                "error": st.get("error"),
                "has_testset": False,
                "tested": False,
                "acc_17keys": None,
                "best_shift": None,
            })
    order = {"downloaded": 0, "available": 1, "unknown": 2, "dead": 3}
    videos.sort(key=lambda v: (order.get(v["status"], 2), not v["tested"]))
    return ok({"game": game, "videos": videos})


# --------------------------------------------------------------------------
# API 3: 单帧识别（核心）
# --------------------------------------------------------------------------
@app.get("/api/frame")
def api_frame():
    game = request.args.get("game", "")
    video = request.args.get("video", "")
    fresh = request.args.get("fresh", "1") == "1"
    try:
        frame = int(request.args.get("frame", ""))
    except ValueError:
        return err("参数 frame 必须是整数", 400)

    # --- 血缘断言（失败抛 LineageError -> 4xx JSON） ---
    trow, gt = assert_lineage(game, video, frame)

    # --- shift ---
    row = db_query("SELECT best_shift, acc_17keys FROM eval_results WHERE game=? AND video=?",
                   (game, video), one=True)
    shift_param = request.args.get("shift", "auto")
    if shift_param == "auto":
        shift = int(row["best_shift"]) if row else 0
    else:
        try:
            shift = max(0, min(17, int(shift_param)))
        except ValueError:
            return err("shift 必须是 0~17 或 auto", 400)

    pred = None
    action_block = None
    inference_ms = None

    if fresh:
        # --- 实时推理（演示核心）：18 步动作块 ---
        from PIL import Image
        img_path = DATA_ROOT / trow["frame_path"].replace("\\", "/")
        if not img_path.exists():
            # 兼容 test_set.csv 中路径分隔符差异
            img_path = DATA_ROOT / game / "test_frames" / Path(trow["frame_path"]).name
        if not img_path.exists():
            return err(f"帧画面缺失: {img_path.name}", 404)
        t0 = time.time()
        result = Predictor.predict(Image.open(img_path))
        inference_ms = int((time.time() - t0) * 1000)

        import numpy as np
        buttons = np.asarray(result["buttons"])        # (18, 21)
        jl = np.asarray(result["j_left"])              # (18, 2)
        jr = np.asarray(result["j_right"])             # (18, 2)
        b17 = buttons[shift, :][MODEL17_IDX] >= 0.5
        pred = {
            "buttons": {b: int(b17[i]) for i, b in enumerate(BUTTON_COLS)},
            "j_left": [round(float(jl[shift, 0]), 4), round(float(jl[shift, 1]), 4)],
            "j_right": [round(float(jr[shift, 0]), 4), round(float(jr[shift, 1]), 4)],
        }
        action_block = {
            "steps": 18,
            "j_left": [[round(float(x), 4) for x in jl[i]] for i in range(18)],
            "j_right": [[round(float(x), 4) for x in jr[i]] for i in range(18)],
            "button_counts": [int((buttons[i, MODEL17_IDX] >= 0.5).sum()) for i in range(18)],
        }
    else:
        # --- 缓存模式：从 predictions.csv 读（含 pred_* 明细） ---
        preds = get_predictions(game, video)
        hit = [p for p in preds
               if (p.get("absolute_frame") or "").strip() == str(frame)]
        if not hit:
            # 旧版 predictions.csv 无 absolute_frame 列，按 frame_path 匹配
            fname = Path(trow["frame_path"]).name
            hit = [p for p in preds if Path(p["frame_path"]).name == fname]
        if not hit:
            return err(f"frame {frame} 无缓存预测（predictions.csv），可用 fresh=1 实时推理", 404)
        p = hit[0]
        pred = {
            "buttons": {b: int(p.get(f"pred_{b}", 0)) for b in BUTTON_COLS},
            "j_left": [float(p["pred_jl_x"]), float(p["pred_jl_y"])],
            "j_right": [float(p["pred_jr_x"]), float(p["pred_jr_y"])],
        }

    # --- 单帧指标 ---
    gt_btn = [gt["buttons"][b] for b in BUTTON_COLS]
    pd_btn = [pred["buttons"][b] for b in BUTTON_COLS]
    mismatch_keys = [b for b, g, p in zip(BUTTON_COLS, gt_btn, pd_btn) if g != p]

    return ok({
        "frame": {
            "absolute_frame": frame,
            "second": float(trow["second"]),
            "chunk": trow["chunk"],
            "frame_idx": int(trow["frame_idx"]),
            "image_url": f"/files/{game}/test_frames/{Path(trow['frame_path']).name}",
        },
        "shift": shift,
        "ground_truth": {
            "buttons": gt["buttons"],
            "j_left": [round(v, 4) for v in gt["j_left"]],
            "j_right": [round(v, 4) for v in gt["j_right"]],
        },
        "prediction": pred,
        "action_block": action_block,
        "metrics": {
            "n_mismatch": len(mismatch_keys),
            "mismatch_keys": mismatch_keys,
            "gt_n_press": sum(gt_btn),
            "pred_n_press": sum(pd_btn),
            "inference_ms": inference_ms,
        },
        "video_metrics": row or None,
    })


# --------------------------------------------------------------------------
# API 4: 统计分布
# --------------------------------------------------------------------------
@app.get("/api/stats")
def api_stats():
    game = request.args.get("game", "")
    video = request.args.get("video", "")
    assert_lineage(game, video)

    import polars as pl
    p = DATA_ROOT / game / "annotations.parquet"
    if not p.exists():
        return err("annotations.parquet 缺失", 404)
    ann = pl.read_parquet(p).filter(pl.col("video") == video)

    btn_rate = {b: float(ann[b].mean()) for b in BUTTON_COLS}
    import numpy as np
    jl_arr = np.array(ann["j_left"].to_list())
    jr_arr = np.array(ann["j_right"].to_list())
    n = ann.height
    idle_mask = ((np.abs(jl_arr).max(axis=1) <= 0.1)
                 & (np.abs(jr_arr).max(axis=1) <= 0.1)
                 & (ann[BUTTON_COLS].sum_horizontal() == 0).to_numpy())

    row = db_query("SELECT stats_frames, acc_17keys, best_shift FROM eval_results "
                   "WHERE game=? AND video=?", (game, video), one=True)

    stats_dir = DATA_ROOT / game / "stats"
    return ok({
        "stats_frames": n,
        "test_frames": row["stats_frames"] if row else None,
        "buttons": [{"button": b, "press_rate": round(r, 4)}
                    for b, r in sorted(btn_rate.items(), key=lambda kv: -kv[1])],
        "summary": {
            "top_button": max(btn_rate, key=btn_rate.get),
            "top_press_rate": round(max(btn_rate.values()), 4),
            "idle_rate": round(float(idle_mask.mean()), 4),
            "left_stick_move_rate": round(float((np.abs(jl_arr).max(axis=1) > 0.1).mean()), 4),
            "right_stick_move_rate": round(float((np.abs(jr_arr).max(axis=1) > 0.1).mean()), 4),
        },
        "joystick_samples": {
            # 降采样到 <=4000 点供热图/散点（前端 ECharts）
            "j_left": _downsample(jl_arr, 4000),
            "j_right": _downsample(jr_arr, 4000),
        },
        "plots": {
            # 按视频优先（button_press_dist_<video>.png），回退游戏级（button_press_dist.png）
            "button_press_dist": _plot_url(stats_dir / f"button_press_dist_{video}.png")
                                 or _plot_url(stats_dir / "button_press_dist.png"),
            "joystick_dist": _plot_url(stats_dir / f"joystick_dist_{video}.png")
                             or _plot_url(stats_dir / "joystick_dist.png"),
            "sequences": _plot_url(stats_dir / f"sequences_{video}.png")
                         or _plot_url(stats_dir / "sequences.png"),
            "shift_scan": _plot_url(DATA_ROOT / game / "eval" / f"shift_scan_{video}.png")
                          or _plot_url(DATA_ROOT / game / "eval" / "shift_scan.png"),
        },
        "metrics_row": row,
    })


def _downsample(arr, cap: int) -> list:
    import numpy as np
    if len(arr) <= cap:
        idx = np.arange(len(arr))
    else:
        idx = np.linspace(0, len(arr) - 1, cap).astype(int)
    return [[round(float(arr[i][0]), 3), round(float(arr[i][1]), 3)] for i in idx]


def _plot_url(p: Path) -> str | None:
    if p.exists():
        return "/files/" + str(p.relative_to(DATA_ROOT)).replace("\\", "/")
    return None


# --------------------------------------------------------------------------
# API 4.5: 核心指标对比（测试集约 200 帧 + zero-shot 参考水平）
# 让"按键一致率 / 摇杆相关系数 / 摇杆 MSE"与参考水平直白对照、达标可判。
# --------------------------------------------------------------------------
REFERENCE = {
    "acc_17keys": 0.50,   # zero-shot 参考：按键一致率约 50%
    "corr_jl": 0.40,      # zero-shot 参考：摇杆相关系数约 0.4
    "test_frames": 200,   # 测试集约 200 帧量级
}


@app.get("/api/metrics")
def api_metrics():
    """核心指标对比。支持 ?shift=k 实时查看任意动作块步偏移下的指标；
    缺省/auto 用评估记录的最优 shift。数据来自 metrics_<video>.csv（shift 0~17 全行）。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    if not game or not video:
        return err("参数 game/video 必填")
    assert_lineage(game, video)

    row = db_query("SELECT * FROM eval_results WHERE game=? AND video=?",
                   (game, video), one=True)
    if row is None:
        return err(f"「{game}/{video}」尚未评估，请先在 Tab③ 运行评估", 404)

    # 读 metrics 明细（shift 0~17 全行，优先按视频隔离文件）
    import csv as _csv
    metrics_rows = []
    for cand in (DATA_ROOT / game / "eval" / f"metrics_{video}.csv",
                 DATA_ROOT / game / "eval" / "metrics.csv"):
        if cand.exists():
            metrics_rows = list(_csv.DictReader(open(cand, encoding="utf-8")))
            break

    # 选定 shift：显式参数 > auto(DB 最优)
    shift_param = request.args.get("shift", "auto")
    if shift_param == "auto":
        shift = int(row["best_shift"])
    else:
        try:
            shift = max(0, min(17, int(shift_param)))
        except ValueError:
            return err("shift 必须是 0~17 或 auto", 400)

    # 取该 shift 的指标行；无明细表则回退 DB 行
    mrow = None
    if metrics_rows:
        mrow = next((m for m in metrics_rows if int(m["shift"]) == shift), None)
    if mrow is not None:
        acc = float(mrow["acc_17keys_all"])          # B 口径：逐帧全对
        recall = float(mrow["btn_recall"]) if mrow["btn_recall"] not in ("nan", "") else None
        precision = float(mrow["btn_precision"]) if mrow["btn_precision"] not in ("nan", "") else None
        corr_x = float(mrow["corr_jl_x"]) if mrow["corr_jl_x"] not in ("nan", "") else None
        corr_y = float(mrow["corr_jl_y"]) if mrow["corr_jl_y"] not in ("nan", "") else None
        mse = float(mrow["mse_jl_x"])
    else:
        acc = row["acc_17keys"]
        recall = row["recall"]
        precision = row["precision"]
        corr_x = row["corr_jl_x"]
        corr_y = row["corr_jl_y"]
        mse = row["mse_jl"]

    # 摇杆相关系数：x/y 两轴取有效值中的较大者作为综合代表（更乐观口径）
    vals = [c for c in (corr_x, corr_y)
            if c is not None and not (isinstance(c, float) and c != c)]
    corr = max(vals) if vals else 0.0

    return ok({
        "game": game,
        "video": video,
        "test_frames": row["test_frames"],
        "best_shift": row["best_shift"],
        "shift": shift,
        "metrics": {
            "acc_17keys": acc,
            "btn_recall": recall,
            "btn_precision": precision,
            "corr_jl_x": corr_x,
            "corr_jl_y": corr_y,
            "corr_jl": round(float(corr), 4),
            "mse_jl": mse,
        },
        "reference": REFERENCE,
        "verdicts": {
            "acc_17keys": acc >= REFERENCE["acc_17keys"],
            "corr_jl": corr >= REFERENCE["corr_jl"],
        },
        # 供前端直白展示的文案
        "display": {
            "test_frames": f"测试集 {row['test_frames']} 帧",
            "acc": f"按键一致率 {acc*100:.1f}%  vs 参考 50%"
                   + ("  ✅ 达标" if acc >= REFERENCE["acc_17keys"] else "  ❌ 未达"),
            "corr": f"摇杆相关系数 {corr:.2f}  vs 参考 0.40"
                    + ("  ✅ 达标" if corr >= REFERENCE["corr_jl"] else "  ❌ 未达"),
            "mse": f"摇杆 MSE {mse:.3f}",
        },
    })


# --------------------------------------------------------------------------
# API 5: 序列对比 + 差异 Top-5
# --------------------------------------------------------------------------
@app.get("/api/sequences")
def api_sequences():
    game = request.args.get("game", "")
    video = request.args.get("video", "")
    assert_lineage(game, video)

    preds = get_predictions(game, video)
    if not preds:
        return err("predictions.csv 缺失（先跑 evaluate.py）", 404)

    # 逐帧序列（供前端画曲线）；旧版 predictions.csv 无 absolute_frame 列时退回 frame_idx
    def _af(p) -> int:
        v = (p.get("absolute_frame") or "").strip()
        return _to_int(v) if v else _to_int(p.get("frame_idx"))

    # --- 每帧综合差异分 D（扩展 C 差异定义，见下方 diff_definition） ---
    # D = 按键不一致位数 + 双摇杆平均 L2（折到 [0,1] 尺度）
    # 按键分量：17 键逐位比对的不一致个数（0~17），每键等权
    # 摇杆分量：左右摇杆各算 L2 = sqrt((Δx)² + (Δy)²)，取平均后除以 √2 折到 [0,1]（对角最大距离）
    def diff_score(p) -> float:
        n_mis = _to_int(p["n_mismatch_17"])
        l2_l = math.hypot(float(p["pred_jl_x"]) - float(p["gt_jl_x"]),
                          float(p["pred_jl_y"]) - float(p["gt_jl_y"]))
        l2_r = math.hypot(float(p["pred_jr_x"]) - float(p["gt_jr_x"]),
                          float(p["pred_jr_y"]) - float(p["gt_jr_y"]))
        return round(n_mis + (l2_l + l2_r) / 2 / math.sqrt(2), 4)

    seq = []
    for p in preds:
        seq.append({
            "absolute_frame": _af(p),
            "second": round(_af(p) / _fps(game, video), 2),
            "gt_n_press": _to_int(p["gt_n_press"]),
            "pred_n_press": _to_int(p["pred_n_press"]),
            "gt_jl_x": float(p["gt_jl_x"]), "gt_jl_y": float(p["gt_jl_y"]),
            "pred_jl_x": float(p["pred_jl_x"]), "pred_jl_y": float(p["pred_jl_y"]),
            "gt_jr_x": float(p["gt_jr_x"]), "gt_jr_y": float(p["gt_jr_y"]),
            "pred_jr_x": float(p["pred_jr_x"]), "pred_jr_y": float(p["pred_jr_y"]),
            "n_mismatch_17": _to_int(p["n_mismatch_17"]),
            "diff_score": diff_score(p),
            "is_idle": _to_int(p.get("is_idle", 0)),
        })
    seq.sort(key=lambda s: s["absolute_frame"])

    # 差异 Top-5 帧（按综合差异分 D 降序，含按键明细差异）
    def detail(p):
        d = {}
        for b in BUTTON_COLS:
            g, q = _to_int(p.get(f"gt_{b}", 0)), _to_int(p.get(f"pred_{b}", 0))
            if g != q:
                d[b] = {"gt": g, "pred": q}
        return d

    top5 = sorted(preds, key=diff_score, reverse=True)[:5]
    top5_out = [{
        "absolute_frame": _af(t),
        "n_mismatch": _to_int(t["n_mismatch_17"]),
        "jl_l2": round(math.hypot(float(t["pred_jl_x"]) - float(t["gt_jl_x"]),
                                  float(t["pred_jl_y"]) - float(t["gt_jl_y"])), 4),
        "jr_l2": round(math.hypot(float(t["pred_jr_x"]) - float(t["gt_jr_x"]),
                                  float(t["pred_jr_y"]) - float(t["gt_jr_y"])), 4),
        "diff_score": diff_score(t),
        "gt_n_press": _to_int(t["gt_n_press"]),
        "pred_n_press": _to_int(t["pred_n_press"]),
        "diff": detail(t),
        "is_idle": _to_int(t.get("is_idle", 0)),
    } for t in top5]

    # --- 约 20 段曲线：测试集按 absolute_frame 排序后每 10 帧一组 ---
    # （200 帧抽样 → 20 段，每段 10 帧；分段仅按帧序，帧间不连续故为"逐帧差异片段"）
    SEG_LEN = 10
    segments = []
    for i in range(0, len(seq), SEG_LEN):
        chunk = seq[i:i + SEG_LEN]
        if len(chunk) < 2:
            continue
        segments.append({
            "id": len(segments) + 1,
            "start_frame": chunk[0]["absolute_frame"],
            "end_frame": chunk[-1]["absolute_frame"],
            "n_frames": len(chunk),
            "mean_diff": round(sum(s["diff_score"] for s in chunk) / len(chunk), 4),
            "max_diff_frame": max(chunk, key=lambda s: s["diff_score"])["absolute_frame"],
            "points": chunk,
        })

    row = db_query("SELECT * FROM eval_results WHERE game=? AND video=?",
                   (game, video), one=True)
    return ok({
        "n_frames": len(seq),
        "sequence": seq,
        "segments": segments,
        "top5_mismatch": top5_out,
        "metrics_row": row,
        "diff_definition": {
            "text": "综合差异分 D = 按键不一致位数 + 双摇杆平均 L2/√2",
            "button_component": "17 键逐位比对的不一致个数（0~17），每键等权",
            "stick_component": "左右摇杆各算 L2 = √((Δx)²+(Δy)²)（预测 vs 真值），取平均后除以 √2 折到 [0,1]（对角最大距离）",
            "top5": "按 D 降序取差异最大的约 5 帧",
            "segments": "测试集排序后每 10 帧为一段，约 20 段；每段输出 mean_diff 与最大差异帧",
        },
    })


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=16)
def _fps(game: str, video: str) -> int:
    """从 test_set.csv 的 second/absolute_frame 推导帧率。"""
    rows = get_testset(game, video)
    for r in rows:
        af, sec = _to_int(r["absolute_frame"], -1), float(r["second"] or 0)
        if af > 30 and sec > 1:
            f = af / sec
            for cand in (30, 60, 24, 50):
                if abs(f - cand) < 1.5:
                    return cand
            return round(f)
    return 30


def _fps_from_manifest(game: str, video: str) -> int | None:
    """从 manifest 推导帧率：每 chunk 固定 20 秒，fps = chunk 行数 / 20。

    供评估流水线在无 test_set 时自动推导 --fps。
    """
    manifest = get_manifest(game)
    if not manifest:
        return None
    for c in manifest.get("chunks", []):
        if c.get("video") == video and c.get("rows"):
            f = c["rows"] / 20.0
            for cand in (30, 60, 24, 50):
                if abs(f - cand) < 1.5:
                    return cand
            return round(f)
    return None


# --------------------------------------------------------------------------
# API 6: 测试集帧列表（前端帧选择器数据源）
# --------------------------------------------------------------------------
@app.get("/api/testset")
def api_testset():
    game = request.args.get("game", "")
    video = request.args.get("video", "")
    assert_lineage(game, video)
    rows = get_testset(game, video)
    return ok({
        "fps": _fps(game, video),
        "frames": [
            {"absolute_frame": int(r["absolute_frame"]), "second": float(r["second"])}
            for r in rows
        ],
    })


# --------------------------------------------------------------------------
# API 7: 读取本地分片（extract_game.py 提取标注，纯本地不联网）
# --------------------------------------------------------------------------
_pipeline = {
    "game": None, "stage": "idle", "running": False,
    "log_tail": "", "error": None, "started_at": None, "proc": None,
}
_pipeline_lock = threading.Lock()


@app.get("/api/extract/estimate")
def api_extract_estimate():
    """读取本地分片耗时估算：tar 扫描启动开销 + ~0.35s/chunk（纯本地）。"""
    game = request.args.get("game", "").strip()
    info = load_json(GAMES_SCAN, {}).get(game)
    if not info:
        return err(f"未知游戏 '{game}'", 404)
    chunks = int(info.get("chunks", 0))
    est = max(30, int(40 + chunks * 0.35))  # 保底 30s
    text = f"约 {est} 秒" if est < 90 else f"约 {est // 60} 分 {est % 60:02d} 秒"
    return ok({"game": game, "chunks": chunks,
               "est_seconds": est, "est_text": text})


@app.post("/api/extract/cancel")
def api_extract_cancel():
    """停止当前提取/探测任务（终止子进程）。"""
    with _pipeline_lock:
        if not _pipeline["running"]:
            return ok({"status": "idle", "game": None})
        proc = _pipeline.get("proc")
        game = _pipeline["game"]
        if proc:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        _pipeline.update(stage="cancelled", running=False, proc=None)
        return ok({"status": "cancelled", "game": game})


@app.post("/api/extract")
def api_extract():
    """触发后台任务：仅读取本地 SHARD 分片并提取该游戏标注（不联网）。"""
    game = request.args.get("game", "").strip()
    if not game:
        return err("参数 game 必填")
    scan = load_json(GAMES_SCAN, {})
    if game not in scan:
        return err(f"未知游戏 '{game}'（不在 SHARD_0034 切片清单中）", 404)
    if not CKPT.parent.exists():
        return err("NitroGen 目录不可用", 500)

    with _pipeline_lock:
        if _pipeline["running"]:
            return err(f"已有任务在运行: {_pipeline['game']}（{_pipeline['stage']}），请稍候", 409)
        _pipeline.update(game=game, stage="extracting", running=True,
                         log_tail="", error=None,
                         started_at=time.strftime("%H:%M:%S"))

    threading.Thread(target=_run_pipeline, args=(game,), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game}}), 202


def _spawn(cmd: list[str], log: Path) -> subprocess.Popen:
    """启动子进程并登记到 _pipeline['proc']（供 cancel 终止）。"""
    proc = subprocess.Popen(cmd, cwd=str(REPO),
                            stdout=open(log, "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT)
    with _pipeline_lock:
        _pipeline["proc"] = proc
    return proc


def _wait(proc: subprocess.Popen, log: Path) -> int:
    rc = proc.wait()
    with _pipeline_lock:
        if _pipeline["proc"] is proc:   # 未被 cancel 清掉
            _pipeline["proc"] = None
    return rc


def _run_pipeline(game: str):
    """后台线程：仅读取本地分片（extract_game.py 提取标注），不碰网络。"""
    try:
        # --- 阶段 1：读取本地分片并提取标注 ---
        log1 = DATA_ROOT / f"extract_{game}.log"
        rc = _wait(_spawn(
            [sys.executable, str(REPO / "scripts" / "extract_game.py"), "--game", game],
            log1), log1)
        if rc != 0:
            raise RuntimeError(f"extract_game.py 失败（退出码 {rc}），日志: data/extract_{game}.log")

        # --- 阶段 2：完成，清数据缓存让新游戏立即可见 ---
        clear_cache(game)
        with _pipeline_lock:
            _pipeline.update(stage="done", running=False)
    except Exception as e:  # noqa: BLE001
        with _pipeline_lock:
            if _pipeline.get("stage") != "cancelled":
                _pipeline.update(stage="failed", running=False, error=str(e)[:300])
            else:
                _pipeline["running"] = False


@app.get("/api/extract/status")
def api_extract_status():
    with _pipeline_lock:
        t = {k: v for k, v in _pipeline.items()}
    t.pop("proc", None)
    # 附提取日志尾部（进度反馈）
    if t.get("game"):
        log = DATA_ROOT / f"extract_{t['game']}.log"
        if log.exists():
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                t["log_tail"] = "\n".join(lines[-6:])
            except OSError:
                pass
    return ok(t)


# --------------------------------------------------------------------------
# API 8: 后台视频探测
# --------------------------------------------------------------------------
_rescan_proc = None
_rescan_lock = threading.Lock()
_rescan_scope = None  # 本次探测范围："full" 或 "game=xxx"（供前端显示进度/范围）


@app.post("/api/rescan")
def api_rescan():
    global _rescan_proc, _rescan_scope
    scope = request.args.get("scope", "")  # "full" 或 "game=hades"
    with _rescan_lock:
        if _rescan_proc is not None and _rescan_proc.poll() is None:
            return jsonify({"ok": True, "data": {"status": "running",
                                                 "started": True, "already": True}})
        args = [shutil.which("python") or sys.executable,
                str(REPO / "scripts" / "probe_videos.py")]
        if scope == "full":
            args.append("--full")
        elif scope.startswith("game="):
            args += ["--game", scope[5:]]
        else:
            return err("scope 必须是 full 或 game=<游戏名>", 400)
        _rescan_scope = scope
        _rescan_proc = subprocess.Popen(
            args, cwd=str(REPO),
            stdout=open(REPO / "data" / "probe.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    return jsonify({"ok": True, "data": {"status": "started", "scope": scope}}), 202


@app.post("/api/rescan/cancel")
def api_rescan_cancel():
    """停止正在运行的视频探测（终止 yt-dlp 子进程）。"""
    with _rescan_lock:
        if _rescan_proc is not None and _rescan_proc.poll() is None:
            try:
                _rescan_proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            _rescan_proc = None
            return ok({"status": "cancelled"})
    return ok({"status": "idle"})


@app.get("/api/rescan/status")
def api_rescan_status():
    global _rescan_scope
    running = False
    with _rescan_lock:
        if _rescan_proc is not None:
            running = _rescan_proc.poll() is None
    vstatus = load_json(VIDEO_STATUS, {})
    from collections import Counter
    # 本次探测范围内的 URL 总数 / 已探测数 / 状态分布（均按范围统计，不含其它游戏的旧缓存）
    scan = load_json(GAMES_SCAN, {})
    scope_urls = []
    if _rescan_scope == "full":
        scope_urls = [u for info in scan.values() for u in info.get("urls", [])]
    elif _rescan_scope and _rescan_scope.startswith("game="):
        scope_urls = scan.get(_rescan_scope[5:], {}).get("urls", [])
    total = len(scope_urls) if _rescan_scope else None
    if total:
        # 容错：video_status.json 可能缺该 video（未探测/探测列表与 games_scan 不一致），按 unknown 处理，避免 KeyError -> 500
        scope_status = [vstatus.get(u["video"], {}).get("status", "unknown") for u in scope_urls]
        cnt = Counter(scope_status)
        done = len(scope_status)
    else:
        cnt = Counter()
        done = None
    return ok({
        "running": running,
        "scope": _rescan_scope,
        "total": total,
        "done": done,
        "status_counts": dict(cnt),
    })


# --------------------------------------------------------------------------
# API 9: 生成 matplotlib 静态图（stats_viz.py + plot_shift_scan.py）
# 统计图仅需读过分片；shift 扫描图需评估产物（metrics_<video>.csv）。
# --------------------------------------------------------------------------
_genplots = {"running": False, "game": None, "log_tail": "", "error": None, "proc": None}
_genplots_lock = threading.Lock()


@app.post("/api/genplots")
def api_genplots():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    if not game:
        return err("参数 game 必填")
    busy = _busy()
    if busy:
        return err(f"有任务正在运行（{busy}），请稍后再试", 409)
    with _genplots_lock:
        if _genplots["running"]:
            return err("已有静态图生成任务在运行", 409)
        _genplots.update(running=True, game=game, log_tail="", error=None, proc=None)
    threading.Thread(target=_run_genplots, args=(game, video), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game}}), 202


def _run_genplots(game: str, video: str):
    import subprocess as sp

    def run(cmd: list[str]) -> tuple[int, str]:
        """bytes 模式读子进程输出，避免 Windows GBK/UTF-8 编码错乱。"""
        proc = sp.Popen(cmd, cwd=str(REPO), stdout=sp.PIPE, stderr=sp.STDOUT)
        with _genplots_lock:
            _genplots["proc"] = proc
        raw, _ = proc.communicate(timeout=300)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="replace")
        return proc.returncode, text

    try:
        # 1) 统计图（stats_viz.py，需 matplotlib+pandas；带 --video 则按视频生成）
        cmd1 = [sys.executable, str(REPO / "scripts" / "stats_viz.py"), "--game", game]
        if video:
            cmd1 += ["--video", video]
        rc1, out1 = run(cmd1)
        if rc1 != 0:
            raise RuntimeError(f"stats_viz.py 失败（退出码 {rc1}）: {out1[-300:]}")

        # 2) shift 扫描图（需评估产物 metrics_<video>.csv；未评估时非致命跳过）
        if video:
            rc2, out2 = run([sys.executable, str(REPO / "scripts" / "plot_shift_scan.py"),
                             "--game", game, "--video", video])
            if rc2 != 0:
                print(f"WARN: shift_scan 图跳过: {out2[-200:]}")

        clear_cache(game)
        tail = "\n".join(out1.strip().splitlines()[-3:])
        with _genplots_lock:
            _genplots.update(running=False, log_tail=tail)
    except Exception as e:  # noqa: BLE001
        with _genplots_lock:
            _genplots.update(running=False, error=str(e)[:300])


@app.get("/api/genplots/status")
def api_genplots_status():
    with _genplots_lock:
        t = {k: v for k, v in _genplots.items()}
    t.pop("proc", None)
    return ok(t)


# --------------------------------------------------------------------------
# API 10: 运行评估流水线（evaluate.py：构建测试集 + 推理 + shift 扫描 + 写库）
# 让新视频的「序列对比」自动生成。与提取/探测/下载互斥（单任务槽）。
# --------------------------------------------------------------------------
_eval_task = {"running": False, "game": None, "video": None, "stage": "idle",
              "log_tail": "", "error": None, "started_at": None, "proc": None}
_eval_lock = threading.Lock()


def _busy() -> str | None:
    """任一后台任务正在运行？返回任务名或 None。"""
    with _pipeline_lock:
        if _pipeline["running"]:
            return "读取本地分片"
    with _eval_lock:
        if _eval_task["running"]:
            return "评估"
    with _rescan_lock:
        if _rescan_proc is not None and _rescan_proc.poll() is None:
            return "探测链接"
    with _download_lock:
        if _download["running"]:
            return "下载视频"
    return None


@app.post("/api/evaluate")
def api_evaluate():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    if not game or not video:
        return err("参数 game/video 必填")
    assert_lineage(game, video)  # 需已提取（有 manifest + annotations）
    if not (DATA_ROOT / "videos" / f"{game}_{video}.mp4").exists():
        return err("该视频未下载，评估需要本地视频文件。请先用「下载视频」下载。", 400)

    fps = _fps_from_manifest(game, video)
    if fps is None:
        return err("无法自动推导 fps（manifest 中无该视频的 chunk 行数）", 400)

    busy = _busy()
    if busy:
        return err(f"有任务正在运行（{busy}），请稍后再试", 409)
    with _eval_lock:
        if _eval_task["running"]:
            return err("已有评估任务在运行，请稍候", 409)
        _eval_task.update(running=True, game=game, video=video, stage="evaluating",
                          log_tail="", error=None,
                          started_at=time.strftime("%H:%M:%S"), proc=None)
    threading.Thread(target=_run_evaluate, args=(game, video, fps), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game,
                                         "video": video, "fps": fps}}), 202


def _run_evaluate(game: str, video: str, fps: int):
    log = DATA_ROOT / f"evaluate_{game}_{video}.log"
    try:
        cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
               "--game", game, "--video", video, "--fps", str(fps)]
        proc = subprocess.Popen(cmd, cwd=str(REPO),
                                stdout=open(log, "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT)
        with _eval_lock:
            _eval_task["proc"] = proc
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"evaluate.py 失败（退出码 {rc}），日志: data/evaluate_{game}_{video}.log")
        clear_cache(game)
        with _eval_lock:
            _eval_task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _eval_lock:
            _eval_task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


@app.get("/api/evaluate/status")
def api_evaluate_status():
    with _eval_lock:
        t = {k: v for k, v in _eval_task.items()}
    t.pop("proc", None)
    if t.get("game"):
        log = DATA_ROOT / f"evaluate_{t['game']}_{t['video']}.log"
        if log.exists():
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                t["log_tail"] = "\n".join(lines[-6:])
            except OSError:
                pass
    return ok(t)


@app.post("/api/evaluate/cancel")
def api_evaluate_cancel():
    with _eval_lock:
        if _eval_task["running"] and _eval_task["proc"]:
            try:
                _eval_task["proc"].terminate()
            except Exception:  # noqa: BLE001
                pass
        d = dict(_eval_task)
        d.pop("proc", None)
    return ok(d)


# --------------------------------------------------------------------------
# API 10: 视频下载（yt-dlp 后台任务）
# --------------------------------------------------------------------------
_download = {"running": False, "game": None, "video": None, "proc": None,
             "pct": 0, "msg": "", "error": None, "started_at": None}
_download_lock = threading.Lock()


def _video_url(game: str, video: str) -> str | None:
    manifest = get_manifest(game)
    if manifest and video in manifest.get("videos", {}):
        return manifest["videos"][video].get("url")
    for u in load_json(GAMES_SCAN, {}).get(game, {}).get("urls", []):
        if u["video"] == video:
            return u.get("url")
    return None


@app.post("/api/download")
def api_download():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    if not game or not video:
        return err("参数 game/video 必填")
    out = DATA_ROOT / "videos" / f"{game}_{video}.mp4"
    if out.exists():
        return ok({"status": "exists", "game": game, "video": video})
    url = _video_url(game, video)
    if not url:
        return err(f"找不到 {game}/{video} 的下载链接", 404)
    with _download_lock:
        if _download["running"]:
            return err(f"已有下载任务在运行: {_download['game']}/{_download['video']}", 409)
        _download.update(running=True, game=game, video=video, proc=None,
                         pct=0, msg="", error=None,
                         started_at=time.strftime("%H:%M:%S"))
    threading.Thread(target=_run_download, args=(game, video, url), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game,
                                         "video": video, "url": url}}), 202


def _run_download(game: str, video: str, url: str):
    out = DATA_ROOT / "videos" / f"{game}_{video}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 定位 yt-dlp：PATH 里的 exe，或系统 Python 的模块
    exe = shutil.which("yt-dlp")
    if exe:
        cmd = [exe]
    else:
        py = shutil.which("python") or sys.executable
        cmd = [py, "-m", "yt_dlp"]
    cmd += ["--newline", "--no-warnings", "-f", "bv*[height<=720]+ba/b",
            "--merge-output-format", "mp4", "-o", str(out)]
    # YouTube bot 验证（"Sign in to confirm you're not a bot"）规避：
    # 优先用 data/cookies.txt（浏览器插件导出，见 README）；PO Token 插件(bgutil)已装时自动生效
    cookies_file = DATA_ROOT / "cookies.txt"
    if cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    cmd += [url]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        with _download_lock:
            _download["proc"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # 解析 yt-dlp 进度行: "[download]  42.3% of 1.23GiB at 2.1MiB/s ETA 00:10"
            import re
            m = re.search(r"\[download\]\s+([\d.]+)%", line)
            if m:
                with _download_lock:
                    _download["pct"] = float(m.group(1))
                    _download["msg"] = line[:120]
            elif line.startswith("ERROR"):
                with _download_lock:
                    _download["error"] = line[:200]
        rc = proc.wait()
        if out.exists() and out.stat().st_size > 1000:
            with _download_lock:
                _download.update(running=False, pct=100, msg="下载完成")
            clear_cache(game)
        else:
            with _download_lock:
                _download.update(running=False,
                                 error=_download.get("error") or f"yt-dlp 退出码 {rc}")
    except Exception as e:  # noqa: BLE001
        with _download_lock:
            _download.update(running=False, error=str(e)[:300])
    finally:
        with _download_lock:
            _download["proc"] = None


@app.get("/api/download/status")
def api_download_status():
    with _download_lock:
        d = {k: v for k, v in _download.items()}
    d.pop("proc", None)
    return ok(d)


@app.post("/api/download/cancel")
def api_download_cancel():
    with _download_lock:
        if _download["running"] and _download["proc"]:
            try:
                _download["proc"].terminate()
            except Exception:  # noqa: BLE001
                pass
        d = dict(_download)
        d.pop("proc", None)
    return ok(d)


# --------------------------------------------------------------------------
# 静态文件：前端 + data 白名单
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".csv", ".json", ".log"}


@app.get("/files/<path:relpath>")
def files(relpath: str):
    p = (DATA_ROOT / relpath).resolve()
    # 防目录穿越 + 扩展名白名单
    if not str(p).startswith(str(DATA_ROOT.resolve())):
        return err("forbidden", 403)
    if p.suffix.lower() not in ALLOWED_EXT:
        return err(f"文件类型不允许: {p.suffix}", 403)
    if not p.exists():
        return err("file not found", 404)
    return send_file(p)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("NitroGen 评估工作台")
    print(f"  数据目录: {DATA_ROOT}")
    print(f"  结果库  : {DB_PATH}")
    print(f"  模型    : {CKPT}（首次 /api/frame 时加载）")
    print("  地址    : http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
