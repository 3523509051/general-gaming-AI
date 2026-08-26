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
import os
import re
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
    if not _NAME_RE.match(game or ""):  # 防 glob 通配/路径穿越
        return []
    return [p.stem[len(game) + 1:]
            for p in (DATA_ROOT / "videos").glob(f"{game}_*.mp4")]


# --------------------------------------------------------------------------
# 数据缓存（进程内，重启失效；标注 parquet 用 polars 惰性加载）
# --------------------------------------------------------------------------
# 安全名白名单：Unicode 字母/数字/下划线/连字符（\w 兼容 HF 数据集游戏名中的口音字符 é/ö 等），
# 且不含 . / \ 等路径穿越字符，防穿越性质不变。
_NAME_RE = re.compile(r"^[\w\-]+$")


def _safe_name(name: str, field: str = "参数") -> str:
    """外部输入安全名校验：只允许 Unicode 字母/数字/下划线/连字符，防止路径穿越。"""
    if not _NAME_RE.match(name or ""):
        raise LineageError(f"{field} 含非法字符（仅允许字母/数字/下划线/连字符）: {name!r}")
    return name


@lru_cache(maxsize=8)
def get_manifest(game: str) -> dict | None:
    # 非法 game 名视为无 manifest（安全失败，避免 DATA_ROOT / game 路径穿越）
    if not _NAME_RE.match(game or ""):
        return None
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

    - 入口先做外部输入安全名校验（防路径穿越）
    - manifest：video 必须属于该 game
    - test_set：请求帧必须在该 video 的测试集内
    - annotations：该帧必须能关联到标注真值
    成功且 frame 不为 None 时返回 (test_row, gt_dict)。
    """
    _safe_name(game, "game")
    _safe_name(video, "video")
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
    # 游戏 -> 所属切片（从各切片独立清单计算；同名游戏跨切片时归首个切片）
    game_shard: dict[str, str] = {}
    shards_dir = DATA_ROOT / "shards"
    if shards_dir.exists():
        for jf in sorted(shards_dir.glob("SHARD_*.games.json")):
            for g in load_json(jf, {}).keys():
                game_shard.setdefault(g, jf.name[:-len(".games.json")])  # SHARD_0000.games.json -> SHARD_0000
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
            "shard": game_shard.get(game, ""),   # 所属切片（前端按此分组）
            # 每个 chunk 固定 20 秒（metadata duration=20）→ 总时长估算
            "video_seconds": chunks * 20,
            "ready_videos": len(dl),
            "extracted": has_manifest,
            "tested": game_tested,
        })
    # 排序：按切片 → 已提取 > 已下载 > 其余，再按 chunks 降序
    games.sort(key=lambda g: (g["shard"], not g["extracted"], -g["ready_videos"], -g["chunks"]))
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
def _remote_infer_frame(url: str, game: str, video: str, frame: int, img_path: Path) -> dict:
    """无本地 GPU 时把单帧推理转发到远程 worker /infer_frame（base64 图片 POST）。

    返回与本地 Predictor.predict 相同的 dict：buttons/j_left/j_right（嵌套 list）。"""
    import base64
    import urllib.parse
    import urllib.request
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    body = urllib.parse.urlencode({
        "game": game, "video": video, "frame": str(frame), "image_b64": b64,
    }).encode("utf-8")
    req = urllib.request.Request(url + "/infer_frame", data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.load(r)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"远程推理失败（{url}）: {e}") from e
    if not j.get("ok"):
        raise RuntimeError(f"远程推理失败: {j.get('error')}")
    return j["data"]


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
        # 单帧推理：远程后端模式且 worker 在线时转发到 A100（无本地 GPU 的轻薄本可用）；
        # 否则本机 Predictor（需 NVIDIA GPU）
        _ensure_backend_loaded()
        if FT_BACKEND.get("mode") == "remote" and FT_BACKEND.get("remote_url"):
            result = _remote_infer_frame(FT_BACKEND["remote_url"], game, video, frame, img_path)
        else:
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

    # 最优 shift：优先从 metrics 明细现算（远程评估产物回传后本地 DB 可能未同步）
    best_shift = int(row["best_shift"])
    if metrics_rows:
        try:
            best_shift = int(max(metrics_rows, key=lambda m: float(m["acc_17keys_all"]))["shift"])
        except (ValueError, KeyError):  # noqa: PERF203
            pass
    # 选定 shift：显式参数 > auto(现算最优)
    shift_param = request.args.get("shift", "auto")
    if shift_param == "auto":
        shift = best_shift
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
        # 可选过滤 IDLE 口径：勾选后按键一致率用非 IDLE 帧统计（仅新版 evaluate.py 产物有该列）
        filter_idle = request.args.get("filter_idle", "0").strip() in ("1", "true", "yes")
        use_nonidle = filter_idle and mrow.get("acc_17keys_all_nonidle", "") not in ("", "nan")
        acc = float(mrow["acc_17keys_all_nonidle"] if use_nonidle else mrow["acc_17keys_all"])  # B 口径：逐帧全对
        recall = float(mrow["btn_recall"]) if mrow["btn_recall"] not in ("nan", "") else None
        precision = float(mrow["btn_precision"]) if mrow["btn_precision"] not in ("nan", "") else None
        corr_x = float(mrow["corr_jl_x"]) if mrow["corr_jl_x"] not in ("nan", "") else None
        corr_y = float(mrow["corr_jl_y"]) if mrow["corr_jl_y"] not in ("nan", "") else None
        mse = float(mrow["mse_jl_x"])
        n_frames_nonidle = int(float(mrow["n_frames_nonidle"])) if mrow.get("n_frames_nonidle", "") not in ("", "nan") else None
    else:
        filter_idle = False
        acc = row["acc_17keys"]
        recall = row["recall"]
        precision = row["precision"]
        corr_x = row["corr_jl_x"]
        corr_y = row["corr_jl_y"]
        mse = row["mse_jl"]
        n_frames_nonidle = None

    # 测试集帧数：优先取 metrics 明细的 n_frames（真实评估帧数，远程评估回传后也正确）；
    # 旧产物无该列时回退本地测试集 CSV 行数，再回退 DB 历史快照
    if mrow is not None and mrow.get("n_frames", "") not in ("", "nan"):
        try:
            real_frames = int(float(mrow["n_frames"]))
        except (ValueError, TypeError):  # noqa: PERF203
            real_frames = len(get_testset(game, video))
    else:
        try:
            real_frames = len(get_testset(game, video))
        except Exception:  # noqa: BLE001
            real_frames = int(row["test_frames"])

    # 摇杆相关系数：x/y 两轴取有效值中的较大者作为综合代表（更乐观口径）
    vals = [c for c in (corr_x, corr_y)
            if c is not None and not (isinstance(c, float) and c != c)]
    corr = max(vals) if vals else 0.0

    return ok({
        "game": game,
        "video": video,
        "test_frames": real_frames,
        "best_shift": best_shift,
        "shift": shift,
        "filter_idle": filter_idle,
        "n_frames_nonidle": n_frames_nonidle,
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
            "test_frames": f"测试集 {real_frames} 帧",
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
    """触发后台任务：仅读取本地 SHARD 分片并提取该游戏标注（不联网）。

    可选参数 shard：指定切片 tar.gz 路径（默认自动发现 HF 缓存全部 SHARD_*.tar.gz）。
    """
    game = request.args.get("game", "").strip()
    shard = request.args.get("shard", "").strip() or None
    if not game:
        return err("参数 game 必填")
    scan = load_json(GAMES_SCAN, {})
    if game not in scan:
        return err(f"未知游戏 '{game}'（不在切片清单中）", 404)
    if not CKPT.parent.exists():
        return err("NitroGen 目录不可用", 500)

    with _pipeline_lock:
        if _pipeline["running"]:
            return err(f"已有任务在运行: {_pipeline['game']}（{_pipeline['stage']}），请稍候", 409)
        _pipeline.update(game=game, stage="extracting", running=True,
                         log_tail="", error=None,
                         started_at=time.strftime("%H:%M:%S"))

    threading.Thread(target=_run_pipeline, args=(game, shard), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game}}), 202


@app.get("/api/shaders")
def api_shaders():
    """列出项目 data/shards/ 中的全部切片（SHARD_*.tar.gz）——供 Web 端选择要导入的切片。

    每个切片附加其独立清单（data/shards/<SHARD>.games.json）的游戏数与视频数，
    让各切片内容分开显示（不合并）。
    """
    shards_dir = DATA_ROOT / "shards"
    found = []
    if shards_dir.exists():
        for tar in sorted(shards_dir.glob("SHARD_*.tar.gz")):
            try:
                size_mb = tar.stat().st_size / 1024 / 1024
            except OSError:
                size_mb = 0
            # 该切片的独立清单（若已扫描）：分开显示其游戏/视频数
            games_json = shards_dir / f"{tar.name[:-7]}.games.json"  # SHARD_0000.tar.gz -> SHARD_0000
            games_count = None
            videos_count = None
            if games_json.exists():
                g = load_json(games_json, {})
                games_count = len(g)
                videos_count = sum(int(v.get("videos", 0)) for v in g.values())
            found.append({
                "name": tar.name,
                "path": str(tar),
                "size_mb": round(size_mb, 1),
                "scanned": games_count is not None,
                "games_count": games_count,
                "videos_count": videos_count,
            })
    return ok({"shards": found})


@app.post("/api/scan_shard")
def api_scan_shard():
    """扫描指定切片，生成该切片的独立清单（data/shards/<SHARD>.games.json）。

    多切片分开显示：每个切片单独一份清单，不合并到 games_scan.json。
    参数 shard：tar.gz 完整路径。
    """
    shard = request.args.get("shard", "").strip()
    if not shard:
        return err("参数 shard 必填")
    if not os.path.exists(shard):
        return err(f"切片文件不存在: {shard}", 404)
    py = sys.executable
    scan_script = REPO / "scripts" / "scan_shard.py"
    try:
        # --merge：扫描生成该切片独立清单的同时，合并进 games_scan.json（全局游戏列表）。
        # 不加 --merge 会把 games_scan.json 覆盖成单切片内容，导致 Web 下拉游戏变少。
        r = subprocess.run(
            [py, str(scan_script), "--shard", shard, "--merge",
             "--out", str(GAMES_SCAN)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600,
        )
        if r.returncode != 0:
            return err(f"scan_shard.py 失败: {(r.stderr or r.stdout)[-300:]}", 500)
        clear_cache("")
        return ok({"scanned": True, "tail": (r.stdout or "")[-200:]})
    except subprocess.TimeoutExpired:
        return err("扫描超时（切片过大）", 500)


@app.post("/api/upload_shard")
def api_upload_shard():
    """拖拽上传切片：识别 SHARD_*.tar.gz 并复制到 data/shards/（不覆盖同名）。"""
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("未收到文件", 400)
    name = os.path.basename(f.filename)  # 防路径穿越
    if not (name.startswith("SHARD_") and name.endswith(".tar.gz")):
        return err(f"文件名应为 SHARD_*.tar.gz，收到: {name}", 400)
    shards_dir = DATA_ROOT / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    dest = shards_dir / name
    if dest.exists():
        return err(f"切片已存在: {name}（data/shards/），无需重复导入", 409)
    try:
        f.save(str(dest))
    except Exception as e:  # noqa: BLE001
        return err(f"保存失败: {str(e)[:150]}", 500)
    return ok({"name": name, "path": str(dest)})


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


def _run_pipeline(game: str, shard: str | None = None):
    """后台线程：仅读取本地分片（extract_game.py 提取标注），不碰网络。

    shard 可选：指定切片 tar.gz 路径，None 时 extract_game.py 自动发现全部本地切片。
    """
    try:
        # --- 阶段 1：读取本地分片并提取标注 ---
        log1 = DATA_ROOT / f"extract_{game}.log"
        cmd = [sys.executable, str(REPO / "scripts" / "extract_game.py"), "--game", game]
        if shard:
            cmd += ["--shard", shard]
        rc = _wait(_spawn(cmd, log1), log1)
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
        args = [sys.executable,   # 强制 venv python，避免 shutil.which 解析到系统 python
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
    global _rescan_proc, _rescan_scope
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
              "log_tail": "", "error": None, "started_at": None, "proc": None,
              "test_size": None}
_eval_lock = threading.Lock()


def _busy() -> str | None:
    """任一后台任务正在运行？返回任务名或 None。"""
    with _pipeline_lock:
        if _pipeline["running"]:
            return "读取本地分片"
    with _eval_lock:
        if _eval_task["running"]:
            return "评估"
    with _ft_lock:
        if _ft_task["running"]:
            return "微调"
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
    # 可选 test_size：前端评估引导可调帧数（默认 evaluate.py 的 200）
    test_size_raw = request.args.get("test_size", "").strip()
    test_size = None
    if test_size_raw:
        try:
            test_size = int(test_size_raw)
            if test_size < 10 or test_size > 5000:
                return err("test_size 应在 10~5000 之间", 400)
        except ValueError:
            return err("test_size 必须是整数", 400)
    # 过滤 IDLE 口径 + 多次评估取均值（与微调面板一致）
    filter_idle = request.args.get("filter_idle", "0").strip() in ("1", "true", "yes")
    repeats_raw = request.args.get("eval_repeats", "").strip()
    eval_repeats = 1
    if repeats_raw:
        try:
            eval_repeats = int(repeats_raw)
            if eval_repeats < 1 or eval_repeats > 5:
                return err("eval_repeats 应在 1~5 之间", 400)
        except ValueError:
            return err("eval_repeats 必须是整数", 400)
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
                          log_tail="", error=None, test_size=test_size,
                          filter_idle=filter_idle, eval_repeats=eval_repeats,
                          started_at=time.strftime("%H:%M:%S"), proc=None)
    # 按微调后端分发：remote 时评估也放 A100（worker /eval_base），本地仅数据检查/上传/回传
    _ensure_backend_loaded()
    backend = FT_BACKEND["mode"]
    remote_url = FT_BACKEND["remote_url"] if backend == "remote" else ""
    if backend == "remote":
        threading.Thread(target=_run_evaluate_remote,
                         args=(game, video, test_size, remote_url, eval_repeats), daemon=True).start()
    else:
        threading.Thread(target=_run_evaluate,
                         args=(game, video, fps, test_size, eval_repeats), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game,
                                         "video": video, "fps": fps,
                                         "test_size": test_size,
                                         "backend": backend,
                                         "filter_idle": filter_idle,
                                         "eval_repeats": eval_repeats}}), 202


def _run_evaluate(game: str, video: str, fps: int, test_size: int | None = None,
                  eval_repeats: int = 1):
    reps = max(1, eval_repeats)
    try:
        for i in range(reps):
            rtag = "ng" if reps == 1 else f"ng_s{i}"
            log = DATA_ROOT / (f"evaluate_{game}_{video}.log" if reps == 1
                               else f"evaluate_{game}_{video}_s{i}.log")
            cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
                   "--game", game, "--video", video, "--fps", str(fps),
                   "--tag", rtag, "--infer-seed", str(i)]
            if test_size is not None:
                cmd += ["--test-size", str(test_size)]
            proc = subprocess.Popen(cmd, cwd=str(REPO),
                                    stdout=open(log, "w", encoding="utf-8"),
                                    stderr=subprocess.STDOUT)
            with _eval_lock:
                _eval_task["proc"] = proc
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"evaluate.py 失败（退出码 {rc}，第 {i+1}/{reps} 次），日志: data/{log.name}")
        # 多副本时复制第一次结果为主副本（metrics_*_ng.csv，供 /api/metrics 等读）
        if reps > 1:
            first = DATA_ROOT / game / "eval" / f"metrics_{video}_ng_s0.csv"
            main = DATA_ROOT / game / "eval" / f"metrics_{video}_ng.csv"
            if first.exists():
                shutil.copy(first, main)
        clear_cache(game)
        with _eval_lock:
            _eval_task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _eval_lock:
            _eval_task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


def _run_evaluate_remote(game: str, video: str, test_size: int | None, remote_url: str,
                         eval_repeats: int = 1):
    """远程评估：数据检查/自动上传 → worker /eval_base 评估 → 回传 ng CSV。"""
    log = DATA_ROOT / f"evaluate_{game}_{video}.log"

    def _log(msg: str):
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        _log(f"=== 远程评估（A100 {remote_url}，test_size={test_size or 200}） ===")
        # 1) 远程数据齐备性：缺则自动上传（视频+manifest+annotations）
        with _eval_lock:
            _eval_task.update(stage="uploading")
        missing = remote_data_check(remote_url, game, video)
        if missing:
            _log(f"远程缺失数据: {missing}，开始自动上传 ...")
            ssh_cfg = FT_BACKEND.get("ssh") or {}
            if not ssh_cfg.get("password"):
                raise RuntimeError("远程缺数据且未配置 SSH 凭据（后端设置里填 SSH 密码后重试）")
            remote_upload_data(ssh_cfg, game, video, log_fn=_log)
            _log("数据上传完成")
        else:
            _log("远程数据齐备")
        # 2) 远程评估（零样本，tag=ng，与微调链基线同副本；eval_repeats>1 生成 _s{i} 副本）
        code, resp = remote_eval_base(remote_url, game, video, test_size, eval_repeats=max(1, eval_repeats))
        if code != 202:
            raise RuntimeError(f"远程评估启动失败（{code}）: {resp}")
        # 3) 轮询到完成（阶段同步到 _eval_task）
        st = _poll_remote_until_done(remote_url, game, video, 0, _log, task=_eval_task)
        if st.get("stage") != "done":
            raise RuntimeError(f"远程评估未成功完成: {st.get('stage')} / {st.get('error')}")
        # 4) 回传 ng 副本（repeats>1 时自动拉全 _s{i}）
        eval_dir = DATA_ROOT / game / "eval"
        remote_fetch_csv(remote_url, game, video, 0,
                         eval_dir, log_fn=_log, tag="ng",
                         repeats=max(1, eval_repeats))
        # 5) 零样本产物提升为主副本：/api/metrics、/api/sequences 读的是
        #    metrics_<video>.csv / predictions_<video>.csv，不回传主副本会显示旧数据
        for stem in (f"metrics_{video}", f"predictions_{video}"):
            src = eval_dir / f"{stem}_ng.csv"
            dst = eval_dir / f"{stem}.csv"
            if src.exists():
                shutil.copy(src, dst)
                _log(f"{dst.name} 已更新为主副本（{src.name}）")
        # 6) 回传测试集：远程评估按 --test-size 构建的 test_set_<video>.csv，
        #    不同步会导致本地帧浏览（/api/testset）与 1000 帧评估口径不一致；
        #    拉取失败不阻塞完成（前端回退旧测试集）
        try:
            import urllib.parse as _up
            import urllib.request as _ur
            (DATA_ROOT / game).mkdir(parents=True, exist_ok=True)
            qs = _up.urlencode({"game": game, "video": video})
            with _ur.urlopen(f"{remote_url}/testset_csv?{qs}", timeout=30) as _r:
                (DATA_ROOT / game / f"test_set_{video}.csv").write_text(
                    _r.read().decode("utf-8"), encoding="utf-8")
            _log(f"test_set_{video}.csv 已回传（远程测试集，{test_size or 200} 帧口径）")
        except Exception as e:  # noqa: BLE001
            _log(f"测试集回传失败（忽略）: {str(e)[:150]}")
        _log("评估结果已回传")
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
# API 11: 微调自动化对照闭环（扩展 A：finetune.py + evaluate.py --ckpt 串联）
# 任务链：补零样本基线副本（如缺）→ 微调 → 微调权重评估 → compare 接口取对照
# --------------------------------------------------------------------------
_ft_task = {"running": False, "game": None, "video": None, "stage": "idle",
            "samples": None, "epochs": None, "batch": None, "gpu_note": "",
            "backend": "local", "remote_url": "", "out": None,
            "test_size": None, "filter_idle": False, "eval_repeats": 1,
            "log_tail": "", "error": None, "started_at": None, "proc": None}
_ft_lock = threading.Lock()

# 微调后端：local=本机 GPU，remote=<http://host:port>=远程 A100 worker
# 通过 /api/finetune/backend 运行时切换；初始为空，GET 时读持久化配置
# ssh 字段（host/port/user/password）仅用于缺数据时自动上传，存 data/remote_worker.json（不入库）
FT_BACKEND = {"mode": "", "remote_url": "", "ssh": {}}
FT_REMOTE_CONFIG = Path(os.environ.get("FT_REMOTE_CONFIG", str(REPO / "data" / "remote_worker.json")))


def _load_remote_config() -> dict:
    """读远程 worker 配置（含 ssh 凭据），失败返回 {}。"""
    try:
        if FT_REMOTE_CONFIG.exists():
            return json.load(open(FT_REMOTE_CONFIG, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _ensure_backend_loaded():
    """确保 FT_BACKEND 已从持久化配置恢复（Web 重启后内存态为空时）。"""
    if not FT_BACKEND["mode"] and not FT_BACKEND["remote_url"]:
        cfg = _load_remote_config()
        if cfg.get("url"):
            FT_BACKEND["mode"] = "remote"
            FT_BACKEND["remote_url"] = cfg["url"]
            FT_BACKEND["ssh"] = cfg.get("ssh") or {}


def remote_worker_health(url: str) -> dict | None:
    """探测远程 worker /health，失败返回 None。"""
    try:
        import urllib.request
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return json.load(r).get("data")
    except Exception:  # noqa: BLE001
        return None


def remote_worker_status(url: str) -> dict | None:
    """查远程 worker /status，失败返回 None。"""
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/status", timeout=10) as r:
            return json.load(r).get("data", {})
    except Exception:  # noqa: BLE001
        return None


def _poll_remote_until_done(url: str, game: str, video: str, samples: int,
                            log_fn, task: dict | None = None) -> dict:
    """轮询远程任务到结束，返回最终 status dict（期间日志打到 log_fn）。
    task 指定同步阶段到哪个本地任务（默认微调 _ft_task；评估传 _eval_task）。"""
    import urllib.request
    task = _ft_task if task is None else task
    lock = _eval_lock if task is _eval_task else _ft_lock
    st = {}
    while True:
        try:
            with urllib.request.urlopen(url + "/status", timeout=10) as r:
                st = json.load(r).get("data", {})
            # 同步远程真实阶段到本地（前端据此显示 1/2 微调 or 2/2 评估）
            r_stage = st.get("stage")
            if r_stage in ("finetuning", "evaluating", "baseline", "uploading"):
                with lock:
                    task["stage"] = r_stage
            tail = st.get("log_tail") or ""
            if tail:
                last = tail.splitlines()[-1]
                if last and not last.startswith("["):
                    log_fn(f"[remote] {last}")
        except Exception as e:  # noqa: BLE001
            log_fn(f"[remote] status 轮询异常: {e}")
        if not st.get("running"):
            return st
        if st.get("stage") == "failed":
            raise RuntimeError(f"远程微调失败: {st.get('error')}")
        time.sleep(10)


def remote_start(url: str, params: dict) -> tuple[int, dict | str]:
    """调远程 worker /start，返回 (status_code, data或error)。"""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url + "/start?" + qs, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except Exception as e:  # noqa: BLE001
        return 500, str(e)


def remote_has_ckpt(url: str, out: str) -> dict:
    """远程权重信息：{"exists": bool, "meta": dict|None}（meta 含训练参数，供跳过微调校验）。"""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"out": out})
    try:
        with urllib.request.urlopen(f"{url}/has_ckpt?{qs}", timeout=8) as r:
            d = json.load(r).get("data", {})
            return {"exists": bool(d.get("exists")), "meta": d.get("meta")}
    except Exception:  # noqa: BLE001
        return {"exists": False, "meta": None}


def _meta_matches(meta, samples: int, epochs: int, batch: int) -> bool:
    """权重 meta 与当前训练参数一致才可跳过微调；无 meta（旧权重）保守跳过以兼容。"""
    if not meta:
        return True
    return (meta.get("samples") == samples and meta.get("epochs") == epochs
            and meta.get("batch") == batch)


def _ckpt_matches(ckpt_path: Path, samples: int, epochs: int, batch: int) -> bool:
    """本地权重存在且训练参数一致才跳过微调；参数变了（如 epochs 1→2）需重训覆盖。"""
    if not ckpt_path.exists():
        return False
    try:
        meta = json.loads(Path(str(ckpt_path) + ".meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return True  # 旧权重无 meta：跳过（兼容）；想重训可删除权重文件
    return _meta_matches(meta, samples, epochs, batch)


def _csv_n_frames(path: Path) -> int | None:
    """读 metrics CSV 的 n_frames（无文件/无该列返回 None）。"""
    try:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        v = rows[0].get("n_frames") if rows else None
        return int(float(v)) if v not in (None, "", "nan") else None
    except Exception:  # noqa: BLE001
        return None


def remote_eval_base(url: str, game: str, video: str, test_size: int | None,
                     eval_repeats: int = 1) -> tuple[int, dict | str]:
    """调远程 worker /eval_base 跑零样本基线评估（A100），返回 (status_code, data或error)。"""
    import urllib.parse
    import urllib.request
    params = {"game": game, "video": video}
    if test_size:
        params["test_size"] = str(test_size)
    if eval_repeats and eval_repeats > 1:
        params["eval_repeats"] = str(eval_repeats)
    qs = urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url + "/eval_base?" + qs, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except Exception as e:  # noqa: BLE001
        return 500, str(e)


def remote_fetch_weight(url: str, out: str, dest: Path, log_fn=None) -> None:
    """从远程 worker /download 分块拉微调权重到本地 dest（避免大文件整读挂起）。"""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"out": out})
    with urllib.request.urlopen(f"{url}/download?{qs}", timeout=3600) as r:
        total = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if log_fn and total % (200 * 1024 * 1024) == 0:
                    log_fn(f"已下载 {total / 1024 / 1024:.0f} MB")
        if log_fn:
            log_fn(f"权重下载完成（{total / 1024 / 1024:.0f} MB）")


def remote_fetch_csv(url: str, game: str, video: str, samples: int,
                     dest_dir: Path, log_fn=None, tag: str | None = None,
                     repeats: int = 1) -> str:
    """从远程 worker 拉评估 CSV 回本地。

    容错：先探测主副本是否存在；若远程只有 _s{i} 副本（如 recover 接管后
    eval_repeats 与原始请求不一致），自动逐个探测 _s0.._sN 拉全副本。
    """
    import urllib.error
    import urllib.parse
    import urllib.request
    tag = tag or f"ng_ft_{samples}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _metrics_exists(rtag: str) -> bool:
        qs = urllib.parse.urlencode({"game": game, "video": video, "tag": rtag})
        try:
            with urllib.request.urlopen(f"{url}/metrics_csv?{qs}", timeout=15) as r:
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    # 决定要拉哪些副本：主副本存在则「主副本 + 全部 _s{i} 副本」（eval_repeats>1 供均值±std）；
    # 主副本缺失则探测 _s0.._sN（如 recover 接管后 eval_repeats 与原始请求不一致）
    if _metrics_exists(tag):
        targets = [tag]
        for i in range(5):
            t = f"{tag}_s{i}"
            if _metrics_exists(t):
                targets.append(t)
            else:
                break
    else:
        targets = []
        for i in range(5):
            t = f"{tag}_s{i}"
            if _metrics_exists(t):
                targets.append(t)
            else:
                break
    if not targets:
        raise RuntimeError(f"远程无 {tag} 的评估副本（metrics_csv 均 404）")

    for rtag in targets:
        qs = urllib.parse.urlencode({"game": game, "video": video, "tag": rtag})
        with urllib.request.urlopen(f"{url}/metrics_csv?{qs}", timeout=120) as r:
            (dest_dir / f"metrics_{video}_{rtag}.csv").write_text(
                r.read().decode("utf-8"), encoding="utf-8")
        if log_fn:
            log_fn(f"metrics_{video}_{rtag}.csv 已回传")
        if rtag == tag:  # 主副本才拉 predictions（重复评估只回传 metrics，对照表不读预测明细）
            try:
                with urllib.request.urlopen(f"{url}/predictions_csv?{qs}", timeout=300) as r:
                    (dest_dir / f"predictions_{video}_{rtag}.csv").write_text(
                        r.read().decode("utf-8"), encoding="utf-8")
                if log_fn:
                    log_fn(f"predictions_{video}_{rtag}.csv 已回传")
            except Exception as e:  # noqa: BLE001
                if log_fn:
                    log_fn(f"predictions 拉取失败（可选，不影响对照）: {e}")
    return tag


def remote_data_check(url: str, game: str, video: str) -> list[str]:
    """调远程 worker /data_check，返回缺失数据列表（空=齐备）。失败按全缺失处理。"""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"game": game, "video": video})
    try:
        with urllib.request.urlopen(f"{url}/data_check?{qs}", timeout=10) as r:
            d = json.load(r).get("data", {})
            return d.get("missing", [])
    except Exception:  # noqa: BLE001
        return ["video", "manifest", "annotations"]


def remote_upload_data(ssh_cfg: dict, game: str, video: str, log_fn=None) -> int:
    """SFTP 自动上传该游戏数据（视频+manifest+annotations）到远程，返回总字节数。"""
    import paramiko
    host = ssh_cfg.get("host")
    port = int(ssh_cfg.get("port", 22))
    user = ssh_cfg.get("user", "root")
    pwd = ssh_cfg.get("password", "")
    if not host or not pwd:
        raise RuntimeError("SSH 配置不完整（host/password），请在后端设置中填写")
    remote_base = "/root/workspace/nitrogen_worker"
    srcs = [
        (DATA_ROOT / "videos" / f"{game}_{video}.mp4", f"{remote_base}/data/videos/{game}_{video}.mp4"),
        (DATA_ROOT / game / "manifest.json", f"{remote_base}/data/{game}/manifest.json"),
        (DATA_ROOT / game / "annotations.parquet", f"{remote_base}/data/{game}/annotations.parquet"),
    ]
    missing = [p.name for p, _ in srcs if not p.exists()]
    if missing:
        raise RuntimeError(f"本地数据缺失，无法上传: {missing}")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pwd, timeout=15)
    c.exec_command(f"mkdir -p {remote_base}/data/videos {remote_base}/data/{game}")
    time.sleep(0.5)
    sftp = c.open_sftp()
    total = 0
    for p, remote in srcs:
        mb = p.stat().st_size / 1024 / 1024
        if log_fn:
            log_fn(f"上传 {p.name}（{mb:.0f} MB）...")
        sftp.put(str(p), remote)
        total += p.stat().st_size
    sftp.close()
    c.close()
    return total


def auto_batch_size(remote_url: str = "") -> tuple[int, str]:
    """按后端显存自动选微调 batch；remote 优先用 worker /health 的显存。"""
    if remote_url:
        h = remote_worker_health(remote_url)
        if h and h.get("mem_gb"):
            mem = h["mem_gb"]
            if mem < 6:
                b = 1
            elif mem < 8:
                b = 2
            elif mem < 16:
                b = 4
            else:
                b = 8
            return b, f"{h.get('gpu','远程GPU')}（{mem:.1f} GB）→ 自动 batch={b}"
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            name = torch.cuda.get_device_name(0)
            if mem < 6:
                b = 1
            elif mem < 8:
                b = 2
            elif mem < 16:
                b = 4
            else:
                b = 8
            return b, f"{name}（{mem:.1f} GB）→ 自动 batch={b}"
    except Exception:  # noqa: BLE001
        pass
    return 2, "未检测到 GPU，保守使用 batch=2（训练可能很慢）"
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            name = torch.cuda.get_device_name(0)
            if mem < 6:
                b = 1
            elif mem < 8:
                b = 2
            elif mem < 16:
                b = 4
            else:
                b = 8
            return b, f"{name}（{mem:.1f} GB）→ 自动 batch={b}"
    except Exception:  # noqa: BLE001
        pass
    return 2, "未检测到 GPU，保守使用 batch=2（训练可能很慢）"


def _ft_stage(cmd: list[str], log: Path) -> int:
    """跑任务链中的一个子阶段（追加写同一日志），记录 proc 供取消。"""
    proc = subprocess.Popen(cmd, cwd=str(REPO),
                            stdout=open(log, "a", encoding="utf-8"),
                            stderr=subprocess.STDOUT)
    with _ft_lock:
        _ft_task["proc"] = proc
    return proc.wait()


def _run_finetune_chain(game: str, video: str, fps: int,
                        samples: int, epochs: int, batch: int,
                        backend: str = "local", remote_url: str = "",
                        test_size: int | None = None, filter_idle: bool = False,
                        eval_repeats: int = 1):
    tag = f"ng_ft_{samples}"
    ckpt_rel = f"NitroGen/{tag}.pt"
    log = DATA_ROOT / f"finetune_{game}_{video}.log"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"[chain] game={game} video={video} samples={samples} "
                f"epochs={epochs} batch={batch} backend={backend} "
                f"test_size={test_size} filter_idle={filter_idle} "
                f"eval_repeats={eval_repeats} out={ckpt_rel}\n")

    def _log(msg: str):
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        # 阶段 0：零样本基线副本（缺失，或测试集帧数与本次不一致时重跑；一次约 2 分钟）
        base_csv = DATA_ROOT / game / "eval" / f"metrics_{video}_ng.csv"
        base_n = _csv_n_frames(base_csv)
        if base_n is not None and (test_size is None or base_n == test_size):
            _log(f"基线副本已存在（n_frames={base_n}），复用")
        else:
            with _ft_lock:
                _ft_task.update(stage="baseline")
            if base_n is not None:
                _log(f"=== 阶段 0/2：基线测试集帧数变化（{base_n} → {test_size}），重跑基线 ===")
            else:
                _log("=== 阶段 0/2：补零样本基线副本（--tag ng） ===")
            if backend == "remote":
                # 零样本基线也放 A100：远程 worker /eval_base 评估，回传 ng CSV
                _ensure_backend_loaded()
                _log(f"基线评估放 A100（{remote_url}，test_size={test_size or 200}）")
                code, resp = remote_eval_base(remote_url, game, video, test_size)
                if code != 202:
                    raise RuntimeError(f"远程基线评估启动失败（{code}）: {resp}")
                st = _poll_remote_until_done(remote_url, game, video, 0, _log)
                if st.get("stage") != "done":
                    raise RuntimeError(f"远程基线评估未成功完成: {st.get('stage')} / {st.get('error')}")
                remote_fetch_csv(remote_url, game, video, 0,
                                 DATA_ROOT / game / "eval", log_fn=_log, tag="ng")
                _log("基线评估结果已回传")
            else:
                base_cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
                            "--game", game, "--video", video, "--fps", str(fps),
                            "--tag", "ng"]
                if test_size is not None:
                    base_cmd += ["--test-size", str(test_size)]
                rc = _ft_stage(base_cmd, log)
                if rc != 0:
                    raise RuntimeError(f"基线评估失败（退出码 {rc}），日志: data/finetune_{game}_{video}.log")

        # 阶段 1：微调（按后端分发）
        with _ft_lock:
            _ft_task.update(stage="finetuning")
        if backend == "remote":
            # 阶段 0.5：数据检查 + 自动上传（远程缺视频/标注时，仅首次慢）
            _ensure_backend_loaded()
            _log("=== 检查远程数据 ===")
            missing = remote_data_check(remote_url, game, video)
            if missing:
                with _ft_lock:
                    _ft_task.update(stage="uploading")
                _log(f"远程缺失数据: {missing}，开始自动上传 ...")
                ssh_cfg = FT_BACKEND.get("ssh") or {}
                if not ssh_cfg.get("password"):
                    raise RuntimeError("远程缺数据且未配置 SSH 凭据（后端设置里填 SSH 密码后重试）")
                remote_upload_data(ssh_cfg, game, video, log_fn=_log)
                _log("数据上传完成")
                with _ft_lock:
                    _ft_task.update(stage="finetuning")
            else:
                _log("远程数据齐备，跳过上传")
            _log(f"=== 阶段 1/2：远程微调+评估（{remote_url}，samples={samples} epochs={epochs} batch={batch}） ===")
            out_name = f"{tag}.pt"
            ckpt_info = remote_has_ckpt(remote_url, out_name)
            skip_ft = ckpt_info.get("exists") and _meta_matches(
                ckpt_info.get("meta"), samples, epochs, batch)
            if skip_ft:
                _log(f"远程权重已存在且训练参数一致（output/{out_name}），跳过微调，仅重新评估（test_size={test_size or 200}）")
            else:
                _log("=== 需要重新微调："
                     + ("远程权重不存在" if not ckpt_info.get("exists")
                        else f"训练参数变化（当前 samples={samples} epochs={epochs} batch={batch} 与已存权重不一致）")
                     + f"，将覆盖 output/{out_name} ===")
            code, resp = remote_start(remote_url, {
                "game": game, "video": video, "samples": samples,
                "epochs": epochs, "batch": batch, "out": out_name,
                "test_size": test_size if test_size else 200,
                "eval_only": "1" if skip_ft else "0",
                "eval_repeats": str(max(1, eval_repeats))})
            if code != 202:
                raise RuntimeError(f"远程微调启动失败（{code}）: {resp}")
            # 轮询远程状态直到完成（复用函数，供 recover 接管）
            st = _poll_remote_until_done(remote_url, game, video, samples, _log)
            if st.get("stage") != "done":
                raise RuntimeError(f"远程微调未成功完成: {st.get('stage')} / {st.get('error')}")
            # 回传评估 CSV（KB 级，权重留远程按需下载）
            _log("=== 拉取远程评估结果 CSV 回本地 ===")
            remote_fetch_csv(remote_url, game, video, samples,
                             DATA_ROOT / game / "eval", log_fn=_log,
                             repeats=max(1, eval_repeats))
            _log(f"对照数据就绪（权重留远程：output/{out_name}，可在面板按需下载）")
        else:
            ckpt_path = REPO / ckpt_rel
            if _ckpt_matches(ckpt_path, samples, epochs, batch):
                _log(f"=== 阶段 1/2：本机权重已存在且训练参数一致（{ckpt_rel}），跳过微调，直接评估 ===")
            else:
                _log(f"=== 阶段 1/2：本机微调（samples={samples} epochs={epochs} batch={batch}） ===")
                rc = _ft_stage([sys.executable, str(REPO / "scripts" / "finetune.py"),
                                "--game", game, "--video", video, "--fps", str(fps),
                                "--samples", str(samples), "--epochs", str(epochs),
                                "--batch", str(batch), "--out", ckpt_rel], log)
                if rc != 0:
                    raise RuntimeError(f"微调失败（退出码 {rc}），日志: data/finetune_{game}_{video}.log")
            # 阶段 2：本机微调权重评估（remote 后端评估已在远程完成，只回传 CSV）
            with _ft_lock:
                _ft_task.update(stage="evaluating")
            reps = max(1, eval_repeats)
            for i in range(reps):
                rtag = tag if reps == 1 else f"{tag}_s{i}"
                _log(f"=== 阶段 2/2：本机微调权重评估（--ckpt {ckpt_rel} --tag {rtag} --infer-seed {i}）[{i+1}/{reps}] ===")
                eval_cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
                            "--game", game, "--video", video, "--fps", str(fps),
                            "--ckpt", ckpt_rel, "--tag", rtag, "--infer-seed", str(i)]
                if test_size is not None:
                    eval_cmd += ["--test-size", str(test_size)]
                rc = _ft_stage(eval_cmd, log)
                if rc != 0:
                    raise RuntimeError(f"微调权重评估失败（退出码 {rc}），日志: data/finetune_{game}_{video}.log")

        clear_cache(game)
        with _ft_lock:
            _ft_task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _ft_lock:
            _ft_task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


@app.post("/api/finetune")
def api_finetune():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    if not game or not video:
        return err("参数 game/video 必填")
    samples_raw = request.args.get("samples", "").strip()
    try:
        samples = int(samples_raw) if samples_raw else 1000
        if samples < 2 or samples > 1000000:
            return err("samples 应在 2~1000000 之间")
    except ValueError:
        return err("samples 必须是整数")
    epochs_raw = request.args.get("epochs", "").strip()
    try:
        epochs = int(epochs_raw) if epochs_raw else 1
        if epochs < 1 or epochs > 5:
            return err("epochs 应在 1~5 之间")
    except ValueError:
        return err("epochs 必须是整数")
    _ensure_backend_loaded()
    batch_raw = request.args.get("batch", "").strip()
    # 后端：显式 > 全局 FT_BACKEND
    backend = FT_BACKEND["mode"]
    remote_url = FT_BACKEND["remote_url"] if backend == "remote" else ""
    if batch_raw:
        try:
            batch = int(batch_raw)
            if batch < 1 or batch > 16:
                return err("batch 应在 1~16 之间")
            gpu_note = f"手动指定 batch={batch}"
        except ValueError:
            return err("batch 必须是整数")
    else:
        batch, gpu_note = auto_batch_size(remote_url)

    # 可选评估参数：测试集帧数（默认 evaluate.py 200）+ 是否过滤 IDLE 帧（对照表显示口径）
    test_size_raw = request.args.get("test_size", "").strip()
    test_size = None
    if test_size_raw:
        try:
            test_size = int(test_size_raw)
            if test_size < 10 or test_size > 5000:
                return err("test_size 应在 10~5000 之间", 400)
        except ValueError:
            return err("test_size 必须是整数", 400)
    filter_idle = request.args.get("filter_idle", "0").strip() in ("1", "true", "yes")
    # 评估重复次数：同权重不同推理 seed 各评估一次，取均值±std（判断差异是真实还是噪声）
    repeats_raw = request.args.get("eval_repeats", "").strip()
    eval_repeats = 1
    if repeats_raw:
        try:
            eval_repeats = int(repeats_raw)
            if eval_repeats < 1 or eval_repeats > 5:
                return err("eval_repeats 应在 1~5 之间", 400)
        except ValueError:
            return err("eval_repeats 必须是整数", 400)

    assert_lineage(game, video)
    if backend == "remote":
        # 远程后端：本地必须有视频供评估用；微调样本源在远程，但本地也需视频+标注评估
        if not (DATA_ROOT / "videos" / f"{game}_{video}.mp4").exists():
            return err("该视频未下载，评估需要本地视频文件。请先下载。", 400)
        if not remote_url:
            return err("远程后端未配置地址（先设置 backend）", 400)
        if remote_worker_health(remote_url) is None:
            return err(f"远程 worker 不可达: {remote_url}", 400)
    else:
        if not (DATA_ROOT / "videos" / f"{game}_{video}.mp4").exists():
            return err("该视频未下载，微调需要本地视频文件。请先下载。", 400)
    fps = _fps_from_manifest(game, video)
    if fps is None:
        return err("无法自动推导 fps（manifest 中无该视频的 chunk 行数）", 400)

    busy = _busy()
    if busy:
        return err(f"有任务正在运行（{busy}），请稍后再试", 409)
    with _ft_lock:
        if _ft_task["running"]:
            return err("已有微调任务在运行，请稍候", 409)
        _ft_task.update(running=True, game=game, video=video, stage="finetuning",
                        samples=samples, epochs=epochs, batch=batch, gpu_note=gpu_note,
                        backend=backend, remote_url=remote_url, out=f"ng_ft_{samples}.pt",
                        test_size=test_size, filter_idle=filter_idle, eval_repeats=eval_repeats,
                        log_tail="", error=None,
                        started_at=time.strftime("%H:%M:%S"), proc=None)
    threading.Thread(target=_run_finetune_chain,
                     args=(game, video, fps, samples, epochs, batch, backend, remote_url,
                           test_size, filter_idle, eval_repeats),
                     daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "game": game,
                                         "video": video, "fps": fps, "samples": samples,
                                         "epochs": epochs, "batch": batch,
                                         "backend": backend, "remote_url": remote_url,
                                         "gpu_note": gpu_note,
                                         "test_size": test_size,
                                         "filter_idle": filter_idle,
                                         "eval_repeats": eval_repeats}}), 202


@app.post("/api/finetune/backend")
def api_finetune_backend_set():
    """设置微调后端：?mode=local 或 ?mode=remote&url=http://host:port
    SSH 参数（ssh_port/ssh_user/ssh_password）仅缺数据自动上传时用，存本地不入库。"""
    mode = request.args.get("mode", "").strip()
    if mode not in ("local", "remote"):
        return err("mode 应为 local 或 remote")
    url = request.args.get("url", "").strip()
    if mode == "remote":
        if not url:
            return err("remote 模式需要 url（如 http://10.14.3.52:56272）")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        url = url.rstrip("/")
        if remote_worker_health(url) is None:
            return err(f"远程 worker 不可达: {url}", 400)
    else:
        url = ""
    # SSH 配置：新填则更新；未填保留原值
    ssh = dict(FT_BACKEND.get("ssh") or {})
    if mode == "remote":
        for key, arg in (("host", "ssh_host"), ("port", "ssh_port"),
                         ("user", "ssh_user"), ("password", "ssh_password")):
            v = request.args.get(arg, "").strip()
            if v:
                ssh[key] = v
        if not ssh.get("host"):
            # 默认从 url 推导 host
            host = url.split("://")[1].split(":")[0] if "://" in url else url.split(":")[0]
            ssh.setdefault("host", host)
        ssh.setdefault("port", "56271")
        ssh.setdefault("user", "root")
    FT_BACKEND["mode"] = mode
    FT_BACKEND["remote_url"] = url
    FT_BACKEND["ssh"] = ssh if mode == "remote" else {}
    # 持久化（password 仅存 data/remote_worker.json，不入库）
    try:
        FT_REMOTE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"mode": mode, "url": url, "ssh": ssh if mode == "remote" else {}},
                  open(FT_REMOTE_CONFIG, "w", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    # 返回时隐藏 password
    safe = {k: v for k, v in ssh.items() if k != "password"}
    return ok({"mode": mode, "remote_url": url, "ssh": safe,
               "health": remote_worker_health(url)})


@app.get("/api/finetune/backend")
def api_finetune_backend_get():
    """查询当前微调后端；未显式设置时尝试读持久化配置，否则默认 local。"""
    _ensure_backend_loaded()
    health = None
    if FT_BACKEND["mode"] == "remote" and FT_BACKEND["remote_url"]:
        health = remote_worker_health(FT_BACKEND["remote_url"])
    ssh_safe = {k: v for k, v in (FT_BACKEND.get("ssh") or {}).items() if k != "password"}
    return ok({
        "mode": FT_BACKEND["mode"],
        "remote_url": FT_BACKEND["remote_url"],
        "ssh": ssh_safe,
        "health": health,
        "available_backends": ["local", "remote"],
    })


@app.get("/api/finetune/status")
def api_finetune_status():
    with _ft_lock:
        t = {k: v for k, v in _ft_task.items()}
    t.pop("proc", None)
    if t.get("game"):
        log = DATA_ROOT / f"finetune_{t['game']}_{t['video']}.log"
        if log.exists():
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                t["log_tail"] = "\n".join(lines[-8:])
            except OSError:
                pass
    return ok(t)


@app.post("/api/finetune/cancel")
def api_finetune_cancel():
    with _ft_lock:
        if _ft_task["running"] and _ft_task["proc"]:
            try:
                _ft_task["proc"].terminate()
            except Exception:  # noqa: BLE001
                pass
        d = dict(_ft_task)
        d.pop("proc", None)
    return ok(d)


@app.get("/api/finetune/compare")
def api_finetune_compare():
    """零样本基线 vs 微调权重对照（读两份 metrics 副本的 best shift 行）。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    samples_raw = request.args.get("samples", "").strip()
    if not game or not video or not samples_raw:
        return err("参数 game/video/samples 必填")
    try:
        samples = int(samples_raw)
    except ValueError:
        return err("samples 必须是整数")
    tag = f"ng_ft_{samples}"

    def best_row(p: Path) -> dict | None:
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            return None
        r = max(rows, key=lambda m: float(m["acc_17keys_all"]))
        # acc_17keys_all_nonidle 列是新版 evaluate.py 才有；旧 metrics 无该列时返回 None（前端显示"—"）
        def _f(col):
            v = r.get(col, "")
            return float(v) if v not in ("", "nan") else None
        return {
            "shift": int(r["shift"]),
            "acc_17keys_all": round(float(r["acc_17keys_all"]), 4),   # B 口径（主）
            "acc_17keys_all_nonidle": round(_f("acc_17keys_all_nonidle"), 4)
                if _f("acc_17keys_all_nonidle") is not None else None,  # B 口径（过滤 IDLE）
            "n_frames_nonidle": int(_f("n_frames_nonidle"))
                if _f("n_frames_nonidle") is not None else None,
            "acc_17keys_bits": round(float(r["acc_17keys_bits"]), 4),  # A 口径（对照）
            "corr_jl_x": float(r["corr_jl_x"]) if r["corr_jl_x"] not in ("nan", "") else None,
            "mse_jl_x": round(float(r["mse_jl_x"]), 4),
            "n_frames": int(r["n_frames"]),
        }

    def multi_rows(prefix: Path) -> list[dict]:
        """读 _s0.._sN 副本（重复评估产物，优先）或单主副本，返回各自 best row 列表。"""
        stem = prefix.with_suffix("")  # metrics_xxx_ng_ft_174000（去掉 .csv 后加 _s{i}）
        sub = []
        for i in range(5):
            p = Path(f"{stem}_s{i}.csv")
            if p.exists():
                r = best_row(p)
                if r:
                    sub.append(r)
        if len(sub) >= 2:  # 多次重复评估产物优先（避免读到旧单次主副本）
            return sub
        if prefix.exists():
            r = best_row(prefix)
            if r:
                return [r]
        return sub

    def stat(rows: list[dict], key: str) -> tuple:
        """样本均值 / 样本内标准差（N<2 时 std=None）。"""
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else None
        return mean, std

    def pack(rows: list[dict]) -> dict:
        b = rows[0]
        am, ast = stat(rows, "acc_17keys_all")
        nam, nast = stat(rows, "acc_17keys_all_nonidle")
        abm, abst = stat(rows, "acc_17keys_bits")
        cm, cst = stat(rows, "corr_jl_x")
        mm, mst = stat(rows, "mse_jl_x")
        return {
            "shift": b["shift"],
            "acc_17keys_all": round(am, 4) if am is not None else None,
            "acc_17keys_all_std": round(ast, 4) if ast is not None else None,
            "acc_17keys_all_nonidle": round(nam, 4) if nam is not None else None,
            "acc_17keys_all_nonidle_std": round(nast, 4) if nast is not None else None,
            "acc_17keys_bits": round(abm, 4) if abm is not None else None,
            "corr_jl_x": round(cm, 4) if cm is not None else None,
            "mse_jl_x": round(mm, 4) if mm is not None else None,
            "n_frames": b["n_frames"],
            "n_frames_nonidle": b["n_frames_nonidle"],
            "repeats": len(rows),
        }

    base_rows = multi_rows(DATA_ROOT / game / "eval" / f"metrics_{video}_ng.csv")
    ft_rows = multi_rows(DATA_ROOT / game / "eval" / f"metrics_{video}_{tag}.csv")
    if not base_rows:
        return err(f"缺少零样本基线副本 metrics_{video}_ng.csv（先跑一次微调链或评估）", 404)
    if not ft_rows:
        return err(f"缺少微调副本 metrics_{video}_{tag}.csv（samples={samples} 的微调尚未完成）", 404)
    base = pack(base_rows)
    ft = pack(ft_rows)

    def delta(k):
        if base[k] is None or ft[k] is None:
            return None
        return round(ft[k] - base[k], 4)

    n_repeat = max(len(base_rows), len(ft_rows))
    return ok({
        "game": game, "video": video, "samples": samples,
        "baseline": base, "finetuned": ft,
        "delta": {
            "acc_17keys_all": delta("acc_17keys_all"),
            "acc_17keys_all_nonidle": delta("acc_17keys_all_nonidle"),
            "acc_17keys_bits": delta("acc_17keys_bits"),
        },
        "note": "论文口径为任务完成率（微调 +10%~52% 相对提升）；本表为离线口径"
                "（按键一致率 B/摇杆相关），两者不等价，差异小属正常。"
                + (f"（各评估 {n_repeat} 次取均值±std，判断差异是否落在推理采样波动内）"
                   if n_repeat > 1 else ""),
    })


# ---- 恢复接管：本地关闭期间远程任务照跑，重开 Web 后自动接管/拉回 ----
@app.get("/api/finetune/recover")
def api_finetune_recover():
    _ensure_backend_loaded()
    if FT_BACKEND["mode"] != "remote" or not FT_BACKEND["remote_url"]:
        return ok({"status": "idle", "reason": "非远程后端"})
    url = FT_BACKEND["remote_url"]
    st = remote_worker_status(url)
    if st is None:
        return ok({"status": "idle", "reason": "远程 worker 不可达"})
    with _ft_lock:
        if _ft_task["running"]:
            return ok({"status": "busy", "reason": "本地已有任务在运行"})
    if st.get("running"):
        game, video = st.get("game"), st.get("video")
        try:
            samples_i = int(st.get("samples"))
        except (TypeError, ValueError):
            return ok({"status": "idle", "reason": "远程任务参数不完整"})
        if not game or not video:
            return ok({"status": "idle", "reason": "远程任务参数不完整"})
        tag = f"ng_ft_{samples_i}"
        log = DATA_ROOT / f"finetune_{game}_{video}.log"
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"[recover] 接管远程任务 game={game} video={video} samples={samples_i}\n")

        def _log(m: str):
            with open(log, "a", encoding="utf-8") as f:
                f.write(m + "\n")

        with _ft_lock:
            _ft_task.update(running=True, game=game, video=video, samples=samples_i,
                            epochs=st.get("epochs"), batch=st.get("batch"),
                            test_size=st.get("test_size"),
                            filter_idle=bool(st.get("filter_idle")),
                            eval_repeats=st.get("eval_repeats") or 1,
                            stage=st.get("stage") or "finetuning", backend="remote",
                            remote_url=url, out=st.get("out"), gpu_note="恢复远程任务",
                            error=None, started_at=st.get("started_at"), proc=None)
        threading.Thread(target=_recover_poll,
                         args=(url, game, video, samples_i, tag, log), daemon=True).start()
        return ok({"status": "adopted", "stage": st.get("stage"), "samples": samples_i,
                   "note": "已接管远程运行中的任务"})
    if st.get("stage") == "done":
        game, video = st.get("game"), st.get("video")
        try:
            samples_i = int(st.get("samples"))
        except (TypeError, ValueError):
            samples_i = None
        if game and video and samples_i:
            eval_dir = DATA_ROOT / game / "eval"
            tag = f"ng_ft_{samples_i}"
            if not (eval_dir / f"metrics_{video}_{tag}.csv").exists():
                remote_fetch_csv(url, game, video, samples_i, eval_dir)
                clear_cache(game)
                return ok({"status": "recovered", "samples": samples_i,
                           "note": "已拉取远程完成任务的评估结果"})
            return ok({"status": "already", "samples": samples_i,
                       "note": "本地已有该任务的对照数据"})
    return ok({"status": "idle", "reason": f"远程无运行中任务（stage={st.get('stage')}）"})


def _recover_poll(url: str, game: str, video: str, samples: int, tag: str, log: Path):
    """接管线程：轮询远程到 done → 拉 CSV → 更新本地状态。"""
    def _log(m: str):
        with open(log, "a", encoding="utf-8") as f:
            f.write(m + "\n")
    try:
        st = _poll_remote_until_done(url, game, video, samples, _log)
        if st.get("stage") != "done":
            raise RuntimeError(f"远程未成功完成: {st.get('stage')} / {st.get('error')}")
        _log("=== 拉取远程评估结果 CSV 回本地 ===")
        remote_fetch_csv(url, game, video, samples, DATA_ROOT / game / "eval", log_fn=_log)
        clear_cache(game)
        with _ft_lock:
            _ft_task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _ft_lock:
            _ft_task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


# ---- 实时日志终端（前端可视化命令窗口） ----
@app.get("/api/finetune/logtail")
def api_finetune_logtail():
    _ensure_backend_loaded()
    out = request.args.get("out", "").strip()
    try:
        lines = min(max(int(request.args.get("lines", "60")), 1), 300)
    except ValueError:
        lines = 60
    if FT_BACKEND["mode"] == "remote" and FT_BACKEND["remote_url"]:
        import urllib.parse
        import urllib.request
        if not out:
            # 无 out：自动取 worker 上次任务的 out（非 running 时也保留），显示历史日志
            st = remote_worker_status(FT_BACKEND["remote_url"])
            if st and st.get("out"):
                out = st["out"]
        qs = urllib.parse.urlencode({"out": out, "lines": lines})
        try:
            with urllib.request.urlopen(f"{FT_BACKEND['remote_url']}/logtail?{qs}", timeout=10) as r:
                return ok(json.load(r).get("data", {}))
        except Exception as e:  # noqa: BLE001
            return err(f"远程日志拉取失败: {e}")
    # local：读本地 finetune 日志
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    p = DATA_ROOT / f"finetune_{game}_{video}.log"
    if not p.exists():
        return ok({"lines": []})
    lines_list = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return ok({"lines": lines_list[-lines:]})


# ---- 按需下载远程微调权重（只拉优质权重，2GB 异步后台） ----
_pw_task = {"running": False, "done": False, "error": None, "dest": None,
            "samples": None, "started_at": None}
_pw_lock = threading.Lock()


@app.post("/api/finetune/pull_weight")
def api_finetune_pull_weight():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    samples_raw = request.args.get("samples", "").strip()
    try:
        samples = int(samples_raw)
    except ValueError:
        return err("samples 必须是整数")
    _ensure_backend_loaded()
    if FT_BACKEND["mode"] != "remote" or not FT_BACKEND["remote_url"]:
        return err("当前不是远程后端，无需拉取远程权重", 400)
    url = FT_BACKEND["remote_url"]
    if remote_worker_health(url) is None:
        return err(f"远程 worker 不可达: {url}", 400)
    tag = f"ng_ft_{samples}"
    dest = REPO / "NitroGen" / f"{tag}.pt"
    if dest.exists():
        return ok({"dest": str(dest), "note": "权重已存在本地"})
    with _pw_lock:
        if _pw_task["running"]:
            return err("已有权重下载任务在运行", 409)
        _pw_task.update(running=True, done=False, error=None,
                        dest=str(dest), samples=samples,
                        started_at=time.strftime("%H:%M:%S"))
    threading.Thread(target=_pull_weight_worker,
                     args=(url, f"{tag}.pt", dest), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "dest": str(dest),
                                         "note": "约需 10~15 分钟，后台异步不阻塞"}}), 202


def _pull_weight_worker(url: str, out: str, dest: Path):
    try:
        remote_fetch_weight(url, out, dest)
        with _pw_lock:
            _pw_task.update(running=False, done=True, error=None)
    except Exception as e:  # noqa: BLE001
        with _pw_lock:
            _pw_task.update(running=False, done=False, error=str(e)[:200])
        if dest.exists():
            try:
                dest.unlink()  # 删除半成品
            except OSError:
                pass


@app.get("/api/finetune/pull_weight/status")
def api_finetune_pull_weight_status():
    with _pw_lock:
        d = dict(_pw_task)
    if d.get("dest"):
        p = Path(d["dest"])
        d["size_mb"] = round(p.stat().st_size / 1024 / 1024, 1) if p.exists() else 0
    return ok(d)


# ---- 界面偏好：保存/恢复上次关闭时选择的游戏和视频（data/web_prefs.json，不入库） ----
PREFS_PATH = DATA_ROOT / "web_prefs.json"


@app.get("/api/prefs")
def api_prefs_get():
    try:
        if PREFS_PATH.exists():
            return ok(json.load(open(PREFS_PATH, encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return ok({})


@app.post("/api/prefs")
def api_prefs_set():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    # 字段级合并：未提供的字段保留（避免 onVideoChange 只传 game/video 时清掉微调参数）
    try:
        prefs = json.load(open(PREFS_PATH, encoding="utf-8")) if PREFS_PATH.exists() else {}
    except Exception:  # noqa: BLE001
        prefs = {}
    if game:
        prefs["game"] = game
    if video:
        prefs["video"] = video
    for k in ("samples", "epochs", "batch", "test_size", "filter_idle", "eval_repeats"):
        if k in request.args:
            v = request.args.get(k, "").strip()
            if v:
                prefs[k] = v
            else:
                prefs.pop(k, None)   # 显式传空 → 清除该字段（batch 留空=自动）
    prefs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(prefs, open(PREFS_PATH, "w", encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return ok(prefs)


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
    try:
        _safe_name(game, "game")
        _safe_name(video, "video")
    except LineageError as e:
        return err(str(e), 400)
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
    # 定位 yt-dlp：PATH 里的 exe，或当前 Python（venv 已装 yt-dlp）的模块
    exe = shutil.which("yt-dlp")
    if exe:
        cmd = [exe]
    else:
        cmd = [sys.executable, "-m", "yt_dlp"]   # venv 已装 yt-dlp
    cmd += ["--newline", "--no-warnings", "-f", "bv*[height<=720]+ba/b",
            "--merge-output-format", "mp4",
            "--retries", "10", "--fragment-retries", "10",
            "-o", str(out)]
    # --retries/--fragment-retries：Twitch HLS 分段流断流/超时自动重试，减少"下载到一半失败"
    # ffmpeg 定位：优先系统 PATH；没有则用 venv 内 imageio-ffmpeg 自带二进制
    # （Twitch 视频为分离的音视频流，merge 成 mp4 必需 ffmpeg，否则 yt-dlp 退出码 1）
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg_exe = None
    if ffmpeg_exe:
        cmd += ["--ffmpeg-location", ffmpeg_exe]
    # Twitch 长下载限流 / YouTube bot 验证规避：
    # 优先 data/cookies.txt（浏览器插件导出，见 README）；否则读 Edge 浏览器登录态
    cookies_file = DATA_ROOT / "cookies.txt"
    if cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    else:
        cmd += ["--cookies-from-browser", "edge"]   # 直接读 Edge 中已登录的 cookie
    cmd += [url]
    # 诊断：yt-dlp 完整输出落盘（长下载中途失败时可查真实原因）
    dl_log = DATA_ROOT / "downloads.log"
    log_lines = []
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
            log_lines.append(line)
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
            # 失败：保留完整日志供诊断，并清理半成品（.part / 残缺 mp4）
            try:
                dl_log.write_text("\n".join(log_lines[-80:]), encoding="utf-8")
            except OSError:
                pass
            for p in (out, out.with_suffix(".mp4.part"), out.with_suffix(".mp4.part")):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            with _download_lock:
                _download.update(running=False, error=(
                    _download.get("error")
                    or f"yt-dlp 退出码 {rc}（详见 data/downloads.log，尾部80行）"))
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
            pid = _download["proc"].pid
            # 杀整个进程树（/T /F）：仅 terminate 杀主进程会让子进程继承 stdout 管道，
            # 导致读取循环永不 EOF、状态卡在 running（"完全不动"）
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=15)
            except Exception:  # noqa: BLE001
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
