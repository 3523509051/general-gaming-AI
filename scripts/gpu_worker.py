# -*- coding: utf-8 -*-
"""远程 GPU Worker：微调专用服务（部署到 A100，本机 Web 通过 HTTP 调用）。

职责：只做「微调」（finetune.py），评估留在本地（evaluate.py --ckpt）。
接口（监听 56272，仅内网）：
    GET  /health         返回 GPU 型号/显存 + 是否空闲
    POST /start?game=&video=&samples=&epochs=&batch=&out=
                        启动微调任务（后台，产物落 output/<out>）
    GET  /status         任务状态 + 日志尾部
    POST /cancel         取消当前任务
    GET  /download?out=  下载微调权重（本地 Web 拉回后评估）

约定：
    - 工作目录 /root/workspace/nitrogen_worker（脚本、数据、venv 所在）
    - 数据：data/videos/<game>_<video>.mp4 + data/<game>/manifest.json + annotations.parquet
    - 产物：output/<out>（文件名如 ng_ft_1000.pt）
    - finetune.py 在 scripts/ 下，与 evaluate 等脚本平级（本 worker 部署时脚本目录含 finetune.py）

启动（在部署机）：
    cd /root/workspace/nitrogen_worker
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/gpu_worker.py --port 56272
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, Response

BASE = Path(__file__).resolve().parent.parent   # nitrogen_worker/
SCRIPTS = BASE / "scripts"
DATA = BASE / "data"
OUT_DIR = BASE / "output"
FINETUNE = SCRIPTS / "finetune.py"
EVALUATE = SCRIPTS / "evaluate.py"

app = Flask(__name__)

_task = {"running": False, "game": None, "video": None, "samples": None,
         "epochs": None, "batch": None, "out": None, "proc": None,
         "test_size": None, "eval_only": False, "eval_repeats": 1,
         "log": None, "log_tail": "", "error": None,
         "started_at": None, "stage": "idle"}
_lock = threading.Lock()


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return {"gpu": name, "mem_gb": round(mem, 1), "available": True}
        return {"gpu": "no-cuda", "mem_gb": None, "available": False}
    except Exception as e:  # noqa: BLE001
        return {"gpu": f"error: {e}", "mem_gb": None, "available": False}


@app.get("/health")
def health():
    return jsonify({"ok": True, "data": {**_gpu_info(), "running": _task["running"]}})


@app.get("/has_ckpt")
def has_ckpt():
    """权重是否存在 + 训练参数 meta（本地 Web 据此决定跳过微调或重训）。"""
    out = request.args.get("out", "").strip()
    p = OUT_DIR / Path(out).name
    exists = bool(out) and p.exists()
    meta = None
    if exists:
        mp = Path(str(p) + ".meta.json")
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = None
    return jsonify({"ok": True, "data": {"exists": exists, "meta": meta}})


@app.post("/eval_base")
def eval_base():
    """零样本基线评估（evaluate.py 无 --ckpt --tag ng），产物 CSV 供 /metrics_csv 回传。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    test_size_raw = request.args.get("test_size", "").strip()
    if not (game and video):
        return jsonify({"ok": False, "error": "参数 game/video 必填"}), 400
    test_size = None
    if test_size_raw:
        try:
            test_size = int(test_size_raw)
            if test_size < 10 or test_size > 5000:
                raise ValueError
        except ValueError:
            return jsonify({"ok": False, "error": "test_size 应为 10~5000 整数"}), 400
    with _lock:
        if _task["running"]:
            return jsonify({"ok": False, "error": "已有任务运行中"}), 409
        _task.update(running=True, game=game, video=video, samples=None,
                     epochs=None, batch=None, out=f"base_{video}", proc=None,
                     test_size=test_size, eval_only=False, log_tail="", error=None,
                     started_at=time.strftime("%H:%M:%S"), stage="baseline")
    threading.Thread(target=_run_eval_base, args=(game, video, test_size), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "stage": "baseline"}}), 202


def _run_eval_base(game: str, video: str, test_size: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f"base_{video}.eval.log"
    try:
        cmd = [str(BASE / ".venv" / "bin" / "python"), str(EVALUATE),
               "--game", game, "--video", video, "--fps", "30",
               "--tag", "ng", "--no-plots"]
        if test_size:
            cmd += ["--test-size", str(test_size)]
        proc = subprocess.Popen(cmd, cwd=str(BASE),
                                stdout=open(log, "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT)
        with _lock:
            _task["proc"] = proc
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"基线评估退出码 {rc}，日志: {log}")
        with _lock:
            _task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _lock:
            _task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


@app.post("/start")
def start():
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    samples = request.args.get("samples", "").strip()
    epochs = request.args.get("epochs", "1").strip()
    batch = request.args.get("batch", "4").strip()
    out = request.args.get("out", "").strip()
    test_size_raw = request.args.get("test_size", "").strip()
    test_size = None
    if test_size_raw:
        try:
            test_size = int(test_size_raw)
            if test_size < 10 or test_size > 5000:
                raise ValueError
        except ValueError:
            return jsonify({"ok": False, "error": "test_size 应为 10~5000 整数"}), 400
    eval_only = request.args.get("eval_only", "0").strip() in ("1", "true", "yes")
    repeats_raw = request.args.get("eval_repeats", "").strip()
    eval_repeats = 1
    if repeats_raw:
        try:
            eval_repeats = int(repeats_raw)
            if eval_repeats < 1 or eval_repeats > 5:
                raise ValueError
        except ValueError:
            return jsonify({"ok": False, "error": "eval_repeats 应为 1~5 整数"}), 400
    if not (game and video and samples and out):
        return jsonify({"ok": False, "error": "参数 game/video/samples/out 必填"}), 400
    try:
        samples_i = int(samples)
        if samples_i < 2 or samples_i > 1000000:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "error": "samples 应为 2~1000000 整数"}), 400
    # out 限定为纯文件名，防路径穿越
    out = Path(out).name
    if not out.endswith(".pt"):
        out += ".pt"
    with _lock:
        if _task["running"]:
            return jsonify({"ok": False, "error": "已有微调任务运行中"}), 409
        _task.update(running=True, game=game, video=video, samples=samples_i,
                     epochs=int(epochs), batch=int(batch), out=out, proc=None,
                     test_size=test_size, eval_only=eval_only, eval_repeats=eval_repeats,
                     log_tail="", error=None,
                     started_at=time.strftime("%H:%M:%S"), stage="finetuning")
    threading.Thread(target=_run, args=(game, video, samples_i, int(epochs),
                                        int(batch), out, test_size, eval_only,
                                        eval_repeats), daemon=True).start()
    return jsonify({"ok": True, "data": {"status": "started", "out": out,
                                         "samples": samples_i}}), 202


def _run(game: str, video: str, samples: int, epochs: int, batch: int, out: str,
         test_size: int | None = None, eval_only: bool = False, eval_repeats: int = 1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = out[:-3] if out.endswith(".pt") else out
    log = OUT_DIR / f"{out}.log"
    eval_log = OUT_DIR / f"{out}.eval.log"
    try:
        # 阶段 1：微调（eval_only 时跳过——权重已存在，仅重新评估）
        if eval_only:
            with _lock:
                _task["stage"] = "evaluating"
            with open(log, "a", encoding="utf-8") as f:
                f.write("[worker] eval_only：权重已存在，跳过微调，直接评估\n")
        else:
            with _lock:
                _task["stage"] = "finetuning"
            cmd = [str(BASE / ".venv" / "bin" / "python"), str(FINETUNE),
                   "--game", game, "--video", video, "--fps", "30",
                   "--samples", str(samples), "--epochs", str(epochs),
                   "--batch", str(batch), "--out", str(OUT_DIR / out)]
            proc = subprocess.Popen(cmd, cwd=str(BASE),
                                    stdout=open(log, "w", encoding="utf-8"),
                                    stderr=subprocess.STDOUT)
            with _lock:
                _task["proc"] = proc
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"finetune 退出码 {rc}，日志: {log}")
        # 阶段 2：评估（A100 上 evaluate.py --ckpt --no-plots，产物 CSV 由 /metrics_csv 回传）
        # eval_repeats>1 时用不同 --infer-seed 评估 N 次，产物 metrics_*_s{i}.csv 供均值±std 对照
        with _lock:
            _task["stage"] = "evaluating"
        reps = max(1, eval_repeats)
        for i in range(reps):
            rtag = tag if reps == 1 else f"{tag}_s{i}"
            log_i = OUT_DIR / f"{out}.eval{'%d' % i}.log" if reps > 1 else eval_log
            cmd = [str(BASE / ".venv" / "bin" / "python"), str(EVALUATE),
                   "--game", game, "--video", video, "--fps", "30",
                   "--ckpt", str(OUT_DIR / out), "--tag", rtag, "--no-plots",
                   "--infer-seed", str(i)]
            if test_size:
                cmd += ["--test-size", str(test_size)]
            proc = subprocess.Popen(cmd, cwd=str(BASE),
                                    stdout=open(log_i, "w", encoding="utf-8"),
                                    stderr=subprocess.STDOUT)
            with _lock:
                _task["proc"] = proc
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"评估退出码 {rc}（第 {i+1}/{reps} 次），日志: {log_i}")
        with _lock:
            _task.update(running=False, stage="done", proc=None)
    except Exception as e:  # noqa: BLE001
        with _lock:
            _task.update(running=False, stage="failed", proc=None, error=str(e)[:300])


@app.get("/status")
def status():
    with _lock:
        t = {k: v for k, v in _task.items()}
    t.pop("proc", None)
    if t.get("out"):
        log = OUT_DIR / f"{t['out']}.log"
        if log.exists():
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                t["log_tail"] = "\n".join(lines[-8:])
            except OSError:
                pass
    return jsonify({"ok": True, "data": t})


@app.post("/cancel")
def cancel():
    with _lock:
        if _task["running"] and _task["proc"]:
            try:
                _task["proc"].terminate()
            except Exception:  # noqa: BLE001
                pass
        d = {k: v for k, v in _task.items()}
        d.pop("proc", None)
    return jsonify({"ok": True, "data": d})


@app.get("/data_check")
def data_check():
    """检查远程该游戏数据是否齐备（本地启动前调用，缺则自动上传）。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    missing = []
    if not (DATA / "videos" / f"{game}_{video}.mp4").exists():
        missing.append("video")
    if not (DATA / game / "manifest.json").exists():
        missing.append("manifest")
    if not (DATA / game / "annotations.parquet").exists():
        missing.append("annotations")
    return jsonify({"ok": True, "data": {
        "ready": not missing, "missing": missing,
        "game": game, "video": video}})


@app.get("/logtail")
def logtail():
    """合并微调+评估日志尾部，供前端实时终端显示。"""
    out = Path(request.args.get("out", "")).name
    try:
        lines = max(1, min(int(request.args.get("lines", "60")), 300))
    except ValueError:
        lines = 60
    parts = []
    # 兼容 eval_repeats>1 时的 {out}.eval{i}.log（重复评估日志）
    candidates = [OUT_DIR / f"{out}.log", OUT_DIR / f"{out}.eval.log"] + \
        sorted(OUT_DIR.glob(f"{out}.eval[0-9].log"))
    for p in candidates:
        if p.exists():
            parts.append(f"----- {p.name} -----")
            parts.extend(p.read_text(encoding="utf-8", errors="replace").splitlines())
    return jsonify({"ok": True, "data": {"lines": parts[-lines:] if parts else []}})


@app.get("/metrics_csv")
def metrics_csv():
    """回传评估指标 CSV 文本（KB 级，替代 2GB 权重回传）。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    samples = request.args.get("samples", "").strip()
    tag_raw = request.args.get("tag", "").strip()
    tag = tag_raw if tag_raw else f"ng_ft_{samples}"
    p = DATA / game / "eval" / f"metrics_{video}_{tag}.csv"
    if not p.exists():
        return jsonify({"ok": False, "error": f"远程无指标文件 {p.name}"}), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="text/csv")


@app.get("/predictions_csv")
def predictions_csv():
    """回传逐帧预测 CSV 文本。"""
    game = request.args.get("game", "").strip()
    video = request.args.get("video", "").strip()
    samples = request.args.get("samples", "").strip()
    tag_raw = request.args.get("tag", "").strip()
    tag = tag_raw if tag_raw else f"ng_ft_{samples}"
    p = DATA / game / "eval" / f"predictions_{video}_{tag}.csv"
    if not p.exists():
        return jsonify({"ok": False, "error": f"远程无预测文件 {p.name}"}), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="text/csv")


@app.get("/download")
def download():
    """按需下载微调权重（2GB，优质权重才拉）。"""
    out = Path(request.args.get("out", "")).name
    if not out.endswith(".pt"):
        out += ".pt"
    p = OUT_DIR / out
    if not p.exists():
        return jsonify({"ok": False, "error": f"产物不存在: {out}"}), 404
    return send_file(p, as_attachment=True, download_name=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=56272)
    ap.add_argument("--host", default="0.0.0.0")   # 仅内网可到，见服务器规范
    args = ap.parse_args()
    print(f"GPU Worker 启动 -> http://{args.host}:{args.port}  {_gpu_info()}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
