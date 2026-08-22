# -*- coding: utf-8 -*-
"""视频下载链接可用性探测（yt-dlp simulate，不下载）。

读取 data/games_scan.json，逐个探测视频 URL 是否仍可访问，
输出 data/video_status.json 供 Web 平台展示状态徽标。

用法（系统 Python，需 yt-dlp）：
    python scripts/probe_videos.py --game hades          # 单游戏（快）
    python scripts/probe_videos.py --full                # 全量 311 URL（10~20 分钟）
    python scripts/probe_videos.py --game hades --refresh # 忽略缓存强制重探

状态枚举：
    downloaded : data/videos/{game}_{video}.mp4 已存在（无需探测，直接标记）
    available  : URL 可访问（yt-dlp 模拟提取成功）
    dead       : URL 失效（视频被删/私有/区域限制）
    unknown    : 未探测

输出（data/video_status.json）：
    {video: {"game", "url", "status", "duration", "error", "checked_at"}}
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
STATUS_FILE = DATA_ROOT / "video_status.json"
VIDEOS_DIR = DATA_ROOT / "videos"


def find_ytdlp_cmd() -> list[str] | None:
    """定位可用的 yt-dlp 调用方式。优先 PATH 里的 yt-dlp，其次各解释器的 yt_dlp 模块。"""
    # 1) PATH 里的 yt-dlp 可执行
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    # 2) 依次尝试：当前解释器 / 系统 python / py 启动器
    for py in [sys.executable, shutil.which("python"), shutil.which("py")]:
        if not py:
            continue
        try:
            r = subprocess.run([py, "-c", "import yt_dlp"], capture_output=True, timeout=20)
            if r.returncode == 0:
                return [py, "-m", "yt_dlp"]
        except Exception:  # noqa: BLE001
            continue
    return None


_YTDLP = None  # 惰性初始化


def probe_oembed(url: str, timeout: int = 15) -> dict | None:
    """YouTube oEmbed 探测：无需 cookie/登录，纯 HTTP 确认视频是否真实存在。

    对 yt-dlp 被 bot 验证拦下（"Sign in to confirm you're not a bot"）的链接，
    用官方 oEmbed 接口二次确认虚实：
      - 返回 JSON（含 title）→ 视频真实存在 → available
      - 404 / 空 → 视频可能被删/私有 → 仍 unknown（保守，不误判 dead）
    仅对 youtube.com 链接生效；其它平台返回 None（不适用）。
    """
    if "youtube.com" not in url and "youtu.be" not in url:
        return None
    api = "https://www.youtube.com/oembed?url=" + urllib.parse.quote(url, safe="")
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", "replace"))
                return {"status": "available", "duration": None,
                        "error": f"oEmbed 确认存在（{body.get('title', '')[:60]}）"}
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", None)
        if code in (404, 410):
            return {"status": "unknown", "duration": None,
                    "error": f"oEmbed 404/410（视频可能已删/私有）: {str(e)[:100]}"}
    return None


def probe_one(video: str, url: str) -> dict:
    """探测单个 URL。已下载的直接标 downloaded；yt-dlp 不可用标 unknown（不误判为 dead）。"""
    global _YTDLP
    # 1) 本地已下载？
    hits = list(VIDEOS_DIR.glob(f"*_{video}.mp4"))
    if hits:
        return {"status": "downloaded", "duration": None, "error": None}

    # 2) yt-dlp 模拟提取（--simulate 不落盘）
    if _YTDLP is None:
        _YTDLP = find_ytdlp_cmd() or False
    if _YTDLP is False:
        return {"status": "unknown", "duration": None,
                "error": "yt-dlp 不可用（未安装或不在 PATH/任何 Python 中）"}

    try:
        r = subprocess.run(
            _YTDLP + ["--simulate", "--no-warnings", "--print", "%(duration)s", url],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout.strip():
            try:
                dur = float(r.stdout.strip().splitlines()[-1])
            except ValueError:
                dur = None
            return {"status": "available", "duration": dur, "error": None}

        # 失败时按错误内容分类：网络错误 ≠ 视频失效，避免误判 dead
        err_lower = (r.stderr or "").lower()
        # 明确的"视频不存在/被删/私有/区域限制/版权"
        gone_markers = [
            "video unavailable", "has been removed", "removed", "private",
            "not available", "does not exist", "unavailable", "video unavailable",
            "copyright", "taken down", "this video", "members only", "gated",
            "playlist unavailable", "no longer exists", "410", "404",
        ]
        # 网络/连接类错误（可能只是没挂梯子，不该判 dead）
        net_markers = [
            "timed out", "timeout", "connection", "network", "getaddrinfo",
            "name or service not known", "unable to connect", "ssl", "tls",
            "certificate", "proxy", "curl error", "errno", "failed to resolve",
            "host", "socket", "read timed", "connection reset", "connection refused",
        ]
        err_lines = (r.stderr or "").strip().splitlines()
        err_msg = err_lines[-1][:200] if err_lines else "probe failed"
        if any(m in err_lower for m in net_markers):
            return {"status": "unknown", "duration": None,
                    "error": f"网络异常（可能未挂梯子，无法判定）: {err_msg}"}
        if any(m in err_lower for m in gone_markers):
            return {"status": "dead", "duration": None, "error": err_msg}
        # 其它未知错误（如 bot 验证）：先用 oEmbed 二次确认虚实，再决定标 available 还是 unknown
        oem = probe_oembed(url)
        if oem:
            return oem
        return {"status": "unknown", "duration": None,
                "error": f"探测异常（无法判定）: {err_msg}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "unknown", "duration": None, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser(description="视频下载链接可用性探测")
    ap.add_argument("--game", default=None, help="只探测指定游戏的视频（快）")
    ap.add_argument("--full", action="store_true", help="全量探测所有游戏")
    ap.add_argument("--refresh", action="store_true", help="忽略已有探测缓存，强制重探")
    args = ap.parse_args()

    scan = json.load(open(DATA_ROOT / "games_scan.json", encoding="utf-8"))
    status = {}
    if STATUS_FILE.exists() and not args.refresh:
        status = json.load(open(STATUS_FILE, encoding="utf-8"))

    # 待探测清单
    targets = []  # (game, video, url)
    for game, info in scan.items():
        if args.game and game != args.game and not args.full:
            continue
        for u in info["urls"]:
            targets.append((game, u["video"], u["url"]))
    if not args.full and args.game:
        pass  # 已按 game 过滤
    elif not args.full:
        # 默认（无参数）：只探测本地已有数据的游戏 + 已有缓存保持
        local_games = {p.name for p in DATA_ROOT.iterdir()
                       if p.is_dir() and (p / "manifest.json").exists()}
        targets = [t for t in targets if t[0] in local_games]

    print(f"probing {len(targets)} urls ...")
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for i, (game, video, url) in enumerate(targets):
        old = status.get(video, {})
        # 缓存有效：已下载/已死的不重探（除非 --refresh）
        if not args.refresh and old.get("status") in ("downloaded", "dead"):
            old.setdefault("game", game)
            old.setdefault("url", url)
            status[video] = old
            continue
        res = probe_one(video, url)
        status[video] = {"game": game, "url": url, **res, "checked_at": now}
        print(f"  [{i+1}/{len(targets)}] {game}/{video}: {res['status']}"
              + (f" ({res.get('duration') or '?'}s)" if res["status"] == "available" else ""), flush=True)
        # 每次都落盘（可中断续跑）
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

    # 统计
    from collections import Counter
    cnt = Counter(v["status"] for v in status.values())
    print(f"\nDONE: {dict(cnt)} -> {STATUS_FILE}")


if __name__ == "__main__":
    main()
