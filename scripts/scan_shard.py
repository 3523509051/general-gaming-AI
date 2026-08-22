# -*- coding: utf-8 -*-
"""一键识别切片（SHARD_0034）内含的游戏与视频下载链接。

流式单遍扫描 tar 内全部 metadata.json（只读小文件，不读 parquet），
按 game 聚合出"游戏 -> 视频数 / chunk 数 / 估算帧数 / 下载链接清单"。

用法（仓库根目录，任意有 Python 的环境均可，无第三方依赖）：
    python scripts/scan_shard.py                          # 自动定位 HF 缓存中的 SHARD_0034
    python scripts/scan_shard.py --shard <tar.gz 路径>     # 显式指定分片
    python scripts/scan_shard.py --out data/games_scan.json

输出（data/games_scan.json）：
    {game: {"videos": N, "chunks": N, "frames": N(估算), "urls": [{video, url, chunks}]}}

说明：
- frames 为估算值（chunk_size × chunks）：最后一个 chunk 常不满，真实标注行数
  以 data/<game>/annotations.parquet 为准（提取后可知）。
- 耗时约 3 分钟（4745 个 chunk 的 metadata 流式读取）。
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
HF_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nvidia--NitroGen" / "snapshots"


def resolve_shard_path(explicit: str | None) -> Path:
    """定位 SHARD_0034.tar.gz（显式路径优先，否则搜索 HF 缓存）。"""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(explicit)
    if HF_SNAPSHOT.exists():
        for snap in sorted(HF_SNAPSHOT.iterdir()):
            cand = snap / "actions" / "SHARD_0034.tar.gz"
            if cand.exists():
                return cand
    raise FileNotFoundError("未在 HF 缓存中找到 SHARD_0034.tar.gz，请用 --shard 显式指定路径")


def main():
    ap = argparse.ArgumentParser(description="一键识别切片含有的游戏与下载链接")
    ap.add_argument("--shard", default=None, help="SHARD_0034.tar.gz 路径（默认自动定位 HF 缓存）")
    ap.add_argument("--out", default=str(DATA_ROOT / "games_scan.json"), help="输出 JSON 路径")
    args = ap.parse_args()

    shard = resolve_shard_path(args.shard)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"shard : {shard}")
    print(f"out   : {out_path}")

    # game -> {"videos": {vid: {"url","source","chunks","frames_est"}}}
    games: dict[str, dict] = {}
    t0 = time.time()
    n_meta = 0

    with tarfile.open(shard, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith("metadata.json"):
                continue
            n_meta += 1
            meta = json.load(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"))
            game = meta.get("game") or "unknown"
            ov = meta.get("original_video") or {}
            vid = ov.get("video_id") or m.name.split("/")[1]
            chunk_size = int(meta.get("chunk_size") or 0)

            g = games.setdefault(game, {"videos": {}, "chunk_size": chunk_size})
            v = g["videos"].setdefault(vid, {
                "url": ov.get("url"),
                "source": ov.get("source"),
                "controller": meta.get("controller_type"),
                "resolution": ov.get("resolution"),
                "chunks": 0,
                "frames_est": 0,
            })
            v["chunks"] += 1
            v["frames_est"] += chunk_size
            if n_meta % 500 == 0:
                print(f"  scanned {n_meta} chunks ... ({time.time()-t0:.0f}s)", flush=True)

    # 组装输出（与既有 games_scan.json 格式一致）
    result = {}
    for game, g in sorted(games.items(), key=lambda kv: -sum(v["chunks"] for v in kv[1]["videos"].values())):
        vids = sorted(g["videos"].values(), key=lambda v: -v["chunks"])
        result[game] = {
            "videos": len(vids),
            "chunks": sum(v["chunks"] for v in vids),
            "frames": sum(v["frames_est"] for v in vids),
            "urls": [
                {"video": vid, "url": v["url"], "chunks": v["chunks"],
                 "source": v["source"], "controller": v["controller"]}
                for vid, v in sorted(g["videos"].items(), key=lambda kv: -kv[1]["chunks"])
            ],
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nDONE: {len(result)} games, {n_meta} chunks scanned in {time.time()-t0:.0f}s")
    print(f"top games: {[g for g in list(result)[:8]]}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
