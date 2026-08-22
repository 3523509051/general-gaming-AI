# -*- coding: utf-8 -*-
"""从 SHARD_0034 分片提取指定游戏的手柄标注（通用脚本）。

用法（仓库根目录）：
    NitroGen\\.venv\\Scripts\\python.exe scripts\\extract_game.py --game hades           # 全量提取 hades
    NitroGen\\.venv\\Scripts\\python.exe scripts\\extract_game.py --game hollow_knight  # 全量提取 hollow_knight
    NitroGen\\.venv\\Scripts\\python.exe scripts\\extract_game.py --game hades --limit 2  # 冒烟验证

设计要点（见 第2天报告.md「动作对齐关系与指标口径」）：
1. 数据源：HF 缓存中 SHARD_0034.tar.gz（符号链接解析到 blobs 实际文件）；
2. 只提取 game == <目标游戏> 的 chunk 的 actions_raw.parquet（口径统一，全量可得）；
   actions_processed.parquet 是否存在记录进 manifest（has_processed），但提取不混用；
3. tar 成员顺序注意：每个 chunk 目录内 actions_raw.parquet 在 metadata.json 之前，
   而 game 字段在 metadata.json 中 —— 首个 chunk 的 parquet 先缓存（pending），
   读到 metadata 确认 game 后再决定写盘或丢弃；
4. 输出（输出目录按 --game 动态命名，避免不同游戏互相覆盖）：
   data/<game>/raw/<video_id>/<chunk_id>.parquet  每chunk一份原始标注
   data/<game>/manifest.json                      chunk级清单（视频/帧数/来源URL等）
   data/<game>/annotations.parquet                合并表（含 video/chunk/frame_idx 溯源列）
5. 合并表列约定（与官方 BUTTON_ACTION_TOKENS 字母序对齐）：
   17 个按键列（小写）+ j_left/j_right（list[f64]，[-1,1]，(-1,-1)=左上，与模型输出同坐标系）。
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import polars as pl

# 数据集 17 个按键列（字母序，与模型 BUTTON_ACTION_TOKENS 的 21 键字母序一致；
# 模型多出的 4 个 PS5 键 right_bottom/right_left/right_right/right_up 在 xboxone 数据中不存在，评估时排除）
BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
TARGET_GAME = "hades"

# SHARD_0034 在 HuggingFace 缓存中的位置（snapshot 符号链接 -> blobs 实体）
HF_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nvidia--NitroGen" / "snapshots"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def resolve_shard_path(explicit: str | None) -> Path:
    """定位 SHARD_0034.tar.gz（支持直接传路径或自动搜索 HF 缓存）。"""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(explicit)
    if HF_SNAPSHOT.exists():
        for snap in sorted(HF_SNAPSHOT.iterdir()):
            cand = snap / "actions" / "SHARD_0034.tar.gz"
            if cand.exists():
                return cand  # 符号链接，tarfile 可直接跟随
    raise FileNotFoundError("未在 HF 缓存中找到 SHARD_0034.tar.gz，请用 --shard 显式指定路径")


def summarize_chunk(video: str, chunk: str, data: bytes, meta: dict) -> dict:
    """读取单个 chunk 的 parquet 并生成 manifest 摘要。"""
    df = pl.read_parquet(io.BytesIO(data))
    missing = [c for c in BUTTON_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{video}/{chunk}: 缺失按键列 {missing}")
    j_left = df["j_left"].list.eval(pl.element().abs() > 0.1).list.any()
    return {
        "video": video,
        "chunk": chunk,
        "rows": df.height,
        "n_button_presses": int(sum(df[c].sum() for c in BUTTON_COLS)),
        "j_left_active_frames": int(j_left.sum()),
        "has_processed": False,  # 由外部回填
        "url": meta.get("url") if meta else None,
        "start_frame": meta.get("start_frame") if meta else None,
        "end_frame": meta.get("end_frame") if meta else None,
    }


def main():
    ap = argparse.ArgumentParser(description="从 SHARD_0034 提取指定游戏的手柄标注")
    ap.add_argument("--shard", default=None, help="SHARD_0034.tar.gz 路径（默认自动定位 HF 缓存）")
    ap.add_argument("--game", default=TARGET_GAME, help=f"要提取的游戏名（默认 {TARGET_GAME}）")
    ap.add_argument("--video", default=None, help="只提取指定 video_id（可选，默认全部）")
    ap.add_argument("--limit", type=int, default=None, help="最多提取的 chunk 数（冒烟验证用）")
    args = ap.parse_args()

    shard = resolve_shard_path(args.shard)
    # 输出目录按游戏动态命名，避免不同游戏互相覆盖
    OUT_ROOT = DATA_ROOT / args.game
    print(f"shard : {shard}")
    print(f"game  : {args.game}")
    print(f"out   : {OUT_ROOT}")
    (OUT_ROOT / "raw").mkdir(parents=True, exist_ok=True)

    manifest = {"shard": str(shard), "game": args.game, "videos": {}, "chunks": []}
    video_game: dict[str, str | None] = {}    # video -> game（读到 metadata 后填充）
    chunk_meta: dict[str, dict] = {}          # "video|chunk" -> metadata 摘要
    processed_seen: set[str] = set()          # "video|chunk" 有 actions_processed.parquet
    pending: dict[str, dict[str, bytes]] = {} # video -> {chunk: parquet bytes}（game 未知时缓存）
    n_extracted = 0
    n_members = 0

    def save_chunk(video: str, chunk: str, data: bytes) -> None:
        nonlocal n_extracted
        if args.limit is not None and n_extracted >= args.limit:
            return
        out_dir = OUT_ROOT / "raw" / video
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{chunk}.parquet").write_bytes(data)
        manifest["chunks"].append(summarize_chunk(video, chunk, data, chunk_meta.get(f"{video}|{chunk}")))
        n_extracted += 1
        if n_extracted == 1 or n_extracted % 10 == 0:
            print(f"  extracted {n_extracted} chunks ...", flush=True)

    with tarfile.open(shard, "r:gz") as tf:
        for m in tf:
            n_members += 1
            parts = m.name.split("/")
            if len(parts) != 4:
                continue
            _, video, chunk, fname = parts
            # 视频过滤：指定了 --video 时只处理该视频
            video_ok = args.video is None or video == args.video

            if fname == "metadata.json":
                meta = json.load(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"))
                game = meta.get("game")
                video_game.setdefault(video, game)
                if game == args.game and video_ok:
                    ov = meta["original_video"]
                    chunk_meta[f"{video}|{chunk}"] = {
                        "url": ov.get("url"),
                        "start_frame": ov.get("start_frame"),
                        "end_frame": ov.get("end_frame"),
                    }
                    # 不依赖 chunk_0000：任何目标游戏的 metadata 都可登记视频
                    # （部分视频缺少 chunk_0000 或其 metadata，此前按 chunk_0000 统计会漏）
                    manifest["videos"].setdefault(
                        video,
                        {
                            "game": game,
                            "url": ov.get("url"),
                            "source": ov.get("source"),
                            "resolution": ov.get("resolution"),
                            "controller_type": meta.get("controller_type"),
                        },
                    )
                # metadata 到达 -> 该视频 game 已知，处理 pending 中缓存的 parquet
                if video in pending:
                    for pchunk, pdata in pending.pop(video).items():
                        if game == args.game and video_ok:
                            save_chunk(video, pchunk, pdata)
                continue

            if fname == "actions_processed.parquet":
                if video_game.get(video) == args.game and video_ok:
                    processed_seen.add(f"{video}|{chunk}")
                continue

            if fname != "actions_raw.parquet":
                continue

            # actions_raw.parquet 成员
            if video in video_game:  # game 已知（非首个 chunk）
                if video_game[video] == args.game and video_ok:
                    save_chunk(video, chunk, tf.extractfile(m).read())
            else:  # 首个 chunk 的 parquet 先于 metadata 出现：缓存待定
                pending.setdefault(video, {})[chunk] = tf.extractfile(m).read()

    # 回填 has_processed 与 metadata 信息（parquet 成员先于同 chunk 的 metadata.json 出现）
    for c in manifest["chunks"]:
        key = f"{c['video']}|{c['chunk']}"
        if key in processed_seen:
            c["has_processed"] = True
        if key in chunk_meta:
            c.update(chunk_meta[key])

    manifest["total_chunks_extracted"] = n_extracted
    manifest["total_frames"] = sum(c["rows"] for c in manifest["chunks"])
    manifest["tar_members_scanned"] = n_members
    manifest["chunks_with_processed"] = len(processed_seen)

    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 合并成单表（附溯源列），供统计与评估脚本使用
    # 注意：个别 chunk 的 j_left/j_right 列可能是 Int 类型（全 0 时 parquet 推断），统一 cast
    frames = []
    for c in sorted(manifest["chunks"], key=lambda x: (x["video"], x["chunk"])):
        df = pl.read_parquet(OUT_ROOT / "raw" / c["video"] / f"{c['chunk']}.parquet")
        df = df.with_columns(
            pl.col("j_left").cast(pl.List(pl.Float64)),
            pl.col("j_right").cast(pl.List(pl.Float64)),
            *[pl.col(c2).cast(pl.Int32) for c2 in BUTTON_COLS],
            pl.lit(c["video"]).alias("video"),
            pl.lit(c["chunk"]).alias("chunk"),
            pl.int_range(0, df.height).alias("frame_idx"),
        )
        frames.append(df)
    if frames:
        merged = pl.concat(frames)
        merged.write_parquet(OUT_ROOT / "annotations.parquet")
        print(f"merged annotations: {merged.height} rows -> {OUT_ROOT / 'annotations.parquet'}")

    print(
        f"DONE: {n_extracted} chunks, {manifest['total_frames']} frames "
        f"({len(manifest['videos'])} videos) from {n_members} tar members"
    )


if __name__ == "__main__":
    main()
