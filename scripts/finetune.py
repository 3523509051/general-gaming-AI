# -*- coding: utf-8 -*-
"""扩展 A：NitroGen 小样本微调（flow-matching，最小可跑版）。

流程：
  1) 加载官方 ng.pt（load_model），复用其 tokenizer / 图像处理器 / 模型结构；
  2) 从 hades 视频抽 N 帧 + 对齐 annotations 标注，构建 (frames, actions) 样本；
  3) tokenizer.train() 打包 actions → model.train() → forward 得 flow-matching loss → 反向传播；
  4) 保存微调权重 ng_finetuned.pt（供 evaluate.py --ckpt 评估对照）。

用法（仓库根目录，venv python；小样本先验证链路）：
    NitroGen\\.venv\\Scripts\\python.exe scripts\\finetune.py --samples 500 --epochs 1 --batch 4
自测：退出码 0 且打印 loss 数值 / 保存 ng_finetuned.pt 即链路通过。
"""
import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 权重已缓存，跳过 HF 联网检查（与 evaluate.py 一致）

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "NitroGen"))

import numpy as np
import polars as pl
import torch

from nitrogen.inference_session import load_model

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
CKPT = REPO / "NitroGen" / "ng.pt"
OUT_CKPT = REPO / "NitroGen" / "ng_finetuned.pt"

BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "left_stick_x", "left_stick_y",
    "right_shoulder", "right_thumb", "right_trigger", "right_stick_x", "right_stick_y",
    "south", "start", "west", "north",
]  # 仅占位；实际以 annotations 列为准
# 数据集动作列（Hades xboxone：17 键 + 左右摇杆）
DATA_BUTTONS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "right_shoulder", "right_thumb",
    "right_trigger", "south", "start", "west", "north",
]
JOY_COLS = ["j_left", "j_right"]  # 每项 [x, y]


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        import shutil
        p = shutil.which("ffmpeg")
        if p:
            return p
        raise RuntimeError("ffmpeg not found")


def build_samples(game: str, video: str, video_file: Path, fps: int,
                  n_samples: int, seed: int) -> list[dict]:
    """抽 n_samples 帧 + 对齐标注，返回样本列表（frame 路径 + 标注）。"""
    manifest = json.load(open(DATA_ROOT / game / "manifest.json", encoding="utf-8"))
    ann = pl.read_parquet(DATA_ROOT / game / "annotations.parquet")
    chunks = sorted([c for c in manifest["chunks"] if c["video"] == video], key=lambda c: c["chunk"])
    total = sum(c["rows"] for c in chunks)
    off = [0]
    for c in chunks[:-1]:
        off.append(off[-1] + c["rows"])

    rng = random.Random(seed)
    picked = sorted(rng.sample(range(total), k=min(n_samples, total)))
    ann_lookup = {(r["video"], r["chunk"], r["frame_idx"]): r
                  for r in ann.filter(pl.col("video") == video).to_dicts()}

    ffmpeg = find_ffmpeg()
    out_dir = DATA_ROOT / game / "finetune_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for g in picked:
        lo, hi = 0, len(chunks) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if off[mid] <= g:
                lo = mid
            else:
                hi = mid - 1
        c = chunks[lo]
        fid = g - off[lo]
        abs_frame = c["start_frame"] + fid
        out_path = out_dir / f"f{abs_frame:05d}.jpg"
        if not (out_path.exists() and out_path.stat().st_size > 0):
            second = abs_frame / fps
            for _ in range(2):
                subprocess.run([ffmpeg, "-y", "-ss", f"{second:.3f}", "-i", str(video_file),
                                "-frames:v", "1", str(out_path)],
                               capture_output=True, timeout=30)
                if out_path.exists() and out_path.stat().st_size > 0:
                    break
                out_path.unlink(missing_ok=True)
        if not (out_path.exists() and out_path.stat().st_size > 0):
            continue
        r = ann_lookup.get((video, c["chunk"], fid))
        if r is None:
            continue
        samples.append({"frame_path": str(out_path), "ann": r})
    print(f"[finetune] 构建样本 {len(samples)} 帧（目标 {n_samples}）")
    return samples


def to_action(ann) -> dict:
    """把标注行转成 (buttons, j_left, j_right)。"""
    buttons = np.array([int(ann.get(b, 0)) for b in DATA_BUTTONS], dtype=np.float32)  # [17]
    jl = np.array(ann.get("j_left", [0.0, 0.0]), dtype=np.float32)
    jr = np.array(ann.get("j_right", [0.0, 0.0]), dtype=np.float32)
    return {"buttons": buttons, "j_left": jl, "j_right": jr}


def main():
    ap = argparse.ArgumentParser(description="NitroGen 小样本微调")
    ap.add_argument("--game", default="hades")
    ap.add_argument("--video", default="v1805686899")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=500, help="训练帧数（小样本，如 500/1k/5k）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4, help="批次（8GB 显存建议 1~4）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_CKPT))
    args = ap.parse_args()

    # 1) 加载模型
    print("[1/4] loading model ...", flush=True)
    t0 = time.time()
    model, tokenizer, img_proc, ckpt_config, game_mapping, _ = load_model(str(CKPT))
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)
    model.train()
    tokenizer.train()

    # 冻结视觉编码器（8GB 显存必需）
    for name, p in model.named_parameters():
        if "siglip" in name or "vision" in name:
            p.requires_grad = False
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"      可训练参数子模块数: {n_train}（视觉编码器已冻结）", flush=True)

    # 2) 构建样本
    print("[2/4] building samples ...", flush=True)
    video_file = DATA_ROOT / "videos" / f"{args.game}_{args.video}.mp4"
    if not video_file.exists():
        raise SystemExit(f"视频未下载: {video_file}")
    samples = build_samples(args.game, args.video, video_file, args.fps,
                            args.samples, args.seed)
    if not samples:
        raise SystemExit("无可用样本，检查视频/标注")
    # 打乱
    random.Random(args.seed).shuffle(samples)

    # 3) 训练
    print("[3/4] training ...", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    batch = []
    step = 0
    total_loss = 0.0
    for epoch in range(args.epochs):
        for s in samples:
            # 图像 -> pixel_values
            img = img_proc(images=[__import__("PIL").Image.open(s["frame_path"]).convert("RGB")],
                           return_tensors="pt")
            pixels = img["pixel_values"]  # [1, 3, 256, 256]
            act = to_action(s["ann"])
            # 单样本形态（与 tokenizer.encode 约定一致，探针已验证）：
            #   frames=[1,3,256,256]（f=1）, dropped_frames=[1]
            #   buttons=[1,T=18,17]（chunks=1,T=action_horizon）, j_left/j_right=[1,T,2]
            T = 18  # action_horizon
            data = {
                "frames": pixels.numpy()[None],                     # [1,1,3,256,256]（B=1,f=1）
                "dropped_frames": np.array([False]),                # [1]
                "buttons": np.repeat(act["buttons"][None], T, axis=0)[None],   # [1,18,17]
                "j_left": np.repeat(act["j_left"][None], T, axis=0)[None],     # [1,18,2]
                "j_right": np.repeat(act["j_right"][None], T, axis=0)[None],   # [1,18,2]
                "game": args.game if game_mapping else None,
            }
            batch.append((data, act))
            if len(batch) == args.batch:
                # 组装 batch 并 forward
                loss = _train_step(model, tokenizer, batch, args.batch)
                total_loss += loss
                step += 1
                if step % 5 == 0:
                    print(f"      step {step}: loss={loss:.4f}", flush=True)
                batch = []
        if batch:  # 尾部不足一批
            loss = _train_step(model, tokenizer, batch, len(batch))
            total_loss += loss
            step += 1
            print(f"      step {step}: loss={loss:.4f}", flush=True)

    avg_loss = total_loss / max(step, 1)
    print(f"      训练完成: {step} steps, avg_loss={avg_loss:.4f}", flush=True)

    # 4) 保存微调权重（仅模型权重 + ckpt_config）
    print("[4/4] saving ...", flush=True)
    torch.save({"model": model.state_dict(), "ckpt_config": ckpt_config.model_dump()}, args.out)
    print(f"      saved -> {args.out}")
    # 记录训练参数（供链路"跳过微调"校验：权重存在但参数变了必须重训）
    try:
        import json
        meta = {"samples": args.samples, "epochs": args.epochs, "batch": args.batch,
                "lr": args.lr, "seed": args.seed, "game": args.game, "video": args.video}
        Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"      meta -> {args.out}.meta.json")
    except Exception as e:  # noqa: BLE001
        print(f"      meta 写入失败（不影响训练）: {e}")
    print("FINETUNE DONE")
    return 0


def _train_step(model, tokenizer, batch, bsz):
    """forward + backward 一步。

    数据形态（与 tokenizer.encode / model.forward 约定一致，探针已验证单样本）：
      - frames        [f=1, C, H, W]（单样本 encode 内部组装为 [1, f, C, H, W]）
      - dropped_frames[1]
      - buttons       [1, T=18, 17]、j_left [1, T, 2]、j_right [1, T, 2]
        （chunks 维=1；T=action_horizon，逐帧动作复制 18 步）
    注意：tokenizer.encode 内部校验 actions.shape[0]==action_horizon，只支持单样本打包，
    因此 batch 场景逐个样本 encode 后再堆叠成模型输入。
    """
    T = 18  # action_horizon
    td = {}
    # 逐样本 encode（每个样本都是探针验证过的单样本形态）
    tokens = []
    for b in batch:
        data = b[0]
        tokenized = tokenizer.encode(data)
        tokens.append(tokenized)
    # 堆叠 batch 维度
    for k in tokens[0].keys():
        v0 = tokens[0][k]
        if isinstance(v0, np.ndarray):
            stacked = np.stack([t[k] for t in tokens])
            td[k] = torch.from_numpy(stacked)
        elif isinstance(v0, torch.Tensor):
            stacked = torch.stack([t[k] for t in tokens])
            td[k] = stacked
        else:
            td[k] = [t[k] for t in tokens]
    # 送 GPU（vl/sa token 参与 masked_scatter，须与模型同设备）
    for k in td:
        if isinstance(td[k], torch.Tensor):
            td[k] = td[k].to("cuda")
    # images/dropped_images 显式组装并送 GPU（单样本 frames=[1,1,3,256,256] -> [B,1,3,256,256]）
    td["images"] = torch.from_numpy(np.stack([b[0]["frames"][0] for b in batch])).to("cuda")
    td["dropped_images"] = torch.from_numpy(np.stack([b[0]["dropped_frames"] for b in batch])).to("cuda")
    td["embodiment_id"] = torch.zeros(bsz, dtype=torch.long, device="cuda")
    td["game_ids"] = torch.zeros(bsz, dtype=torch.long, device="cuda")
    td["has_real_action"] = torch.ones(bsz, dtype=torch.bool, device="cuda")
    # actions/actions_mask：encode 输出 [T,25]（含 17->25 pad），堆叠后 [B,T,25]
    td["actions"] = td["actions"].float().to("cuda")
    td["actions_mask"] = td["actions_mask"].float().to("cuda")

    out = model(td)
    loss = out["loss"]
    loss.backward()
    return float(loss.detach().cpu())


if __name__ == "__main__":
    sys.exit(main())
