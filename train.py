"""Train / evaluate Spectral EoMT on the prepared HSI-AgriFoodAnomaly cache.

  python train.py --config configs/hsi_spec_branch.yaml
  python train.py --config configs/hsi_spec_branch.yaml --set train.iters=200 model.backbone=vit_base_patch14_reg4_dinov2.lvd142m
  python train.py --config configs/hsi_spec_branch.yaml --eval_only --ckpt runs/hsi_spec_branch/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hsiadapt.data import AFAData, TrainCrops, collate
from hsiadapt.evaluate import evaluate
from hsiadapt.loss import SetCriterion
from hsiadapt.model import SpectralEoMT, mask_anneal_probs


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        cfg = deep_merge(parent, cfg)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: dict, sets: list[str]):
    for s in sets:
        key, val = s.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)


def _seed_worker(worker_id: int):
    s = torch.initial_seed() % 2**31  # distinct per worker and per run
    np.random.seed(s)
    random.seed(s)


def build_model(cfg: dict, data: AFAData) -> SpectralEoMT:
    m = cfg["model"]
    return SpectralEoMT(
        backbone=m["backbone"], pretrained=m.get("pretrained", True), in_chans=data.in_chans,
        wavelengths=data.meta.get("wavelengths"), modality=cfg["data"]["modality"],
        img_size=cfg["data"]["crop"], num_classes=data.num_classes,
        num_queries=m.get("num_queries", 32), num_query_blocks=m.get("num_query_blocks", 4),
        pe_init=m.get("pe_init", "wavelength_vis"), spectral_branch=m.get("spectral_branch", False),
        spectral_dim=m.get("spectral_dim", 128), inject=m.get("inject", "query_blocks"),
        freeze=m.get("freeze", "early"), train_patch_embed=m.get("train_patch_embed", True),
        lora_rank=m.get("lora_rank", 0), lora_alpha=m.get("lora_alpha", 16.0),
        num_upscale=m.get("num_upscale", 2), drop_path=m.get("drop_path", 0.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.iters=500")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(cfg.get("out_dir", "runs"), name)
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dc, tc, ec = cfg["data"], cfg["train"], cfg.get("eval", {})
    train_data = AFAData(dc["root"], "train", dc["modality"], dc.get("class_mode", "binary"))
    model = build_model(cfg, train_data).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params: {n_all/1e6:.2f}M total, {n_train/1e6:.2f}M trainable | device={device}")

    tile, stride = dc["crop"], int(dc["crop"] * ec.get("overlap_stride", 0.75))

    if args.eval_only:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["model"])
        data = AFAData(dc["root"], args.split, dc["modality"], dc.get("class_mode", "binary"))
        res = evaluate(model, data, tile, stride, device, ec.get("thr", 0.5), ec.get("min_area", 50),
                       os.path.join(out_dir, f"preds_{args.split}") if args.save_preds else None)
        print(json.dumps(res, indent=1))
        with open(os.path.join(out_dir, f"eval_{args.split}.json"), "w") as f:
            json.dump(res, f, indent=1)
        return

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    val_data = AFAData(dc["root"], "val", dc["modality"], dc.get("class_mode", "binary"))
    iters, bs = tc["iters"], tc["batch_size"]
    ds = TrainCrops(train_data, dc["crop"], iters * bs, dc.get("fg_prob", 0.7))
    dl = DataLoader(ds, batch_size=bs, num_workers=tc.get("workers", 4), collate_fn=collate,
                    drop_last=True, persistent_workers=tc.get("workers", 4) > 0,
                    worker_init_fn=_seed_worker)
    crit = SetCriterion(train_data.num_classes, loss_res=tc.get("loss_res", dc["crop"] // 2)).to(device)
    opt = torch.optim.AdamW(model.param_groups(tc["lr"], tc.get("backbone_lr_mult", 0.1),
                                               tc.get("weight_decay", 0.05)))
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = tc.get("warmup", 200)
    amp = device.type == "cuda"
    ma = cfg["model"].get("mask_anneal", {"start": 0.1, "end": 0.6})
    use_masked = cfg["model"].get("masked_attn", True)

    best, log_f = -1.0, open(os.path.join(out_dir, "log.jsonl"), "a")
    model.train()
    t0 = time.time()
    for step, batch in enumerate(dl, start=1):
        f = step / iters
        scale = step / warm if step < warm else (1 - (step - warm) / max(iters - warm, 1)) ** 0.9
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        mp = mask_anneal_probs(step, iters, model.nqb, ma["start"], ma["end"]) if use_masked else None
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            outs = model(x, mp)
        loss, logs = crit(outs, batch["targets"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip", 1.0))
        opt.step()
        if step % tc.get("log_every", 20) == 0 or step == 1:
            rec = {"step": step, **logs, "gnorm": float(gn), "lr": opt.param_groups[0]["lr"],
                   "mask_p": mp, "it_s": step / (time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % tc.get("eval_every", 1000) == 0 or step == iters:
            res = evaluate(model, val_data, tile, stride, device, ec.get("thr", 0.5), ec.get("min_area", 50))
            res["step"] = step
            print("[val]", json.dumps(res), flush=True)
            log_f.write(json.dumps({"val": res}) + "\n"); log_f.flush()
            state = {"model": model.state_dict(), "step": step, "val": res}
            torch.save(state, os.path.join(out_dir, "last.pt"))
            if res["iou"] > best:
                best = res["iou"]
                torch.save(state, os.path.join(out_dir, "best.pt"))
        if step >= iters:
            break
    print(f"[{name}] done. best val FO-IoU={best:.4f}. Test with:\n"
          f"  python train.py --config {args.config} --eval_only --ckpt {out_dir}/best.pt --split test")


if __name__ == "__main__":
    main()
