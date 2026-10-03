"""Sliding-window inference over full cubes and foreign-object metrics."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from PIL import Image

from .data import AFAData


def _starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


@torch.no_grad()
def predict_image(model, data: AFAData, i: int, tile: int, stride: int, device, batch: int = 8,
                  amp_dtype=torch.bfloat16) -> np.ndarray:
    """Returns (K, H, W) float32 class probability maps for sample i."""
    s = data.samples[i]
    H, W = s["H"], s["W"]
    K = model.num_classes
    acc = np.zeros((K, max(H, tile), max(W, tile)), np.float32)
    cnt = np.zeros((max(H, tile), max(W, tile)), np.float32)
    coords = [(y, x) for y in _starts(H, tile, stride) for x in _starts(W, tile, stride)]
    for b in range(0, len(coords), batch):
        cb = coords[b : b + batch]
        x = torch.from_numpy(np.stack([data.read(i, y, xx, tile, tile) for y, xx in cb])).to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            p = model.semantic(x).float().cpu().numpy()
        for (y, xx), pi in zip(cb, p):
            acc[:, y : y + tile, xx : xx + tile] += pi
            cnt[y : y + tile, xx : xx + tile] += 1
    return (acc / np.maximum(cnt, 1))[:, :H, :W]


@torch.no_grad()
def evaluate(model, data: AFAData, tile: int, stride: int, device, thr: float = 0.5,
             min_area: int = 50, save_dir: str | None = None) -> dict:
    """Foreign-object (any class) metrics. Pixel metrics are pooled over the split;
    image-level detection flags an image if predicted FO area >= min_area pixels."""
    was_training = model.training
    model.eval()
    tp = fp = fn = 0
    per_img_iou, img_tp, img_fp, img_fn, img_tn = [], 0, 0, 0, 0
    t0 = time.time()
    npix = 0
    for i, s in enumerate(data.samples):
        prob = predict_image(model, data, i, tile, stride, device)
        fo = prob.sum(0).clip(0, 1) if prob.shape[0] > 1 else prob[0]
        pred = fo >= thr
        gt = data.masks[i]
        a, b_, c = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        tp, fp, fn = tp + a, fp + b_, fn + c
        npix += gt.size
        if gt.any():
            per_img_iou.append(a / max(a + b_ + c, 1))
        has_gt, has_pred = bool(gt.any()), int(pred.sum()) >= min_area
        img_tp += has_gt and has_pred
        img_fn += has_gt and not has_pred
        img_fp += (not has_gt) and has_pred
        img_tn += (not has_gt) and (not has_pred)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray((fo * 255).astype(np.uint8)).save(os.path.join(save_dir, s["stem"] + "_prob.png"))
    dt = time.time() - t0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * prec * rec / max(prec + rec, 1e-8),
        "precision": prec,
        "recall": rec,
        "mean_img_iou": float(np.mean(per_img_iou)) if per_img_iou else 0.0,
        "img_recall": img_tp / max(img_tp + img_fn, 1),
        "img_fpr": img_fp / max(img_fp + img_tn, 1),
        "mpix_per_s": npix / 1e6 / max(dt, 1e-6),
    }
    model.train(was_training)
    return out
