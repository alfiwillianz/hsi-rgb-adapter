"""Mask-classification loss (Mask2Former/EoMT style) with Hungarian matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _pair_bce(logits, targets):
    # logits (Q, P), targets (N, P) -> (Q, N) mean BCE
    pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    return (pos @ targets.T + neg @ (1 - targets).T) / logits.shape[1]


def _pair_dice(logits, targets):
    p = logits.sigmoid()
    num = 2 * p @ targets.T
    den = p.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (num + 1) / (den + 1)


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, w_class: float = 2.0, w_bce: float = 5.0,
                 w_dice: float = 5.0, eos_coef: float = 0.1, loss_res: int = 224):
        super().__init__()
        self.K = num_classes
        self.w_class, self.w_bce, self.w_dice = w_class, w_bce, w_dice
        self.loss_res = loss_res
        w = torch.ones(num_classes + 1)
        w[-1] = eos_coef
        self.register_buffer("class_weight", w)

    def _prep_targets(self, targets, device):
        out = []
        for t in targets:
            m = t["masks"].to(device).float()
            if m.numel() and m.shape[-1] != self.loss_res:
                # max-pool keeps thin objects (threads, shards) alive at low resolution
                m = F.adaptive_max_pool2d(m[None], self.loss_res)[0]
            out.append({"masks": m.flatten(1), "labels": t["labels"].to(device)})
        return out

    @torch.no_grad()
    def _match(self, cls, mlog, tgt):
        idx = []
        for b, t in enumerate(tgt):
            if t["labels"].numel() == 0:
                idx.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            prob = cls[b].float().softmax(-1)[:, t["labels"]]
            C = (-self.w_class * prob + self.w_bce * _pair_bce(mlog[b].float(), t["masks"])
                 + self.w_dice * _pair_dice(mlog[b].float(), t["masks"]))
            qi, ti = linear_sum_assignment(C.cpu().numpy())
            idx.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(ti, dtype=torch.long)))
        return idx

    def _single(self, cls, mlog, tgt, num_masks):
        B, Q = cls.shape[:2]
        mlog = F.interpolate(mlog.float(), size=(self.loss_res, self.loss_res), mode="bilinear",
                             align_corners=False).flatten(2)
        idx = self._match(cls, mlog, tgt)
        tcls = torch.full((B, Q), self.K, dtype=torch.long, device=cls.device)
        pm, tm = [], []
        for b, (qi, ti) in enumerate(idx):
            if len(qi):
                tcls[b, qi] = tgt[b]["labels"][ti]
                pm.append(mlog[b, qi])
                tm.append(tgt[b]["masks"][ti])
        l_cls = F.cross_entropy(cls.float().transpose(1, 2), tcls, weight=self.class_weight)
        if pm:
            pm, tm = torch.cat(pm), torch.cat(tm)
            l_bce = F.binary_cross_entropy_with_logits(pm, tm, reduction="none").mean(1).sum() / num_masks
            p = pm.sigmoid()
            l_dice = (1 - (2 * (p * tm).sum(1) + 1) / (p.sum(1) + tm.sum(1) + 1)).sum() / num_masks
        else:
            l_bce = l_dice = mlog.sum() * 0.0
        return {"cls": l_cls, "bce": l_bce, "dice": l_dice}

    def forward(self, outputs, targets):
        tgt = self._prep_targets(targets, outputs[-1][0].device)
        num_masks = max(sum(t["labels"].numel() for t in tgt), 1)
        total, logs = 0.0, {}
        for li, (cls, mlog) in enumerate(outputs):
            l = self._single(cls, mlog, tgt, num_masks)
            layer = self.w_class * l["cls"] + self.w_bce * l["bce"] + self.w_dice * l["dice"]
            total = total + layer
            if li == len(outputs) - 1:
                logs = {k: float(v.detach()) for k, v in l.items()}
        logs["total"] = float(total.detach())
        return total, logs
