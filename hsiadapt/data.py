"""Datasets over the cache written by tools/prepare.py.

modality:
  hsi   all binned bands, per-band standardized with train statistics
  rgb   the RGB projection shipped with the dataset, ImageNet-normalized
  pca3  first 3 PCA components of the standardized bands (PCA-to-RGB baseline)
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


class AFAData:
    """Shared loader: memmaps cubes, keeps masks / instance maps in RAM."""

    def __init__(self, root: str, split: str, modality: str = "hsi", class_mode: str = "binary"):
        self.root, self.split, self.modality, self.class_mode = root, split, modality, class_mode
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        self.samples = self.meta["splits"][split]
        self.mean = np.asarray(self.meta["mean"], np.float32)[:, None, None]
        self.std = np.asarray(self.meta["std"], np.float32)[:, None, None]
        self.pca = np.asarray(self.meta["pca_components"], np.float32)
        self.pca_mean = np.asarray(self.meta["pca_mean"], np.float32)[:, None, None]
        self.pca_std = np.asarray(self.meta["pca_std"], np.float32)[:, None, None]
        d = os.path.join(root, split)
        self.cubes, self.rgbs, self.masks, self.insts = [], [], [], []
        for s in self.samples:
            if modality == "rgb":
                if not s.get("rgb"):
                    raise FileNotFoundError(f"No RGB projection for {s['stem']}")
                self.rgbs.append(np.array(Image.open(os.path.join(d, s["rgb"])).convert("RGB")))
            else:
                self.cubes.append(np.load(os.path.join(d, s["stem"] + ".npy"), mmap_mode="r"))
            self.masks.append(np.array(Image.open(os.path.join(d, s["stem"] + "_mask.png"))) > 127)
            self.insts.append(np.array(Image.open(os.path.join(d, s["stem"] + "_inst.png"))).astype(np.int32))
        self.num_material = len(self.meta.get("material_labels", []))

    @property
    def in_chans(self) -> int:
        return self.meta["num_bands"] if self.modality == "hsi" else 3

    @property
    def num_classes(self) -> int:
        return max(self.num_material, 1) if self.class_mode == "material" else 1

    def read(self, i: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """Normalized (C, h, w) float32 crop; zero-padded past the image border."""
        s = self.samples[i]
        H, W = s["H"], s["W"]
        y1, x1 = min(y0 + h, H), min(x0 + w, W)
        if self.modality == "rgb":
            a = self.rgbs[i][y0:y1, x0:x1].transpose(2, 0, 1).astype(np.float32) / 255.0
            a = (a - IMAGENET_MEAN) / IMAGENET_STD
        else:
            a = (np.asarray(self.cubes[i][:, y0:y1, x0:x1], np.float32) - self.mean) / self.std
            if self.modality == "pca3":
                a = np.einsum("kc,chw->khw", self.pca, a)
                a = (a - self.pca_mean) / self.pca_std
        if a.shape[1] != h or a.shape[2] != w:
            a = np.pad(a, ((0, 0), (0, h - a.shape[1]), (0, w - a.shape[2])))
        return a

    def targets(self, i: int, y0: int, x0: int, h: int, w: int):
        inst = np.zeros((h, w), np.int32)
        src = self.insts[i][y0 : y0 + h, x0 : x0 + w]
        inst[: src.shape[0], : src.shape[1]] = src
        return inst


class TrainCrops(Dataset):
    def __init__(self, data: AFAData, crop: int, length: int, fg_prob: float = 0.7,
                 augment: bool = True, min_instance_px: int = 4):
        self.d, self.crop, self.length = data, crop, length
        self.fg_prob, self.augment, self.min_px = fg_prob, augment, min_instance_px
        self.fg_coords = [np.argwhere(m) for m in data.masks]
        # sample cubes with foreground more often but never starve the anomaly-free ones
        w = np.array([1.0 + (len(c) > 0) for c in self.fg_coords])
        self.weights = w / w.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        d, c = self.d, self.crop
        i = int(np.random.choice(len(d.samples), p=self.weights))
        H, W = d.samples[i]["H"], d.samples[i]["W"]
        fg = self.fg_coords[i]
        if len(fg) and random.random() < self.fg_prob:
            cy, cx = fg[random.randrange(len(fg))]
            y0 = int(np.clip(cy - random.randint(c // 8, c - c // 8), 0, max(H - c, 0)))
            x0 = int(np.clip(cx - random.randint(c // 8, c - c // 8), 0, max(W - c, 0)))
        else:
            y0, x0 = random.randint(0, max(H - c, 0)), random.randint(0, max(W - c, 0))
        img = d.read(i, y0, x0, c, c)
        inst = d.targets(i, y0, x0, c, c)
        if self.augment:
            if random.random() < 0.5:
                img, inst = img[:, :, ::-1], inst[:, ::-1]
            if random.random() < 0.5:
                img, inst = img[:, ::-1, :], inst[::-1, :]
            k = random.randint(0, 3)
            img, inst = np.rot90(img, k, (1, 2)), np.rot90(inst, k)
        return to_sample(img, inst, d.samples[i]["instance_labels"], d.class_mode, self.min_px)


def to_sample(img, inst, inst_labels, class_mode, min_px=1):
    ids = [v for v in np.unique(inst) if v != 0]
    masks, labels = [], []
    for v in ids:
        m = inst == v
        if m.sum() < min_px:
            continue
        masks.append(m)
        lab = inst_labels[v - 1] if class_mode == "material" and v - 1 < len(inst_labels) else 0
        labels.append(lab)
    h, w = inst.shape
    return {
        "image": torch.from_numpy(np.ascontiguousarray(img)),
        "masks": torch.from_numpy(np.stack(masks)) if masks else torch.zeros(0, h, w, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
        "semantic": torch.from_numpy(np.ascontiguousarray(inst > 0)),
    }


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "semantic": torch.stack([b["semantic"] for b in batch]),
        "targets": [{"masks": b["masks"], "labels": b["labels"]} for b in batch],
    }
