"""Create a tiny fake dataset in the HSI-AgriFoodAnomaly layout, for smoke tests.

  python tools/make_synthetic.py --out data/synthetic_raw
  python tools/prepare.py --root data/synthetic_raw --out data/synthetic --bin 4
  python train.py --config configs/smoke.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hsiadapt.envi import write_cube  # noqa: E402

MATERIALS = ["Plastic", "Textile", "Metal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--bands", type=int, default=40)
    ap.add_argument("--H", type=int, default=150)
    ap.add_argument("--W", type=int, default=170)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 1000, args.bands)
    background = 2000 + 800 * np.sin(wl / 90.0)  # "oat flakes"
    # materials look like the background in the visible range but differ in NIR
    mats = [background + np.where(wl > 700, 600 * (k + 1), 40 * k) for k in range(len(MATERIALS))]
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        d = os.path.join(args.out, split)
        for sub in ("HSI-Hybercube", "Annotation/PNG", "Annotation/JSON", "RGB/PNG"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        items = []
        for i in range(n):
            stem = f"UseCase_1_(Avoine1)_Anomaly_Easy_{split}_{i}"
            cube = background[:, None, None] + rng.normal(0, 60, (args.bands, args.H, args.W))
            mask = Image.new("L", (args.W, args.H), 0)
            draw = ImageDraw.Draw(mask)
            labels = []
            for _ in range(rng.integers(0 if i == 0 else 1, 4)):
                k = int(rng.integers(len(MATERIALS)))
                cx, cy = rng.integers(15, args.W - 15), rng.integers(15, args.H - 15)
                rx, ry = rng.integers(4, 14), rng.integers(2, 10)
                poly = [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
                obj = Image.new("L", (args.W, args.H), 0)
                ImageDraw.Draw(obj).polygon(poly, fill=1)
                om = np.array(obj, bool)
                cube[:, om] = mats[k][:, None] + rng.normal(0, 60, (args.bands, om.sum()))
                draw.polygon(poly, fill=255)
                labels.append({
                    "points": [[100 * x / args.W, 100 * y / args.H] for x, y in poly],
                    "polygonlabels": [MATERIALS[k]],
                    "original_width": args.W, "original_height": args.H,
                })
            write_cube(os.path.join(d, "HSI-Hybercube", stem), np.clip(cube, 0, 65535).astype(np.uint16), wl.tolist())
            mask.save(os.path.join(d, "Annotation/PNG", stem + ".png"))
            vis = [np.argmin(abs(wl - w)) for w in (610, 545, 465)]
            rgb = cube[vis].transpose(1, 2, 0)
            rgb = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-6) * 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(d, "RGB/PNG", stem + ".png"))
            items.append({"image": f"/data/upload/1/abc123-{stem}.png", "label": labels})
        with open(os.path.join(d, "Annotation/JSON", f"{split}_synthetic.json"), "w") as f:
            json.dump(items, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
