"""Convert the raw HSI-AgriFoodAnomaly release into a fast training cache.

For every cube it writes
  <out>/<split>/<stem>.npy        (C', H, W) float16, bands averaged in groups of --bin
  <out>/<split>/<stem>_mask.png   binary foreign-object mask (0/255)
  <out>/<split>/<stem>_inst.png   uint16 instance map (0 = background)
  <out>/<split>/<stem>_rgb.png    RGB projection shipped with the dataset (if found)
and a <out>/meta.json with binned wavelengths, train-set per-band mean/std, a 3-component
PCA (for the PCA-to-RGB baseline), material label names and the sample list.

Usage:
  python tools/prepare.py --root data/raw --out data/afa_bin5 --bin 5

Split folders may be named exactly train/ val/ test/ or train_*/ val_*/ test_*/ (as in the release).
If a cube has no RGB projection, a true-colour composite is synthesized from the cube.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hsiadapt.envi import open_cube, wavelengths  # noqa: E402


def find_dir(split_dir: str, *patterns: str) -> str | None:
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(split_dir, p)))
        if hits:
            return hits[0]
    return None


def split_dir(root: str, split: str) -> str:
    """Resolve <root>/<split>, or <root>/<split>_* (the release ships e.g. 'train_UseCase_1_(Avoine1)')."""
    exact = os.path.join(root, split)
    if os.path.isdir(exact):
        return exact
    hits = sorted(d for d in glob.glob(os.path.join(glob.escape(root), split + "_*")) if os.path.isdir(d))
    if len(hits) > 1:
        raise RuntimeError(f"Several folders match split '{split}': {hits}")
    return hits[0] if hits else exact


def pseudo_rgb(cube, wl: list[float] | None) -> np.ndarray:
    """True-colour composite from the bands nearest 610/545/465 nm, 0.5-99.5 percentile stretch
    per channel (per image). Used only when the release has no RGB projection."""
    C = cube.shape[0]
    if wl:
        idx = [int(np.argmin(np.abs(np.asarray(wl) - t))) for t in (610.0, 545.0, 465.0)]
    else:
        idx = [int(C * f) for f in (0.35, 0.25, 0.12)]
    out = []
    for i in idx:
        b = np.asarray(cube[i], np.float32)
        lo, hi = np.percentile(b[::4, ::4], (0.5, 99.5))
        out.append(np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1))
    return (np.stack(out, -1) * 255).astype(np.uint8)


def load_label_studio(json_dir: str | None) -> dict[str, list[dict]]:
    """Map image stem -> list of {'points': [[x%,y%],...], 'label': str|None, 'W','H'}."""
    out: dict[str, list[dict]] = {}
    if not json_dir:
        return out
    for jf in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        with open(jf) as f:
            data = json.load(f)
        for item in data if isinstance(data, list) else [data]:
            stem = os.path.splitext(os.path.basename(str(item.get("image", ""))))[0]
            polys = []
            for lab in item.get("label", []) or []:
                pts = lab.get("points")
                if not pts:
                    continue
                names = lab.get("polygonlabels") or lab.get("labels") or []
                polys.append({
                    "points": pts,
                    "label": names[0] if names else None,
                    "W": lab.get("original_width"),
                    "H": lab.get("original_height"),
                })
            out[stem] = polys
    return out


def match_polys(stem: str, table: dict[str, list[dict]]) -> list[dict] | None:
    if stem in table:
        return table[stem]
    for k, v in table.items():  # Label Studio sometimes prefixes "<hash>-"
        if k.endswith(stem) or stem.endswith(k):
            return v
    return None


def rasterize(polys: list[dict], H: int, W: int, label_ids: dict[str, int]):
    inst = Image.new("I;16", (W, H), 0)
    draw = ImageDraw.Draw(inst)
    labels = []
    for i, p in enumerate(polys, start=1):
        pw, ph = p["W"] or W, p["H"] or H
        xy = [(x * pw / 100.0 * W / pw, y * ph / 100.0 * H / ph) for x, y in p["points"]]
        if len(xy) >= 3:
            draw.polygon(xy, fill=i)
        labels.append(label_ids.get(p["label"], 0) if p["label"] else 0)
    return np.array(inst, dtype=np.uint16), labels


def load_mask(path: str | None, H: int, W: int) -> np.ndarray:
    if not path or not os.path.exists(path):
        return np.zeros((H, W), np.uint8)
    m = Image.open(path).convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def bin_cube(cube, out_path: str, b: int, chunk: int = 64) -> int:
    C, H, W = cube.shape
    Cb = C // b
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(Cb, H, W))
    for y in range(0, H, chunk):
        blk = np.asarray(cube[: Cb * b, y : y + chunk, :], dtype=np.float32)
        out[:, y : y + chunk, :] = blk.reshape(Cb, b, -1, W).mean(1).astype(np.float16)
    out.flush()
    return Cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing train/ val/ test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", type=int, default=5, help="average every N adjacent bands")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--stat_stride", type=int, default=8, help="pixel subsampling for stats/PCA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta: dict = {"bin": args.bin, "splits": {}, "material_labels": []}
    label_ids: dict[str, int] = {}
    wl_binned = None

    # first pass over JSON to collect material label names (stable ids)
    for split in args.splits:
        sd = split_dir(args.root, split)
        tbl = load_label_studio(find_dir(sd, "Annotation/JSON", "Annotation/json"))
        for polys in tbl.values():
            for p in polys:
                if p["label"] and p["label"] not in label_ids:
                    label_ids[p["label"]] = len(label_ids)
    meta["material_labels"] = sorted(label_ids, key=label_ids.get)

    for split in args.splits:
        sd = split_dir(args.root, split)
        cube_dir = find_dir(sd, "HSI-Hy*", "HSI*", "*cube*")
        mask_dir = find_dir(sd, "Annotation/PNG", "Annotation/png")
        rgb_dir = find_dir(sd, "RGB/PNG", "RGB")
        polys_tbl = load_label_studio(find_dir(sd, "Annotation/JSON", "Annotation/json"))
        if cube_dir is None:
            print(f"[skip] no cube dir in {sd}")
            continue
        od = os.path.join(args.out, split)
        os.makedirs(od, exist_ok=True)
        samples = []
        for hdr in sorted(glob.glob(os.path.join(cube_dir, "*.hdr"))):
            stem = os.path.basename(hdr)
            for ext in (".bil.hdr", ".bsq.hdr", ".bip.hdr", ".hdr"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            cube, h = open_cube(hdr)
            C, H, W = cube.shape
            Cb = bin_cube(cube, os.path.join(od, stem + ".npy"), args.bin)
            wl = wavelengths(h)
            if wl is not None and wl_binned is None:
                wl_binned = np.asarray(wl[: Cb * args.bin]).reshape(Cb, args.bin).mean(1).tolist()

            mask = load_mask(os.path.join(mask_dir, stem + ".png") if mask_dir else None, H, W)
            Image.fromarray(mask * 255).save(os.path.join(od, stem + "_mask.png"))

            polys = match_polys(stem, polys_tbl)
            if polys:
                inst, inst_labels = rasterize(polys, H, W, label_ids)
                inst[mask == 0] = 0  # trust the PNG mask for extent
                src = "polygons"
            else:
                inst, n = ndimage.label(mask, structure=np.ones((3, 3)))
                inst = inst.astype(np.uint16)
                inst_labels = [0] * int(n)
                src = "components"
            Image.fromarray(inst).save(os.path.join(od, stem + "_inst.png"))

            rgb_path = None
            if rgb_dir:
                cand = os.path.join(rgb_dir, stem + ".png")
                if os.path.exists(cand):
                    im = Image.open(cand).convert("RGB")
                    if im.size != (W, H):
                        im = im.resize((W, H), Image.BILINEAR)
                    rgb_path = stem + "_rgb.png"
                    im.save(os.path.join(od, rgb_path))
            rgb_source = "release" if rgb_path else "synthesized_truecolor"
            if rgb_path is None:
                rgb_path = stem + "_rgb.png"
                Image.fromarray(pseudo_rgb(cube, wl)).save(os.path.join(od, rgb_path))
            samples.append({
                "stem": stem, "H": H, "W": W, "fg_pixels": int(mask.sum()),
                "num_instances": int(inst.max()), "instance_labels": inst_labels,
                "instance_source": src, "rgb": rgb_path, "rgb_source": rgb_source,
            })
            print(f"[{split}] {stem}: {C}->{Cb} bands, {H}x{W}, fg={mask.mean():.4f}, "
                  f"inst={inst.max()} ({src}), rgb={rgb_source}", flush=True)
        meta["splits"][split] = samples
        meta["num_bands"] = Cb if samples else meta.get("num_bands")

    # train statistics: per-band mean/std and 3-comp PCA on normalized pixels
    tr = meta["splits"].get("train", [])
    if tr:
        pix = []
        for s in tr:
            arr = np.load(os.path.join(args.out, "train", s["stem"] + ".npy"), mmap_mode="r")
            sub = np.asarray(arr[:, :: args.stat_stride, :: args.stat_stride], dtype=np.float32)
            pix.append(sub.reshape(sub.shape[0], -1))
        pix = np.concatenate(pix, 1)
        mean, std = pix.mean(1), pix.std(1) + 1e-6
        z = (pix - mean[:, None]) / std[:, None]
        cov = z @ z.T / z.shape[1]
        evals, evecs = np.linalg.eigh(cov)
        comps = evecs[:, ::-1][:, :3].T  # (3, C)
        proj = comps @ z
        meta.update(
            mean=mean.tolist(), std=std.tolist(), pca_components=comps.tolist(),
            pca_mean=proj.mean(1).tolist(), pca_std=(proj.std(1) + 1e-6).tolist(),
            pca_explained=(evals[::-1][:3] / evals.sum()).tolist(),
        )
    meta["wavelengths"] = wl_binned
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("material labels:", meta["material_labels"] or "(none found -> binary only)")
    print("wrote", os.path.join(args.out, "meta.json"))


if __name__ == "__main__":
    main()
