# Status

_Last updated: 2026-10-04_

## Done
- [x] Codebase scaffolded: data prep, Spectral EoMT model, loss, eval, train loop, ablation configs.
- [x] `tools/prepare.py` verified on **synthetic** data (band binning matches the raw cube; polygon
      JSON matched despite Label Studio hash prefixes; material labels collected).
- [x] `tools/smoke_test.py` passes on hal9000 (RTX 5060 Ti): inflation preserves RGB response,
      spectral branch is a no-op at init, shapes OK, LoRA trainability OK, loss 44.6 → 0.76 on a
      fixed batch, annealing ends at plain attention.
- [x] Dataset downloaded to `~/data/afa/` (train 37 GiB, val 7 GiB, test 17 GiB), MD5 checked.
- [x] Release layout verified (2026-10-03): `{train,val,test}_UseCase_1_(Avoine1)/` with
      `HSI-Hybercube/`, `Annotation/{JSON,PNG}/` and `RGB/{PNG,TIFF}/`. Cubes: BIL uint16, reflectance
      ×10000, 300 bands from 381.3 nm, lines=1000 × samples=900; masks match (900×1000 WxH, 0/255,
      89 masks for 89 train cubes). `prepare.py` resolves `<split>_*` folders and now refuses
      size-mismatched masks/RGB instead of resizing.
- [x] Checked the dataset's GitHub repo (2026-10-03): no RGB conversion utility (RGB ships inside
      the zips); cubes are reflectance-calibrated (their loader divides by `reflectance scale factor`).
      Their `json2png.py` revealed Label Studio names drop the parentheses; matching fixed.
- [x] Zips extracted into `data/raw/` (needed `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`).

- [x] `prepare.py` ran on the real data (2026-10-03) -> `data/afa_bin5` (15 GB, 60 bands, 385-1011 nm,
      log in `data/prepare.log`). 147 cubes: train 89, val 17, test 41. Findings:
      - `Normal_*` masks are blank 900x1000 RGB PNGs (transposed vs. the cube); all zeros, now accepted as empty.
      - Test `Normal_11/12/13` RGB PNGs are transposed vs. their cubes. They are the anti-transpose
        (corr 0.97-0.99 with cube bands, <=0.2 for every other orientation). `prepare.py` now
        auto-orients RGB (refuses below corr 0.9) and records `rgb_transform` per sample in `meta.json`.
      - Saturated uint16 cubes (max 65535) overflowed float16 when binning; now clipped at 65504. Some
        cubes still hold values >> 10000 (max per-band std 2104), so normalisation is dominated by highlights.
      - Val has no Label Studio JSON (only `Note.txt`): val instances come from connected components.
        Train polygons 80/80 anomalous cubes, test 38/38.
      - Material labels found: `ForeignBody`, `abnormal`. Every train instance label is id 0 (`ForeignBody`),
        so `class_mode: material` has no variety on train; also id 0 collides with "unlabelled" (0).

- [x] RGB/mask overlay checked (2026-10-03): shipped RGB is aligned with the masks. `configs/smoke.yaml`
      and a 2000-iter real run work on GPU: ~4.9 it/s, ~6 GB for ViT-S/448/batch 8 (RGB input).
- [x] **A (`rgb_eomt`) done (2026-10-04)**, 3 seeds, 4000 iters (~14 min each), checkpoint chosen on val.
      Schedule length was selected on val with seed 0 (final val FO-IoU: 2k 0.80, 4k 0.81, 8k 0.73,
      20k 0.42; best-over-run for 20k only 0.64). `base.yaml` now uses `iters: 4000`, `eval_every: 500`
      for all configs. The 20k collapse is unexplained: lower train loss, but the model under-segments
      (precision 0.95, recall 0.42) even on train cubes, missing low-contrast objects and stacking
      several queries on one object. Archived in `runs/rgb_eomt_20k_s0`. Watch for it in HSI runs.

## In progress
- [ ] D (`hsi_last_blocks`) next.

## Next
1. Decide on `class_mode: material` (see findings: effectively one class on train).
2. Check the highlight/saturation handling in normalisation (values >> 10000).
3. Run D, E, then B, C, F, G, H with the same 4000-iter recipe. Three seeds for D/E.
   Check D's schedule length on val too (a new patch embed may want a different length; keep it
   the same for all configs or document why not).
4. Modal volume `afa-bin5` (L4 account) holds the cache for parallel runs; nothing launched there yet.
6. Delete the zips in `~/data/afa/` once prepare is verified.

## Untested / open questions
- Real-release folder names and JSON format (see AGENTS.md "fragile" section).
- Whether the shipped RGB PNGs are pixel-aligned with the cubes (same size is enforced). Visually
  check one overlay of RGB + mask.
- Throughput and memory for ViT-S at 448 crops / batch 8 with the 60-band input.

## Results
Params: 23.57M total, 8.97M trainable (ViT-S/14 reg4, RGB). Test eval ~14 Mpix/s (full cube, sliding window).

| config | seed | val FO-IoU (best) | test FO-IoU | test F1 | test P / R | img-recall | img-FPR |
|---|---|---|---|---|---|---|---|
| A rgb_eomt | 0 | 0.814 | 0.804 | 0.891 | 0.905 / 0.878 | 1.00 | 0.33 |
| A rgb_eomt | 1 | 0.818 | 0.806 | 0.892 | 0.922 / 0.865 | 1.00 | 1.00 |
| A rgb_eomt | 2 | 0.775 | 0.753 | 0.859 | 0.924 / 0.803 | 1.00 | 0.33 |
| **A mean ± std** | | 0.803 ± 0.024 | **0.787 ± 0.030** | 0.881 ± 0.019 | 0.917 / 0.848 | 1.00 | 0.56 ± 0.39 |

img-FPR is over only 3 anomaly-free test images (1 in val), so it is very noisy.
