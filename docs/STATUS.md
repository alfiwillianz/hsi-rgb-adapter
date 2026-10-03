# Status

_Last updated: 2026-10-05_

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

- [x] **D (`hsi_last_blocks`) done (2026-10-05)**, 3 seeds, 4000 iters. First attempts were bad (val IoU
      ~0.55, below RGB) and traced to the recipe, not the data. Seed-0 val sweep:
      - `pe_init` (wavelength_vis / wavelength / mean): no real difference.
      - 8k iters worse (0.46 final); masked-attn off (H-style) did not help D (0.575) nor RGB (0.79).
      - Zero-shot check: A's trained RGB weights with the patch embed inflated to 60 bands score 0.66
        (0.74 with `clip_z=3`) on HSI val with no HSI training, so the spectral input is fine.
      - Cause: the inflated C-band patch-embed weight counted as a "new" parameter and trained at the
        full LR (1e-4), 10x the RGB patch embed (1e-5), drifting the input to the frozen blocks.
        `train.pe_lr_mult`: 1.0 -> 0.55, 0.1 -> 0.75, 0.03 / 0.01 -> 0.81 (with `clip_z=3`).
      - `data.clip_z` (new, off by default): per-band z-scores of saturated highlights reach ~100.
      Final settings live in `configs/hsi_base.yaml` (`pe_lr_mult: 0.03`, `clip_z: 3`), inherited by all
      HSI configs C-H (not by A or B). Details/curves in `runs/sweepD/`.

## In progress
- [ ] E (`hsi_spec_branch`, the proposed method) next, then C, F, G, H, B.

## Next
1. Decide on `class_mode: material` (see findings: effectively one class on train).
2. Run E, then B, C, F, G, H with the same 4000-iter recipe (HSI ones via `hsi_base.yaml`). Three
   seeds for E. Re-check on val that `pe_lr_mult`/`clip_z` are sensible for E/G (full FT) before the
   final runs; pca3 (B) has its own 3-channel patch embed and keeps the A recipe.
3. The A-vs-D gap is small (test 0.787 vs 0.807, within A's seed spread), so the "RGB -> HSI" claim
   needs harder cases (the dark-film cubes) or a per-object breakdown; look at per-cube IoU.
4. Modal volume `afa-bin5` (L4 account) holds the cache for parallel runs; nothing launched there yet.
5. Delete the zips in `~/data/afa/` once prepare is verified.

## Untested / open questions
- Real-release folder names and JSON format (see AGENTS.md "fragile" section).
- The 20k-iter RGB collapse (see A above) is unexplained; the recipe is only validated at 4000 iters.
- HSI test inference was 2.3-2.7 Mpix/s vs ~14 for RGB (reading 60-band memmaps, probably I/O bound,
  not the model). Re-measure FPS with the cache warm / in RAM before quoting it.
- Training speed: RGB ~4.9 it/s, HSI ~3.9 it/s (ViT-S, 448 crops, batch 8).

## Results
Params: 23.57M total, 8.97M trainable (ViT-S/14 reg4, RGB). Test eval ~14 Mpix/s (full cube, sliding window).

| config | seed | val FO-IoU (best) | test FO-IoU | test F1 | test P / R | img-recall | img-FPR |
|---|---|---|---|---|---|---|---|
| A rgb_eomt | 0 | 0.814 | 0.804 | 0.891 | 0.905 / 0.878 | 1.00 | 0.33 |
| A rgb_eomt | 1 | 0.818 | 0.806 | 0.892 | 0.922 / 0.865 | 1.00 | 1.00 |
| A rgb_eomt | 2 | 0.775 | 0.753 | 0.859 | 0.924 / 0.803 | 1.00 | 0.33 |
| **A mean ± std** | | 0.803 ± 0.024 | **0.787 ± 0.030** | 0.881 ± 0.019 | 0.917 / 0.848 | 1.00 | 0.56 ± 0.39 |

| D hsi_last_blocks | 0 | 0.807 | 0.811 | 0.896 | 0.921 / 0.872 | 1.00 | 0.00 |
| D hsi_last_blocks | 1 | 0.793 | 0.806 | 0.892 | 0.908 / 0.877 | 1.00 | 0.00 |
| D hsi_last_blocks | 2 | 0.801 | 0.803 | 0.890 | 0.907 / 0.874 | 1.00 | 0.67 |
| **D mean ± std** | | 0.800 ± 0.007 | **0.807 ± 0.004** | 0.893 ± 0.003 | 0.912 / 0.874 | 1.00 | 0.22 ± 0.39 |

D: 27.86M params, 13.26M trainable.

img-FPR is over only 3 anomaly-free test images (1 in val), so it is very noisy.
