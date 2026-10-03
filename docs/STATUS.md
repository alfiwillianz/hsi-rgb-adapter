# Status

_Last updated: 2026-10-03_

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

## In progress
- [ ] Verify one RGB + mask overlay visually, then smoke config on GPU.

## Next
1. Decide on `class_mode: material` (see findings: effectively one class on train).
2. Check the highlight/saturation handling in normalisation (values >> 10000).
3. Run `configs/smoke.yaml` end to end once (dataloader, val loop, checkpointing are untested on GPU).
4. Short real runs (`--set train.iters=2000 train.eval_every=500`) for A (`rgb_eomt`) and D
   (`hsi_last_blocks`) to confirm val IoU moves and to measure it/s and memory.
5. Full runs: A, D, E, then B, C, F, G, H. Three seeds for A/D/E.
6. Delete the zips in `~/data/afa/` once prepare is verified.

## Untested / open questions
- Real-release folder names and JSON format (see AGENTS.md "fragile" section).
- Whether the shipped RGB PNGs are pixel-aligned with the cubes (same size is enforced). Visually
  check one overlay of RGB + mask.
- Throughput and memory for ViT-S at 448 crops / batch 8 with the 60-band input.

## Results
_(fill in: config, seed, val FO-IoU, test FO-IoU / F1 / img-recall / img-FPR, trainable params, FPS)_
