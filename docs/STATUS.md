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
- [x] Release layout seen: `{train,val,test}_UseCase_1_(Avoine1)/` with `HSI-Hybercube/` and
      `Annotation/{JSON,PNG}/`, **no RGB folder**. `prepare.py` fixed to resolve `<split>_*` folders
      and synthesize true-colour RGB (2026-10-03).

- [x] Checked the dataset's GitHub repo (2026-10-03): no RGB conversion utility and no separate RGB
      download; cubes are reflectance-calibrated (their loader divides by `reflectance scale factor`).
      Their `json2png.py` revealed Label Studio names drop the parentheses; matching fixed.

## In progress
- [ ] Extracting zips into `data/raw/` (needs `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`; plain unzip
      aborts with a false "zip bomb" error).

## Next
1. After extraction, confirm there is still no `RGB/` folder (`find data/raw -maxdepth 3 -type d`).
2. `python tools/prepare.py --root data/raw --out data/afa_bin5 --bin 5`. Verify in the log: every cube says
   `(polygons)`, the material label list (decides whether `class_mode: material` is possible), fg
   fractions look sane, `rgb=` matches expectations, and `meta.json` has
   wavelengths spanning about 400–1000 nm.
3. Run `configs/smoke.yaml` end to end once (dataloader, val loop, checkpointing are untested on GPU).
4. Short real runs (`--set train.iters=2000 train.eval_every=500`) for A (`rgb_eomt`) and D
   (`hsi_last_blocks`) to confirm val IoU moves and to measure it/s and memory.
5. Full runs: A, D, E, then B, C, F, G, H. Three seeds for A/D/E.
6. Delete the zips in `~/data/afa/` once prepare is verified.

## Untested / open questions
- Real-release folder names and JSON format (see AGENTS.md "fragile" section).
- The synthesized true-colour RGB uses a per-image percentile stretch. Check a few by eye, and
  decide whether a global stretch is fairer for the RGB baseline.
- Throughput and memory for ViT-S at 448 crops / batch 8 with the 60-band input.

## Results
_(fill in: config, seed, val FO-IoU, test FO-IoU / F1 / img-recall / img-FPR, trainable params, FPS)_
