# AGENTS.md: hsi-rgb-adapter

Guide for coding agents working in this repo. Read this first, then `docs/RESEARCH.md` for the
scientific context and `docs/STATUS.md` for where things stand.

## What this project is

Research code that adapts an **RGB-pretrained DINOv2 ViT** to **hyperspectral (HSI) foreign-object
segmentation** with an **encoder-only mask transformer (EoMT)**: no pixel decoder, no transformer
decoder. Queries join the patch tokens in the last ViT blocks, and light heads predict classes and
masks.

Research question: does EoMT's "the pretrained ViT is the decoder's prior" survive the RGB → 300-band
VNIR domain shift, and what is the *smallest* change to the ViT that recovers spectral information RGB
cannot express? The proposed method is a wavelength-aware spectral patch embed plus trainable query
blocks plus a **zero-initialized spectral branch**, compared against RGB, PCA-to-RGB, frozen-ViT,
LoRA and full-finetune baselines.

Dataset: HSI-AgriFoodAnomaly (foreign objects on an oat-product conveyor; 147 cubes, 400–1000 nm,
300 bands, 1000×900, binary pixel masks + Label Studio polygons, paired RGB projections).
Owner: Alfi Willianz (github.com/alfiwillianz). The target is a paper; correctness and clean
ablations matter more than speed.

## Layout

```
hsiadapt/
  envi.py        ENVI .hdr + .bil/.bsq/.bip memmap reader, writer for synthetic data
  data.py        AFAData (memmapped cache, modalities hsi|rgb|pca3), TrainCrops, collate
  model.py       SpectralEoMT, patch-embed inflation, spectral branch, LoRA, mask annealing
  loss.py        Hungarian matching + CE/BCE/dice (Mask2Former/EoMT style), aux losses
  evaluate.py    full-cube sliding-window inference + foreign-object metrics
train.py         train / --eval_only entry point, YAML configs with `base:` inheritance and --set
configs/         base.yaml + one file per ablation (A–H, see README), smoke.yaml
tools/
  prepare.py     raw dataset -> cache (band binning, masks, instances, RGB, stats, PCA, meta.json)
  make_synthetic.py  tiny fake dataset in the real layout
  smoke_test.py  unit checks (no downloads); must pass before and after model changes
```

## Machine and data (owner's setup)

- Training box `hal9000`: Ryzen 5 7500F (6 cores), 32 GB DDR5, RTX 5060 Ti 16 GB, CachyOS, headless,
  reached over SSH/Tailscale. One GPU, so run experiments sequentially.
- Raw data (cold, DRAM-less SSD): `~/Projects/afa/raw/`
- Prepared cache (fast SSD with DRAM): `~/Active/afa/afa_bin5/`. Point configs here with
  `--set data.root=$HOME/Active/afa/afa_bin5`, or edit `configs/base.yaml`.
- Zips downloaded to `~/data/afa/` (to delete after prepare succeeds).
- Never modify or delete anything under `raw/`. Re-running `prepare.py` must stay possible.

## Commands

```bash
pip install -r requirements.txt
python tools/smoke_test.py                                   # must print "all checks passed"
python tools/prepare.py --root <dir with train/ val/ test/> --out ~/Active/afa/afa_bin5 --bin 5
python train.py --config configs/hsi_spec_branch.yaml [--set key.sub=value ...]
python train.py --config configs/<cfg>.yaml --eval_only --ckpt runs/<cfg>/best.pt --split test --save_preds
```

Synthetic end-to-end check: `tools/make_synthetic.py` → `prepare.py --bin 4` → `configs/smoke.yaml`.
Quick real-data sanity run: `--set train.iters=2000 train.eval_every=500`.

Outputs go to `runs/<config name>/`: `config.yaml` (resolved), `log.jsonl` (train + val records),
`best.pt` (best val FO-IoU), `last.pt`, `eval_<split>.json`, `preds_<split>/`.

## Conventions

- **Experiments are configs.** A new ablation is a new YAML with `base: base.yaml` plus the minimal
  overrides. Don't hard-code experiment choices in Python. Keep `configs/base.yaml` as the shared
  default, and when changing a default, check every config that relies on it.
- **Select by val, report test.** Pick checkpoints and hyperparameters on `val` only. Report `test`
  once per final config, and never tune on test.
- Seeds: `seed` in config; report mean ± std over at least 3 seeds for headline numbers.
- Keep RGB / PCA / HSI runs on the **same recipe** (crop, iters, LR, frozen blocks), so the only
  difference is the input. That comparison is the core result.
- Code style: plain PyTorch + timm, type hints, small modules, no new heavy dependencies without a
  reason. Python ≥ 3.10.
- Run `tools/smoke_test.py` after touching `model.py`, `loss.py` or `data.py`. Extend it when adding a
  model feature: zero-init / no-op-at-init properties are cheap to test and easy to break.
- `runs/`, `data/`, `*.npy`, `*.pt` are git-ignored. Don't commit checkpoints or data.
- Commits: small, one logical change each, Conventional Commits style, imperative and lower-case
  after the prefix: `feat:` new capability · `fix:` bug fix · `exp:` new/changed experiment configs or
  run results · `docs:` docs and STATUS updates · `refactor:` · `perf:` · `test:` · `chore:` deps,
  tooling, cleanup. Examples: `fix: handle nested UseCase folder in prepare.py`,
  `exp: add pe_init sweep configs`. The owner reviews diffs, so don't rewrite history on `main`.

## Things that are fragile, read before editing

- `model.py` relies on timm ViT internals: `patch_embed` (NHWC output with `dynamic_img_size=True`),
  `_pos_embed`, `patch_drop`, `norm_pre`, `num_prefix_tokens`, and Block attributes (`norm1`,
  `attn.qkv`, `attn.q_norm`/`k_norm`, `attn.proj`, `ls1/ls2`, `drop_path1/2`, `mlp`, `norm2`).
  `block_forward` re-implements a timm Block forward so it can pass a boolean attention mask. If timm
  is upgraded, re-run the smoke test, and if blocks change, update `block_forward` to match exactly.
- Token order inside the query blocks is `[queries | prefix (cls+registers) | patches]`. Spectral
  injections must only touch patch tokens. Offsets are computed from this order.
- Patch-embed inflation assumes timm channel order **R, G, B**, with centres in `RGB_CENTERS_NM`. Every
  `pe_init` mode must keep "spectrally flat input → same response as RGB" (smoke test checks it).
- Masked attention must anneal to zero by the end of training (`mask_anneal.end < 1`). Inference
  never uses masks, so a model still relying on them at the end would degrade at test time.
- Loss targets are downsampled with **max-pool** (`loss_res`) to keep thin objects (threads, shards)
  from vanishing. Keep this in mind before switching to area/bilinear.
- `prepare.py` assumptions **not yet verified on the real release** (see `docs/STATUS.md`): split
  folders `train/ val/ test/`, cube folder name `HSI-Hybercube` (sic, globbed as `HSI-Hy*`),
  `Annotation/PNG` and `Annotation/JSON`, `RGB/PNG`, mask PNG name = cube stem, Label Studio JSON
  with `image` + `label[].points` in percent and optional `polygonlabels`. If prepare logs
  `(components)` instead of `(polygons)`, the JSON matching failed. Investigate before training
  `class_mode: material`.

## Experiment plan (configs)

A `rgb_eomt` · B `pca3_eomt` · C `hsi_pe_only` · D `hsi_last_blocks` · E `hsi_spec_branch`
(proposed) · E2 `hsi_spec_branch_all` · F `hsi_lora` · G `hsi_full_ft` · H `hsi_no_maskattn`.
Order to run: A and D first (the RGB → HSI gap), then E, then the rest. Extra sweeps: `pe_init`,
`--bin` 3/5/10, `num_query_blocks` 2/4/6, ViT-S vs ViT-B (`vit_base_patch14_reg4_dinov2.lvd142m`,
batch 4 on 16 GB).

Metrics from `evaluate.py`: pooled pixel FO-IoU (model selection), F1, precision, recall, mean
per-image IoU, image-level recall / false-positive rate, Mpix/s. For the paper also report params,
trainable params and FPS. The efficiency comparison against HSI-Adapter is part of the claim.

## When unsure

Ask the owner rather than guessing on anything that changes the experimental protocol (splits,
metrics, what gets frozen, test-set usage). Purely engineering choices (logging, speed, refactors
that keep smoke tests green) don't need to wait.
