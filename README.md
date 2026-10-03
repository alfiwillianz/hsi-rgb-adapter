# hsi-rgb-adapter

Adapting an **RGB-pretrained DINOv2 ViT** to **hyperspectral foreign-object segmentation** with an
**encoder-only mask transformer** (EoMT): no pixel decoder and no transformer decoder. Learnable
queries join the patch tokens in the last ViT blocks, and light heads read out class and mask
predictions.

Research question: EoMT's claim is that a strongly pretrained ViT *is* the decoder's prior. Does that
survive the shift from RGB to 300-band VNIR spectra, and what is the smallest change to the ViT that
recovers the spectral information RGB cannot express?

Target data: [HSI-AgriFoodAnomaly](https://github.com/lsllabisen/HSI-AgriFoodAnomaly-Dataset)
(147 conveyor cubes, 400–1000 nm, 300 bands, 1000×900, pixel masks + polygons, paired RGB).

## Model

```
HSI cube (C bands) ──► spectral patch embed (C-band conv, init from RGB weights by wavelength)
        │                        │
        │                 ViT blocks 1..L1 (frozen DINOv2)
        │                        │
        │              + N queries ──► ViT blocks L1+1..L (trainable) ──► class head / mask head
        │                                ▲   ▲   ▲   ▲
        └─► spectral branch (per-pixel MLP → patch tokens) ── zero-init injections (optional)
```

* **Spectral patch embed** (`pe_init`): `wavelength` (each band interpolates the nearest RGB
  channels), `wavelength_vis` (non-visible bands start at zero, the default), `mean` (timm-style
  repeat), `zero`. All of them preserve the RGB response for a spectrally flat input.
* **EoMT**: queries are added before the last `num_query_blocks` blocks. The mask head is a
  query MLP dotted with 4×-upsampled patch features. Masked attention is used early in training and
  annealed to plain attention, so inference has no masking.
* **Spectral branch** (`spectral_branch: true`): a per-pixel spectral MLP whose patch tokens are
  added through **zero-initialized** projections (ControlNet / HyperSAM style), so step 0 equals the
  RGB-initialized model. This removes the 3-channel bottleneck without a ViT-Adapter.
* **Freezing**: `none` | `early` (blocks before the queries) | `blocks` (all). Optional LoRA on frozen
  blocks.

## Setup

```bash
pip install -r requirements.txt
# DINOv2 weights download from the Hugging Face hub on first run (timm).
```

1. Download the dataset from <https://doi.org/10.57745/QTLG7X> (non-commercial research use).
2. Build the cache (bands averaged in groups of 5 → 60 bands, float16, about 108 MB per cube):

```bash
python tools/prepare.py --root /path/to/Anomaly_Easy --out data/afa_bin5 --bin 5
```

   This prints the material labels it found in the polygon JSON. If there are any,
   `data.class_mode: material` gives multi-class instance segmentation; otherwise use binary.
3. Sanity-check the code on your GPU (no downloads, about a minute):

```bash
python tools/smoke_test.py
python tools/make_synthetic.py --out data/synthetic_raw
python tools/prepare.py --root data/synthetic_raw --out data/synthetic --bin 4
python train.py --config configs/smoke.yaml
```

## Experiments

| id | config | input | what trains | question |
|----|--------|-------|-------------|----------|
| A | `rgb_eomt` | shipped RGB | query blocks + heads | RGB reference |
| B | `pca3_eomt` | PCA→3 | same | naive HSI→RGB |
| C | `hsi_pe_only` | HSI | patch embed + queries + heads | frozen ViT (PMT predicts collapse) |
| D | `hsi_last_blocks` | HSI | patch embed + query blocks | simplest spectral EoMT |
| E | `hsi_spec_branch` | HSI | D + zero-init spectral branch | **proposed** |
| E2 | `hsi_spec_branch_all` | HSI | branch into every block | where to inject |
| F | `hsi_lora` | HSI | patch embed + LoRA r=16 | PEFT baseline |
| G | `hsi_full_ft` | HSI | everything | upper bound / overfit |
| H | `hsi_no_maskattn` | HSI | D without masked attention | does annealing matter on HSI |

```bash
python train.py --config configs/hsi_spec_branch.yaml
python train.py --config configs/hsi_spec_branch.yaml --eval_only --ckpt runs/hsi_spec_branch/best.pt --split test --save_preds
# overrides:  --set model.pe_init=mean train.iters=10000 model.backbone=vit_base_patch14_reg4_dinov2.lvd142m
```

Metrics (foreign object vs background, sliding window over the full cube): pooled pixel IoU / F1 /
precision / recall, mean per-image IoU, image-level recall and false-positive rate, and throughput.
Also report params and FPS against HSI-Adapter for the efficiency claim.

Suggested extra ablations: `pe_init` (4 modes), `--bin` (3 / 5 / 10), `num_query_blocks` (2 / 4 / 6),
backbone size (S / B), and crop size versus small-object recall (threads and shards are thin compared
with a 14-px patch).

## Notes / known limits

* Defaults are sized for one 16 GB GPU (ViT-S, 448 crops, batch 8, bf16). ViT-B fits with batch 4.
* The train split is small (89 cubes). Crops are foreground-biased (`fg_prob`) and augmented with
  flips and 90° rotations.
* Instance masks come from the polygon JSON when it is available, or from connected components of the
  PNG mask otherwise (`instance_source` in `meta.json`).

## Related work

EoMT (Kerssies et al., CVPR 2025) · PMT (frozen-encoder EoMT) · EoSeg · HSI-Adapter (2025) ·
HyperSAM (2026) · SpectraDINO (2026) · DEFLECT/UPE (ICCV 2025) · SpecTrack (wavelength-aware
inflation) · Panopticon · DOFA · ChannelViT · ExPLoRA.
