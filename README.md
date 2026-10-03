# hsi-rgb-adapter

Adapting an **RGB-pretrained DINOv2 ViT** [1], [2] to **hyperspectral foreign-object segmentation**
with an **encoder-only mask transformer** (EoMT) [3]: no pixel decoder and no transformer decoder. Learnable
queries join the patch tokens in the last ViT blocks, and light heads read out class and mask
predictions.

Research question: EoMT's claim is that a strongly pretrained ViT *is* the decoder's prior. Does that
survive the shift from RGB to 300-band VNIR spectra, and what is the smallest change to the ViT that
recovers the spectral information RGB cannot express?

Target data: HSI-AgriFoodAnomaly [4]
(147 conveyor cubes, 400–1000 nm, 300 bands, 1000×900, pixel masks + polygons, paired RGB images).

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
  channels, cf. [5]), `wavelength_vis` (non-visible bands start at zero, the default; init choice
  studied in [6]), `mean` (timm-style repeat), `zero`. All of them preserve the RGB response for a
  spectrally flat input.
* **EoMT** [3]: queries are added before the last `num_query_blocks` blocks. The mask head is a
  query MLP dotted with 4×-upsampled patch features. Masked attention [7] is used early in training and
  annealed to plain attention, so inference has no masking.
* **Spectral branch** (`spectral_branch: true`): a per-pixel spectral MLP whose patch tokens are
  added through **zero-initialized** projections (ControlNet [8] / HyperSAM [9] style), so step 0 equals the
  RGB-initialized model. This removes the 3-channel bottleneck without a ViT-Adapter as in [10].
* **Freezing**: `none` | `early` (blocks before the queries) | `blocks` (all). Optional LoRA [11] on frozen
  blocks.

## Setup

```bash
pip install -r requirements.txt
# DINOv2 weights download from the Hugging Face hub on first run (timm).
```

1. Download the dataset [4] from <https://doi.org/10.57745/QTLG7X> (non-commercial research use) and
   extract the three zips into `data/raw/`. Info-ZIP may refuse them as a "zip bomb" (false positive):
   use `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip ...` or `bsdtar -xf ...`.
2. Build the cache (bands averaged in groups of 5 → 60 bands, float16, about 108 MB per cube):

```bash
python tools/prepare.py --root data/raw --out data/afa_bin5 --bin 5
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
| C | `hsi_pe_only` | HSI | patch embed + queries + heads (cf. Panopticon-PE [12]) | frozen ViT (PMT [13] predicts collapse) |
| D | `hsi_last_blocks` | HSI | patch embed + query blocks | simplest spectral EoMT |
| E | `hsi_spec_branch` | HSI | D + zero-init spectral branch | **proposed** |
| E2 | `hsi_spec_branch_all` | HSI | branch into every block | where to inject |
| F | `hsi_lora` | HSI | patch embed + LoRA r=16 [11] | PEFT baseline |
| G | `hsi_full_ft` | HSI | everything | upper bound / overfit |
| H | `hsi_no_maskattn` | HSI | D without masked attention | does annealing matter on HSI |

```bash
python train.py --config configs/hsi_spec_branch.yaml
python train.py --config configs/hsi_spec_branch.yaml --eval_only --ckpt runs/hsi_spec_branch/best.pt --split test --save_preds
# overrides:  --set model.pe_init=mean train.iters=10000 model.backbone=vit_base_patch14_reg4_dinov2.lvd142m
```

Metrics (foreign object vs background, sliding window over the full cube): pooled pixel IoU / F1 /
precision / recall, mean per-image IoU, image-level recall and false-positive rate, and throughput.
Also report params and FPS against HSI-Adapter [10] for the efficiency claim.

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

Encoder-only segmentation: EoMT [3], its frozen-encoder follow-up PMT [13], and EoSeg [14] (medical).
RGB foundation models adapted to spectral data: HSI-Adapter [10], HyperSAM [9], SpectraDINO [15],
DEFLECT [6], SpecTrack [5]. Spectrum-flexible models: Panopticon [12], DOFA [16], ChannelViT [17].
Domain adaptation of ViTs: ExPLoRA [18]. The dataset's own baseline [4] is patch-level classification.

## References

[1] M. Oquab *et al.*, "DINOv2: Learning robust visual features without supervision," *Trans. Mach.
Learn. Res.*, 2024.

[2] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, "Vision transformers need registers," in *Proc.
Int. Conf. Learn. Represent. (ICLR)*, 2024.

[3] T. Kerssies *et al.*, "Your ViT is secretly an image segmentation model," in *Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit. (CVPR)*, 2025, pp. 25303–25313.

[4] M. E. A. Bechar, N. Abdallah Saab, O. Assainova, N. Settouti, and M. El Bouz,
"HSI-AgriFoodAnomaly: A hyperspectral dataset for anomaly detection in the agri-food industrial
inspection," Recherche Data Gouv, V1, 2025, doi: 10.57745/QTLG7X.

[5] X. Tan, Y. Qin, and M. Hu, "SpecTrack: Spectral prompt guided adaptive experts for multispectral
object tracking," 2026, *arXiv:2607.05988*.

[6] R. Thoreau, V. Marsocci, and D. Derksen, "Parameter-efficient adaptation of geospatial foundation
models through embedding deflection," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2025.

[7] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, "Masked-attention mask transformer
for universal image segmentation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*,
2022, pp. 1290–1299.

[8] L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion
models," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023, pp. 3836–3847.

[9] L. Pang *et al.*, "HyperSAM: A promptable foundation model for hyperspectral remote sensing," 2026,
*arXiv:2609.37340*.

[10] J. V. Hurtado, R. Mohan, and A. Valada, "Hyperspectral adapter for semantic segmentation with
vision foundation models," 2025, *arXiv:2509.20107*.

[11] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf.
Learn. Represent. (ICLR)*, 2022.

[12] L. Waldmann *et al.*, "Panopticon: Advancing any-sensor foundation models for Earth observation,"
in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2025.

[13] N. Cavagnero, N. Norouzi, G. Dubbelman, and D. de Geus, "PMT: Plain mask transformer for image and
video segmentation with frozen vision encoders," 2026, *arXiv:2603.25398*.

[14] X. Li *et al.*, "Does your ViT still need U-Net for segmentation?," 2026, *arXiv:2607.00223*.

[15] Y. Nalcakan, H. Ju, I. Park, S. Yeo, Y. Jin, and S. Kim, "SpectraDINO: Bridging the spectral gap in
vision foundation models via lightweight adapters," 2026, *arXiv:2605.02258*.

[16] Z. Xiong *et al.*, "Neural plasticity-inspired multimodal foundation model for Earth observation,"
2024, *arXiv:2403.15356*.

[17] Y. Bao, S. Sivanandan, and T. Karaletsos, "Channel vision transformers: An image is worth 1 × 16 ×
16 words," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.

[18] S. Khanna, M. Irgau, D. B. Lobell, and S. Ermon, "ExPLoRA: Parameter-efficient extended
pre-training to adapt vision transformers under domain shifts," in *Proc. Int. Conf. Mach. Learn.
(ICML)*, PMLR, vol. 267, 2025.
