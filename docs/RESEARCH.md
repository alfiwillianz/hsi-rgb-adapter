# Research context

## Idea in one paragraph

EoMT (Kerssies et al., CVPR 2025) showed that a large, well-pretrained plain ViT (DINOv2) needs no
adapter, pixel decoder or transformer decoder for segmentation. Appending learnable queries before
the last few blocks plus tiny heads matches Mask2Former and runs several times faster. That result
rests on the pretrained features being the decoder's prior. Hyperspectral food inspection breaks the
assumption twice: the input has hundreds of bands, and the discriminative signal is material
spectra rather than RGB texture. This project asks how little must change in an RGB-pretrained EoMT
to work on HSI, and where it breaks.

## Why the design looks like this (decisions + evidence)

- **Query blocks must be trainable.** The EoMT authors' follow-up, PMT (arXiv 2603.25398), reports
  that EoMT with a *frozen* encoder collapses badly (about 6.8 PQ). So "frozen ViT + learned spectral
  patch embed" is expected to fail. We keep it only as baseline C to confirm this on HSI.
- **Wavelength-aware patch-embed inflation.** Initialize each band's patch-embed weights from the
  nearest RGB channels by wavelength, preserving total response (as in SpecTrack, arXiv 2607.05988).
  Non-visible bands start at zero by default (`wavelength_vis`), following evidence that zero-init of
  extra bands can beat mean init (arXiv 2503.15969). DEFLECT/UPE (ICCV 2025, arXiv 2503.09493) found
  init choice unstable across datasets, so `pe_init` is an ablation, not a fixed choice.
- **Zero-init spectral branch instead of an HSI→3-channel stem.** Any stem that makes HSI "look RGB"
  must discard what RGB can't express, which is exactly the material signal. HSI-Adapter (arXiv
  2509.20107) compensates with ViT-Adapter-style interaction blocks, but that reintroduces the
  machinery EoMT removed. HyperSAM (arXiv 2609.37340) shows zero-initialized (ControlNet-style)
  injection from a spectral side branch into a frozen RGB model. We do a lighter version: a per-pixel
  spectral MLP whose patch tokens are added into the query blocks through zero-init projections, so
  the model starts exactly as the RGB-initialized EoMT.
- **SpectraDINO (arXiv 2605.02258)**: with a fully frozen backbone, adapters didn't help LWIR
  segmentation; upper blocks had to co-adapt. This is consistent with training the query blocks.
- **Mask annealing** is kept from EoMT. Whether it matters on small HSI data is ablation H.

## Closest prior work (what we must beat or distinguish from)

| work | what it does | difference from ours |
|---|---|---|
| HSI-Adapter (2025) | frozen ViT + spectral transformer→3ch + ViT-Adapter interaction + decoder | heavy; we are encoder-only, no adapter |
| HyperSAM (2026) | frozen SAM3 + trainable HSI side encoder, zero-init injection | promptable, side ViT, remote sensing |
| EoSeg (arXiv 2607.00223) | encoder-only ViT seg for medical images; compares DINOv2/v3, SigLIP and a remote-sensing FM as backbones | medical RGB, not HSI |
| PMT (2026) | light decoder on frozen VFM features | frozen encoder; baseline candidate |
| Panopticon | any-sensor DINOv2, channel cross-attention patch embed; "-PE" retrains only the patch embed | pretraining-scale; our C is the PE-only analogue |
| DOFA | wavelength-conditioned hypernetwork patch embed | foundation model, not adaptation |
| ChannelViT | per-band tokens | token count × bands, too costly at 60–300 bands |
| SpectralGPT PEFT (arXiv 2505.15334), MBTI (2607.12782) | LoRA / Kronecker PEFT for HSI classification | classification, not segmentation |

## Dataset

HSI-AgriFoodAnomaly: Bechar et al., Recherche Data Gouv 2025, doi:10.57745/QTLG7X;
code at github.com/lsllabisen/HSI-AgriFoodAnomaly-Dataset. 89/17/41 train/val/test cubes, 300 bands
over 400–1000 nm, 1000×900. Non-commercial research licence. Their baseline is **patch-level binary
classification** with 2D CNNs (MobileNetV2, ResNet, TinyNet, ...), so a proper segmentation benchmark
on it is itself a contribution. They report that RGB models struggle on visually ambiguous anomalies,
which is the motivation for the RGB vs HSI comparison (configs A vs D/E).

## Expected risks

- Small train set (89 cubes): overfitting, noisy comparisons. Use multiple seeds and keep early blocks
  frozen.
- Thin objects (threads, shards) versus 14-px patches: watch per-category recall. Mask resolution is 4×
  the patch grid. If small objects fail, try a smaller crop stride, `num_upscale: 3`, or a higher
  `loss_res`.
- Raw DN versus reflectance: the cache is standardized per band with train statistics. Check whether
  the cubes are calibrated, and whether illumination drifts between scenes.

## Owner's related work

PCAE: A Partial Convolution Autoencoder for Hyperspectral Anomaly Detection in Food Safety Inspection
(IEEE Access; owner is third author). Same application domain, so cite and position against it.
