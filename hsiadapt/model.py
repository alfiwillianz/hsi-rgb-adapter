"""Spectral EoMT: an Encoder-only Mask Transformer on an RGB-pretrained DINOv2 ViT,
adapted to hyperspectral input.

Pieces (all switchable from the config):
  * spectral patch embedding: C-band conv initialized from the RGB patch embed
    (wavelength-aware, visible-only, mean-repeat, or zero init)
  * EoMT: N learnable queries appended before the last `num_query_blocks` ViT blocks,
    class head + mask head over upscaled patch tokens, masked attention annealed away
  * optional zero-initialized spectral branch: a per-pixel spectral MLP whose patch tokens
    are added to the ViT patch tokens through zero-init projections (ControlNet-style),
    so at step 0 the network is exactly the RGB-initialized model
  * freezing schemes and optional LoRA on frozen blocks
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# timm/DINOv2 expect RGB order: channel 0 = R, 1 = G, 2 = B
RGB_CENTERS_NM = (610.0, 545.0, 465.0)


# ----------------------------------------------------------------------------- patch embed init
def inflate_patch_weight(w_rgb: torch.Tensor, wl: list[float] | None, mode: str) -> torch.Tensor:
    """Build a (D, C, p, p) weight from the (D, 3, p, p) RGB weight.

    mode:
      wavelength      each band interpolates the two nearest RGB channels by wavelength;
                      bands outside the visible range copy the closest channel
      wavelength_vis  same, but bands outside 400-700 nm start at zero
      mean            every band gets mean(RGB) * 3 / C   (timm-style repeat)
      zero            all zeros (the model starts blind; sanity baseline)
    Each RGB channel's total weight is preserved, so a spectrally flat input gives the same
    first-layer response as the RGB model.
    """
    D, _, p, _ = w_rgb.shape
    if mode == "zero":
        return torch.zeros(D, len(wl) if wl else 3, p, p)
    if mode == "mean" or not wl:
        C = len(wl) if wl else 3
        return w_rgb.mean(1, keepdim=True).repeat(1, C, 1, 1) * 3.0 / C
    C = len(wl)
    centers = sorted(((c, i) for i, c in enumerate(RGB_CENTERS_NM)))  # ascending: B, G, R
    alpha = torch.zeros(C, 3)
    for c, lam in enumerate(wl):
        if mode == "wavelength_vis" and not (400.0 <= lam <= 700.0):
            continue
        if lam <= centers[0][0]:
            alpha[c, centers[0][1]] = 1.0
        elif lam >= centers[-1][0]:
            alpha[c, centers[-1][1]] = 1.0
        else:
            for (l0, i0), (l1, i1) in zip(centers[:-1], centers[1:]):
                if l0 <= lam <= l1:
                    t = (lam - l0) / (l1 - l0)
                    alpha[c, i0], alpha[c, i1] = 1 - t, t
                    break
    norm = alpha.sum(0).clamp(min=1e-6)  # bands per RGB channel
    return torch.einsum("cr,drhw->dchw", alpha / norm, w_rgb)


# ----------------------------------------------------------------------------- small modules
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ScaleBlock(nn.Module):
    """2x upsampling of patch features, as in EoMT."""

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        return self.norm(self.dw(self.act(self.up(x))))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):  # some timm code paths read .weight
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


class SpectralEncoder(nn.Module):
    """Per-pixel spectral MLP (1x1 convs) followed by patchification -> (B, N, dim)."""

    def __init__(self, in_chans: int, patch: int, dim: int = 128, hidden: int = 96):
        super().__init__()
        self.pix = nn.Sequential(
            nn.Conv2d(in_chans, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
        )
        self.patchify = nn.Conv2d(hidden, dim, patch, patch)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.patchify(self.pix(x)).flatten(2).transpose(1, 2))


class ZeroInject(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, s):
        return self.proj(s)


def block_forward(blk: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    """timm ViT Block forward with an optional boolean attention mask (True = attend)."""
    if attn_mask is None:
        return blk(x)
    a = blk.attn
    y = blk.norm1(x)
    B, N, C = y.shape
    qkv = a.qkv(y).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = getattr(a, "q_norm", nn.Identity())(q)
    k = getattr(a, "k_norm", nn.Identity())(k)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    o = o.transpose(1, 2).reshape(B, N, C)
    o = getattr(a, "norm", nn.Identity())(o)
    o = a.proj_drop(a.proj(o))
    x = x + blk.drop_path1(blk.ls1(o))
    return x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))


# ----------------------------------------------------------------------------- model
class SpectralEoMT(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch14_reg4_dinov2.lvd142m",
        pretrained: bool = True,
        in_chans: int = 3,
        wavelengths: list[float] | None = None,
        modality: str = "hsi",
        img_size: int = 448,
        num_classes: int = 1,
        num_queries: int = 32,
        num_query_blocks: int = 4,
        pe_init: str = "wavelength_vis",
        spectral_branch: bool = False,
        spectral_dim: int = 128,
        inject: str = "query_blocks",  # query_blocks | all
        freeze: str = "early",  # none | early | blocks
        train_patch_embed: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        num_upscale: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.vit = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
            img_size=img_size, drop_path_rate=drop_path,
        )
        D = self.vit.embed_dim
        p = self.vit.patch_embed.patch_size[0]
        self.patch, self.num_queries, self.num_classes = p, num_queries, num_classes
        L = len(self.vit.blocks)
        self.L1 = L - num_query_blocks
        self.nqb = num_query_blocks

        if modality == "hsi" and in_chans != 3:
            old = self.vit.patch_embed.proj
            new = nn.Conv2d(in_chans, D, p, p, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(inflate_patch_weight(old.weight.detach().cpu(), wavelengths, pe_init))
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            self.vit.patch_embed.proj = new
            if hasattr(self.vit.patch_embed, "in_chans"):
                self.vit.patch_embed.in_chans = in_chans

        self.queries = nn.Embedding(num_queries, D)
        self.class_head = nn.Linear(D, num_classes + 1)  # last = no-object
        self.mask_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.upscale = nn.Sequential(*[ScaleBlock(D) for _ in range(num_upscale)])

        self.spec_enc, self.inject_idx = None, []
        if spectral_branch:
            self.spec_enc = SpectralEncoder(in_chans, p, spectral_dim)
            self.inject_idx = list(range(self.L1, L)) if inject == "query_blocks" else list(range(L))
            self.injectors = nn.ModuleDict({str(i): ZeroInject(spectral_dim, D) for i in self.inject_idx})

        self._setup_trainable(freeze, train_patch_embed, lora_rank, lora_alpha)

    # ------------------------------------------------------------------ trainability
    def _setup_trainable(self, freeze, train_pe, lora_rank, lora_alpha):
        v = self.vit
        frozen_blocks = {"none": [], "early": list(range(self.L1)), "blocks": list(range(len(v.blocks)))}[freeze]
        for name in ("cls_token", "reg_token", "pos_embed"):
            t = getattr(v, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad_(freeze == "none")
        for i in frozen_blocks:
            v.blocks[i].requires_grad_(False)
        if freeze == "blocks":
            v.norm.requires_grad_(False)
        v.patch_embed.requires_grad_(train_pe)
        if lora_rank > 0:
            for i in frozen_blocks:
                attn = v.blocks[i].attn
                attn.qkv = LoRALinear(attn.qkv, lora_rank, lora_alpha)
                attn.proj = LoRALinear(attn.proj, lora_rank, lora_alpha)
        # parameters that came with the pretrained ViT (for a smaller LR)
        self._pretrained_ids = {id(p) for n, p in v.named_parameters() if ".A" not in n[-2:] and ".B" not in n[-2:]}
        self._pretrained_ids -= {id(v.patch_embed.proj.weight)} if v.patch_embed.proj.weight.shape[1] != 3 else set()

    def param_groups(self, lr: float, backbone_lr_mult: float, wd: float, pe_lr_mult: float | None = None):
        """pe_lr_mult: LR multiplier for the patch embed (None = old behaviour: an inflated C-band
        weight counts as new and gets the full LR, a pretrained 3-band one the backbone LR)."""
        pe_ids = {id(p) for p in self.vit.patch_embed.parameters()} if pe_lr_mult is not None else set()
        new, pre, pe = [], [], []
        for p in self.parameters():
            if p.requires_grad:
                if id(p) in pe_ids:
                    pe.append(p)
                else:
                    (pre if id(p) in self._pretrained_ids else new).append(p)
        groups = [
            {"params": new, "lr": lr, "weight_decay": wd},
            {"params": pre, "lr": lr * backbone_lr_mult, "weight_decay": wd},
        ]
        if pe:
            groups.append({"params": pe, "lr": lr * pe_lr_mult, "weight_decay": wd})
        return groups

    # ------------------------------------------------------------------ heads
    def _predict(self, x, gh, gw):
        n = self.vit.norm(x)
        Q, P = self.num_queries, self.vit.num_prefix_tokens
        q = n[:, :Q]
        pt = n[:, Q + P :].transpose(1, 2).reshape(n.shape[0], -1, gh, gw)
        feats = self.upscale(pt)
        cls = self.class_head(q)
        masks = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q), feats)
        return cls, masks

    def _attn_mask(self, mask_logits, prob, gh, gw, T):
        B, Q = mask_logits.shape[:2]
        P = self.vit.num_prefix_tokens
        m = F.adaptive_avg_pool2d(mask_logits.float(), (gh, gw)).flatten(2) > 0  # B,Q,N
        m = m | ~m.any(-1, keepdim=True)  # empty mask -> attend everywhere
        keep = torch.rand(B, Q, 1, device=m.device) < prob  # which queries stay masked
        m = m | ~keep
        full = torch.ones(B, 1, T, T, dtype=torch.bool, device=m.device)
        full[:, 0, :Q, Q + P :] = m
        return full

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, mask_probs: list[float] | None = None):
        """Returns a list of (class_logits, mask_logits) from each query block (aux) + final.

        mask_probs: per query block, the probability that a query's attention is restricted
        to its predicted mask (masked attention). None / zeros = plain attention (inference).
        """
        B = x.shape[0]
        spec = self.spec_enc(x) if self.spec_enc is not None else None
        t = self.vit.patch_embed(x)
        gh, gw = t.shape[1], t.shape[2]
        t = self.vit._pos_embed(t)
        t = self.vit.patch_drop(t)
        t = self.vit.norm_pre(t)
        P, Q = self.vit.num_prefix_tokens, self.num_queries
        outs = []
        for i, blk in enumerate(self.vit.blocks):
            if i == self.L1:
                t = torch.cat([self.queries.weight[None].expand(B, -1, -1), t], 1)
            if spec is not None and str(i) in getattr(self, "injectors", {}):
                off = (Q if i >= self.L1 else 0) + P
                t = torch.cat([t[:, :off], t[:, off:] + self.injectors[str(i)](spec)], 1)
            attn_mask = None
            if i >= self.L1 and self.training:
                cls, mlog = self._predict(t, gh, gw)
                outs.append((cls, mlog))
                pr = mask_probs[i - self.L1] if mask_probs else 0.0
                if pr > 0:
                    attn_mask = self._attn_mask(mlog.detach(), pr, gh, gw, t.shape[1])
            t = block_forward(blk, t, attn_mask)
        outs.append(self._predict(t, gh, gw))
        return outs

    @torch.no_grad()
    def semantic(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, H, W) per-class probability maps (Mask2Former semantic inference)."""
        cls, mlog = self.forward(x)[-1]
        probs = cls.softmax(-1)[..., :-1]
        m = F.interpolate(mlog.float(), size=x.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        return torch.einsum("bqk,bqhw->bkhw", probs.float(), m).clamp(0, 1)


def mask_anneal_probs(step: int, total: int, nqb: int, start: float, end: float) -> list[float]:
    """EoMT-style staggered annealing: block j goes 1 -> 0 linearly over its own slice of
    [start, end] (fractions of training). After `end`, attention is unmasked everywhere."""
    f = step / max(total, 1)
    span = (end - start) / nqb
    out = []
    for j in range(nqb):
        s = start + j * span
        e = s + span
        out.append(1.0 if f < s else max(0.0, 1.0 - (f - s) / max(e - s, 1e-8)))
    return out
