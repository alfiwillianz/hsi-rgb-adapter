"""Unit checks to run once on the GPU box before real training (takes ~1 min, no downloads).

  python tools/smoke_test.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hsiadapt.loss import SetCriterion  # noqa: E402
from hsiadapt.model import SpectralEoMT, inflate_patch_weight, mask_anneal_probs  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BB = "vit_small_patch14_reg4_dinov2"
wl = torch.linspace(400, 1000, 20).tolist()


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        sys.exit(1)


# 1. inflation preserves the RGB response for a spectrally flat input
w = torch.randn(8, 3, 14, 14)
for mode in ("wavelength", "wavelength_vis", "mean"):
    wi = inflate_patch_weight(w, wl, mode)
    check(f"inflate[{mode}] flat-spectrum response == RGB",
          torch.allclose(wi.sum(1), w.sum(1), atol=1e-5))

# 2. zero-init spectral branch: output identical to the model without it at step 0
kw = dict(backbone=BB, pretrained=False, in_chans=20, wavelengths=wl, img_size=112, num_queries=8)
m0 = SpectralEoMT(**kw).to(dev).eval()
m1 = SpectralEoMT(**kw, spectral_branch=True).to(dev).eval()
m1.load_state_dict(m0.state_dict(), strict=False)
x = torch.randn(2, 20, 112, 112, device=dev)
with torch.no_grad():
    a, b = m0(x)[-1], m1(x)[-1]
check("zero-init spectral branch is a no-op at init", torch.allclose(a[1], b[1], atol=1e-5))

# 3. shapes, aux outputs, masked attention, trainable split
m = SpectralEoMT(**kw, spectral_branch=True, freeze="early").to(dev).train()
outs = m(x, mask_probs=[1.0, 1.0, 0.5, 0.0])
check("4 aux + 1 final outputs", len(outs) == 5)
cls, ml = outs[-1]
check("class logits (B,Q,K+1)", tuple(cls.shape) == (2, 8, 2))
check("mask logits at 4x patch grid", tuple(ml.shape) == (2, 8, 32, 32))
frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
check("early blocks frozen", any(n.startswith("vit.blocks.0.") for n in frozen)
      and not any(n.startswith("vit.blocks.11.") for n in frozen))
check("semantic map shape", tuple(m.eval().semantic(x).shape) == (2, 1, 112, 112))

# 4. LoRA variant: only LoRA + new params train inside blocks
ml_ = SpectralEoMT(**kw, freeze="blocks", lora_rank=4)
tr = [n for n, p in ml_.named_parameters() if p.requires_grad and ".blocks." in n]
check("LoRA: only A/B trainable in blocks", tr and all(n.endswith((".A", ".B")) for n in tr))

# 5. a few optimisation steps reduce the loss on a fixed batch
m = SpectralEoMT(**kw, spectral_branch=True).to(dev).train()
crit = SetCriterion(1, loss_res=56).to(dev)
masks = torch.zeros(2, 112, 112, dtype=torch.bool)
masks[0, 20:40, 30:60] = True
masks[1, 70:75, 10:90] = True  # thin "thread"
tgts = [{"masks": masks[:1], "labels": torch.tensor([0])},
        {"masks": masks[1:], "labels": torch.tensor([0])}]
opt = torch.optim.AdamW(m.param_groups(3e-4, 0.1, 0.0))
losses = []
for step in range(30):
    loss, _ = crit(m(x, mask_anneal_probs(step, 30, 4, 0.1, 0.6)), tgts)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"     loss {losses[0]:.3f} -> {losses[-1]:.3f}")
check("loss decreases on a fixed batch", losses[-1] < 0.7 * losses[0])
check("annealing ends at plain attention", mask_anneal_probs(29, 30, 4, 0.1, 0.6) == [0.0] * 4)
print("all checks passed")
