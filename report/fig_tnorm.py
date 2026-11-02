"""Same activations, two coordinates: (a) value / tensor sigma (as in MR-GPTQ Fig. 2) and (b) block-normalized
t = 6|x| / block max (what E2M1 rounds), energy-weighted, without rotation and after RHT16.
Qwen2.5-0.5B FPROP activations of every 4th decoder layer, 2 x 1024 DCLM tokens, CPU. -> figs/fig26_raw_vs_blocknorm.png"""
import os
import json, sys
from pathlib import Path
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import hadamard
from hf_common import load_hf

ORANGE, BLUE, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#1f2328", "#6b7280", "#e5e7eb"
torch.manual_seed(1); G = 16; H = hadamard(G, "cpu")
model, tok = load_hf("Qwen/Qwen2.5-0.5B", "cpu")
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 2 * 1025: break
batch = torch.tensor(ids[:2 * 1025]).view(2, 1025)
z = {"I": [], "R": []}; tt = {"I": [], "R": []}; ee = {"I": [], "R": []}; mx_share = {"I": [0., 0.], "R": [0., 0.]}


def hook(mod, inp, out):
    X = inp[0].detach().reshape(-1, mod.in_features).float(); K = X.shape[1]
    sg = (torch.randint(0, 2, (K,)) * 2 - 1).float()
    for T, A in (("I", X), ("R", ((X * sg).view(-1, K // G, G) @ H).view(X.shape))):
        z[T].append((A / A.std()).flatten()[::7])                      # subsample for the raw histogram
        Ab = A.view(-1, K // G, G); mx = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        t = 6 * Ab.abs() / mx; e = Ab ** 2; ismax = t >= 6 - 1e-6
        tt[T].append(t[~ismax]); ee[T].append(e[~ismax] / e.sum())
        mx_share[T][0] += (mx ** 2).sum().item() / e.sum().item(); mx_share[T][1] += 1


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers.") and int(n.split(".")[1]) % 4 == 0: m.register_forward_hook(hook)
with torch.no_grad(): model(batch[:, :-1], target=batch[:, 1:])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
lab = {"I": "no rotation", "R": "RHT16"}; col = {"I": ORANGE, "R": BLUE}
for T in ("I", "R"):
    v = torch.cat(z[T]); kurt = ((v - v.mean()) ** 4).mean() / v.var() ** 2 - 3
    a1.hist(v.clamp(-5, 5).numpy(), bins=161, range=(-5, 5), density=True, histtype="step", linewidth=1.6, color=col[T],
            label=f"{lab[T]} (excess kurtosis {kurt:.1f})")
a1.set_xlim(-5, 5); a1.set_xlabel("value / tensor $\\sigma$  (MR-GPTQ view)"); a1.set_ylabel("density")
a1.set_title("(a) raw values: rotation removes the tails", color=INK, fontsize=8.5)
a1.legend(fontsize=6.5, frameon=False, loc="upper right")
LEV = [0, .5, 1, 1.5, 2, 3, 4, 6]
for T in ("I", "R"):
    t = torch.cat(tt[T]); w = torch.cat(ee[T]); w = w / w.sum() * (1 - mx_share[T][0] / mx_share[T][1])
    top = w[(t >= 4)].sum().item()
    a2.hist(t.numpy(), bins=60, range=(0, 6), weights=w.numpy(), histtype="step", linewidth=1.6, color=col[T],
            label=f"{lab[T]}, energy: [4,6) {100 * top:.0f}%, block max {100 * mx_share[T][0] / mx_share[T][1]:.0f}%")
    cw = torch.full_like(t, (G - 1) / G / t.numel())          # element count share (the block max is 1 of G elements)
    a2.hist(t.numpy(), bins=60, range=(0, 6), weights=cw.numpy(), histtype="step", linewidth=1.1, linestyle=(0, (2, 1.5)),
            color=col[T], label=f"{lab[T]}, count: [4,6) {100 * cw[t >= 4].sum().item():.0f}%")
    print(T, "count share t<0.25 %.3f, max bin %.3f" % (cw[t < 0.25].sum().item(), torch.histc(t, 60, 0, 6).max().item() * cw[0].item()))
a2.axvspan(4, 6, color=GRID, zorder=0); a2.text(5.0, 0.62, "[4,6)", color=MUTED, ha="center", va="top", fontsize=7, transform=a2.get_xaxis_transform())
for l in LEV: a2.axvline(l, ymax=.04, color=MUTED, linewidth=1)
a2.set_xlim(0, 6); a2.set_xlabel("$t = 6|x|\\,/$ block max (ticks = E2M1 levels)")
a2.set_ylabel("share per bin"); a2.set_title("(b) block-normalized: solid = energy, dashed = count", color=INK, fontsize=8.5)
a2.legend(fontsize=6.5, frameon=False, loc="upper left")
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, linewidth=.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "figs/fig26_raw_vs_blocknorm.png", dpi=160, bbox_inches="tight")
for T in ("I", "R"):
    t = torch.cat(tt[T]); w = torch.cat(ee[T]); w = w / w.sum()
    print(T, "max share %.2f" % (mx_share[T][0] / mx_share[T][1]), "non-max energy in t<1 %.2f, [1,4) %.2f, [4,6) %.2f" %
          (w[t < 1].sum(), w[(t >= 1) & (t < 4)].sum(), w[t >= 4].sum()))
