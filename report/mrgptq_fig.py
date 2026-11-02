"""Reproduce the MR-GPTQ (arXiv 2509.23202) Fig. 2 style: aggregate weight / activation distributions of a Llama-3.1-8B,
native vs rotated, with Laplace / Gaussian fits and excess kurtosis. The paper does not say how the values were normalized
or which rotation was used, so several normalizations are measured and the one closest to the paper's kurtosis values
(act 8.75 -> 0.02, weight 1.47 -> 0.05) is drawn.
Normalizations (before aggregation, all followed by a global rescale to unit variance):
  tensor  value / tensor std
  row     value / std of its row (token for activations, output channel for weights)
  group   value / std of its G-element group along the reduction axis
Rotation: randomized block Hadamard of size G along the reduction axis (G = 32 as in MR-GPTQ's experiments), or "full" = the
largest power-of-two block dividing K (4096, or 2048 for down_proj). Activation rows of BOS tokens (attention sinks with
massive activations) are dropped when DROP_SINK=1.
Every 4th decoder layer, all 7 linear layers, 2 x 1024 DCLM tokens, CPU. -> figs/fig27_mrgptq_style.png"""
import json, math, os, sys
from pathlib import Path
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import hadamard

HF = os.environ.get("HF", "meta-llama/Llama-3.1-8B")
torch.manual_seed(1)
tok = AutoTokenizer.from_pretrained(HF)
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).eval()
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += [tok.bos_token_id] + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= 2 * 1024: break
batch = torch.tensor(ids[:2 * 1024]).view(2, 1024)
GS = (32, "full"); NORMS = ("tensor", "row", "group")
DROP = os.environ.get("DROP_SINK", "1") == "1"
keep_rows = (batch.flatten() != tok.bos_token_id)
S = {}                                                        # (op, rot, G, norm) -> list of subsampled values
HS = {g: hadamard(g, "cpu") for g in (32, 2048, 4096)}
bsz = lambda G, K: G if G != "full" else (4096 if K % 4096 == 0 else 2048)


def norm(M, how, G):
    if how == "tensor": return M / M.std()
    if how == "row": return M / M.std(1, keepdim=True).clamp(min=1e-12)
    Mb = M.view(M.shape[0], -1, G if G != "full" else 32); return (Mb / Mb.std(-1, keepdim=True).clamp(min=1e-12)).view(M.shape)


def add(op, M):
    K = M.shape[1]; sg = (torch.randint(0, 2, (K,)) * 2 - 1).float(); step = max(1, M.numel() // 200000)
    for G in GS:
        b = bsz(G, K); R = ((M * sg).view(M.shape[0], -1, b) @ HS[b]).view(M.shape)
        for rot, A in (("native", M), ("rotated", R)):
            for how in NORMS:
                S.setdefault((op, rot, G, how), []).append(norm(A, how, G).flatten()[::step])


def hook(mod, inp, out):
    X = inp[0].detach().reshape(-1, mod.in_features).float()
    add("act", X[keep_rows] if DROP else X); add("weight", mod.weight.detach().float())


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and ".layers." in n and int(n.split(".layers.")[1].split(".")[0]) % 4 == 0:
        m.register_forward_hook(hook)
with torch.no_grad(): model(input_ids=batch, use_cache=False)

kurt = lambda v: (((v - v.mean()) / v.std()) ** 4).mean().item() - 3
K = {k: kurt(torch.cat(v)) for k, v in S.items()}
paper = {("act", "native"): 8.75, ("act", "rotated"): 0.02, ("weight", "native"): 1.47, ("weight", "rotated"): 0.05}
print("excess kurtosis (paper: act 8.75 -> 0.02, weight 1.47 -> 0.05)")
for G in GS:
    for how in NORMS:
        print(f"G={G!s:4s} {how:6s} " + "  ".join(f"{op} {rot} {K[(op, rot, G, how)]:7.2f}" for op in ("act", "weight") for rot in ("native", "rotated")))
best = min(((G, how) for G in GS for how in NORMS),
           key=lambda c: sum(abs(math.log1p(max(K[(op, rot, *c)], -0.9)) - math.log1p(v)) for (op, rot), v in paper.items()))
print("closest to the paper:", best)
if os.environ.get("BEST"): g, h = os.environ["BEST"].split(","); best = (g if g == "full" else int(g), h); print("drawn:", best)

RED, GREEN, INK = "#e8706a", "#6bbf6b", "#1f2328"            # paper-like colors for a side-by-side comparison
fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
x = torch.linspace(-5, 5, 400)
lap = torch.exp(-math.sqrt(2) * x.abs()) / math.sqrt(2); gau = torch.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
for ax, (op, rot) in zip(axes, [("act", "native"), ("act", "rotated"), ("weight", "native"), ("weight", "rotated")]):
    v = torch.cat(S[(op, rot, *best)]); v = v / v.std()
    ax.hist(v[v.abs() < 5].numpy(), bins=150, range=(-5, 5), density=True, color=RED if rot == "native" else GREEN)
    fit, name = (lap, "Laplace fit") if rot == "native" else (gau, "Gaussian fit")
    ax.plot(x, fit, linestyle=":" if op == "act" else "--", color="black", linewidth=1.3, label=name)
    lbl = {"act": "activations", "weight": "weights"}[op]
    ax.plot([], [], " ", label=f"{'Native' if rot == 'native' else 'Rotated'} {lbl}")
    ax.plot([], [], " ", label=f"Kurtosis={K[(op, rot, *best)]:.2f}  (paper {paper[(op, rot)]})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5); ax.legend(fontsize=6.5, loc="upper right", handlelength=1.5, frameon=False); ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(-5, 5)
axes[0].set_ylabel("Density")
rot_txt = "full-row randomized Hadamard" if best[0] == "full" else f"randomized block Hadamard, G={best[0]}"
fig.suptitle(f"{HF.split('/')[-1]}: values / {best[1]} std, {rot_txt}, every 4th layer (MR-GPTQ Fig. 2 style)", fontsize=8.5, color=INK)
fig.tight_layout(); fig.savefig(D / "figs/fig27_mrgptq_style.png", dpi=160, bbox_inches="tight")
