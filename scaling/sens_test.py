"""Perturbation-based sensitivity for FP8/FP4 mixed precision (forward GEMMs, W4A4-style, no rotation).

Model: an open-source HF causal LM in bf16. Every decoder Linear's FPROP (X W^T) can run as
  bf16 | fp4 (E2M1 RTN, 1x16 blocks along K, FP32 absmax scale) | fp8 (E4M3, 1x16 blocks) | mixed (channels idx in FP8,
  rest FP4) | noise (bf16 output + Gaussian noise with the same energy as this Linear's FP4 output error).
Sensitivities are measured on a calibration set; every mixed-precision choice is evaluated on a disjoint eval set.

Part 1 (layer level, perturbation = quantize one Linear at a time):
  dL_q[l]     = L(only l in FP4) - L(bf16)
  dL_noise[l] = L(only l gets equal-energy Gaussian noise) - L(bf16)
  sqnr[l]     = output SQNR of l in FP4
  Then keep a FLOP budget (10%, 25%) of Linears in FP8, rest FP4, chosen by dL_q, dL_noise, worst SQNR,
  first/last layers, random (3 seeds); report eval loss.
Part 2 (channel level, all Linears FP4 except k% of K channels per Linear in FP8), channels chosen by:
  mag     rms(X_c) * rms(W_c)
  outerr  ||Q(X_c) Q(W_c)^T - X_c W_c^T||^2                 (output perturbation caused by channel c)
  grad    sum_seq | a_c^T G b_c - x_c^T G w_c |              (gradient-weighted perturbation, G = dL/dY)
  random
"""
import os
import argparse, json, math, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, quantize
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--tag", default="qwen2.5-0.5b")
ap.add_argument("--out", required=True); ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--neval", type=int, default=16)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0); random.seed(0)
wrap, tok = load_hf(args.hf, dev)
hfm = wrap.m
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
T = 1024
need = (args.ncal + args.neval) * (T + 1)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
seqs = torch.tensor(ids[:need], device=dev).view(-1, T + 1)
cal, ev = seqs[:args.ncal], seqs[args.ncal:]


def fp4(A):
    return quantize(A, E2M1, "rtn", "fp32")[0]


def fp8(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    s = (Ab.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return ((Ab / s).to(torch.float8_e4m3fn).float() * s).view(N, R)


class QLinear(torch.nn.Module):
    def __init__(self, lin, name):
        super().__init__()
        self.lin, self.name, self.mode, self.idx = lin, name, "bf16", None
        self.cap = None

    def forward(self, x):
        if self.mode == "bf16":
            y = self.lin(x)
            if self.cap is not None:
                self.cap["X"].append(x.detach().reshape(-1, x.shape[-1]).float().cpu())
                y.register_hook(lambda g: self.cap["G"].append(g.detach().reshape(-1, g.shape[-1]).float().cpu()))
            return y
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        if self.mode == "fp4":
            y = fp4(xs) @ fp4(W).T
        elif self.mode == "fp8":
            y = fp8(xs) @ fp8(W).T
        elif self.mode == "noise":
            y0 = xs @ W.T; e = fp4(xs) @ fp4(W).T - y0
            y = y0 + torch.randn_like(y0) * e.pow(2).mean().sqrt()
        elif self.mode == "mixed":
            idx = self.idx; K = xs.shape[1]
            mask = torch.ones(K, dtype=torch.bool, device=dev); mask[idx] = False; rest = mask.nonzero().squeeze(1)
            y = fp8(xs[:, idx].contiguous()) @ fp8(W[:, idx].contiguous()).T + fp4(xs[:, rest].contiguous()) @ fp4(W[:, rest].contiguous()).T
        if self.lin.bias is not None:
            y = y + self.lin.bias.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)


Q = {}
for lname, layer in [(n, m) for n, m in hfm.named_modules() if n.endswith("layers")][0:1]:
    for i, blk in enumerate(layer):
        for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
            for cn, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    q = QLinear(child, f"layers.{i}.{pn}.{cn}"); setattr(parent, cn, q); Q[q.name] = q
names = list(Q)
flops = {n: Q[n].lin.weight.numel() for n in names}
print(f"[{args.tag}] {len(names)} Linears", flush=True)


@torch.no_grad()
def loss_on(data, bs=4):
    tot = 0.0
    for i in range(0, len(data), bs):
        b = data[i:i + bs]
        logits = hfm(input_ids=b[:, :-1]).logits
        tot += F.cross_entropy(logits.float().flatten(0, 1), b[:, 1:].flatten()).item() * len(b)
    return tot / len(data)


def set_all(mode):
    for q in Q.values():
        q.mode, q.idx = mode, None


res = dict(model=args.tag)
set_all("bf16"); res["bf16_cal"] = loss_on(cal); res["bf16_eval"] = loss_on(ev)
set_all("fp4"); res["fp4_eval"] = loss_on(ev)
set_all("fp8"); res["fp8_eval"] = loss_on(ev)
print(f"[{args.tag}] eval loss bf16 {res['bf16_eval']:.4f} fp8 {res['fp8_eval']:.4f} fp4 {res['fp4_eval']:.4f}", flush=True)

# ---------------- Part 1: layer-level perturbation
set_all("bf16"); base = res["bf16_cal"]
dq, dn, sq = {}, {}, {}
for n in names:
    Q[n].mode = "fp4"; dq[n] = loss_on(cal) - base
    Q[n].mode = "noise"; dn[n] = loss_on(cal) - base
    Q[n].mode = "bf16"
# per-layer FP4 output SQNR on calibration activations (captured once)
for q in Q.values():
    q.cap = {"X": [], "G": []}
wrap.zero_grad(set_to_none=True)
for i in range(0, len(cal), 4):
    b = cal[i:i + 4]
    loss = F.cross_entropy(hfm(input_ids=b[:, :-1]).logits.float().flatten(0, 1), b[:, 1:].flatten())
    loss.backward()
for n, q in Q.items():
    X = torch.cat(q.cap["X"])[:4096].to(dev); W = q.lin.weight.float()
    Y = X @ W.T; Yh = fp4(X) @ fp4(W).T
    sq[n] = -10 * math.log10(((Yh - Y) ** 2).sum().item() / (Y ** 2).sum().item())
res["layer"] = {n: dict(dL_q=dq[n], dL_noise=dn[n], sqnr=sq[n], params=flops[n]) for n in names}


def rank_corr(a, b):
    ra = {k: i for i, k in enumerate(sorted(a, key=a.get))}; rb = {k: i for i, k in enumerate(sorted(b, key=b.get))}
    n = len(a); d = sum((ra[k] - rb[k]) ** 2 for k in a)
    return 1 - 6 * d / (n * (n * n - 1))


res["spearman_q_noise"] = rank_corr(dq, dn)
res["spearman_q_negsqnr"] = rank_corr(dq, {k: -v for k, v in sq.items()})
print(f"[{args.tag}] Spearman(dL_q, dL_noise)={res['spearman_q_noise']:.3f}  Spearman(dL_q, -SQNR)={res['spearman_q_negsqnr']:.3f}", flush=True)
total = sum(flops.values())
nlayers = 1 + max(int(n.split(".")[1]) for n in names)


def pick(order, budget):
    sel, used = [], 0
    for n in order:
        if used + flops[n] <= budget * total:
            sel.append(n); used += flops[n]
    return sel


res["layer_mix"] = {}
for budget in (0.10, 0.25):
    choices = {"dL_q": sorted(names, key=lambda n: -dq[n] / flops[n]), "dL_noise": sorted(names, key=lambda n: -dn[n] / flops[n]),
               "worst_SQNR": sorted(names, key=lambda n: sq[n]),
               "first_last": sorted(names, key=lambda n: min(int(n.split(".")[1]), nlayers - 1 - int(n.split(".")[1])))}
    for s in range(3):
        r = names[:]; random.Random(s).shuffle(r); choices[f"random{s}"] = r
    out = {}
    for cname, order in choices.items():
        sel = set(pick(order, budget))
        for n in names:
            Q[n].mode = "fp8" if n in sel else "fp4"
        out[cname] = loss_on(ev)
    out["random_mean"] = sum(out.pop(f"random{s}") for s in range(3)) / 3
    res["layer_mix"][str(budget)] = out
    print(f"[{args.tag}] layer budget {budget}: " + ", ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)

# ---------------- Part 2: channel-level selection
scores = {}
for n, q in Q.items():
    X = torch.cat(q.cap["X"]).to(dev); G = torch.cat(q.cap["G"]).to(dev); W = q.lin.weight.float()
    Xq, Wq = fp4(X), fp4(W)
    mag = X.pow(2).mean(0).sqrt() * W.pow(2).mean(0).sqrt()
    outerr = Xq.pow(2).sum(0) * Wq.pow(2).sum(0) + X.pow(2).sum(0) * W.pow(2).sum(0) - 2 * (Xq * X).sum(0) * (Wq * W).sum(0)
    gs = torch.zeros_like(mag)
    for s in range(0, X.shape[0], T):                                # per sequence: |a^T G b - x^T G w| per channel
        Xs, Xqs, Gs = X[s:s + T], Xq[s:s + T], G[s:s + T]
        gs += ((Xqs * (Gs @ Wq)).sum(0) - (Xs * (Gs @ W)).sum(0)).abs()
    scores[n] = {"mag": mag, "outerr": outerr, "grad": gs}
    q.cap = None
res["channel_mix"] = {}
for k in (0.01, 0.03):
    out = {}
    for crit in ["mag", "outerr", "grad", "random"]:
        for n, q in Q.items():
            K = q.lin.in_features; m = max(BLK, int(math.ceil(k * K / BLK)) * BLK)
            if crit == "random":
                idx = torch.randperm(K, device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:m]
            else:
                idx = scores[n][crit].topk(m).indices
            q.mode, q.idx = "mixed", idx
        out[crit] = loss_on(ev)
    res["channel_mix"][str(k)] = out
    print(f"[{args.tag}] channel k={k}: " + ", ".join(f"{c} {v:.4f}" for c, v in out.items()), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print(f"[{args.tag}] done -> {args.out}", flush=True)
