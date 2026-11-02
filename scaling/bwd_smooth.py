"""Per-channel / per-token scaling on the BACKWARD GEMMs (open-source model, gradient error).

Forward is fixed at E2M1 (FP4, no transform). Reference = the same forward with an exact (BF16) backward.
One backward GEMM is quantized at a time (the other stays BF16), and the error of all decoder-Linear weight
gradients is measured (relative L2 and cosine) on evaluation sequences disjoint from calibration.
  DGRAD dX = dY W        reduce over output channels O: operands dY [N,O], W^T [K,O]
  WGRAD dW = dY^T X      reduce over tokens N:         operands dY^T [O,N], X^T [K,N]
Variants (grids E2M1 and E2M1-4/6):
  none | H16 | UFP4 (uniform + H16) | smooth (SmoothQuant alpha 0.5, computed per call on the reduction axis)
  DGRAD learned: static per-O-channel scale learned on calibration batches (STE, DGRAD output error)
  WGRAD learned-alpha: per-token s_n = (max|dY_n| / max|X_n|)^alpha, alpha per Linear picked on calibration batches
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--tag", default="qwen2.5-0.5b"); ap.add_argument("--out", required=True)
ap.add_argument("--ncal", type=int, default=2); ap.add_argument("--neval", type=int, default=4)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
wrap, tok = load_hf(args.hf, dev); hfm = wrap.m
for p in hfm.parameters(): p.requires_grad_(False)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); T = 1024
need = (args.ncal + args.neval) * (T + 1)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
seqs = torch.tensor(ids[:need], device=dev).view(-1, T + 1)
cal, ev = seqs[:args.ncal], seqs[args.ncal:]
H = hadamard(BLK, dev)
SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=torch.Generator(device=dev).manual_seed(0)) * 2 - 1).float()


def e2m1(A): return quantize(A.contiguous(), E2M1, "rtn", "fp32")[0]
def unif(A): return quantize(A.contiguous(), UNIF, "rtn", "fp32")[0]


def q46(A):
    A = A.contiguous(); N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def rht(A): return ((A * SG[:A.shape[1]]).view(A.shape[0], -1, BLK) @ H).view(A.shape)
def ste(A): return A + (e2m1(A.detach()) - A).detach()


def qgemm(A, B, v, grid, s_static=None, alpha=None):
    """A [M,R], B [P,R] -> A_hat @ B_hat^T under variant v."""
    qf = {"e2m1": e2m1, "e46": q46}[grid]
    if v == "none":
        return qf(A) @ qf(B).T
    if v == "h16":
        return qf(rht(A)) @ qf(rht(B)).T
    if v == "ufp4":
        return unif(rht(A)) @ unif(rht(B)).T
    if v == "smooth":
        s = (A.abs().amax(0).clamp(min=1e-12) / B.abs().amax(0).clamp(min=1e-12)).sqrt()
    elif v == "learned":
        s = s_static
    elif v == "learned_alpha":
        s = A.abs().amax(0).clamp(min=1e-12) ** alpha / B.abs().amax(0).clamp(min=1e-12) ** alpha
    return qf(A / s) @ qf(B * s).T


CFG = {"gemm": None, "v": "none", "grid": "e2m1", "phase": "eval"}


class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mod):
        ctx.save_for_backward(x, w); ctx.mod = mod
        xs = x.reshape(-1, x.shape[-1]).float()
        y = e2m1(xs) @ e2m1(w.float()).T
        if b is not None: y = y + b.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors; m = ctx.mod
        xs = x.reshape(-1, x.shape[-1]).float(); dy = gy.reshape(-1, gy.shape[-1]).float(); W = w.float()
        if CFG["phase"] == "calib":
            m.cal.append((dy.cpu(), xs.cpu()))
        g = CFG["gemm"]
        if g == "dgrad":
            dx = qgemm(dy, W.t(), CFG["v"], CFG["grid"], s_static=m.s_dgrad)
        else:
            dx = dy @ W
        if g == "wgrad":
            dw = qgemm(dy.t(), xs.t(), CFG["v"], CFG["grid"], alpha=m.alpha)
        else:
            dw = dy.t() @ xs
        m.grad_acc += dw
        return dx.view(*x.shape).to(x.dtype), None, None, None


class QLin(torch.nn.Module):
    def __init__(self, lin, name):
        super().__init__()
        self.lin, self.name = lin, name
        self.grad_acc = torch.zeros_like(lin.weight, dtype=torch.float32)
        self.cal, self.s_dgrad, self.alpha = [], None, 0.5

    def forward(self, x):
        return Fn.apply(x, self.lin.weight, self.lin.bias, self)


Q = []
for i, blk in enumerate([m for n, m in hfm.named_modules() if n.endswith("layers")][0]):
    for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
        for cn, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QLin(child, f"layers.{i}.{pn}.{cn}"); setattr(parent, cn, q); Q.append(q)


def run(data):
    for q in Q: q.grad_acc.zero_()
    for s in data:
        loss = F.cross_entropy(hfm(input_ids=s[None, :-1], use_cache=False).logits.float()[0], s[1:]); loss.backward()
    return torch.cat([q.grad_acc.flatten() for q in Q])


# ---- calibration pass (captures dY, X per Linear), then fit learned variants
CFG.update(gemm=None, phase="calib"); run(cal); CFG["phase"] = "eval"
for q in Q:
    dY = torch.cat([a for a, _ in q.cal]).to(dev); X = torch.cat([b for _, b in q.cal]).to(dev); q.cal = []
    W = q.lin.weight.detach().float()
    # DGRAD learned static per-O-channel scale (init SmoothQuant 0.5 on the calibration batch)
    Y = dY @ W; nY = (Y * Y).sum()
    v = (0.5 * (torch.log(dY.abs().amax(0).clamp(min=1e-12)) - torch.log(W.t().abs().amax(0).clamp(min=1e-12)))).clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=1e-2); best = (1e30, None)
    for it in range(200):
        s = torch.exp(v); loss = ((ste(dY / s) @ ste(W.t() * s).T - Y) ** 2).sum() / nY
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), torch.exp(v.detach()))
    q.s_dgrad = best[1]
    # WGRAD learned alpha (per-token scale exponent), grid search on the calibration batch
    A, B = dY.t().contiguous(), X.t().contiguous(); G = A @ B.T; nG = (G * G).sum(); errs = {}
    for a in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        s = A.abs().amax(0).clamp(min=1e-12) ** a / B.abs().amax(0).clamp(min=1e-12) ** a
        errs[a] = (((e2m1(A / s) @ e2m1(B * s).T - G) ** 2).sum() / nG).item()
    q.alpha = min(errs, key=errs.get)
print(f"[{args.tag}] calibrated; WGRAD alphas: " + json.dumps({a: sum(q.alpha == a for q in Q) for a in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)}), flush=True)

CFG.update(gemm=None); Gr = run(ev); nGr = Gr.norm()
res = dict(model=args.tag)
for gemm, vs in [("dgrad", ["none", "h16", "ufp4", "smooth", "learned"]), ("wgrad", ["none", "h16", "ufp4", "smooth", "learned_alpha"])]:
    for v in vs:
        for grid in (["e2m1"] if v == "ufp4" else ["e2m1", "e46"]):
            CFG.update(gemm=gemm, v=v, grid=grid); G = run(ev)
            r = dict(grad_relerr=((G - Gr).norm() / nGr).item(), grad_cos=(G @ Gr / (G.norm() * nGr)).item())
            res[f"{gemm}|{v}|{grid}"] = r
            print(f"[{args.tag}] {gemm} {v:14s} {grid:5s} grad relerr {r['grad_relerr']:.4f}  cos {r['grad_cos']:.5f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
