"""Which GEMM operand roles are sensitive to FP4? (open-source HF model, one training step, no weight update)

Six roles per Linear:  FPROP: fwd_x, fwd_w | DGRAD: dgrad_dy, dgrad_wT | WGRAD: wgrad_dyT, wgrad_xT
A role can be BF16 (reference), FP4 (E2M1 RTN, 1x16 blocks along that GEMM's reduction axis, FP32 absmax scale, no
rotation), FP8 (E4M3, 1x16), or NOISE (BF16 + Gaussian noise with the energy of that operand's FP4 error).
Measured against the all-BF16 step on the same batches:
  grad_relerr = ||g - g_ref|| / ||g_ref||   over all decoder Linear weight gradients (concatenated)
  grad_cos    = cos(g, g_ref)
  dloss       = loss - loss_ref (only forward roles change it)
Part A: one role FP4 (or NOISE) in every layer, the rest BF16.
Part B: one (layer, role) FP4 at a time -> sensitivity map.
Part C: all roles FP4 except one role kept FP8 (which role is worth FP8).
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, quantize
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--tag", default="qwen2.5-0.5b")
ap.add_argument("--out", required=True); ap.add_argument("--nseq", type=int, default=4)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
wrap, tok = load_hf(args.hf, dev)
hfm = wrap.m
for p in hfm.parameters():
    p.requires_grad_(False)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); T = 1024
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= args.nseq * (T + 1): break
data = torch.tensor(ids[:args.nseq * (T + 1)], device=dev).view(args.nseq, T + 1)
ROLES = ["fwd_x", "fwd_w", "dgrad_dy", "dgrad_wT", "wgrad_dyT", "wgrad_xT"]


def fp4(A):
    return quantize(A.contiguous(), E2M1, "rtn", "fp32")[0]


def fp8(A):
    A = A.contiguous(); N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    s = (Ab.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return ((Ab / s).to(torch.float8_e4m3fn).float() * s).view(N, R)


def apply(mode, A):
    """A: [rows, R] with the reduction axis last."""
    A = A.float()
    if mode == "bf16": return A
    if mode == "fp4": return fp4(A)
    if mode == "fp8": return fp8(A)
    if mode == "noise":
        e = fp4(A) - A
        return A + torch.randn_like(A) * e.pow(2).mean().sqrt()
    raise ValueError(mode)


class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mod):
        ctx.save_for_backward(x, w); ctx.mod = mod
        xs = x.reshape(-1, x.shape[-1])
        y = apply(mod.cfg["fwd_x"], xs) @ apply(mod.cfg["fwd_w"], w).T
        if b is not None: y = y + b.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors; c = ctx.mod.cfg
        xs = x.reshape(-1, x.shape[-1]); dy = gy.reshape(-1, gy.shape[-1])
        dx = apply(c["dgrad_dy"], dy) @ apply(c["dgrad_wT"], w.t()).T
        dw = apply(c["wgrad_dyT"], dy.t()) @ apply(c["wgrad_xT"], xs.t()).T
        ctx.mod.grad_acc += dw
        return dx.view(*x.shape).to(x.dtype), None, None, None


class QLin(torch.nn.Module):
    def __init__(self, lin, name, layer):
        super().__init__()
        self.lin, self.name, self.layer = lin, name, layer
        self.cfg = {r: "bf16" for r in ROLES}
        self.grad_acc = torch.zeros_like(lin.weight, dtype=torch.float32)

    def forward(self, x):
        return Fn.apply(x, self.lin.weight, self.lin.bias, self)


Q = []
layers = [m for n, m in hfm.named_modules() if n.endswith("layers")][0]
for i, blk in enumerate(layers):
    for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
        for cn, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QLin(child, f"layers.{i}.{pn}.{cn}", i); setattr(parent, cn, q); Q.append(q)
NL = len(layers)


def run():
    for q in Q: q.grad_acc.zero_()
    tot = 0.0
    for s in range(args.nseq):
        b = data[s:s + 1]
        loss = F.cross_entropy(hfm(input_ids=b[:, :-1]).logits.float().flatten(0, 1), b[:, 1:].flatten())
        loss.backward(); tot += loss.item()
    return tot / args.nseq, torch.cat([q.grad_acc.flatten() for q in Q])


def setcfg(fn):
    for q in Q:
        for r in ROLES:
            q.cfg[r] = fn(q, r)


torch.manual_seed(0)
setcfg(lambda q, r: "bf16"); L0, G0 = run(); nG0 = G0.norm()


def metrics(L, G):
    return dict(dloss=L - L0, grad_relerr=((G - G0).norm() / nG0).item(), grad_cos=(G @ G0 / (G.norm() * nG0)).item())


res = dict(model=args.tag, loss_ref=L0, partA={}, partB={}, partC={})
for mode in ["fp4", "noise"]:
    for r in ROLES:
        torch.manual_seed(1)
        setcfg(lambda q, rr: mode if rr == r else "bf16")
        res["partA"][f"{mode}:{r}"] = metrics(*run())
        print(f"[A] {mode:5s} {r:10s} " + json.dumps({k: round(v, 5) for k, v in res["partA"][f'{mode}:{r}'].items()}), flush=True)
for mode in ["all_fp4", "all_fp8"]:
    setcfg(lambda q, rr: mode.split("_")[1])
    res["partA"][mode] = metrics(*run())
    print(f"[A] {mode} " + json.dumps({k: round(v, 5) for k, v in res["partA"][mode].items()}), flush=True)
for keep in ROLES:
    setcfg(lambda q, rr: "fp8" if rr == keep else "fp4")
    res["partC"][f"fp8:{keep}"] = metrics(*run())
    print(f"[C] all FP4 but {keep} FP8 " + json.dumps({k: round(v, 5) for k, v in res["partC"][f'fp8:{keep}'].items()}), flush=True)
for li in range(NL):
    for r in ROLES:
        setcfg(lambda q, rr: "fp4" if (rr == r and q.layer == li) else "bf16")
        res["partB"][f"{li}:{r}"] = metrics(*run())
    print(f"[B] layer {li} done", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print(f"[{args.tag}] done -> {args.out}", flush=True)
