"""Model quality (held-out loss / BPB) with learned per-channel-group transforms for FP4 FPROP (W4A4, PTQ-style).

Every decoder Linear runs Y = Q(X T) Q(W T^{-T})^T where T acts on the input-channel axis K, block-diagonal per
16-channel group (the quantization block):
  GROUP-DIAG  T_g = diag(exp(p_g))     (learned smoothing; 16 scalars per group)
  GROUP-INV   T_g = I + U_g            (general invertible 16x16 per group)
T is learned per Linear (Adam, straight-through estimator) to minimize the E2M1 FPROP output error on calibration
activations, then frozen. Baselines: BF16, E2M1 (no rotation), E2M1+H16, UFP4 (uniform E1M2 + H16), E2M1-4/6.
Calibration and evaluation texts are disjoint (DCLM val). BPB = cross-entropy / ln2 / bytes of the predicted tokens.
"""
import os
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ncal", type=int, default=4); ap.add_argument("--neval", type=int, default=16); ap.add_argument("--steps", type=int, default=250)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
wrap, tok = load_hf(args.hf, dev)
hfm = wrap.m
H = hadamard(BLK, dev)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); T = 1024
need = (args.ncal + args.neval) * (T + 1)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
seqs = torch.tensor(ids[:need], device=dev).view(-1, T + 1)
cal, ev = seqs[:args.ncal], seqs[args.ncal:]
ev_bytes = sum(len(tok.t.decode(s[1:].tolist()).encode("utf-8")) for s in ev)


def e2m1(A): return quantize(A, E2M1, "rtn", "fp32")[0]
def unif(A): return quantize(A, UNIF, "rtn", "fp32")[0]


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def ste(qf):
    return lambda A: A + (qf(A.detach()) - A).detach()


def bmul(A, M):
    N, R = A.shape
    if M.dim() == 2:
        return (A.view(N, R // BLK, BLK) @ M).view(N, R)
    return (A.view(N, R // BLK, 1, BLK) @ M.unsqueeze(0)).view(N, R)


def mats(kind, P):
    if kind == "diag":
        return torch.diag_embed(torch.exp(P)), torch.diag_embed(torch.exp(-P))
    M = torch.eye(BLK, device=dev) + P
    return M, torch.linalg.inv(M).transpose(-1, -2)


class QL(torch.nn.Module):
    def __init__(self, lin, name):
        super().__init__()
        self.lin, self.name, self.mode = lin, name, "bf16"
        self.T = {}          # kind -> (M for X, Minv_T for W)
        self.cal = []

    def forward(self, x):
        if self.mode == "capture":
            self.cal.append(x.detach().reshape(-1, x.shape[-1]).float().cpu())
            return self.lin(x)
        if self.mode == "bf16":
            return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        grid, tf = self.mode.split("+")                       # e.g. "e2m1+none", "unif+h16", "e46+inv"
        qf = {"e2m1": e2m1, "unif": unif, "e46": q46}[grid]
        if tf == "h16":
            xs, W = bmul(xs * SG[:xs.shape[1]], H), bmul(W * SG[:W.shape[1]], H)
        elif tf in ("diag", "inv"):
            M, Mi = self.T[tf]; xs, W = bmul(xs, M), bmul(W, Mi)
        y = qf(xs) @ qf(W).T
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)


SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=torch.Generator(device=dev).manual_seed(0)) * 2 - 1).float()
Q = []
layers = [m for n, m in hfm.named_modules() if n.endswith("layers")][0]
for i, blk in enumerate(layers):
    for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
        for cn, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QL(child, f"layers.{i}.{pn}.{cn}"); setattr(parent, cn, q); Q.append(q)


@torch.no_grad()
def evaluate():
    nats = 0.0; ntok = 0
    for s in ev:
        logits = hfm(input_ids=s[None, :-1]).logits.float()[0]
        nats += F.cross_entropy(logits, s[1:], reduction="sum").item(); ntok += len(s) - 1
    return dict(loss=nats / ntok, bpb=nats / math.log(2) / ev_bytes)


# ---- calibration activations
for q in Q: q.mode = "capture"
with torch.no_grad():
    for s in cal: hfm(input_ids=s[None, :-1])
t0 = time.time()
for kind in ("diag", "inv"):
    for q in Q:
        X = torch.cat(q.cal).to(dev); W = q.lin.weight.float(); Y = X @ W.T; nY = (Y * Y).sum()
        G = W.shape[1] // BLK
        P = torch.zeros((G, BLK) if kind == "diag" else (G, BLK, BLK), device=dev, requires_grad=True)
        opt = torch.optim.Adam([P], lr=1e-2 if kind == "diag" else 3e-3)
        best = (float("inf"), None)
        for it in range(args.steps):
            M, Mi = mats(kind, P)
            loss = ((ste(e2m1)(bmul(X, M)) @ ste(e2m1)(bmul(W, Mi)).T - Y) ** 2).sum() / nY
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best[0]: best = (loss.item(), P.detach().clone())
        with torch.no_grad():
            q.T[kind] = mats(kind, best[1])
        del X, Y
    print(f"[{args.tag}] learned {kind} for {len(Q)} Linears ({time.time()-t0:.0f}s)", flush=True)
for q in Q: q.cal = []

res = dict(model=args.tag, hf=args.hf, ncal_tokens=args.ncal * T, neval_tokens=args.neval * T)
for name, mode in [("BF16", "bf16"), ("E2M1 회전 없음", "e2m1+none"), ("E2M1 + H16", "e2m1+h16"), ("UFP4 (균일 + H16)", "unif+h16"),
                   ("E2M1-4/6", "e46+none"), ("E2M1 + 그룹별 대각", "e2m1+diag"), ("E2M1-4/6 + 그룹별 대각", "e46+diag"),
                   ("E2M1 + 그룹별 가역", "e2m1+inv"), ("E2M1-4/6 + 그룹별 가역", "e46+inv"), ("균일 + 그룹별 대각", "unif+diag")]:
    for q in Q: q.mode = mode
    res[name] = evaluate()
    print(f"[{args.tag}] {name:24s} loss {res[name]['loss']:.4f}  BPB {res[name]['bpb']:.4f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
