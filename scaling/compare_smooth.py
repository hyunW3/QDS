"""Per-channel smoothing variants for E2M1 FP4 FPROP (W4A4 PTQ), held-out BPB. Same data split as learned_eval.py.

  SQ(alpha)      SmoothQuant closed form s = max|X|^a / max|W|^(1-a); alpha = 0.5 and the best of a grid (by calib BPB)
  DIAG (ours)    per-channel log-scale learned with STE to minimise THIS Linear's E2M1 output error (init s = 1)
  DIAG+SHIFT     same plus a per-channel shift delta: X' = (X - delta)/s, bias += delta W^T (exact)
  LET-style      OmniQuant-style learnable equivalent transformation (our re-implementation): scale + shift shared by
                 Linears that read the same input (q/k/v; gate/up), init from SQ(0.5), optimised per decoder block to
                 minimise the BLOCK output error (FP block inputs). OmniQuant's learnable weight clipping (LWC) is omitted.
                 Shifts start at 0: OmniQuant's (max+min)/2 init blew up on massive-activation channels under 1x16 block absmax.
Every variant is evaluated with E2M1 and E2M1-4/6 (4/6 = per-block scale choice absmax->6 or ->4).
"""
import os
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, quantize
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ncal", type=int, default=4); ap.add_argument("--neval", type=int, default=16); ap.add_argument("--steps", type=int, default=250)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
wrap, tok = load_hf(args.hf, dev); hfm = wrap.m
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); T = 1024
need = (args.ncal + args.neval) * (T + 1)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
seqs = torch.tensor(ids[:need], device=dev).view(-1, T + 1)
cal, ev = seqs[:args.ncal], seqs[args.ncal:]
nbytes = lambda S: sum(len(tok.t.decode(s[1:].tolist()).encode("utf-8")) for s in S)
ev_bytes, cal_bytes = nbytes(ev), nbytes(cal)


def e2m1(A): return quantize(A, E2M1, "rtn", "fp32")[0]


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def ste(A): return A + (e2m1(A.detach()) - A).detach()


class QL(torch.nn.Module):
    """mode: bf16 | capture | quant. Transform params live in self.p[name] = (log_s, delta or None) or a shared holder."""
    def __init__(self, lin, name):
        super().__init__()
        self.lin, self.name, self.mode, self.cal = lin, name, "bf16", []
        self.p = {}; self.cur = None; self.qf = e2m1; self.train_ste = False

    def forward(self, x):
        if self.mode == "capture":
            self.cal.append(x.detach().reshape(-1, x.shape[-1]).float().cpu()); return self.lin(x)
        if self.mode == "bf16":
            return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        b = self.lin.bias.float() if self.lin.bias is not None else None
        if self.cur is not None:
            ls, dl = self.cur() if callable(self.cur) else self.cur
            s = torch.exp(ls)
            if dl is not None:
                xs = xs - dl
                extra = dl @ W.T
                b = extra if b is None else b + extra
            xs, W = xs / s, W * s
        qf = ste if self.train_ste else self.qf
        y = qf(xs) @ qf(W).T
        if b is not None: y = y + b
        return y.view(*x.shape[:-1], -1).to(x.dtype)


Q, BLOCKS = [], []
layers = [m for n, m in hfm.named_modules() if n.endswith("layers")][0]
for i, blk in enumerate(layers):
    bq = {}
    for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
        for cn, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QL(child, f"layers.{i}.{pn}.{cn}"); setattr(parent, cn, q); Q.append(q); bq[cn] = q
    BLOCKS.append((blk, bq))


@torch.no_grad()
def bpb(data, nb):
    nats = 0.0
    for s in data:
        logits = hfm(input_ids=s[None, :-1], use_cache=False).logits.float()[0]
        nats += F.cross_entropy(logits, s[1:], reduction="sum").item()
    return nats / math.log(2) / nb


def set_mode(mode, key=None, qf=e2m1):
    for q in Q:
        q.mode, q.qf = mode, qf
        q.cur = None if key is None else q.p[key]


res = dict(model=args.tag)
set_mode("bf16"); res["BF16"] = bpb(ev, ev_bytes)
# ---- calibration activations (per Linear) and block inputs/outputs (for LET-style)
blk_io = []
pre = [blk.register_forward_pre_hook(lambda m, a, k, i=i: blk_io.append(("in", i, a, k)), with_kwargs=True) for i, (blk, _) in enumerate(BLOCKS)]
post = [blk.register_forward_hook(lambda m, a, o, i=i: blk_io.append(("out", i, o))) for i, (blk, _) in enumerate(BLOCKS)]
set_mode("capture")
with torch.no_grad():
    for s in cal: hfm(input_ids=s[None, :-1], use_cache=False)
for h in pre + post: h.remove()
bins = {i: [] for i in range(len(BLOCKS))}; bouts = {i: [] for i in range(len(BLOCKS))}
for e in blk_io:
    if e[0] == "in":
        a, k = e[2], e[3]
        bins[e[1]].append(((a[0] if a else k["hidden_states"]).detach(), {kk: v for kk, v in k.items() if kk != "hidden_states"}))
    else:
        o = e[2]; bouts[e[1]].append((o[0] if isinstance(o, tuple) else o).detach())
del blk_io
X_of = {}
for q in Q:
    X_of[q.name] = torch.cat(q.cal); q.cal = []
    W = q.lin.weight.detach().float(); X = X_of[q.name].to(dev)
    ax, aw = X.abs().amax(0).clamp(min=1e-8), W.abs().amax(0).clamp(min=1e-8)
    for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        q.p[f"SQ{a}"] = (torch.log(ax ** a / aw ** (1 - a)), None)
# ---- SmoothQuant
set_mode("quant")
sq_cal = {}
for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    set_mode("quant", f"SQ{a}"); sq_cal[a] = bpb(cal, cal_bytes)
best_a = min(sq_cal, key=sq_cal.get); res["SQ_best_alpha"] = best_a; res["SQ_calib_bpb"] = sq_cal
print(f"[{args.tag}] SmoothQuant calib BPB by alpha: {sq_cal} -> best {best_a}", flush=True)
# ---- layer-wise learned DIAG / DIAG+SHIFT
t0 = time.time()
for shift in (False, True):
    key = "DIAG+SHIFT" if shift else "DIAG"
    for q in Q:
        X = X_of[q.name].to(dev); W = q.lin.weight.detach().float(); Y = X @ W.T; nY = (Y * Y).sum()
        ls = torch.zeros(W.shape[1], device=dev, requires_grad=True)
        dl = torch.zeros(W.shape[1], device=dev, requires_grad=True) if shift else None
        # shift init 0: a (max+min)/2 init explodes on massive-activation channels under block-absmax FP4
        params = [ls] + ([dl] if shift else [])
        opt = torch.optim.Adam(params, lr=1e-2); best = (float("inf"), None)
        for it in range(args.steps):
            s = torch.exp(ls); Xs = (X - dl) if shift else X
            Yh = ste(Xs / s) @ ste(W * s).T + ((dl @ W.T) if shift else 0)
            loss = ((Yh - Y) ** 2).sum() / nY
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best[0]: best = (loss.item(), (ls.detach().clone(), dl.detach().clone() if shift else None))
        q.p[key] = best[1]
    print(f"[{args.tag}] learned {key} ({time.time()-t0:.0f}s)", flush=True)
# ---- LET-style, block-wise, shared per input, init SQ(0.5)
GROUPS = [("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",)]
for i, (blk, bq) in enumerate(BLOCKS):
    holders = []
    for g in GROUPS:
        members = [bq[n] for n in g if n in bq]
        if not members: continue
        X = X_of[members[0].name].to(dev)
        ax = X.abs().amax(0).clamp(min=1e-8)
        aw = torch.stack([m.lin.weight.detach().float().abs().amax(0) for m in members]).amax(0).clamp(min=1e-8)
        ls = torch.log(ax ** 0.5 / aw ** 0.5).clone().requires_grad_(True)
        dl = torch.zeros_like(ax).requires_grad_(True)                      # init 0 (see layer-wise note)
        holders.append((members, ls, dl))
        for m in members:
            m.cur = (lambda ls=ls, dl=dl: (ls, dl)); m.mode = "quant"; m.train_ste = True; m.qf = e2m1
    params = [t for _, ls, dl in holders for t in (ls, dl)]
    opt = torch.optim.Adam(params, lr=5e-3); best = (float("inf"), None)
    for it in range(150):
        tot = 0.0
        for (h, kw), tgt in zip(bins[i], bouts[i]):
            out = blk(h, **kw); out = out[0] if isinstance(out, tuple) else out
            loss = ((out.float() - tgt.float()) ** 2).mean() / (tgt.float() ** 2).mean()
            loss.backward(); tot += loss.item()
        opt.step(); opt.zero_grad()
        if tot < best[0]: best = (tot, [(ls.detach().clone(), dl.detach().clone()) for _, ls, dl in holders])
    for (members, _, _), (ls, dl) in zip(holders, best[1]):
        for m in members:
            m.p["LET"] = (ls, dl); m.mode = "bf16"; m.train_ste = False; m.cur = None
print(f"[{args.tag}] LET-style done ({time.time()-t0:.0f}s)", flush=True)
# ---- evaluate
for name, key in [("E2M1 (no transform)", None), ("SmoothQuant α=0.5", "SQ0.5"), (f"SmoothQuant α={best_a} (best)", f"SQ{best_a}"),
                  ("그룹별 대각 (ours, layer-wise)", "DIAG"), ("그룹별 대각 + shift (layer-wise)", "DIAG+SHIFT"), ("LET-style (block-wise, shared, +shift)", "LET")]:
    for qn, qf in [("E2M1", e2m1), ("E2M1-4/6", q46)]:
        set_mode("quant", key, qf); res[f"{name} | {qn}"] = bpb(ev, ev_bytes)
        print(f"[{args.tag}] {name:42s} {qn:8s} BPB {res[f'{name} | {qn}']:.4f} ({100*(res[f'{name} | {qn}']/res['BF16']-1):+.2f}%)", flush=True)
json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
