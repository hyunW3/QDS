"""Zero-overhead (foldable) per-channel scaling for E2M1 FPROP (W4A4 PTQ), held-out BPB; same split as learned_eval.py.

Variants (all learned with STE on E2M1, no shift unless stated):
  DIAG-perLinear   one scale vector per Linear, layer-wise objective (= previous "그룹별 대각")
  DIAG-shared      q/k/v share one vector, gate/up share one vector; layer-wise objective = sum of member errors
  DIAG-fold        DIAG-shared + o_proj scale tied across query heads of one KV group (GQA), so every scale can be
                   folded into the previous op: q/k/v & gate/up -> RMSNorm weight, o_proj -> v_proj rows,
                   down_proj -> up_proj rows. For pre-norm models the folded model is built and checked for equality.
  LET-noshift      DIAG-fold parametrisation, optimised per decoder block on the block output (OmniQuant-style)
SmoothQuant alpha=0.5 (shared per input, foldable) as reference. Evaluated with E2M1 and E2M1-4/6.
"""
import os
import argparse, copy, json, math, sys, time
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
cfg = hfm.config
PRE_NORM = hasattr(hfm.model.layers[0], "input_layernorm")
Hq, Hkv = cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
REP = Hq // Hkv
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
    def __init__(self, lin, name):
        super().__init__()
        self.lin, self.name, self.mode, self.cal, self.s, self.qf, self.train = lin, name, "bf16", [], None, e2m1, False

    def forward(self, x):
        if self.mode == "capture":
            self.cal.append(x.detach().reshape(-1, x.shape[-1]).float().cpu()); return self.lin(x)
        if self.mode == "bf16": return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        if self.s is not None:
            s = self.s() if callable(self.s) else self.s
            xs, W = xs / s, W * s
        qf = ste if self.train else self.qf
        y = qf(xs) @ qf(W).T
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)


BLOCKS = []
for i, blk in enumerate(hfm.model.layers):
    d = {}
    for pn, parent in [("self_attn", blk.self_attn), ("mlp", blk.mlp)]:
        for cn, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QL(child, f"layers.{i}.{pn}.{cn}"); setattr(parent, cn, q); d[cn] = q
    BLOCKS.append((blk, d))
Q = [q for _, d in BLOCKS for q in d.values()]
GROUPS = [("attn_in", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("mlp_in", ("gate_proj", "up_proj")), ("down", ("down_proj",))]


@torch.no_grad()
def bpb():
    nats = 0.0
    for s in ev:
        logits = hfm(input_ids=s[None, :-1], use_cache=False).logits.float()[0]
        nats += F.cross_entropy(logits, s[1:], reduction="sum").item()
    return nats / math.log(2) / ev_bytes


def set_all(mode, scales=None, qf=e2m1):
    for q in Q:
        q.mode, q.qf = mode, qf
        q.s = None if scales is None else scales.get(q.name)


res = dict(model=args.tag, pre_norm=PRE_NORM, gqa_rep=REP)
set_all("bf16"); res["BF16"] = bpb()
# ---- capture inputs per Linear and per block
blk_io = {i: [] for i in range(len(BLOCKS))}; blk_out = {i: [] for i in range(len(BLOCKS))}
hooks = []
for i, (blk, _) in enumerate(BLOCKS):
    hooks.append(blk.register_forward_pre_hook(lambda m, a, k, i=i: blk_io[i].append(((a[0] if a else k["hidden_states"]).detach(), {kk: v for kk, v in k.items() if kk != "hidden_states"})), with_kwargs=True))
    hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: blk_out[i].append((o[0] if isinstance(o, tuple) else o).detach())))
set_all("capture")
with torch.no_grad():
    for s in cal: hfm(input_ids=s[None, :-1], use_cache=False)
for h in hooks: h.remove()
X_of = {q.name: torch.cat(q.cal) for q in Q}
for q in Q: q.cal = []
set_all("bf16")


def expand_o(v):           # kv*hd -> Hq*hd (query heads of one KV group share the scale)
    return v.view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(Hkv * REP * hd)


def make_params(d, init):
    """per group: log-scale parameter (o_proj parametrised per KV channel when tie_o)."""
    P = {}
    for g, names in GROUPS:
        mem = [d[n] for n in names if n in d]
        if not mem: continue
        X = X_of[mem[0].name].to(dev)
        if init == "sq":
            ax = X.abs().amax(0).clamp(min=1e-8)
            aw = torch.stack([m.lin.weight.detach().float().abs().amax(0) for m in mem]).amax(0).clamp(min=1e-8)
            v = torch.log(ax.sqrt() / aw.sqrt())
        else:
            v = torch.zeros(X.shape[1], device=dev)
        if g == "o":
            v = v.view(Hkv, REP, hd).mean(1).reshape(-1)          # tie across query heads of a KV group
        P[g] = (mem, v.clone().requires_grad_(True))
    return P


def scale_of(g, v):
    return torch.exp(expand_o(v) if g == "o" else v)


def assign(P, train):
    for g, (mem, v) in P.items():
        for m in mem:
            m.s = (lambda g=g, v=v: scale_of(g, v)); m.train = train; m.mode = "quant"; m.qf = e2m1


def freeze(P):
    return {m.name: scale_of(g, v.detach()).clone() for g, (mem, v) in P.items() for m in mem}


scales = {}
# SmoothQuant shared (foldable)
sq = {}
for blk, d in BLOCKS:
    sq.update(freeze(make_params(d, "sq")))
scales["SmoothQuant α=0.5 (shared, foldable)"] = sq
t0 = time.time()
# DIAG-perLinear
pl = {}
for q in Q:
    X = X_of[q.name].to(dev); W = q.lin.weight.detach().float(); Y = X @ W.T; nY = (Y * Y).sum()
    v = torch.zeros(W.shape[1], device=dev, requires_grad=True); opt = torch.optim.Adam([v], lr=1e-2); best = (1e30, None)
    for it in range(args.steps):
        s = torch.exp(v); loss = ((ste(X / s) @ ste(W * s).T - Y) ** 2).sum() / nY
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), torch.exp(v.detach()))
    pl[q.name] = best[1]
scales["DIAG per-Linear"] = pl
print(f"[{args.tag}] DIAG per-Linear ({time.time()-t0:.0f}s)", flush=True)
# DIAG-shared / DIAG-fold (layer-wise objective = sum of member output errors)
for name, tie, sqinit in [("DIAG shared", False, False), ("DIAG fold (shared + GQA tie)", True, False), ("DIAG shared, SQ init", False, True)]:
    out = {}
    for blk, d in BLOCKS:
        for g, names in GROUPS:
            mem = [d[n] for n in names if n in d]
            X = X_of[mem[0].name].to(dev)
            K = X.shape[1]
            if sqinit:
                ax = X.abs().amax(0).clamp(min=1e-8)
                aw = torch.stack([m.lin.weight.detach().float().abs().amax(0) for m in mem]).amax(0).clamp(min=1e-8)
                v = torch.log(ax.sqrt() / aw.sqrt()).clone().requires_grad_(True)
            else:
                v = torch.zeros(Hkv * hd if (g == "o" and tie) else K, device=dev, requires_grad=True)
            Ws = [m.lin.weight.detach().float() for m in mem]; Ys = [X @ W.T for W in Ws]; nYs = [(Y * Y).sum() for Y in Ys]
            opt = torch.optim.Adam([v], lr=1e-2); best = (1e30, None)
            for it in range(args.steps):
                s = torch.exp(expand_o(v) if (g == "o" and tie) else v)
                Xq = ste(X / s)
                loss = sum(((Xq @ ste(W * s).T - Y) ** 2).sum() / nY for W, Y, nY in zip(Ws, Ys, nYs))
                opt.zero_grad(); loss.backward(); opt.step()
                if loss.item() < best[0]: best = (loss.item(), torch.exp(expand_o(v.detach()) if (g == "o" and tie) else v.detach()))
            for m in mem: out[m.name] = best[1]
    scales[name] = out
    print(f"[{args.tag}] {name} ({time.time()-t0:.0f}s)", flush=True)
# LET-noshift: fold parametrisation, block-wise objective, init SmoothQuant
let = {}
for i, (blk, d) in enumerate(BLOCKS):
    P = make_params(d, "sq"); assign(P, True)
    opt = torch.optim.Adam([v for _, v in P.values()], lr=5e-3); best = (1e30, None)
    for it in range(150):
        tot = 0.0
        for (h, kw), tgt in zip(blk_io[i], blk_out[i]):
            o = blk(h, **kw); o = o[0] if isinstance(o, tuple) else o
            loss = ((o.float() - tgt.float()) ** 2).mean() / (tgt.float() ** 2).mean(); loss.backward(); tot += loss.item()
        opt.step(); opt.zero_grad()
        if tot < best[0]: best = (tot, freeze(P))
    let.update(best[1])
    for m in d.values(): m.mode, m.s, m.train = "bf16", None, False
scales["LET-noshift (block-wise, fold)"] = let
let_u = {}
for i, (blk, d) in enumerate(BLOCKS):
    P = {}
    for n, m in d.items():
        X = X_of[m.name].to(dev); W = m.lin.weight.detach().float()
        v = torch.log(X.abs().amax(0).clamp(min=1e-8).sqrt() / W.abs().amax(0).clamp(min=1e-8).sqrt())
        P[n] = ([m], v.clone().requires_grad_(True))
    assign(P, True)
    opt = torch.optim.Adam([v for _, v in P.values()], lr=5e-3); best = (1e30, None)
    for it in range(150):
        tot = 0.0
        for (h, kw), tgt in zip(blk_io[i], blk_out[i]):
            o = blk(h, **kw); o = o[0] if isinstance(o, tuple) else o
            loss = ((o.float() - tgt.float()) ** 2).mean() / (tgt.float() ** 2).mean(); loss.backward(); tot += loss.item()
        opt.step(); opt.zero_grad()
        if tot < best[0]: best = (tot, freeze(P))
    let_u.update(best[1])
    for m in d.values(): m.mode, m.s, m.train = "bf16", None, False
scales["LET-noshift per-Linear (block-wise, unshared)"] = let_u
print(f"[{args.tag}] LET-noshift unshared ({time.time()-t0:.0f}s)", flush=True)
print(f"[{args.tag}] LET-noshift ({time.time()-t0:.0f}s)", flush=True)
# ---- evaluate
for name, sc in [("E2M1 (no transform)", None)] + list(scales.items()):
    for qn, qf in [("E2M1", e2m1), ("E2M1-4/6", q46)]:
        set_all("quant", sc, qf); res[f"{name} | {qn}"] = bpb()
        print(f"[{args.tag}] {name:40s} {qn:8s} BPB {res[f'{name} | {qn}']:.4f} ({100*(res[f'{name} | {qn}']/res['BF16']-1):+.2f}%)", flush=True)
# ---- fold check (pre-norm models): put DIAG-fold scales into norm / v / up weights, then run plain E2M1
if PRE_NORM:
    sc = scales["DIAG fold (shared + GQA tie)"]
    with torch.no_grad():
        for i, (blk, d) in enumerate(BLOCKS):
            s_in, s_mlp = sc[d["q_proj"].name], sc[d["gate_proj"].name]
            s_o, s_dn = sc[d["o_proj"].name], sc[d["down_proj"].name]
            blk.input_layernorm.weight.div_(s_in.to(blk.input_layernorm.weight.dtype))
            for n in ("q_proj", "k_proj", "v_proj"): d[n].lin.weight.mul_(s_in.to(d[n].lin.weight.dtype))
            blk.post_attention_layernorm.weight.div_(s_mlp.to(blk.post_attention_layernorm.weight.dtype))
            for n in ("gate_proj", "up_proj"): d[n].lin.weight.mul_(s_mlp.to(d[n].lin.weight.dtype))
            s_kv = s_o.view(Hkv, REP, hd)[:, 0, :].reshape(-1)          # tied -> one value per KV channel
            v = d["v_proj"].lin; v.weight.div_(s_kv[:, None].to(v.weight.dtype))
            if v.bias is not None: v.bias.div_(s_kv.to(v.bias.dtype))
            d["o_proj"].lin.weight.mul_(s_o.to(d["o_proj"].lin.weight.dtype))
            d["up_proj"].lin.weight.div_(s_dn[:, None].to(d["up_proj"].lin.weight.dtype))
            d["down_proj"].lin.weight.mul_(s_dn.to(d["down_proj"].lin.weight.dtype))
    set_all("bf16"); res["fold: BF16 after folding"] = bpb()
    set_all("quant", None, e2m1); res["fold: plain E2M1 on folded weights"] = bpb()
    set_all("quant", None, q46); res["fold: plain E2M1-4/6 on folded weights"] = bpb()
    for k in ("fold: BF16 after folding", "fold: plain E2M1 on folded weights", "fold: plain E2M1-4/6 on folded weights"):
        print(f"[{args.tag}] {k:40s} BPB {res[k]:.4f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
