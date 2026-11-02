"""Can the learned channel scale be made overhead-free on OLMo-2 (norm AFTER each sublayer, so q/k/v and gate/up read the raw
residual stream and there is no RMSNorm in front of them to absorb 1/s)?

  per-layer shared s (attn_in, mlp_in per layer)  -> not foldable; needs x/s inside the activation-quant kernel
  global residual s (ONE s for every layer's attn_in and mlp_in) -> foldable: residual stream basis h' = h / s,
        embed_tokens /= s, every post_attention_layernorm / post_feedforward_layernorm weight /= s, q/k/v/gate/up columns *= s,
        and one elementwise h' * s before the final norm (RMSNorm is not invariant to a non-uniform s).
  o_proj / down_proj scales are folded per layer into v_proj / up_proj rows in both variants (as for Llama).
Same E2M1 / E2M1-4/6 quantizers, STE objective and DCLM BPB split as compare_smooth2.py (so numbers are comparable).
"""
import os
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D))
from fp4lib import E2M1, BLK, quantize
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="allenai/OLMo-2-0425-1B"); ap.add_argument("--tag", default="olmo2-1b"); ap.add_argument("--out", required=True)
ap.add_argument("--cal", default=os.environ.get("DCLM_VAL", "data/dclm_val.jsonl"))
ap.add_argument("--ncal", type=int, default=4); ap.add_argument("--neval", type=int, default=16); ap.add_argument("--steps", type=int, default=250)
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
wrap, tok = load_hf(args.hf, dev); hfm = wrap.m; cfg = hfm.config
Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq; REP = Hq // Hkv
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); T = 1024
need = (args.ncal + args.neval) * (T + 1); ids = []
for line in open(args.cal):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
seqs = torch.tensor(ids[:need], device=dev).view(-1, T + 1); cal, ev = seqs[:args.ncal], seqs[args.ncal:]
ev_bytes = sum(len(tok.t.decode(s[1:].tolist()).encode("utf-8")) for s in ev)


def e2m1(A): return quantize(A, E2M1, "rtn", "fp32")[0]


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2; o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


def ste(A): return A + (e2m1(A.detach()) - A).detach()


class QL(torch.nn.Module):
    def __init__(self, lin):
        super().__init__(); self.lin, self.mode, self.s, self.qf, self.cal = lin, "bf16", None, e2m1, []

    def forward(self, x):
        if self.mode == "capture": self.cal.append(x.detach().reshape(-1, x.shape[-1]).float()); return self.lin(x)
        if self.mode == "bf16": return self.lin(x)
        xs = x.reshape(-1, x.shape[-1]).float(); W = self.lin.weight.float()
        if self.s is not None: xs, W = xs / self.s, W * self.s
        y = self.qf(xs) @ self.qf(W).T
        if self.lin.bias is not None: y = y + self.lin.bias.float()
        return y.view(*x.shape[:-1], -1).to(x.dtype)


BLOCKS = []
for blk in hfm.model.layers:
    d = {}
    for parent in (blk.self_attn, blk.mlp):
        for cn, ch in list(parent.named_children()):
            if isinstance(ch, torch.nn.Linear): q = QL(ch); setattr(parent, cn, q); d[cn] = q
    BLOCKS.append((blk, d))
Q = [q for _, d in BLOCKS for q in d.values()]


@torch.no_grad()
def bpb():
    nats = 0.0
    for s in ev: nats += F.cross_entropy(hfm(input_ids=s[None, :-1], use_cache=False).logits.float()[0], s[1:], reduction="sum").item()
    return nats / math.log(2) / ev_bytes


def set_all(mode, scales=None, qf=e2m1):
    for _, d in BLOCKS:
        for n, q in d.items(): q.mode, q.qf, q.s = mode, qf, (None if scales is None else scales.get(id(q)))


res = dict(model=args.tag); set_all("bf16"); res["BF16"] = bpb()
set_all("capture")
with torch.no_grad():
    for s in cal: hfm(input_ids=s[None, :-1], use_cache=False)
X = {id(q): torch.cat(q.cal) for q in Q}
for q in Q: q.cal = []
set_all("bf16")
expand = lambda u: u.view(Hkv, 1, hd).expand(Hkv, REP, hd).reshape(-1)


def learn(pairs, K, tie=False):
    """pairs: list of (X, [W...]); one log-scale vector shared by all of them (sum of normalised output errors)."""
    v = torch.zeros(Hkv * hd if tie else K, device=dev, requires_grad=True)
    tg = [(Xi, [(W, Xi @ W.T) for W in Ws]) for Xi, Ws in pairs]; nY = [[(Y * Y).sum() for _, Y in t] for _, t in tg]
    opt = torch.optim.Adam([v], lr=1e-2); best = (1e30, None)
    for _ in range(args.steps):
        s = torch.exp(expand(v) if tie else v); loss = 0.0
        for (Xi, t), n in zip(tg, nY):
            Xq = ste(Xi / s)
            for (W, Y), ny in zip(t, n): loss = loss + ((Xq @ ste(W * s).T - Y) ** 2).sum() / ny
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), torch.exp(expand(v.detach()) if tie else v.detach()))
    return best[1]


t0 = time.time()
local = {}                                        # per-layer o / down scales (foldable in both variants)
perlayer = {}                                     # per-layer attn_in / mlp_in scales (not foldable on OLMo-2)
for blk, d in BLOCKS:
    local[id(d["o_proj"])] = learn([(X[id(d["o_proj"])], [d["o_proj"].lin.weight.float()])], cfg.hidden_size, tie=True)
    local[id(d["down_proj"])] = learn([(X[id(d["down_proj"])], [d["down_proj"].lin.weight.float()])], cfg.intermediate_size)
    s_in = learn([(X[id(d["q_proj"])], [d[n].lin.weight.float() for n in ("q_proj", "k_proj", "v_proj")])], cfg.hidden_size)
    s_mlp = learn([(X[id(d["gate_proj"])], [d[n].lin.weight.float() for n in ("gate_proj", "up_proj")])], cfg.hidden_size)
    for n in ("q_proj", "k_proj", "v_proj"): perlayer[id(d[n])] = s_in
    for n in ("gate_proj", "up_proj"): perlayer[id(d[n])] = s_mlp
print(f"per-layer scales learned ({time.time()-t0:.0f}s)", flush=True)
# one global residual-stream scale for all attn_in and mlp_in inputs (subsample tokens to bound memory)
pairs = []
for blk, d in BLOCKS:
    pairs.append((X[id(d["q_proj"])][::2], [d[n].lin.weight.float() for n in ("q_proj", "k_proj", "v_proj")]))
    pairs.append((X[id(d["gate_proj"])][::2], [d[n].lin.weight.float() for n in ("gate_proj", "up_proj")]))
s_glob = learn(pairs, cfg.hidden_size); del pairs
print(f"global scale learned ({time.time()-t0:.0f}s)", flush=True)
glob = dict(local); [glob.__setitem__(id(d[n]), s_glob) for _, d in BLOCKS for n in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")]
per = dict(local); per.update(perlayer)
# geometric mean of the per-layer vectors as a training-free global alternative
gm = torch.exp(torch.stack([perlayer[id(d[n])].log() for _, d in BLOCKS for n in ("q_proj", "gate_proj")]).mean(0))
glob_gm = dict(local); [glob_gm.__setitem__(id(d[n]), gm) for _, d in BLOCKS for n in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")]
for name, sc in [("E2M1 (no transform)", None), ("per-layer s (in-kernel x/s, not foldable)", per),
                 ("global residual s (learned jointly, foldable)", glob), ("global residual s = geo-mean of per-layer s", glob_gm)]:
    for qn, qf in [("E2M1", e2m1), ("E2M1-4/6", q46)]:
        set_all("quant", sc, qf); res[f"{name} | {qn}"] = bpb()
        print(f"[{args.tag}] {name:48s} {qn:8s} BPB {res[f'{name} | {qn}']:.4f} ({100*(res[f'{name} | {qn}']/res['BF16']-1):+.2f}%)", flush=True)

# ---- actually fold the global variant and check function preservation, then quantize the folded model with no scales
with torch.no_grad():
    sg = s_glob.to(torch.bfloat16)
    hfm.model.embed_tokens.weight.div_(sg)
    for blk, d in BLOCKS:
        blk.post_attention_layernorm.weight.div_(sg); blk.post_feedforward_layernorm.weight.div_(sg)
        for n in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"): d[n].lin.weight.mul_(sg)
        s_o, s_dn = local[id(d["o_proj"])], local[id(d["down_proj"])]
        s_kv = s_o.view(Hkv, REP, hd)[:, 0, :].reshape(-1)
        d["v_proj"].lin.weight.div_(s_kv[:, None].to(torch.bfloat16)); d["o_proj"].lin.weight.mul_(s_o.to(torch.bfloat16))
        d["up_proj"].lin.weight.div_(s_dn[:, None].to(torch.bfloat16)); d["down_proj"].lin.weight.mul_(s_dn.to(torch.bfloat16))
    fn = hfm.model.norm; orig_fwd = fn.forward
    fn.forward = lambda h, _f=orig_fwd, _s=s_glob: _f(h * _s.to(h.dtype))      # the one non-foldable elementwise op
set_all("bf16"); res["fold: BF16 after folding"] = bpb()
for qn, qf in [("E2M1", e2m1), ("E2M1-4/6", q46)]:
    set_all("quant", None, qf); res[f"fold: plain {qn} on folded weights"] = bpb()
for k in [k for k in res if k.startswith("fold")]: print(f"[{args.tag}] {k:48s} BPB {res[k]:.4f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
