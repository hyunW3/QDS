"""If not shrinkage bias, then what? Grid-placement vs post-normalization distribution (llama 300M).

For each block Linear and operand (fwd_x, fwd_w, dgrad_dy, wgrad_dyT), with and without 16x16 RHT, collect a
subsample of block-normalized magnitudes t = |x| / (absmax/6) in [0, 6], each with weight s^2 (block scale^2),
so that weighted sums reproduce absolute-domain MSE. Also per-block crest factor absmax / RMS.

Analysis:
  (a) per-bin MSE contribution of E2M1 vs uniform (where the error comes from)
  (b) crest factor distribution before/after RHT
  (c) weighted Lloyd-Max 8-level magnitude grid with 0 and 6 fixed (the absmax maps exactly to 6), fitted per
      operand x transform; compare its SQNR and level placement with E2M1 and uniform.
"""
import os
import json, sys, math
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D))
from fp4lib import hadamard, rht, BLK
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs

dev = "cuda"; torch.manual_seed(0)
PER = 4096  # blocks sampled per (linear, operand, transform)
ck = Path(os.environ.get("LINGUA_CKPTS", "checkpoints")) / "llama300M"
cfg = OmegaConf.load(ck / "params.json")
model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
model.load_state_dict(torch.load(ck / "consolidated.pth", weights_only=True, map_location="cpu", mmap=True)["model"])
model = model.to(dev).train()
tok = build_tokenizer("tiktoken", str(ROOT / "tokenizer/llama3/tokenizer.model"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
x = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
H = hadamard(BLK, dev); saved = {}
T = {}; W8 = {}; CREST = {}


def collect(key, A):
    N, K = A.shape
    Ab = A.float().view(-1, BLK)
    idx = torch.randperm(Ab.shape[0], device=dev)[:PER]
    Ab = Ab[idx]
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = amax / 6
    t = (Ab.abs() / s)
    w = (s * s).expand_as(t)
    rms = Ab.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-30)
    T.setdefault(key, []).append(t.flatten().cpu().numpy())
    W8.setdefault(key, []).append((w / w.mean()).flatten().cpu().numpy())   # per-tensor normalized weights
    CREST.setdefault(key, []).append((amax / rms).flatten().cpu().numpy())


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).to(torch.bfloat16).float()
        Wt = mod.weight.detach().to(torch.bfloat16).float()
        with torch.no_grad():
            for opn, A in {"fwd_x": X, "fwd_w": Wt, "dgrad_dy": dY, "wgrad_dyT": dY.T.contiguous()}.items():
                collect((opn, "I"), A)
                sg = (torch.randint(0, 2, (A.shape[1],), device=dev) * 2 - 1).float()
                collect((opn, "H"), rht(A, sg, H))
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach().to(torch.bfloat16)))
        m.register_full_backward_hook(bwd(n))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x[:, :-1], target=x[:, 1:], attn_impl="sdpa")
loss.backward()

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.]); UNIF = np.arange(8) * 6 / 7


def rtn(t, g):
    return g[np.searchsorted((g[1:] + g[:-1]) / 2, t)]


def nmse(t, w, g):
    q = rtn(t, g); return float((w * (q - t) ** 2).sum() / (w * t * t).sum())


def lloyd(t, w, iters=200):
    g = UNIF.copy()
    for _ in range(iters):
        th = (g[1:] + g[:-1]) / 2
        cell = np.searchsorted(th, t)
        for i in range(1, 7):                     # 0 and 6 fixed
            m = cell == i
            if m.any(): g[i] = (w[m] * t[m]).sum() / w[m].sum()
        g = np.sort(g)
    return g


out = {}
for key in T:
    t = np.concatenate(T[key]); w = np.concatenate(W8[key]); c = np.concatenate(CREST[key])
    g_opt = lloyd(t, w)
    # per-bin MSE contribution (share of total squared error), for E2M1 and UNIF
    contrib = {}
    for gn, g in [("E2M1", E2M1), ("UNIF", UNIF)]:
        q = rtn(t, g); e = w * (q - t) ** 2; lv = np.searchsorted(g, q)
        contrib[gn] = [float(e[lv == i].sum() / e.sum()) for i in range(8)]
    hist, edges = np.histogram(t, bins=60, range=(0, 6), weights=w * t * t)   # energy-weighted density
    out["|".join(key)] = dict(
        sqnr={gn: -10 * math.log10(nmse(t, w, g)) for gn, g in [("E2M1", E2M1), ("UNIF", UNIF), ("LloydMax", g_opt)]},
        lloyd=g_opt.tolist(), contrib=contrib,
        crest=dict(mean=float(c.mean()), p10=float(np.percentile(c, 10)), p50=float(np.median(c)), p90=float(np.percentile(c, 90))),
        crest_hist=np.histogram(np.clip(c, 1, 4), bins=60, range=(1, 4))[0].tolist(),
        energy_hist=(hist / hist.sum()).tolist(),
        count_hist=(np.histogram(t, bins=60, range=(0, 6))[0] / len(t)).tolist())
json.dump(out, open(D / "stage0/cause_probe.json", "w"))
for k, v in out.items():
    print(f"{k:16s} SQNR E2M1 {v['sqnr']['E2M1']:.2f} UNIF {v['sqnr']['UNIF']:.2f} LM {v['sqnr']['LloydMax']:.2f} | "
          f"crest p50 {v['crest']['p50']:.2f} | LM grid {np.round(v['lloyd'], 2).tolist()}")
