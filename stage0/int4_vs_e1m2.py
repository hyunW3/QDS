"""Check: E1M2 vs INT4 on real tensors (llama 300M).

E1M2   magnitudes {0,.25,...,1.75}, scale = absmax/1.75
INT4s  symmetric  {-7..7},          scale = absmax/7      (expected identical to E1M2: grid/7 == grid/1.75, ratio 4 = 2^2)
INT4a  full range {-8..7},          scale = absmax/8      (positive side clips at 7 -> asymmetric)
Also E2M1 for reference. Transforms I/H, block scale fp32/e4m3, RTN. Tensor SQNR/alpha and FPROP GEMM output.
"""
import os
import json, sys, math
from pathlib import Path
import torch
from omegaconf import OmegaConf

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, quantize, hadamard, rht, BLK, _round_scale
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs

dev = "cuda"; torch.manual_seed(0)
INT4S = torch.arange(8, dtype=torch.float32)          # 0..7


def q_int4_asym(A, scale_fmt):
    N, K = A.shape
    Ab = A.float().view(N, K // BLK, BLK)
    s = _round_scale(Ab.abs().amax(-1, keepdim=True) / 8.0, scale_fmt)
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round(Ab / s).clamp(-8, 7)               # round-half-even; +absmax (=+8) clips to 7
    return (q * s).view(N, K)


def Q(name, A, sf):
    if name == "E2M1": return quantize(A, E2M1, "rtn", sf)[0]
    if name == "E1M2": return quantize(A, UNIF, "rtn", sf)[0]
    if name == "INT4sym": return quantize(A, INT4S, "rtn", sf)[0]
    if name == "INT4asym(-8..7)": return q_int4_asym(A, sf)


NAMES = ["E2M1", "E1M2", "INT4sym", "INT4asym(-8..7)"]
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
H = hadamard(BLK, dev); saved = {}; recs = []


def ops(X, W, dY):
    return {"fwd_x": X, "fwd_w": W, "dgrad_dy": dY, "wgrad_dyT": dY.T.contiguous()}


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).to(torch.bfloat16).float()
        W = mod.weight.detach().to(torch.bfloat16).float()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for tr in ["I", "H"]:
                for sf in ["fp32", "e4m3"]:
                    sx = (torch.randint(0, 2, (X.shape[1],), device=dev) * 2 - 1).float()
                    Xt, Wt = (rht(X, sx, H), rht(W, sx, H)) if tr == "H" else (X, W)
                    Y = Xt @ Wt.T
                    outs = {}
                    OPS = {}
                    for opn, A in ops(X, W, dY).items():   # same RHT signs for every grid
                        if tr == "H":
                            s = (torch.randint(0, 2, (A.shape[1],), device=dev) * 2 - 1).float()
                            A = rht(A, s, H)
                        OPS[opn] = A
                    for n in NAMES:
                        r = dict(layer=name, tr=tr, scale=sf, grid=n)
                        for opn, A in OPS.items():
                            Ah = Q(n, A, sf); nA = (A * A).sum()
                            r[opn] = dict(nmse=(((Ah - A) ** 2).sum() / nA).item(), alpha=((Ah * A).sum() / nA).item())
                        Yh = Q(n, Xt, sf) @ Q(n, Wt, sf).T; nY = (Y * Y).sum()
                        r["fprop_Y"] = dict(nmse=(((Yh - Y) ** 2).sum() / nY).item(), alpha=((Yh * Y).sum() / nY).item())
                        outs[n] = Yh
                        recs.append(r)
                    recs[-2]["maxdiff_vs_E1M2"] = (outs["INT4sym"] - outs["E1M2"]).abs().max().item()
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach().to(torch.bfloat16)))
        m.register_full_backward_hook(bwd(n))
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x[:, :-1], target=x[:, 1:], attn_impl="sdpa")
loss.backward()
json.dump(recs, open(D / "stage0/int4_vs_e1m2.json", "w"))

db = lambda v: -10 * math.log10(v)
from statistics import mean
print("| T | scale | grid | fwd_x SQNR / α | fwd_w | dgrad_dy | wgrad_dyT | FPROP Y SQNR / α |")
print("|---|---|---|---|---|---|---|---|")
for tr in ["I", "H"]:
    for sf in ["fp32", "e4m3"]:
        for n in NAMES:
            rs = [r for r in recs if r["tr"] == tr and r["scale"] == sf and r["grid"] == n]
            cell = lambda k: f"{mean(db(r[k]['nmse']) for r in rs):.2f} / {mean(r[k]['alpha'] for r in rs):.4f}"
            print(f"| {tr} | {sf} | {n} | " + " | ".join(cell(k) for k in ["fwd_x", "fwd_w", "dgrad_dy", "wgrad_dyT", "fprop_Y"]) + " |")
md = [r["maxdiff_vs_E1M2"] for r in recs if "maxdiff_vs_E1M2" in r]
print(f"\nmax |Y(INT4sym) - Y(E1M2)| over all linears/configs: {max(md):.3e}")
