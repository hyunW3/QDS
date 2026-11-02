"""Does per-block SQNR fall with the crest factor (CF = max/RMS of a 16-value block) on real 8B tensors, as the uniform-grid law
SQNR_dB ~ 6.02 b + 4.77 - 20 log10 CF (arXiv 2608.25188, Fig. 2 right) predicts -- and does it for the E2M1 (NVFP4) grid?
CPU only (the GPUs of this node are busy). One forward pass captures the Linear-group inputs of a few layers; weights are read
directly. Every 1x16 block gets its CF and its SQNR under each quantizer; blocks are binned by CF.
Quantizers (absmax per block): INT4 (symmetric, 15 levels +-7, fp32 scale), E2M1 (fp32 scale), NVFP4 (E2M1 + E4M3 block scale and
FP32 tensor scale), NVFP4-4/6; each also after a 16x16 block Hadamard (rotation changes the block, so its own CF is used).
"""
import argparse, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cal", required=True); ap.add_argument("--layers", default="0,8,16,24,-1"); ap.add_argument("--nwin", type=int, default=2)
args = ap.parse_args()
torch.set_num_threads(16)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).eval()
NL = len(model.model.layers); LAY = [int(x) % NL for x in args.layers.split(",")]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); TH = (E2M1[1:] + E2M1[:-1]) / 2
_H = torch.ones(1, 1)
while _H.shape[0] < 16: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
H16 = _H / 4.0


def q_int4(B):                       # B [n,16]
    s = B.abs().amax(-1, keepdim=True) / 7.0; s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(B / s).clamp(-7, 7) * s


def q_e2m1(B, scale="fp32", four_six=False, gs=None):
    amax = B.abs().amax(-1, keepdim=True); best = None
    for d in ([6.0, 4.0] if four_six else [6.0]):
        s = amax / d
        if scale == "e4m3": s = (s / gs).clamp(max=448).to(torch.float8_e4m3fn).float() * gs
        s = torch.where(s == 0, torch.ones_like(s), s)
        Q = torch.sign(B) * E2M1[torch.bucketize((B.abs() / s).clamp(max=6), TH)] * s; e = (Q - B).pow(2).sum(-1, keepdim=True)
        best = (Q, e) if best is None else (torch.where(e < best[1], Q, best[0]), torch.minimum(e, best[1]))
    return best[0]


def analyse(A):
    """A [N,K] fp32 -> dict quantizer -> (cf [nb], sqnr_db [nb])."""
    B = A.reshape(-1, 16).double().float()
    keep = B.pow(2).sum(-1) > 0; B = B[keep]
    out = {}
    for rot in (False, True):
        Bt = B @ H16 if rot else B
        cf = Bt.abs().amax(-1) / Bt.pow(2).mean(-1).sqrt()
        gs4 = (Bt.abs().amax() / 4.0 / 448.0); gs6 = (Bt.abs().amax() / 6.0 / 448.0)
        for name, Q in [("INT4", q_int4(Bt)), ("E2M1", q_e2m1(Bt)), ("NVFP4", q_e2m1(Bt, "e4m3", False, gs6)), ("NVFP4-4/6", q_e2m1(Bt, "e4m3", True, gs4))]:
            sq = 10 * torch.log10(Bt.pow(2).sum(-1) / (Q - Bt).pow(2).sum(-1).clamp(min=1e-30))
            out[name + ("+H16" if rot else "")] = (cf, sq)
    return out


ids = []
for line in open(args.cal):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= (600 + args.nwin) * 1024: break
win = torch.tensor(ids[600 * 1024:(600 + args.nwin) * 1024]).view(args.nwin, 1024)
cap = {}
for i in LAY:
    blk = model.model.layers[i]
    for g, m in [("attn_in", blk.self_attn.q_proj), ("o", blk.self_attn.o_proj), ("mlp_in", blk.mlp.gate_proj), ("down", blk.mlp.down_proj)]:
        m.register_forward_pre_hook(lambda mod, a, key=(i, g): cap.setdefault(key, []).append(a[0][0].float()))
with torch.no_grad():
    for s in win: model(input_ids=s[None], use_cache=False)
BINS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.01]
res = dict(model=args.tag, layers=LAY, bins=BINS, curves={}, fits={})
acc = {}
for (i, g), xs in cap.items():
    blk = model.model.layers[i]
    W = {"attn_in": blk.self_attn.q_proj, "o": blk.self_attn.o_proj, "mlp_in": blk.mlp.gate_proj, "down": blk.mlp.down_proj}[g].weight.float()
    for op, A in [("X", torch.cat(xs)), ("W", W)]:
        for q, (cf, sq) in analyse(A).items():
            acc.setdefault((op, q), []).append((cf, sq))
    print(f"[cf {args.tag}] layer {i} {g}", flush=True)
for (op, q), parts in acc.items():
    cf = torch.cat([p[0] for p in parts]); sq = torch.cat([p[1] for p in parts])
    curve = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (cf >= lo) & (cf < hi)
        curve.append(dict(lo=lo, hi=hi, n=int(m.sum()), frac=m.float().mean().item(), sqnr=sq[m].mean().item() if m.any() else None))
    x = 20 * torch.log10(cf); A_ = torch.stack([x, torch.ones_like(x)], 1)
    slope, icpt = torch.linalg.lstsq(A_, sq[:, None]).solution[:, 0].tolist()
    res["curves"][f"{op}|{q}"] = curve
    res["fits"][f"{op}|{q}"] = dict(slope_per_20log10CF=slope, intercept=icpt, mean_sqnr=sq.mean().item(), mean_cf=cf.mean().item(),
                                    corr=torch.corrcoef(torch.stack([x, sq]))[0, 1].item(), n=int(cf.numel()))
json.dump(res, open(args.out, "w"), indent=1)
print("CF_DONE", flush=True)
