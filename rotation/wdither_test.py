"""Direction 2 (weights): randomization as dithering of the (slowly varying) weight quantization error.

W is nearly constant across steps, so with a fixed transform its quantization error repeats every step (systematic).
With a transform resampled per step, the error changes; SGD effectively averages it. Measure, for every decoder Linear,
the relative error of the average of S independently-transformed quantized weights, back-transformed:
    err(S) = || mean_s T_s^{-1}(Q(T_s(W))) - W || / ||W||       (blocks along K, the FPROP reduction axis)
Variants: I (no transform; err constant), H16-resample (fresh signs), RPERM (fresh random permutation of K).
Grids: E2M1, E2M1-4/6, UNIF.
"""
import argparse, json, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"; gen = torch.Generator(device=dev); gen.manual_seed(0)
H = hadamard(BLK, dev)
m = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16)


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    o = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        o.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(o[1][1] < o[0][1], o[1][0], o[0][0]).view(N, R)


GR = {"E2M1": lambda A: quantize(A, E2M1, "rtn", "fp32")[0], "E2M1-4/6": q46, "UNIF": lambda A: quantize(A, UNIF, "rtn", "fp32")[0]}
S_LIST = [1, 2, 4, 8, 16, 32]
res = []
names = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and ".layers." in n]
for n, mod in names[::3]:                               # every 3rd Linear to keep it quick
    W = mod.weight.detach().float().to(dev); nW = W.norm()
    for gname, qf in GR.items():
        for v in ["I", "H16-resample", "RPERM"]:
            acc = torch.zeros_like(W); curve = []
            for s in range(1, max(S_LIST) + 1):
                if v == "I":
                    Wh = qf(W)
                elif v == "RPERM":
                    p = torch.randperm(W.shape[1], device=dev, generator=gen); inv = torch.argsort(p)
                    Wh = qf(W[:, p].contiguous())[:, inv]
                else:
                    sg = (torch.randint(0, 2, (W.shape[1],), device=dev, generator=gen) * 2 - 1).float()
                    Wt = ((W * sg).view(W.shape[0], -1, BLK) @ H).view(W.shape)
                    Wh = (qf(Wt).view(W.shape[0], -1, BLK) @ H.T).view(W.shape) * sg
                acc += Wh
                if s in S_LIST:
                    curve.append(((acc / s - W).norm() / nW).item())
            res.append(dict(model=args.tag, layer=n, grid=gname, variant=v, S=S_LIST, curve=curve))
json.dump(res, open(args.out, "w"))
print(f"[{args.tag}] {len(res)} series -> {args.out}")
