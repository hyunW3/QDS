"""Direction 2: does RHT's *randomization* matter, separately from its Gaussianization? (open-source model, no training)

Proxy for training: accumulate the WGRAD dW = sum_b dY_b^T X_b over many batches (as over steps / micro-batches)
with each batch quantized independently, and track the relative error of the accumulated gradient
    err(K) = ||sum_{b<=K} (dW_hat_b - dW_b)|| / ||sum_{b<=K} dW_b||.
If per-batch quantization errors are independent and zero-mean, err(K) falls like 1/sqrt(K); a systematic (correlated)
error component makes it plateau. Also reported: mean cosine between each batch's error and the first batch's error, and the cosine
between the accumulated error and the true accumulated gradient (systematic shrinkage/expansion along the gradient).

Variants (applied along the token axis of WGRAD, shared by dY^T and X^T):
  I            no transform (deterministic RTN)
  H16-fixed    16x16 RHT with the same random signs for every batch
  H16-resample RHT with fresh random signs per batch
  RPERM        fresh random token permutation per batch (re-grouping into 16-blocks, no mixing)
  SR-dY        no transform, stochastic rounding on dY (unbiased reference)
Grids: E2M1, E2M1-4/6, UNIF (E1M2).
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize, hadamard
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nbatch", type=int, default=32); ap.add_argument("--bsz", type=int, default=2); ap.add_argument("--seqlen", type=int, default=1024)
ap.add_argument("--layers", default="0,8,16")
args = ap.parse_args()
dev = "cuda"; gen = torch.Generator(device=dev); gen.manual_seed(1)
H = hadamard(BLK, dev)
model, tok = load_hf(args.hf, dev)
keep = {int(x) for x in args.layers.split(",")}
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
need = args.nbatch * args.bsz * (args.seqlen + 1)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= need: break
data = torch.tensor(ids[:need], device=dev).view(args.nbatch, args.bsz, args.seqlen + 1)


def q46(A, rnd="rtn"):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    outs = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        outs.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(outs[1][1] < outs[0][1], outs[1][0], outs[0][0]).view(N, R)


GRIDS = {"E2M1": lambda A, r: quantize(A, E2M1, r, "fp32", gen=gen)[0], "E2M1-4/6": lambda A, r: q46(A),
         "UNIF": lambda A, r: quantize(A, UNIF, r, "fp32", gen=gen)[0]}
FIXED_SG = None
VARIANTS = ["I", "H16-fixed", "H16-resample", "RPERM", "SR-dY"]


def transform(name, Ntok):
    global FIXED_SG
    if name in ("I", "SR-dY"):
        return lambda M: M
    if name == "RPERM":
        p = torch.randperm(Ntok, device=dev, generator=gen)
        return lambda M: M[:, p]
    if name == "H16-fixed":
        if FIXED_SG is None:
            FIXED_SG = (torch.randint(0, 2, (Ntok,), device=dev, generator=gen) * 2 - 1).float()
        sg = FIXED_SG
    else:
        sg = (torch.randint(0, 2, (Ntok,), device=dev, generator=gen) * 2 - 1).float()
    return lambda M: ((M * sg).view(M.shape[0], -1, BLK) @ H).view(M.shape)


acc = {}     # (layer, variant, grid) -> dict(true, err, errs list for cosines)
saved = {}
records = []


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()          # [N, K]
        dY = go[0].detach().reshape(-1, mod.out_features).float()         # [N, O]
        with torch.no_grad():
            A, B = dY.T.contiguous(), X.T.contiguous()                    # reduce over tokens N
            G = A @ B.T
            for v in VARIANTS:
                f = transform(v, A.shape[1])
                At, Bt = f(A), f(B)
                for gname, qf in GRIDS.items():
                    if v == "SR-dY" and gname == "E2M1-4/6":
                        continue
                    Ah = qf(At, "sr" if v == "SR-dY" else "rtn"); Bh = qf(Bt, "rtn")
                    E = Ah @ Bh.T - G
                    k = (name, v, gname)
                    if k not in acc:
                        acc[k] = dict(true=torch.zeros_like(G), err=torch.zeros_like(G), prev=[], cos=[])
                    a = acc[k]; a["true"] += G; a["err"] += E
                    e = (E.flatten() / E.norm()).half()
                    if a["prev"]:
                        a["cos"].append(float((e.float() @ a["prev"][0].float())))
                    else:
                        a["prev"].append(e)                                # error of the first batch only
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers.") and int(n.split(".")[1]) in keep:
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))

curve = {}
for b in range(args.nbatch):
    loss = model(data[b, :, :-1], target=data[b, :, 1:]); loss.backward()
    for k, a in acc.items():
        curve.setdefault(k, []).append((a["err"].norm() / a["true"].norm()).item())
out = []
for k, a in acc.items():
    t, e = a["true"].flatten(), a["err"].flatten()
    out.append(dict(model=args.tag, layer=k[0], variant=k[1], grid=k[2], curve=curve[k],
                    mean_pair_cos=sum(a["cos"]) / max(1, len(a["cos"])),
                    cos_err_true=float(e @ t / (e.norm() * t.norm()))))
json.dump(out, open(args.out, "w"))
print(f"[{args.tag}] done, {len(out)} series -> {args.out}", flush=True)
