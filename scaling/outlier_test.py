"""Outlier handling without a global RHT (open-source models).

Reduction-axis indices ("channels") are scored by rms(A_col) * rms(B_col) (contribution to the GEMM). Variants:
  I                  no transform
  H16                RHT on everything (reference; UNIF+H16 = UFP4)
  PERM               sort by score (outliers grouped into the first blocks)
  PERM+oRHT(k)       PERM, then H16 only on the outlier blocks (top k% channels, rounded up to whole 16-blocks)
  PERM+oFP8(k)       PERM, outlier channels quantized to FP8 E4M3 (1x16 absmax scale), rest FP4   (Atom/QUIK-like)
  PERM+oBF16(k)      PERM, outlier channels kept unquantized (LLM.int8()-like upper bound)
  SMOOTH             SmoothQuant-style diagonal scaling, s = max|A_c|^0.5 / max|B_c|^0.5 (A/s, B*s)
The remaining (non-outlier) channels are quantized with E2M1, E2M1-4/6 or the uniform grid.
Channel axes (FPROP: K, DGRAD: O) are fitted on a calibration batch and evaluated on another batch; WGRAD reduces over
tokens, so its scores are computed online on the evaluated batch.
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
args = ap.parse_args()
dev = "cuda"; gen = torch.Generator(device=dev); gen.manual_seed(1)
H = hadamard(BLK, dev)
model, tok = load_hf(args.hf, dev)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 8 * 1025: break
batches = torch.tensor(ids[:8 * 1025], device=dev).view(2, 4, 1025)


def q46(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30); g = E2M1.to(dev); th = (g[1:] + g[:-1]) / 2
    outs = []
    for gm in (6.0, 4.0):
        s = amax / gm; q = g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)]; Ah = torch.sign(Ab) * q * s
        outs.append((Ah, (Ah - Ab).pow(2).sum(-1, keepdim=True)))
    return torch.where(outs[1][1] < outs[0][1], outs[1][0], outs[0][0]).view(N, R)


def fp8_block(A):
    N, R = A.shape; Ab = A.view(N, R // BLK, BLK)
    s = (Ab.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return ((Ab / s).to(torch.float8_e4m3fn).float() * s).view(N, R)


QS = {"E2M1": lambda A: quantize(A, E2M1, "rtn", "fp32")[0], "E2M1-4/6": q46, "UNIF": lambda A: quantize(A, UNIF, "rtn", "fp32")[0]}


def h16(M):
    sg = SG[: M.shape[1]]
    return ((M * sg).view(M.shape[0], -1, BLK) @ H).view(M.shape)


SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=gen) * 2 - 1).float()


def variants(Aref, Bref):
    """returns {name: (transform f, n_outlier_channels or 0, mode)}; outliers are the first n channels after f."""
    R = Aref.shape[1]
    score = Aref.pow(2).mean(0).sqrt() * Bref.pow(2).mean(0).sqrt()
    order = score.argsort(descending=True)
    V = {"I": (lambda M: M, 0, None), "H16": (h16, 0, None), "PERM": (lambda M: M[:, order], 0, None)}
    for k in (0.01, 0.03):
        n = min(R, max(BLK, int(math.ceil(k * R / BLK)) * BLK))
        def orht(M, n=n):
            P = M[:, order].clone(); P[:, :n] = h16(P[:, :n]); return P
        V[f"PERM+oRHT({int(k*100)}%)"] = (orht, 0, None)
        V[f"PERM+oFP8({int(k*100)}%)"] = (lambda M: M[:, order], n, "fp8")
        V[f"PERM+oBF16({int(k*100)}%)"] = (lambda M: M[:, order], n, "bf16")
    s = (Aref.abs().amax(0).clamp(min=1e-12) ** 0.5) / (Bref.abs().amax(0).clamp(min=1e-12) ** 0.5)
    V["SMOOTH"] = ("smooth", s, None)
    return V


fout = open(args.out, "w"); saved = {}; calib = {}; phase = {"p": "calib"}


def evaluate(lname, gemm, A, B, Aref, Bref, online):
    Y = A @ B.T; nY = (Y * Y).sum()
    for vn, (f, n, mode) in variants(Aref, Bref).items():
        if f == "smooth":
            At, Bt = A / n, B * n; n = 0
        else:
            At, Bt = f(A), f(B)
        rec = dict(model=args.tag, layer=lname, gemm=gemm, v=vn, online=online)
        for qn, qf in QS.items():
            if n:
                hi = (lambda M: M) if mode == "bf16" else fp8_block
                Ah = torch.cat([hi(At[:, :n]), qf(At[:, n:].contiguous())], 1)
                Bh = torch.cat([hi(Bt[:, :n]), qf(Bt[:, n:].contiguous())], 1)
            else:
                Ah, Bh = qf(At), qf(Bt)
            Yh = Ah @ Bh.T
            rec[qn] = (((Yh - Y) ** 2).sum() / nY).item()
        fout.write(json.dumps(rec) + "\n")


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).float()
        W = mod.weight.detach().float()
        if phase["p"] == "calib":
            calib[name] = (X.cpu(), dY.cpu()); return
        Xc, dYc = (t.to(dev) for t in calib.pop(name))
        with torch.no_grad():
            evaluate(name, "fprop", X, W, Xc, W, False)
            Wt = W.T.contiguous(); evaluate(name, "dgrad", dY, Wt, dYc, Wt, False)
            dYt, Xt = dY.T.contiguous(), X.T.contiguous(); evaluate(name, "wgrad", dYt, Xt, dYt, Xt, True)
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
for ph, b in [("calib", batches[0]), ("eval", batches[1])]:
    phase["p"] = ph
    loss = model(b[:, :-1], target=b[:, 1:]); loss.backward()
    print(f"[{args.tag}] {ph} loss {loss.item():.4f}", flush=True)
fout.close()
