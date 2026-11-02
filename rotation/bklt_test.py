"""Can a shared orthogonal transform pile block energy onto ONE coordinate (the exactly-represented absmax)?

Block-diagonal KLT: for every 16-index group on the reduction axis, rotate into the eigenbasis of that group's
covariance (largest eigenvalue first) computed on operand A of a calibration batch. Shared by all rows and applied
to both operands, so the GEMM is unchanged. Compared with I, H16, PERM (sort by RMS) and PERM+BKLT.
FPROP (reduce K) and DGRAD (reduce O) are fitted on a calibration batch and evaluated on another batch; WGRAD reduces
over tokens, so its BKLT is fitted online on the evaluated batch (groups of 16 consecutive tokens).

Metrics per operand A: absmax energy share (max^2 / sum^2 per block), mean t^2; GEMM-output SQNR for E2M1, E2M1-4/6,
uniform.
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


QS = {"E2M1": lambda A: quantize(A, E2M1, "rtn", "fp32")[0], "E2M1-4/6": q46, "UNIF": lambda A: quantize(A, UNIF, "rtn", "fp32")[0]}


def bklt_mats(Aref):
    R = Aref.shape[1]
    G = Aref.view(-1, R // BLK, BLK).transpose(0, 1)            # [groups, rows, 16]
    C = G.transpose(1, 2) @ G                                   # [groups, 16, 16]
    _, V = torch.linalg.eigh(C.double())
    return V.flip(-1).float()                                   # columns: largest eigenvalue first


def apply_bd(M, Q):
    N, R = M.shape
    return (M.view(N, R // BLK, 1, BLK) @ Q.unsqueeze(0)).view(N, R)


def tfs(Aref):
    R = Aref.shape[1]
    sg = (torch.randint(0, 2, (R,), device=dev, generator=gen) * 2 - 1).float()
    order = Aref.pow(2).mean(0).argsort(descending=True)
    Qb = bklt_mats(Aref); Qp = bklt_mats(Aref[:, order])
    return {"I": lambda M: M,
            "H16": lambda M: ((M * sg).view(M.shape[0], -1, BLK) @ H).view(M.shape),
            "PERM": lambda M: M[:, order],
            "BKLT": lambda M: apply_bd(M, Qb),
            "PERM+BKLT": lambda M: apply_bd(M[:, order], Qp)}


fout = open(args.out, "w"); saved = {}; calib = {}; phase = {"p": "calib"}


def evaluate(lname, gemm, A, B, Aref, online):
    Y = A @ B.T; nY = (Y * Y).sum()
    for tn, f in tfs(Aref).items():
        At, Bt = f(A), f(B)
        Ab = At.view(-1, BLK); e = Ab.pow(2); amax2 = e.amax(-1)
        rec = dict(model=args.tag, layer=lname, gemm=gemm, tf=tn, online=online and tn in ("PERM", "BKLT", "PERM+BKLT"),
                   absmax_energy=(amax2.sum() / e.sum()).item(), mean_t2=(e / amax2.clamp(min=1e-30)[:, None]).mean().item())
        for qn, qf in QS.items():
            Yh = qf(At) @ qf(Bt).T
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
            evaluate(name, "fprop", X, W, Xc, False)
            evaluate(name, "dgrad", dY, W.T.contiguous(), dYc, False)
            dYt = dY.T.contiguous(); evaluate(name, "wgrad", dYt, X.T.contiguous(), dYt, True)
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
