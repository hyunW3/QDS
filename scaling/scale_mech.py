"""Why a diagonal scale after the placement analysis? Mechanism probe on every decoder Linear's FPROP (Y = X W^T).

Decomposition (dfp_test.py): pred = sum_k (ea_k b_k + a_k eb_k) = sum_k w_k (rA_k + rB_k),  w_k = a_k b_k,
rA_k = ea_k / a_k (per-channel relative error of X), rB_k likewise. A diagonal scale X/s, W*s leaves every w_k, hence
S = sum_k w_k (the joint-concentration / Delta_FP term), exactly unchanged; it can act only through the per-channel
relative errors, i.e. through level placement. A rotation changes w (S) and placement together.
Per layer and per (T, rule), T in {I, H16 (random-sign block Hadamard), SM (s = sqrt(max|X_k|/max|W_k|)),
QDS (alpha search of s = max|X_k|^a / max|W_k|^(1-a) on the E2M1@6 GEMM error, as in placement_test.py)},
rule in {@6, @4/6}, this logs
  gemm                 measured ||Y^ - Y||^2 / ||Y||^2
  S, nu, c             the three factors of pred (all normalized by ||Y||^2)
  side_A, side_B       shares of pred from the X-side (sum ea_k b_k) and W-side (sum a_k eb_k) terms
  A_out, A_mate, A_rest  split of the X-side term over channels: top-1% outlier channels (by max|X_k| of the raw X),
                       their block-mates (other channels of the same 16-channel block), and channels in other blocks
  A_dz, A_top          share of the X-side term from elements quantized in the dead zone (6|x|/blockmax < 0.25, rounds
                       to 0) and from elements in the coarsest interval [4, 6)
For the channel-indexed splits a rotated operand is attributed by its (mixed) coordinate index, so they are only
reported for the non-mixing transforms (I, SM, QDS).
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import hadamard
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda:0"; gen = torch.Generator(device=dev); gen.manual_seed(1)
G = 16
H = hadamard(G, dev)
SG = (torch.randint(0, 2, (1 << 17,), device=dev, generator=gen) * 2 - 1).float()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2


def q_at(Ab, m):
    s = (Ab.abs().amax(-1, keepdim=True) / m).clamp(min=1e-30)
    return torch.sign(Ab) * E2M1[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)] * s


def quant(M, rule):
    """-> dequantized M and t = 6|x| / blockmax of every element."""
    N, R = M.shape; Mb = M.view(N, R // G, G)
    Q = q_at(Mb, 6.)
    if rule == "@4/6":
        Q4 = q_at(Mb, 4.)
        pick = (Q4 - Mb).pow(2).sum(-1, keepdim=True) < (Q - Mb).pow(2).sum(-1, keepdim=True)
        Q = torch.where(pick, Q4, Q)
    t = 6 * Mb.abs() / Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return Q.view(N, R), t.view(N, R)


def qds(A, B):
    ra, rb = A.abs().amax(0).clamp(min=1e-12), B.abs().amax(0).clamp(min=1e-12)
    Y = A @ B.T; best = None
    for i in range(11):
        a = i / 10; s = ra.pow(a) / rb.pow(1 - a)
        e = ((quant(A / s, "@6")[0] @ quant(B * s, "@6")[0].T - Y) ** 2).sum().item()
        if best is None or e < best[0]: best = (e, s)
    return best[1]


def evaluate(lname, A, B):
    R = A.shape[1]
    if R % G or A.shape[0] == 0: return
    Y = A @ B.T; nY = (Y * Y).sum().item()
    if nY == 0: return
    sg = SG[:R]
    sm = (A.abs().amax(0).clamp(min=1e-12) / B.abs().amax(0).clamp(min=1e-12)).sqrt()
    sq = qds(A, B)
    T = {"I": (A, B),
         "H16": (((A * sg).view(A.shape[0], -1, G) @ H).view(A.shape), ((B * sg).view(B.shape[0], -1, G) @ H).view(B.shape)),
         "SM": (A / sm, B * sm), "QDS": (A / sq, B * sq)}
    n_out = max(1, math.ceil(0.01 * R))
    out_ch = torch.zeros(R, dtype=torch.bool, device=dev); out_ch[A.abs().amax(0).argsort(descending=True)[:n_out]] = True
    blk = out_ch.view(-1, G).any(-1, keepdim=True).expand(-1, G).reshape(-1)
    mate = blk & ~out_ch; rest = ~blk
    rec = dict(model=args.tag, layer=lname, K=R)
    for tn, (At, Bt) in T.items():
        for rule in ("@6", "@4/6"):
            Aq, tA = quant(At, rule); Bq, _ = quant(Bt, rule)
            EA, EB = Aq - At, Bq - Bt
            a, b = At.pow(2).sum(0), Bt.pow(2).sum(0); ea, eb = EA.pow(2).sum(0), EB.pow(2).sum(0)
            sA, sB = (ea * b).sum().item(), (a * eb).sum().item(); pred = sA + sB
            S = (a * b).sum().item(); nu = ea.sum().item() / a.sum().item() + eb.sum().item() / b.sum().item()
            r = dict(gemm=((Aq @ Bq.T - Y) ** 2).sum().item() / nY, S=S / nY, nu=nu, c=pred / (S * nu),
                     side_A=sA / pred, side_B=sB / pred)
            ew = EA.pow(2) * b[None]                   # element contributions to the X-side term
            r["A_dz"] = (ew[tA < 0.25].sum() / sA).item(); r["A_top"] = (ew[(tA >= 4) & (tA < 6 - 1e-6)].sum() / sA).item()
            if tn != "H16":
                r["A_out"] = ((ea * b)[out_ch].sum() / sA).item(); r["A_mate"] = ((ea * b)[mate].sum() / sA).item()
                r["A_rest"] = ((ea * b)[rest].sum() / sA).item()
            rec[f"{tn}{rule}"] = r
    fout.write(json.dumps(rec) + "\n")


model, tok = load_hf(args.hf, dev)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batch = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
fout = open(args.out, "w")
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        m.register_forward_hook(lambda mod, i, o, n=n: evaluate(n, i[0].detach().reshape(-1, mod.in_features).float(),
                                                                 mod.weight.detach().float()))
with torch.no_grad():
    loss = model(batch[:, :-1], target=batch[:, 1:])
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
fout.close()
