"""Does the KLT concentrate energy, and is it the kind of concentration a block-scaled FP4 quantizer can use?

Classical transform coding (allocation-flexible: more bits to high-variance coordinates) is optimized by the KLT, which
maximizes the energy in the leading coordinates. NVFP4 spends the same 4 bits on every element and one absmax scale per
16-element block per token, so what it can use is concentration *inside a block of one token* (the block maximum is
exact). This probe measures both kinds for FPROP (A = X, B = W; basis from X^T X) and DGRAD (A = dY, B = W^T; basis from
dY^T dY), basis fitted on a calibration batch, evaluated on a second batch:

  cross-coordinate (what the KLT maximizes, averaged over tokens):
    top1 / top10     share of A's energy in the top 1% / 10% coordinates (columns)
    cg_db            transform coding gain 10 log10(arith. mean / geom. mean of the coordinate variances)
  within-block (what the absmax block quantizer can use):
    amax_share       share of block energy held by the block maximum (exactly represented)
    top46            share of block energy with 6|x|/max in [4, 6) (E2M1's coarsest interval)
    blk_spread       share of a token's energy in its top 10% blocks (energy concentrated in few blocks)
  quantization: GEMM SQNR (dB) for E2M1, E2M1-4/6 and the uniform grid (1x16, FP32 block scale)
Transforms: I, H16 (random signs), KLT (global eigenbasis, eigenvalues descending), BKLT (eigenbasis per 16-channel group).
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
ap.add_argument("--layers", default="0,4,8,12,16,20,24"); ap.add_argument("--threads", type=int, default=16)
args = ap.parse_args()
torch.set_num_threads(args.threads)
dev = "cpu"; G = 16
H = hadamard(G, dev)
gen = torch.Generator(); gen.manual_seed(1)
SG = (torch.randint(0, 2, (1 << 15,), generator=gen) * 2 - 1).float()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); TH = (E2M1[1:] + E2M1[:-1]) / 2
keep = {int(x) for x in args.layers.split(",")}


def q_e2m1(M, four_six=False):
    N, R = M.shape; Mb = M.view(N, R // G, G); amax = Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    def q(m):
        s = amax / m; return torch.sign(Mb) * E2M1[torch.bucketize((Mb.abs() / s).clamp(max=6), TH)] * s
    Q = q(6.0)
    if four_six:
        Q4 = q(4.0); Q = torch.where((Q4 - Mb).pow(2).sum(-1, keepdim=True) < (Q - Mb).pow(2).sum(-1, keepdim=True), Q4, Q)
    return Q.view(N, R)


def q_unif(M):
    N, R = M.shape; Mb = M.view(N, R // G, G); s = (Mb.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-30)
    return ((Mb / s).round().clamp(-7, 7) * s).view(N, R)


def within(M):
    N, R = M.shape; Mb = M.view(N, R // G, G); e = Mb.pow(2); be = e.sum(-1)
    amax = e.amax(-1); t = 6 * Mb.abs() / Mb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    top46 = e[(t >= 4) & (t < 6 - 1e-6)].sum() / e.sum()
    nb = R // G; k = max(1, math.ceil(0.1 * nb))
    spread = (be.topk(k, dim=-1).values.sum(-1) / be.sum(-1).clamp(min=1e-30)).mean()
    return dict(amax_share=(amax.sum() / e.sum()).item(), top46=top46.item(), blk_spread=spread.item())


def cross(M):
    v = M.pow(2).mean(0); R = v.numel(); vs = v.sort(descending=True).values
    cg = 10 * math.log10(v.mean().item() / math.exp(v.clamp(min=1e-30).log().mean().item()))
    return dict(top1=(vs[: max(1, R // 100)].sum() / vs.sum()).item(), top10=(vs[: max(1, R // 10)].sum() / vs.sum()).item(), cg_db=cg)


def bases(Acal):
    R = Acal.shape[1]
    C = (Acal.T @ Acal).double()
    w, U = torch.linalg.eigh(C); U = U[:, w.argsort(descending=True)].float()
    Ub = torch.zeros(R, R)
    for b in range(0, R, G):
        Cb = C[b:b + G, b:b + G]; wb, Vb = torch.linalg.eigh(Cb); Ub[b:b + G, b:b + G] = Vb[:, wb.argsort(descending=True)].float()
    return U, Ub


def tf(M, name, U, Ub):
    if name == "I": return M
    if name == "H16": return ((M * SG[: M.shape[1]]).view(M.shape[0], -1, G) @ H).view(M.shape)
    if name == "KLT": return M @ U
    if name == "BKLT": return M @ Ub


fout = open(args.out, "w"); saved = {}; calib = {}; phase = {"p": "calib"}


def evaluate(lname, gemm, A, B, Acal):
    if A.shape[1] % G: return
    U, Ub = bases(Acal)
    Y = A @ B.T; nY = (Y * Y).sum().item()
    for name in ("I", "H16", "KLT", "BKLT"):
        At, Bt = tf(A, name, U, Ub), tf(B, name, U, Ub)
        rec = dict(model=args.tag, layer=lname, gemm=gemm, T=name, K=A.shape[1], **cross(At), **within(At))
        for qn, qf in (("E2M1", q_e2m1), ("E2M1-4/6", lambda M: q_e2m1(M, True)), ("UNIF", q_unif)):
            rec[qn] = -10 * math.log10(((qf(At) @ qf(Bt).T - Y) ** 2).sum().item() / nY)
        fout.write(json.dumps(rec) + "\n"); fout.flush()


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).float()
        if phase["p"] == "calib":
            calib[name] = (X, dY); return
        Xc, dYc = calib.pop(name); W = mod.weight.detach().float()
        with torch.no_grad():
            evaluate(name, "fprop", X, W, Xc)
            evaluate(name, "dgrad", dY, W.T.contiguous(), dYc)
    return h


model, tok = load_hf(args.hf, dev)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batches = torch.tensor(ids[:4 * 1025]).view(2, 2, 1025)
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers.") and int(n.split(".")[1]) in keep:
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
for ph, b in (("calib", batches[0]), ("eval", batches[1])):
    phase["p"] = ph
    loss = model(b[:, :-1], target=b[:, 1:]); loss.backward()
    print(f"[{args.tag}] {ph} loss {loss.item():.4f}", flush=True)
fout.close()
