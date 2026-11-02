"""Can a different rotation keep E2M1 error out of its coarse top bins (4, 6)? (llama checkpoints)

Every transform is an orthogonal Q on the GEMM reduction axis applied to BOTH operands, so (AQ)(BQ)^T = AB^T.
Quantization: 1x16 blocks along the reduction axis, absmax scale (FP32), RTN.

Transforms
  I                no transform
  H{2,4,8}         block-diagonal Hadamard of size k (< 16: partial mixing inside a 16-block) + random signs
  H16              standard RHT (UFP4 / NVFP4 recipe)
  H{32,64,128}     wider mixing (control)
  DCT16, Haar16    energy-compacting orthogonal 16x16 (sparsify instead of Gaussianize) + random signs
  PERM             reorder reduction index by RMS of operand A (similar magnitudes share a block)
  PERM+H16         PERM then H16
  ADAPT            per 16-block choice between I and H16 (lower normalized operand MSE)
  SIGN16           H16 with the best of 16 random sign vectors (lowest GEMM error)
  KLT              eigenbasis of A^T A on the reduction axis (channel axes only)
Scale-side comparator (E2M1 only)
  4/6, H16+4/6     per-block choice of absmax -> 6 or -> 4 (Four-Over-Six style)

Data-dependent choices (PERM, ADAPT, SIGN16, KLT) are fitted on a calibration batch and evaluated on a different
batch for FPROP (reduce K) and DGRAD (reduce O). For WGRAD the reduction axis is tokens, which change every
batch, so they are computed online on the evaluated batch itself (marked online=True; KLT skipped there).
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF, BLK, quantize
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root")); sys.path.insert(0, str(ROOT / "lingua"))
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None); ap.add_argument("--hf", default=None); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="", help="comma list of block indices to probe (default all)")
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)
gen = torch.Generator(device=dev); gen.manual_seed(1)


# ---------------- orthogonal building blocks
def hadamard(n):
    H = torch.ones(1, 1, device=dev)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def dct(n):
    k = torch.arange(n, device=dev).float()
    M = torch.cos(math.pi / n * (k[None, :] + 0.5) * k[:, None]) * math.sqrt(2 / n)
    M[0] /= math.sqrt(2)
    return M.T            # columns = basis vectors; x @ M gives coefficients


def haar_rows(n):
    """orthonormal Haar matrix, rows = basis vectors."""
    if n == 1:
        return torch.ones(1, 1, device=dev)
    top = torch.kron(haar_rows(n // 2), torch.tensor([[1., 1.]], device=dev))
    bot = torch.kron(torch.eye(n // 2, device=dev), torch.tensor([[1., -1.]], device=dev))
    M = torch.cat([top, bot], 0)
    return M / M.norm(dim=1, keepdim=True)


def haar(n):
    return haar_rows(n).T.contiguous()


MATS = {f"H{k}": hadamard(k) for k in [2, 4, 8, 16, 32, 64, 128]}
MATS["DCT16"] = dct(16); MATS["Haar16"] = haar(16)
for k, M in MATS.items():
    assert torch.allclose(M @ M.T, torch.eye(M.shape[0], device=dev), atol=1e-5), k


def blockdiag(A, M):
    k = M.shape[0]; N, R = A.shape
    return (A.view(N, R // k, k) @ M).view(N, R)


def rand_signs(R):
    return (torch.randint(0, 2, (R,), device=dev, generator=gen) * 2 - 1).float()


# ---------------- quantizers
def q_e2m1(A):
    return quantize(A, E2M1, "rtn", "fp32")


def q_unif(A):
    return quantize(A, UNIF, "rtn", "fp32")


def q_multiscale(A, grid, cands):
    """per-block min-MSE choice among scales s = amax / c (c in cands); values beyond the grid max saturate."""
    N, R = A.shape
    Ab = A.float().view(N, R // BLK, BLK)
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    g = grid.to(dev); th = (g[1:] + g[:-1]) / 2
    best, berr = None, None
    for gm in cands:
        s = amax / gm
        q = g[torch.bucketize((Ab.abs() / s).clamp(max=float(g[-1])), th)]
        Ah = torch.sign(Ab) * q * s
        err = (Ah - Ab).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, berr = Ah, err
        else:
            m = err < berr
            best = torch.where(m, Ah, best); berr = torch.where(m, err, berr)
    return best.view(N, R), None, None


q_46 = lambda A: q_multiscale(A, E2M1, (6.0, 4.0))
q_463 = lambda A: q_multiscale(A, E2M1, (6.0, 4.0, 3.0))
q_unif_clip = lambda A: q_multiscale(A, UNIF, (1.75, 1.75 / 0.9, 1.75 / 0.8))   # UNIF max level 1.75; clip c in {1,.9,.8}
q_e2m1_clip = lambda A: q_multiscale(A, E2M1, (6.0, 6.0 / 0.9, 6.0 / 0.8))


def lvl46_share(A, Ah, q):
    if q is None:
        return None
    e = (Ah - A).pow(2).reshape(-1)
    return (e[q >= 4].sum() / e.sum().clamp(min=1e-30)).item()


# ---------------- transform factory: returns f(M) -> M Q on the last axis
def build_tf(name, Aref, Bref, Yref):
    R = Aref.shape[1]
    if name == "I":
        return lambda M: M
    if name in MATS:
        sg = rand_signs(R); Mx = MATS[name]
        return lambda M: blockdiag(M * sg, Mx)
    if name == "PERM" or name == "PERM+H16":
        order = Aref.pow(2).mean(0).argsort(descending=True)
        if name == "PERM":
            return lambda M: M[:, order]
        sg = rand_signs(R)
        return lambda M: blockdiag(M[:, order] * sg, MATS["H16"])
    if name == "ADAPT":
        sg = rand_signs(R)
        fH = lambda M: blockdiag(M * sg, MATS["H16"])
        err = torch.zeros(R // BLK, 2, device=dev)
        for j, f in enumerate([lambda M: M, fH]):
            for T in (Aref, Bref):
                Tt = f(T); Th = q_e2m1(Tt)[0]
                err[:, j] += ((Th - Tt) ** 2).view(T.shape[0], R // BLK, BLK).sum((0, 2)) / (T * T).sum()
        useH = (err[:, 1] < err[:, 0]).repeat_interleave(BLK)
        return lambda M: torch.where(useH[None, :], fH(M), M)
    if name == "SIGN16":
        best, bsg = None, None
        for _ in range(16):
            sg = rand_signs(R)
            At = blockdiag(Aref * sg, MATS["H16"]); Bt = blockdiag(Bref * sg, MATS["H16"])
            e = ((q_e2m1(At)[0] @ q_e2m1(Bt)[0].T - Yref) ** 2).sum().item()
            if best is None or e < best:
                best, bsg = e, sg
        return lambda M: blockdiag(M * bsg, MATS["H16"])
    if name == "KLT":
        C = Aref.T @ Aref
        _, V = torch.linalg.eigh(C.double())
        Q = V.flip(1).float()
        return lambda M: M @ Q
    raise ValueError(name)


TFS = ["I", "H2", "H4", "H8", "H16", "H32", "H64", "H128", "DCT16", "Haar16", "PERM", "PERM+H16", "ADAPT", "SIGN16", "KLT"]
DATA_DEP = {"PERM", "PERM+H16", "ADAPT", "SIGN16", "KLT"}

# ---------------- model + two batches
if args.hf:
    from hf_common import load_hf
    model, tok = load_hf(args.hf, dev)
else:
    ck = Path(args.ckpt)
    cfg = OmegaConf.load(ck / "params.json")
    model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
    model.load_state_dict(torch.load(ck / "consolidated.pth", weights_only=True, map_location="cpu", mmap=True)["model"])
    model = model.to(dev).train()
    tok = build_tokenizer("tiktoken", str(ROOT / "tokenizer/llama3/tokenizer.model"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 8 * 1025: break
batches = torch.tensor(ids[:8 * 1025], device=dev).view(2, 4, 1025)   # [calib, eval]
keep = set(int(i) for i in args.layers.split(",")) if args.layers else None

saved, calib = {}, {}
phase = {"name": "calib"}
fout = open(args.out, "w")


def evaluate(lname, gemm, A, B, Aref, Bref, online):
    """A [M,R], B [P,R]; refs are the tensors data-dependent transforms are fitted on."""
    Y = A @ B.T
    Yref = Aref @ Bref.T
    nY = (Y * Y).sum()
    for tn in TFS:
        if tn == "KLT" and online:
            continue
        f = build_tf(tn, Aref, Bref, Yref)
        At, Bt = f(A), f(B)
        quants = [("E2M1", q_e2m1), ("UNIF", q_unif)]
        if tn in ("I", "H16", "H4", "PERM"):
            quants += [("E2M1-4/6", q_46), ("E2M1-4/6/3", q_463), ("E2M1-clip", q_e2m1_clip), ("UNIF-clip", q_unif_clip)]
        for gn, qf in quants:
            Ah, _, qa = qf(At); Bh = qf(Bt)[0]
            Yh = Ah @ Bh.T
            rec = dict(model=args.tag, layer=lname, gemm=gemm, tf=tn, grid=gn,
                       online=bool(online and tn in DATA_DEP),
                       g_nmse=(((Yh - Y) ** 2).sum() / nY).item(), g_alpha=((Yh * Y).sum() / nY).item(),
                       A_nmse=(((Ah - At) ** 2).sum() / (At * At).sum()).item(),
                       B_nmse=(((Bh - Bt) ** 2).sum() / (Bt * Bt).sum()).item(),
                       A_lvl46=lvl46_share(At, Ah, qa) if gn == "E2M1" else None)
            fout.write(json.dumps(rec) + "\n")


def fwd_hook(name):
    def h(mod, i, o):
        saved[name] = i[0].detach().to(torch.bfloat16)
    return h


def bwd_hook(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float()
        dY = go[0].detach().reshape(-1, mod.out_features).to(torch.bfloat16).float()
        W = mod.weight.detach().to(torch.bfloat16).float()
        if phase["name"] == "calib":
            calib[name] = (X.cpu(), dY.cpu())
            return
        Xc, dYc = (t.to(dev) for t in calib.pop(name))
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            evaluate(name, "fprop", X, W, Xc, W, False)
            evaluate(name, "dgrad", dY, W.T.contiguous(), dYc, W.T.contiguous(), False)
            dYt, Xt = dY.T.contiguous(), X.T.contiguous()
            evaluate(name, "wgrad", dYt, Xt, dYt, Xt, True)
        del Xc, dYc
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        if keep is not None and int(n.split(".")[1]) not in keep:
            continue
        m.register_forward_hook(fwd_hook(n)); m.register_full_backward_hook(bwd_hook(n))

for ph, b in [("calib", batches[0]), ("eval", batches[1])]:
    phase["name"] = ph
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(b[:, :-1], target=b[:, 1:], attn_impl="sdpa")
    loss.backward()
    print(f"[{args.tag}] {ph} loss {loss.item():.4f}", flush=True)
fout.close()
print(f"[{args.tag}] done -> {args.out}", flush=True)
