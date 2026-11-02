"""Does the joint energy-concentration Delta_FP (Ordentlich & Polyanskiy 2026a, arXiv 2601.17187) explain why a block
Hadamard hurts E2M1, or is it single-operand level placement?

For a GEMM Y = A B^T (reduction axis k last) and quantized operands A^ = A + E_A, B^ = B + E_B, assuming the two
operands' errors are independent and dropping E_A E_B^T:
  ||Y^ - Y||^2 ~= pred = sum_k ( ea_k b_k + a_k eb_k ),  a_k = sum_i A_ik^2, ea_k = sum_i E_A,ik^2 (b, eb likewise)
Exact three-factor split (all normalized by ||Y||^2, which a shared orthogonal transform does not change):
  pred = S * nu * c
  S  = sum_k a_k b_k                  joint energy concentration; S * n / (||A||^2 ||B||^2) = mean Delta_FP. Under the IEIN
                                      (relative-error) model of O&P, ea_k / a_k is the same for every k, so c = 1 and the
                                      rotation effect is entirely S.
  nu = ||E_A||^2/||A||^2 + ||E_B||^2/||B||^2   single-operand tensor NMSE (level placement vs block-normalized values)
  c  = pred / (S nu)                  covariance of per-channel NMSE with per-channel joint energy
Rotation gain (dB) = 10 log10(X_I / X_H) for each factor; gain_pred = gain_S + gain_nu + gain_c exactly.
Formats: E2M1 (fp32 block scale, absmax->6), MXE2M1 (E8M0 OCP floor scale), INT4 (+-7), FP8 E4M3 (absmax->448, the
O&P / IEIN control). Block sizes G in {16, 32, 128}; transform I or H_G (block Hadamard with shared random signs).
Operands: FPROP (X, W), DGRAD (dY, W^T), WGRAD (dY^T, X^T) of every decoder Linear, one 4 x 1024-token DCLM-val batch.
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
ap.add_argument("--moe_expert_stride", type=int, default=0, help="tap routed experts e %% stride == 0 (0 = none)")
ap.add_argument("--moe_layer_stride", type=int, default=1)
ap.add_argument("--auto", action="store_true", help="shard the model over all visible GPUs (device_map=auto)")
ap.add_argument("--layers", default=None, help="comma list of decoder-layer indices (default: all)")
args = ap.parse_args()
dev = "cuda:0"; gen = torch.Generator(device=dev); gen.manual_seed(1)
GS = (16, 32, 128)
HS = {g: hadamard(g, dev) for g in GS}
SG = (torch.randint(0, 2, (1 << 16,), device=dev, generator=gen) * 2 - 1).float()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev)
TH = (E2M1[1:] + E2M1[:-1]) / 2

if args.auto:
    import moe_tap
    ng = torch.cuda.device_count()
    model, tok = moe_tap.load_auto(args.hf, {i: ("10GiB" if i == 0 else "22GiB") for i in range(ng)})
else:
    model, tok = load_hf(args.hf, dev)
ROOT = Path(os.environ.get("LINGUA_ROOT", "lingua_root"))
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batch = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
keep = None if args.layers is None else {int(x) for x in args.layers.split(",")}


def q_e2m1(Ab, mx):
    amax = Ab.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if mx:   # OCP MX: shared exponent floor(log2 amax) - emax(E2M1)=2, elements saturate at 6
        s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    else:
        s = amax / 6
    q = E2M1[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)]
    return torch.sign(Ab) * q * s


def q_int4(Ab):
    s = (Ab.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-30)
    return (Ab / s).round().clamp(-7, 7) * s


def q_fp8(Ab):
    s = (Ab.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return (Ab / s).to(torch.float8_e4m3fn).float() * s


FMTS = {"E2M1": lambda Ab: q_e2m1(Ab, False), "MXE2M1": lambda Ab: q_e2m1(Ab, True), "INT4": q_int4, "FP8": q_fp8}


def blockq(M, fmt, g):
    N, R = M.shape
    return FMTS[fmt](M.view(N, R // g, g)).view(N, R)


def rot(M, g):
    return ((M * SG[: M.shape[1]]).view(M.shape[0], -1, g) @ HS[g]).view(M.shape)


fout = open(args.out, "w"); saved = {}


def orot(M, g, ob):
    """H_G only on the blocks flagged in ob ([R/g] bool); the other blocks stay in place, untransformed."""
    Mb = (M * SG[: M.shape[1]]).view(M.shape[0], -1, g)
    out = M.view(M.shape[0], -1, g).clone()
    out[:, ob] = Mb[:, ob] @ HS[g]
    return out.view(M.shape)


def evaluate(lname, gemm, A, B):
    """T: I (none), H (all blocks), oH (only blocks holding a top-1% outlier channel, score rms(A_k) rms(B_k)).
    share_out = fraction of pred (GEMM error) coming from those outlier blocks; the rest comes from ordinary blocks."""
    R = A.shape[1]
    Y = A @ B.T; nY = (Y * Y).sum().item()
    score = A.pow(2).mean(0).sqrt() * B.pow(2).mean(0).sqrt()
    oc = score.argsort(descending=True)[: max(1, int(math.ceil(0.01 * R)))]
    for g in GS:
        if R % g: continue
        ob = torch.zeros(R // g, dtype=torch.bool, device=dev); ob[oc // g] = True
        ok = ob.repeat_interleave(g)
        for tn in ("I", "H", "oH"):
            At, Bt = {"I": (A, B), "H": (rot(A, g), rot(B, g)), "oH": (orot(A, g, ob), orot(B, g, ob))}[tn]
            a, b = At.pow(2).sum(0), Bt.pow(2).sum(0)
            S = (a * b).sum().item() / nY
            dfp = R * (a * b).sum().item() / (a.sum().item() * b.sum().item())
            rec = dict(model=args.tag, layer=lname, gemm=gemm, G=g, T=tn, K=R, S=S, dfp=dfp,
                       n_oblk=int(ob.sum()), frac_oblk=ob.float().mean().item(), Eshare_out=(a[ok].sum() / a.sum()).item())
            for fn in FMTS:
                EA, EB = blockq(At, fn, g) - At, blockq(Bt, fn, g) - Bt
                ea, eb = EA.pow(2).sum(0), EB.pow(2).sum(0)
                pk = ea * b + a * eb
                pred = pk.sum().item() / nY
                nu = (ea.sum() / a.sum() + eb.sum() / b.sum()).item()
                Yh = (At + EA) @ (Bt + EB).T
                act = ((Yh - Y) ** 2).sum().item() / nY
                rec[fn] = dict(act=act, pred=pred, nu=nu, nuA=(ea.sum() / a.sum()).item(), c=pred / (S * nu),
                               share_out=(pk[ok].sum() / pk.sum()).item(),
                               pred_out=pk[ok].sum().item() / nY, pred_rest=pk[~ok].sum().item() / nY)
            fout.write(json.dumps(rec) + "\n")


def bwd(name):
    def h(mod, gi, go):
        X = saved.pop(name).reshape(-1, mod.in_features).float().to(dev)
        dY = go[0].detach().reshape(-1, mod.out_features).float().to(dev)
        W = mod.weight.detach().float().to(dev)
        with torch.no_grad():
            evaluate(name, "fprop", X, W)
            evaluate(name, "dgrad", dY, W.T.contiguous())
            evaluate(name, "wgrad", dY.T.contiguous(), X.T.contiguous())
    return h


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and n.startswith("layers."):
        if keep is not None and int(n.split(".")[1]) not in keep: continue
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
if args.moe_expert_stride:
    import moe_tap
    print("expert modules tapped:", moe_tap.install(model.m, evaluate, dev, args.moe_expert_stride, args.moe_layer_stride), flush=True)
loss = model(batch[:, :-1], target=batch[:, 1:]); loss.backward()
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
fout.close()
