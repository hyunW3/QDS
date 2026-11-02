"""Cheap, quantization-free statistics per GEMM operand pair, to predict when QDS or OCS helps (joined with
placement_test.py records by (layer, gemm); same batch and seed, so the tensors are identical).

For a GEMM Y = A B^T (reduction axis k last), 1x16 blocks:
  crest_A / crest_B   mean block crest factor max|x| / rms(x) over non-zero blocks
  conc_A / conc_B     energy share of the top-1% reduction channels (columns)
  tok_A               energy share of the top-1% rows of A (token-level outliers, e.g. attention sinks)
  rho_A / rho_B       crest(M) / crest(M / colrms(M)): > 1 when block maxima sit on persistently large channels
                      (channel-aligned outliers, removable by a per-channel scale); ~ 1 when they are token-specific
  sq_crest            mean operand crest after the closed-form SmoothQuant scale s = sqrt(max|A_k| / max|B_k|),
                      relative to before (< 1 = per-channel scaling flattens the blocks)
  dfp                 mean Delta_FP = K sum_k a_k b_k / (sum a)(sum b)
"""
import os
import argparse, json, math, sys
from pathlib import Path
import torch
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from hf_common import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--moe_expert_stride", type=int, default=0); ap.add_argument("--moe_layer_stride", type=int, default=1)
ap.add_argument("--auto", action="store_true"); ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
dev = "cpu" if args.cpu else "cuda:0"; G = 16
if args.cpu: torch.set_num_threads(32)


def crest(M):
    Mb = M.view(M.shape[0], -1, G)
    mx = Mb.abs().amax(-1); rms = Mb.pow(2).mean(-1).sqrt()
    ok = rms > 0
    return (mx[ok] / rms[ok]).mean().item()


def stats(A, B):
    R = A.shape[1]; k = max(1, math.ceil(0.01 * R))
    a, b = A.pow(2).sum(0), B.pow(2).sum(0)
    ra = A.pow(2).sum(1)
    cA, cB = crest(A), crest(B)
    colA = (a / A.shape[0]).sqrt().clamp(min=1e-12); colB = (b / B.shape[0]).sqrt().clamp(min=1e-12)
    s = (A.abs().amax(0).clamp(min=1e-12) / B.abs().amax(0).clamp(min=1e-12)).sqrt()
    return dict(crest_A=cA, crest_B=cB,
                conc_A=(a.topk(k).values.sum() / a.sum()).item(), conc_B=(b.topk(k).values.sum() / b.sum()).item(),
                tok_A=(ra.topk(max(1, math.ceil(0.01 * A.shape[0]))).values.sum() / ra.sum()).item(),
                rho_A=cA / crest(A / colA), rho_B=cB / crest(B / colB),
                sq_crest=(crest(A / s) + crest(B * s)) / (cA + cB),
                dfp=R * (a * b).sum().item() / (a.sum().item() * b.sum().item()))


if args.auto:
    import moe_tap
    model, tok = moe_tap.load_auto(args.hf, {i: ("10GiB" if i == 0 else "22GiB") for i in range(torch.cuda.device_count())})
else:
    model, tok = load_hf(args.hf, dev)
ids = []
for line in open(os.environ.get("DCLM_VAL", "data/dclm_val.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
    if len(ids) >= 4 * 1025: break
batch = torch.tensor(ids[:4 * 1025], device=dev).view(4, 1025)
fout = open(args.out, "w"); saved = {}


def evaluate(lname, gemm, A, B):
    if A.shape[1] % G or A.shape[0] == 0 or B.shape[0] == 0: return
    fout.write(json.dumps(dict(model=args.tag, layer=lname, gemm=gemm, **stats(A, B))) + "\n")


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
        m.register_forward_hook(lambda mod, i, o, n=n: saved.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(bwd(n))
if args.moe_expert_stride:
    import moe_tap
    print("expert modules tapped:", moe_tap.install(model.m, evaluate, dev, args.moe_expert_stride, args.moe_layer_stride), flush=True)
loss = model(batch[:, :-1], target=batch[:, 1:]); loss.backward()
print(f"[{args.tag}] loss {loss.item():.4f}", flush=True)
fout.close()
