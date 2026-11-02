"""Static energy-CONCENTRATING block transforms before absmax for E2M1 (the opposite of rotation), on real 8B tensors.
For each sampled layer and Linear group, the input X (eval windows) and weights W are transformed by a 16x16 block-diagonal
orthogonal Q (X' = X Q, W' = W Q, so the product is unchanged) and quantized (W4A4):
  none     identity
  H16      block Hadamard (flattening; the usual rotation)
  BKLT     per-block PCA of X (eigenvectors of X_b^T X_b, descending): compacts X's average energy into the first coordinates
  BKLT_W   per-block PCA of W instead (compact the weight side)
  HHmax    per block, the Householder reflection that maps X's principal direction onto the channel that is most often the
           block maximum (rank-1 concentration onto the absmax slot)
Reported per transform: GEMM output NMSE (NVFP4-4/6 and NVFP4), absmax-element energy share of X blocks, X block crest factor,
and the Ordentlich-Polyanskiy joint concentration D_FP = n * sum_i (x_i^2/|x|^2)(w_i^2/|w|^2) per block (lower is better for FP).
"""
import argparse, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cal", default="dclm_val_head2000.jsonl"); ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,-1")
ap.add_argument("--nfit", type=int, default=2); ap.add_argument("--neval", type=int, default=2)
args = ap.parse_args(); dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(dev).eval()
NL = len(model.model.layers); LAY = sorted({int(x) % NL for x in args.layers.split(",")})
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2
_H = torch.ones(1, 1, device=dev)
while _H.shape[0] < 16: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
H16 = _H / 4.0


def q(A, four_six):
    N, K = A.shape; Ab = A.view(N, K // 16, 16); amax = Ab.abs().amax(-1, keepdim=True)
    gs = (amax.max() / (4.0 if four_six else 6.0) / 448.0).clamp(min=1e-30); best = None
    for d in ([6.0, 4.0] if four_six else [6.0]):
        s = ((amax / d) / gs).clamp(max=448).to(torch.float8_e4m3fn).float() * gs; s = torch.where(s == 0, torch.ones_like(s), s)
        Q = torch.sign(Ab) * E2M1[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)] * s; e = (Q - Ab).pow(2).sum(-1, keepdim=True)
        best = (Q, e) if best is None else (torch.where(e < best[1], Q, best[0]), torch.minimum(e, best[1]))
    return best[0].view(N, K)


def apply(A, Qb):                    # A [N,K], Qb [K/16,16,16] block-diagonal orthogonal
    N, K = A.shape; return torch.einsum("nbi,bij->nbj", A.view(N, K // 16, 16), Qb).reshape(N, K)


def pca_blocks(A):
    N, K = A.shape; Ab = A.view(N, K // 16, 16)
    C = torch.einsum("nbi,nbj->bij", Ab, Ab)
    ev, V = torch.linalg.eigh(C.double()); return V.flip(-1).float()            # columns in descending eigenvalue order


def hh_max(Xfit):
    """per block: Householder mapping X's principal direction v onto e_j, j = channel most often holding the block max."""
    N, K = Xfit.shape; Ab = Xfit.view(N, K // 16, 16); V = pca_blocks(Xfit)[:, :, 0]          # [b,16]
    j = torch.mode(Ab.abs().argmax(-1), dim=0).values                                        # [b]
    e = torch.nn.functional.one_hot(j, 16).float()
    v = V * torch.sign((V * e).sum(-1, keepdim=True) + 1e-12)                                # orient toward e_j
    u = v - e; nu = u.norm(dim=-1, keepdim=True)
    u = torch.where(nu > 1e-6, u / nu.clamp(min=1e-12), torch.zeros_like(u))
    return torch.eye(16, device=dev)[None] - 2 * u[:, :, None] * u[:, None, :]


ids = []
for line in open(args.cal):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= (600 + args.nfit + args.neval) * 1024: break
win = torch.tensor(ids[600 * 1024:(600 + args.nfit + args.neval) * 1024], device=dev).view(-1, 1024)
cap = {}
for i in LAY:
    blk = model.model.layers[i]
    for g, m in [("attn_in", blk.self_attn.q_proj), ("o", blk.self_attn.o_proj), ("mlp_in", blk.mlp.gate_proj), ("down", blk.mlp.down_proj)]:
        m.register_forward_pre_hook(lambda mod, a, key=(i, g): cap.setdefault(key, []).append(a[0][0].float()))
with torch.no_grad():
    for s in win: model(input_ids=s[None], use_cache=False)
res = dict(model=args.tag, layers=LAY, rows=[])
for (i, g), xs in sorted(cap.items()):
    blk = model.model.layers[i]
    mods = {"attn_in": [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj], "o": [blk.self_attn.o_proj],
            "mlp_in": [blk.mlp.gate_proj, blk.mlp.up_proj], "down": [blk.mlp.down_proj]}[g]
    Xfit = torch.cat(xs[:args.nfit]); X = torch.cat(xs[args.nfit:]); Ws = [m.weight.float() for m in mods]
    K = X.shape[1]; I = torch.eye(16, device=dev)[None].expand(K // 16, 16, 16)
    TF = {"none": I, "H16": H16[None].expand(K // 16, 16, 16), "BKLT": pca_blocks(Xfit), "BKLT_W": pca_blocks(torch.cat(Ws)), "HHmax": hh_max(Xfit)}
    with torch.no_grad():
        for name, Qb in TF.items():
            Xt = apply(X, Qb); Xb = Xt.view(X.shape[0], K // 16, 16)
            e = Xb.pow(2); share = (e.amax(-1) / e.sum(-1).clamp(min=1e-30)).mean().item()
            cf = (Xb.abs().amax(-1) / Xb.pow(2).mean(-1).sqrt().clamp(min=1e-30)).mean().item()
            row = dict(layer=i, group=g, tf=name, x_absmax_share=share, x_crest=cf)
            for qn, f46 in [("nvfp4_46", True), ("nvfp4", False)]:
                Xq = q(Xt, f46); num = den = dfp = 0.0
                for W in Ws:
                    Wt = apply(W, Qb); Y = X @ W.T
                    num += (Xq @ q(Wt, f46).T - Y).pow(2).sum().item(); den += Y.pow(2).sum().item()
                row[f"out_nmse_{qn}"] = num / den
            # joint concentration D_FP (Ordentlich-Polyanskiy) on token-avg x energy vs per-row w energy, per block, averaged
            xe = Xb.pow(2).mean(0); xe = xe / xe.sum(-1, keepdim=True)                         # [b,16]
            Wt = apply(Ws[0], Qb).view(Ws[0].shape[0], K // 16, 16).pow(2); we = Wt / Wt.sum(-1, keepdim=True).clamp(min=1e-30)
            row["d_fp"] = (16 * (we * xe[None]).sum(-1)).mean().item()
            res["rows"].append(row)
    print(f"[conc {args.tag}] layer {i} {g}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
print("CONC_DONE", flush=True)
