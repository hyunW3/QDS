"""Why does the LET shift hurt block-scaled FP4 (M5)?  For the attn_in / mlp_in inputs of every layer, with the LET scale s and
shift d learned by ptq8b.py --method let_46 (saved with --save_scales), compare X/s (scale only) and (X-d)/s (scale + shift):
  - per token class: first token (BOS / position 0), massive tokens (row absmax > 20x the median row absmax of that layer), rest,
  - NVFP4-4/6 activation NMSE, block absmax inflation (absmax with shift / without), fraction of blocks whose absmax grew,
  - GEMM output NMSE with the exact compensation bias W d (so any difference is quantization only),
  - |d| relative to the per-channel median |x| of ordinary tokens (a shift tuned on a few huge tokens moves everyone else).
"""
import argparse, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--scales", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cal", default="dclm_val_head2000.jsonl"); ap.add_argument("--nwin", type=int, default=4); ap.add_argument("--skip", type=int, default=600)
args = ap.parse_args(); dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2


def q46(A, G=16):
    N, K = A.shape; Ab = A.view(N, K // G, G); amax = Ab.abs().amax(-1, keepdim=True)
    gs = (amax.max() / 4.0 / 448.0).clamp(min=1e-30); best = None
    for d in (6.0, 4.0):
        s = ((amax / d) / gs).clamp(max=448).to(torch.float8_e4m3fn).float() * gs; s = torch.where(s == 0, torch.ones_like(s), s)
        Ah = torch.sign(Ab) * E2M1[torch.bucketize((Ab.abs() / s).clamp(max=6), TH)] * s; e = (Ah - Ab).pow(2).sum(-1, keepdim=True)
        best = (Ah, e) if best is None else (torch.where(e < best[1], Ah, best[0]), torch.minimum(e, best[1]))
    return best[0].view(N, K), amax.squeeze(-1)


S = torch.load(args.scales, map_location="cpu")["scales"]
need = (args.skip + args.nwin) * 1024; ids = []
for line in open(args.cal):
    ids += [tok.bos_token_id] * (tok.bos_token_id is not None) + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) >= need: break
win = torch.tensor(ids[args.skip * 1024:need], device=dev).view(args.nwin, 1024)


class _Stop(Exception): pass


res = dict(model=args.tag, rows=[])
for i, blk in enumerate(model.model.layers):
    if i not in S or f"{i}.shift" not in S: continue
    want = {"attn_in": blk.self_attn.q_proj, "mlp_in": blk.mlp.gate_proj}
    Ws = {"attn_in": [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj], "mlp_in": [blk.mlp.gate_proj, blk.mlp.up_proj]}
    got = {g: [] for g in want}; hs = [m.register_forward_pre_hook(lambda m, a, g=g: got[g].append(a[0][0].float())) for g, m in want.items()]
    hs.append(blk.register_forward_hook(lambda *a: (_ for _ in ()).throw(_Stop())))
    with torch.no_grad():
        for s in win:
            try: model(input_ids=s[None], use_cache=False)
            except _Stop: pass
    for h in hs: h.remove()
    for g in want:
        X = torch.cat(got[g]); s = S[i][g].to(dev).float(); d = S[f"{i}.shift"][g]
        if d is None: continue
        d = d.to(dev).float()
        pos = torch.arange(X.shape[0], device=dev) % 1024
        ramax = X.abs().amax(1); massive = ramax > 20 * ramax.median()
        cls = {"first": pos == 0, "massive": massive & (pos != 0), "rest": (~massive) & (pos != 0)}
        Xa, Xb = X / s, (X - d) / s
        qa, ama = q46(Xa); qb, amb = q46(Xb)
        row = dict(layer=i, group=g, n_massive=int(massive.sum().item()),
                   d_over_median=(d.abs() / X[cls["rest"]].abs().median(0).values.clamp(min=1e-6)).median().item(),
                   d_over_median_p99=(d.abs() / X[cls["rest"]].abs().median(0).values.clamp(min=1e-6)).quantile(0.99).item())
        for c, m in cls.items():
            if m.sum() == 0: continue
            na = ((qa[m] - Xa[m]).pow(2).sum() / Xa[m].pow(2).sum()).item(); nb = ((qb[m] - Xb[m]).pow(2).sum() / Xb[m].pow(2).sum()).item()
            infl = (amb[m] / ama[m].clamp(min=1e-12))
            # error of the reconstructed input in the ORIGINAL basis, relative to the original signal (the shift itself is exact)
            ea = ((qa[m] * s - X[m]).pow(2).sum() / X[m].pow(2).sum()).item(); eb = ((qb[m] * s + d - X[m]).pow(2).sum() / X[m].pow(2).sum()).item()
            row.update({f"{c}_nmse_scale": na, f"{c}_nmse_shift": nb, f"{c}_err_orig_scale": ea, f"{c}_err_orig_shift": eb,
                        f"{c}_absmax_inflation_median": infl.median().item(), f"{c}_frac_blocks_grew": (infl > 1.0).float().mean().item()})
        # GEMM output error (activation quantization only, exact W d bias; weights kept BF16 to isolate the input side)
        oa = ob = ny = 0.0
        for lin in Ws[g]:
            W = lin.weight.float(); Y = X @ W.T
            oa += (qa * s @ W.T - Y).pow(2).sum().item(); ob += ((qb * s + d) @ W.T - Y).pow(2).sum().item(); ny += Y.pow(2).sum().item()
        row.update(out_nmse_scale=oa / ny, out_nmse_shift=ob / ny)
        res["rows"].append(row)
    print(f"[shift {args.tag}] layer {i}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print("SHIFT_DONE")
