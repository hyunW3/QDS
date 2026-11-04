"""Local vs propagated error of a saved scale set (follow-up to diag_fmt.py, which ruled out activation-scale coupling).

(i) local: with FP-model inputs (as in training), per (layer, group) GEMM output relative error
      weight-only   ||(X/s) Q(W s)^T - X W^T||^2 / ||X W^T||^2
      W4A4          ||Q(X/s) Q(W s)^T - X W^T||^2 / ||X W^T||^2
    on 8 calibration segments (training data) and 8 WikiText-2 test sequences.
(ii) propagated: fold + weight-only NVFP4 4/6 (activations BF16), then per-layer residual-stream deviation from the FP model on the
    same 8 test sequences, ||h_q - h_fp|| / ||h_fp|| over tokens 1.. (token 0 skipped: its massive activation dominates the norm).
"""
import argparse, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--scales", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--local_only", action="store_true")   # skip (ii); (i) also reports the error over tokens 1.. (token 0 excluded)
ap.add_argument("--fold_only", action="store_true")    # skip (i); in (ii) fold the scales but do NOT quantize (BF16 fold numerics)
args = ap.parse_args()
dev = "cuda"; torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
cfg = model.config; Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq; REP = Hq // Hkv
LAYERS = model.model.layers; G = 16
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2


def rnd(A, s): return torch.sign(A) * E2M1[torch.bucketize((A.abs() / s).clamp(max=6), TH)] * s


def q46(A):   # = ptq8b_s.nvfp4(four_six=True), E4M3 block scales
    N, K = A.shape; Ab = A.float().view(N, K // G, G); amax = Ab.abs().amax(-1, keepdim=True)
    gs = (amax.max() / 4.0 / 448.0).clamp(min=1e-30)
    cs = [((amax / d) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs for d in (6.0, 4.0)]
    cs = [torch.where(c == 0, torch.ones_like(c), c) for c in cs]
    a = rnd(Ab, cs[0]); b = rnd(Ab, cs[1])
    return torch.where((b - Ab).pow(2).sum(-1, keepdim=True) < (a - Ab).pow(2).sum(-1, keepdim=True), b, a).view(N, K)


def qrows(A, R): return torch.cat([q46(A[i:i + R]) for i in range(0, A.shape[0], R)])


T = 2048
test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]
test = test[: args.n * T].view(args.n, T)
train = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]), return_tensors="pt").input_ids[0]
gen = torch.Generator().manual_seed(0); st = torch.randint(0, train.numel() - T - 1, (128,), generator=gen)
cal = torch.stack([train[a:a + T] for a in st.tolist()])[: args.n]

S = torch.load(args.scales, map_location="cpu")["scales"]
GROUPS = {"attn_in": ("self_attn", ("q_proj", "k_proj", "v_proj")), "o": ("self_attn", ("o_proj",)),
          "mlp_in": ("mlp", ("gate_proj", "up_proj")), "down": ("mlp", ("down_proj",))}


def local_hook(i, g):
    def h(m, a):
        X = a[0].reshape(-1, a[0].shape[-1]).float(); s = S[i][g].to(dev).float()
        par = getattr(LAYERS[i], GROUPS[g][0])
        Xs = X / s; Xq = qrows(Xs, 8192)
        for n in GROUPS[g][1]:                 # Q(W s) recomputed per call: caching all layers would not fit 24 GB
            W = getattr(par, n).weight.float(); Wq = qrows(W * s, 4096)
            Y = X @ W.T; d = (Y * Y).sum().item()
            E0 = (Xs @ Wq.T - Y) ** 2; E1 = (Xq @ Wq.T - Y) ** 2
            ACC[(i, g)][0] += E0.sum().item(); ACC[(i, g)][1] += E1.sum().item(); ACC[(i, g)][2] += d
            ACC[(i, g)][3] += E0[1:].sum().item(); ACC[(i, g)][4] += E1[1:].sum().item(); ACC[(i, g)][5] += (Y[1:] * Y[1:]).sum().item()
            ACC[(i, g)][6] += d - (Y[1:] * Y[1:]).sum().item()        # token 0's share of ||Y||^2 (batch 1: row 0 = token 0)
    return h


res = dict(hf=args.hf, scales=args.scales)
hooks = []
for i in range(len(LAYERS)):
    for g, (pn, names) in GROUPS.items():
        hooks.append(getattr(getattr(LAYERS[i], pn), names[0]).register_forward_pre_hook(local_hook(i, g)))
for name, data in (() if args.fold_only else (("cal", cal), ("test", test))):
    ACC ={(i, g): [0.0] * 7 for i in range(len(LAYERS)) for g in GROUPS}
    for j in range(args.n): model(input_ids=data[j:j + 1].to(dev), use_cache=False)
    res[f"local_{name}"] = {f"{i}.{g}": [v[0] / v[2], v[1] / v[2], v[3] / v[5], v[4] / v[5], v[6] / v[2]] for (i, g), v in ACC.items()}
    mean = lambda k: sum(x[k] for x in res[f"local_{name}"].values()) / len(res[f"local_{name}"])
    res[f"local_{name}_mean"] = [mean(k) for k in range(5)]
    print(f"local {name}: mean rel err weight-only {mean(0):.5f}  W4A4 {mean(1):.5f} | tokens 1..: weight-only {mean(2):.5f}  W4A4 {mean(3):.5f} | token-0 share of ||Y||^2 {mean(4):.3f}", flush=True)
    for i in range(3):
        print("   layer", i, " ".join(f"{g}: all {res[f'local_{name}'][f'{i}.{g}'][1]:.4f} tok1.. {res[f'local_{name}'][f'{i}.{g}'][3]:.4f} tok0share {res[f'local_{name}'][f'{i}.{g}'][4]:.2f}" for g in GROUPS), flush=True)
for h in hooks: h.remove()
torch.cuda.empty_cache()
if args.local_only:
    json.dump(res, open(args.out, "w"), indent=1); raise SystemExit

hfp = [model(input_ids=test[j:j + 1].to(dev), use_cache=False, output_hidden_states=True).hidden_states for j in range(args.n)]
hfp = [[x[0, 1:].cpu() for x in hs] for hs in hfp]   # BF16 on CPU (~4 GB for 8 sequences)

for i, blk in enumerate(LAYERS):   # fold exactly like ptq8b_s.fold, then weight-only NVFP4 4/6 (4096-row chunks as in eval)
    a, m = blk.self_attn, blk.mlp; sc = {g: S[i][g].to(dev) for g in GROUPS}
    blk.input_layernorm.weight.div_(sc["attn_in"].to(torch.bfloat16))
    for l in (a.q_proj, a.k_proj, a.v_proj): l.weight.mul_(sc["attn_in"].to(torch.bfloat16))
    blk.post_attention_layernorm.weight.div_(sc["mlp_in"].to(torch.bfloat16))
    for l in (m.gate_proj, m.up_proj): l.weight.mul_(sc["mlp_in"].to(torch.bfloat16))
    s_kv = sc["o"].view(Hkv, REP, hd)[:, 0, :].reshape(-1)
    a.v_proj.weight.div_(s_kv[:, None].to(torch.bfloat16))
    if a.v_proj.bias is not None: a.v_proj.bias.div_(s_kv.to(torch.bfloat16))
    a.o_proj.weight.mul_(sc["o"].to(torch.bfloat16))
    m.up_proj.weight.div_(sc["down"][:, None].to(torch.bfloat16)); m.down_proj.weight.mul_(sc["down"].to(torch.bfloat16))
    for l in (() if args.fold_only else (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj)):
        l.weight.copy_(qrows(l.weight.float(), 4096).to(torch.bfloat16))
dev_l = [0.0] * len(hfp[0]); nll = 0.0
for j in range(args.n):
    out = model(input_ids=test[j:j + 1].to(dev), use_cache=False, output_hidden_states=True)
    for k, x in enumerate(out.hidden_states):
        r = x[0, 1:].float(); f = hfp[j][k].to(dev).float(); dev_l[k] += ((r - f).norm() / f.norm()).item() / args.n
    nll += F.cross_entropy(out.logits[0, :-1].float(), test[j, 1:].to(dev), reduction="sum").item()
res["prop_dev"] = dev_l; res["ppl_weightonly_n"] = math.exp(nll / (args.n * (T - 1)))
print("residual deviation by layer:", " ".join(f"{x:.3f}" for x in dev_l), flush=True)
print(f"weight-only ppl on {args.n} test seqs: {res['ppl_weightonly_n']:.3f}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
