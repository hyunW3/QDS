"""W4A4 NVFP4 PTQ on an MoE model (OLMoE layout: MHA attention, routed experts with 3D gate_up_proj / down_proj).

All quantized GEMMs (attention q/k/v/o and every routed expert's gate_up and down) use NVFP4: E2M1, 1x16 blocks, E4M3
block scales with an FP32 tensor scale (the ptq8b quantizer), RTN, weights quantized once, activations per call.
Router, embeddings, norms and lm_head stay BF16.

Methods:
  bf16          reference
  e2m1          plain NVFP4
  e2m1_46       + per-block 4/6 scale choice
  e2m1_h16      16-point block Hadamard on both operands (online)
  ufp4          uniform grid + H16
  sq_46         closed-form SmoothQuant diag (alpha 0.5), folded, + 4/6
  qds_46        QDS: per-group diag scale s = max|X|^a / max|W|^(1-a), a in {0..1} chosen by the FP4 GEMM output error
                (E2M1, FP32 block scale, absmax -> 6, as in the QDS definition), folded, + 4/6.
                Groups: attn_in {q,k,v} -> input_layernorm; o -> v_proj rows; moe_in {all experts' gate_up, router} ->
                post_attention_layernorm (router weight absorbs s exactly); down_e (per expert) -> that expert's up rows.
  ocs_46        outlier channel splitting on every quantized GEMM (static top-1% channels from calibration), + 4/6
  qds_ocs_46    qds_46, then OCS on the routed experts only
  qds           QDS without 4/6 (orthogonality check)
  qds_46_h      QDS-4/6 + block Hadamard (QDS fitted through the rotated quantizer)
  --rht         random-sign block Hadamard for the rotating methods (default: plain block Hadamard)
Eval: WikiText-2 perplexity (2048-token windows, as in ptq8b), optional lm-eval Avg5 (--lm).
"""
import os
import argparse, json, math, sys, time, types
from pathlib import Path
import torch
import torch.nn.functional as F
D = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(D))
from fp4lib import E2M1, UNIF
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="allenai/OLMoE-1B-7B-0125"); ap.add_argument("--tag", default="olmoe-1b-7b")
ap.add_argument("--method", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--Tcal", type=int, default=1024)
ap.add_argument("--lm", action="store_true"); ap.add_argument("--lm_limit", type=float, default=None)
ap.add_argument("--rht", action="store_true", help="random +-1 signs per reduction channel (fixed seed) before the block Hadamard")
ap.add_argument("--cal", default=os.environ.get("DCLM_VAL", "data/dclm_val.jsonl"))
args = ap.parse_args()
dev = "cuda"; G = 16
M = args.method
GRID = UNIF if M == "ufp4" else E2M1
FS = "_46" in M; ROT = M in ("e2m1_h16", "ufp4", "qds_46_h")
QDS_M = ("sq_46", "qds", "qds_46", "qds_ocs_46", "qds_46_h")
_H = torch.ones(1, 1, device=dev)
while _H.shape[0] < G: _H = torch.cat([torch.cat([_H, _H], 1), torch.cat([_H, -_H], 1)], 0)
HG = _H / math.sqrt(G)
_SG = {}


def sgn(K, device):
    """per-K random +-1 (seed 1234 + K, same as ptq8b_v3 --rht); None = plain block Hadamard."""
    if not args.rht: return None
    if K not in _SG:
        g = torch.Generator(device="cpu"); g.manual_seed(1234 + K)
        _SG[K] = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float()
    return _SG[K].to(device)


# ---------------------------------------------------------------- NVFP4 quantizer (same as ptq8b, E4M3 path)
def tensor_gs(amax, gmax, four_six):
    return (amax.max() / (4.0 if four_six else gmax) / 448.0).clamp(min=1e-30)


def nvfp4(A, grid=GRID, four_six=FS, rot=ROT):
    N, K = A.shape
    sg = sgn(K, A.device) if rot else None
    Ab = (A.float() if sg is None else A.float() * sg).view(N, K // G, G)
    if rot: Ab = Ab @ HG
    g = grid.to(A.device); th = (g[1:] + g[:-1]) / 2; gmax = g[-1]
    amax = Ab.abs().amax(-1, keepdim=True); gs = tensor_gs(amax, gmax, four_six)
    def q(d):
        s = ((amax / d) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=gmax), th)] * s
    Ah = q(gmax)
    if four_six:
        A4 = q(4.0)
        Ah = torch.where((A4 - Ab).pow(2).sum(-1, keepdim=True) < (Ah - Ab).pow(2).sum(-1, keepdim=True), A4, Ah)
    if rot: Ah = Ah @ HG.T
    Ah = Ah.view(N, K)
    return Ah if sg is None else Ah * sg


def q_fit(A, rot=ROT):   # QDS fitting quantizer: E2M1, FP32 block scale, absmax -> 6 (through the rotation if the method rotates)
    N, K = A.shape; sg = sgn(K, A.device) if rot else None
    Ab = (A.float() if sg is None else A.float() * sg).view(N, K // G, G); g = E2M1.to(A.device); th = (g[1:] + g[:-1]) / 2
    if rot: Ab = Ab @ HG
    s = (Ab.abs().amax(-1, keepdim=True) / 6).clamp(min=1e-30)
    Ah = torch.sign(Ab) * g[torch.bucketize((Ab.abs() / s).clamp(max=6), th)] * s
    if rot: Ah = Ah @ HG.T
    Ah = Ah.view(N, K)
    return Ah if sg is None else Ah * sg


def pad16(M_):
    r = (-M_.shape[1]) % G
    return M_ if r == 0 else torch.cat([M_, M_.new_zeros(M_.shape[0], r)], 1)


def ocs_x(x, idx):
    if idx is None: return x
    x2 = x.clone(); x2[:, idx] *= 0.5
    return pad16(torch.cat([x2, x[:, idx] * 0.5], 1))


def ocs_w(W, idx):
    return W if idx is None else pad16(torch.cat([W, W[:, idx]], 1))


def qmm(x, Wq, idx):
    """x [N,K] (bf16) -> quantized activation @ pre-quantized weight^T (both in the original basis)."""
    return (nvfp4(ocs_x(x.float(), idx)) @ Wq.T).to(x.dtype)


# ---------------------------------------------------------------- model + data
tok = AutoTokenizer.from_pretrained(args.hf)
# eager experts: the default grouped_mm path needs compute capability 9.0 (fails on Blackwell sm_120)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="eager").to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
LAYERS = model.model.layers
NE = model.config.num_experts; TOPK = model.config.num_experts_per_tok; I = model.config.intermediate_size
ids = []
for line in open(args.cal):
    ids += tok.encode(json.loads(line)["text"]) + [tok.eos_token_id]
    if len(ids) >= args.ncal * args.Tcal: break
cal = torch.tensor(ids[:args.ncal * args.Tcal], device=dev).view(args.ncal, args.Tcal)


@torch.no_grad()
def capture():
    """per layer: attn_in X, o X, moe_in X (normed), all on CPU fp32; folds are function-preserving, one pass suffices."""
    got = {i: {"attn_in": [], "o": [], "moe_in": []} for i in range(len(LAYERS))}; hs = []
    for i, L in enumerate(LAYERS):
        hs.append(L.self_attn.q_proj.register_forward_pre_hook(lambda m, a, i=i: got[i]["attn_in"].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))
        hs.append(L.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: got[i]["o"].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))
        hs.append(L.mlp.register_forward_pre_hook(lambda m, a, i=i: got[i]["moe_in"].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))
    for s in cal: model(input_ids=s[None], use_cache=False)
    for h in hs: h.remove()
    return {i: {k: torch.cat(v) for k, v in d.items()} for i, d in got.items()}


def route(L, X):
    logits = X @ L.mlp.gate.weight.float().T
    return torch.topk(torch.softmax(logits, -1), TOPK, -1).indices


def group_err(Xs, Ws, s):
    e = 0.0
    for X, W in zip(Xs, Ws):
        Y = X @ W.T; e += ((q_fit(X / s) @ q_fit(W * s).T - Y).pow(2).sum() / Y.pow(2).sum().clamp(min=1e-30)).item()
    return e


def fit_s(Xs, Ws, alphas):
    ax = torch.stack([X.abs().amax(0) for X in Xs]).amax(0).clamp(min=1e-8)
    aw = torch.stack([W.abs().amax(0) for W in Ws]).amax(0).clamp(min=1e-8)
    best = None
    for a in alphas:
        s = ax.pow(a) / aw.pow(1 - a)
        e = group_err(Xs, Ws, s) if len(alphas) > 1 else 0.0
        if best is None or e < best[0]: best = (e, s, a)
    return best[1], best[2]


S_MOE = {}


@torch.no_grad()
def fold_scales(caps):
    alphas = [0.5] if M == "sq_46" else [i / 10 for i in range(11)]
    log = {}
    for i, L in enumerate(LAYERS):
        c = {k: v.to(dev) for k, v in caps[i].items()}
        A = L.self_attn
        # attn_in
        Ws = [A.q_proj.weight.float(), A.k_proj.weight.float(), A.v_proj.weight.float()]
        s, a1 = fit_s([c["attn_in"]] * 3, Ws, alphas)
        L.input_layernorm.weight.div_(s.to(torch.bfloat16))
        for lin in (A.q_proj, A.k_proj, A.v_proj): lin.weight.mul_(s.to(torch.bfloat16))
        # o (MHA: v_proj rows <-> o_proj cols one-to-one); captured o input must be divided by s after the fold
        s, a2 = fit_s([c["o"]], [A.o_proj.weight.float()], alphas)
        A.v_proj.weight.div_(s[:, None].to(torch.bfloat16)); A.o_proj.weight.mul_(s.to(torch.bfloat16))
        # moe_in: experts' gate_up on their routed tokens, router absorbs s exactly
        X = c["moe_in"]; r = route(L, X); E = L.mlp.experts
        Xs, Ws = [], []
        for e in range(NE):
            tok_e = (r == e).any(-1).nonzero().squeeze(1)
            if tok_e.numel(): Xs.append(X[tok_e]); Ws.append(E.gate_up_proj[e].float())
        s, a3 = fit_s(Xs, Ws, alphas)
        S_MOE[i] = s
        L.post_attention_layernorm.weight.div_(s.to(torch.bfloat16))
        E.gate_up_proj.mul_(s.to(torch.bfloat16)); L.mlp.gate.weight.mul_(s.to(torch.bfloat16))
        # down_e: per expert, input = act(gate) * up on routed tokens (computed with the folded weights / inputs)
        Xn = X / s
        a4 = []
        for e in range(NE):
            tok_e = (r == e).any(-1).nonzero().squeeze(1)
            if not tok_e.numel(): continue
            g, u = (Xn[tok_e] @ E.gate_up_proj[e].float().T).chunk(2, -1); h = E.act_fn(g) * u
            s, a = fit_s([h], [E.down_proj[e].float()], alphas); a4.append(a)
            E.gate_up_proj[e, I:].div_(s[:, None].to(torch.bfloat16)); E.down_proj[e].mul_(s.to(torch.bfloat16))
        log[i] = dict(attn_in=a1, o=a2, moe_in=a3, down_mean=sum(a4) / max(1, len(a4)))
        del c
    return log


def top_idx(X, W):
    K = X.shape[1]; sc = X.pow(2).mean(0).sqrt() * W.float().pow(2).mean(0).sqrt()
    return sc.argsort(descending=True)[: max(1, math.ceil(0.01 * K))]


# ---------------------------------------------------------------- quantized forward
class QLin(torch.nn.Module):
    def __init__(self, lin, idx):
        super().__init__(); self.idx = idx
        self.register_buffer("w", nvfp4(ocs_w(lin.weight.float(), idx)))
    def forward(self, x):
        return qmm(x.reshape(-1, x.shape[-1]), self.w, self.idx).view(*x.shape[:-1], -1)


def quantize_experts(E, idx_gu, idx_dn):
    E.q_gu = [nvfp4(ocs_w(E.gate_up_proj[e].float(), idx_gu[e])).to(torch.bfloat16) for e in range(NE)]
    E.q_dn = [nvfp4(ocs_w(E.down_proj[e].float(), idx_dn[e])).to(torch.bfloat16) for e in range(NE)]
    E.idx_gu, E.idx_dn = idx_gu, idx_dn
    E.gate_up_proj = None; E.down_proj = None   # free the BF16 originals

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            e = int(e[0]); pos, t = torch.where(mask[e])
            gate, up = qmm(hidden_states[t], self.q_gu[e].float(), self.idx_gu[e]).chunk(2, dim=-1)
            h = qmm(self.act_fn(gate) * up, self.q_dn[e].float(), self.idx_dn[e])
            out.index_add_(0, t, (h * top_k_weights[t, pos, None]).to(out.dtype))
        return out
    E.forward = types.MethodType(forward, E)


t0 = time.time(); res = dict(model=args.tag, hf=args.hf, method=M, rht=args.rht, ncal=args.ncal, Tcal=args.Tcal)
if M != "bf16":
    caps = capture() if M in QDS_M + ("ocs_46",) else None
    if M in QDS_M:
        res["alphas"] = fold_scales(caps)
    use_ocs_attn = M == "ocs_46"; use_ocs_exp = M in ("ocs_46", "qds_ocs_46")
    with torch.no_grad():
        for i, L in enumerate(LAYERS):
            A = L.self_attn
            if use_ocs_attn:
                # after any fold the captured inputs must be rescaled; ocs_46 has no fold, so captured inputs are exact
                Xa, Xo = caps[i]["attn_in"].to(dev), caps[i]["o"].to(dev)
                ia = top_idx(Xa, torch.cat([A.q_proj.weight, A.k_proj.weight, A.v_proj.weight]))
                io = top_idx(Xo, A.o_proj.weight)
            else:
                ia = io = None
            for n in ("q_proj", "k_proj", "v_proj"): setattr(A, n, QLin(getattr(A, n), ia))
            A.o_proj = QLin(A.o_proj, io)
            E = L.mlp.experts
            if use_ocs_exp:
                X = caps[i]["moe_in"].to(dev)
                if M == "qds_ocs_46":   # expert inputs after the moe_in fold are X / s
                    X = X / S_MOE[i]
                r = route(L, X); igu, idn = [], []
                for e in range(NE):
                    tok_e = (r == e).any(-1).nonzero().squeeze(1)
                    if not tok_e.numel(): igu.append(None); idn.append(None); continue
                    Xe = X[tok_e]; igu.append(top_idx(Xe, E.gate_up_proj[e]))
                    g, u = (Xe @ E.gate_up_proj[e].float().T).chunk(2, -1); h = E.act_fn(g) * u
                    idn.append(top_idx(h, E.down_proj[e]))
            else:
                igu = idn = [None] * NE
            quantize_experts(E, igu, idn)
    del caps; torch.cuda.empty_cache()
res["quant_sec"] = time.time() - t0


@torch.no_grad()
def wikitext_ppl(T=2048):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    t = tok(text, return_tensors="pt").input_ids[0]
    n = t.numel() // T; nll = 0.0
    for j in range(n):
        s = t[j * T:(j + 1) * T].to(dev)[None]
        nll += F.cross_entropy(model(input_ids=s, use_cache=False).logits[0, :-1].float(), s[0, 1:], reduction="sum").item()
    return math.exp(nll / (n * (T - 1)))


t1 = time.time(); res["wikitext2_ppl"] = wikitext_ppl(); res["ppl_sec"] = time.time() - t1
print(f"[{args.tag}/{M}] wikitext2 ppl {res['wikitext2_ppl']:.3f} (quant {res['quant_sec']:.0f}s, ppl {res['ppl_sec']:.0f}s)", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
if args.lm:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    r = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=8), tasks=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"],
                                num_fewshot=0, limit=args.lm_limit, bootstrap_iters=0)
    main = {}
    for t, v in r["results"].items():
        main[t] = v.get("acc_norm,none", v.get("acc,none")) if t != "winogrande" else v["acc,none"]
    res["lm_eval_main"] = main; res["avg5"] = sum(main.values()) / len(main)
    print(f"[{args.tag}/{M}] Avg5 {100*res['avg5']:.2f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
