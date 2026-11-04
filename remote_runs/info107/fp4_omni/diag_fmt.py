"""Why does learned QDS (s = 1 init, --ste fmt, no clamp) lose end-to-end quality while its calibration block losses match qdsres?

Loads saved per-layer scales, folds them exactly like ptq8b_s.py, quantizes weights with NVFP4 4/6 (4096-row chunks, as in
eval), and evaluates WikiText-2 test ppl (seq 2048) under four activation-quantization variants:
  tensor_bs1   eval quantizer as in ptq8b_s.py; one 2048-token sequence per call -> per-tensor scale over that sequence
  tensor_bs4   same quantizer, 4 sequences per call -> per-tensor scale shared by 8192 tokens of 4 sequences (lm-eval-like)
  token_bs1    per-token FP32 scale instead of per-tensor (counterfactual: removes cross-token coupling of the E4M3 range)
  bf16act_bs1  activations not quantized (weight-side damage only)
It also counts activation blocks whose E4M3 block scale underflows to 0 (block zeroed) per (layer, group), on 16 calibration
segments (the training data: WikiText-2 train, seed 0 windows) and on the first 16 test sequences (tensor_bs1).
"""
import argparse, json, math, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--scales", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nstat", type=int, default=16)
args = ap.parse_args()
dev = "cuda"; torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
cfg = model.config; Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq; REP = Hq // Hkv
LAYERS = model.model.layers; G = 16
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.], device=dev); TH = (E2M1[1:] + E2M1[:-1]) / 2


def rnd(A, s): return torch.sign(A) * E2M1[torch.bucketize((A.abs() / s).clamp(max=6), TH)] * s


def nvfp4_46(A, per_token=False, stat=None):
    """ptq8b_s.nvfp4(four_six=True) with E4M3 block scales; per_token: FP32 tensor scale per row instead of per call."""
    N, K = A.shape; Ab = A.float().view(N, K // G, G); amax = Ab.abs().amax(-1, keepdim=True)
    gs = (amax.amax((1, 2), keepdim=True) if per_token else amax.max()) / 4.0 / 448.0
    gs = gs.clamp(min=1e-30)
    cs = [((amax / d) / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs for d in (6.0, 4.0)]
    if stat is not None:
        stat[0] += int(((cs[0] == 0) & (amax > 0)).sum()); stat[1] += int((amax > 0).sum())
    cs = [torch.where(c == 0, torch.ones_like(c), c) for c in cs]
    Ah = rnd(Ab, cs[0]); e = (Ah - Ab).pow(2).sum(-1, keepdim=True)
    A2 = rnd(Ab, cs[1]); e2 = (A2 - Ab).pow(2).sum(-1, keepdim=True)
    return torch.where(e2 < e, A2, Ah).view(N, K)


MODE = {"act": "tensor"}; STATS = {"on": None}


class QLinD(torch.nn.Module):
    def __init__(self, lin, key):
        super().__init__(); self.key = key
        W = lin.weight.data
        self.register_buffer("w", torch.cat([nvfp4_46(W[i:i + 4096]) for i in range(0, W.shape[0], 4096)]).to(torch.bfloat16))
        self.register_buffer("b", None if lin.bias is None else lin.bias.detach().clone())

    def forward(self, x):
        xs = x.reshape(-1, x.shape[-1])
        if MODE["act"] == "bf16": xq = xs
        else:
            st = None
            if STATS["on"] is not None: st = STATS["on"].setdefault(self.key, [0, 0])
            xq = torch.cat([nvfp4_46(xs[i:i + 8192], MODE["act"] == "token", st) for i in range(0, xs.shape[0], 8192)]).to(torch.bfloat16)
        return F.linear(xq, self.w, self.b).view(*x.shape[:-1], -1)


def fold(blk, sc):
    a, m = blk.self_attn, blk.mlp
    s_in, s_o, s_mlp, s_dn = (sc[g].to(dev) for g in ("attn_in", "o", "mlp_in", "down"))
    blk.input_layernorm.weight.div_(s_in.to(torch.bfloat16))
    for l in (a.q_proj, a.k_proj, a.v_proj): l.weight.mul_(s_in.to(torch.bfloat16))
    blk.post_attention_layernorm.weight.div_(s_mlp.to(torch.bfloat16))
    for l in (m.gate_proj, m.up_proj): l.weight.mul_(s_mlp.to(torch.bfloat16))
    s_kv = s_o.view(Hkv, REP, hd)[:, 0, :].reshape(-1)
    a.v_proj.weight.div_(s_kv[:, None].to(torch.bfloat16))
    if a.v_proj.bias is not None: a.v_proj.bias.div_(s_kv.to(torch.bfloat16))
    a.o_proj.weight.mul_(s_o.to(torch.bfloat16))
    m.up_proj.weight.div_(s_dn[:, None].to(torch.bfloat16))
    m.down_proj.weight.mul_(s_dn.to(torch.bfloat16))


GROUP = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o", "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down"}
S = torch.load(args.scales, map_location="cpu")["scales"]
for i, blk in enumerate(LAYERS):
    fold(blk, S[i])
    for parent in (blk.self_attn, blk.mlp):
        for n in list(GROUP):
            if hasattr(parent, n): setattr(parent, n, QLinD(getattr(parent, n), (i, GROUP[n])))
torch.cuda.empty_cache()

T = 2048
test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]
nseq = test.numel() // T; test = test[: nseq * T].view(nseq, T)
train = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]), return_tensors="pt").input_ids[0]
gen = torch.Generator().manual_seed(0); st = torch.randint(0, train.numel() - T - 1, (128,), generator=gen)
cal = torch.stack([train[a:a + T] for a in st.tolist()])                   # = ptq8b_s.cal_batch() for seed 0, wikitext


def nll(batch):
    h = model.model(input_ids=batch.to(dev), use_cache=False).last_hidden_state
    return sum(F.cross_entropy(model.lm_head(h[b]).float()[:-1], batch[b, 1:].to(dev), reduction="sum").item() for b in range(batch.shape[0]))


def ppl(bs):
    tot = 0.0
    for j in range(0, nseq - nseq % bs, bs): tot += nll(test[j:j + bs])
    return math.exp(tot / ((nseq - nseq % bs) * (T - 1)))


def underflow(data):
    STATS["on"] = {}; MODE["act"] = "tensor"
    for j in range(args.nstat): nll(data[j:j + 1])
    r = STATS["on"]; STATS["on"] = None
    per = {f"{k[0]}.{k[1]}": v[0] / max(v[1], 1) for k, v in r.items()}
    tot = sum(v[0] for v in r.values()) / max(sum(v[1] for v in r.values()), 1)
    return tot, per


res = dict(hf=args.hf, scales=args.scales); t0 = time.time()
res["underflow_cal"], res["underflow_cal_per"] = underflow(cal)
res["underflow_test"], res["underflow_test_per"] = underflow(test)
for name, mode, bs in (("tensor_bs1", "tensor", 1), ("tensor_bs4", "tensor", 4), ("token_bs1", "token", 1), ("bf16act_bs1", "bf16", 1)):
    MODE["act"] = mode; res[name] = ppl(bs)
    print(f"{name}: ppl {res[name]:.3f} ({time.time()-t0:.0f}s)", flush=True)
print(f"underflowed activation blocks: cal {100*res['underflow_cal']:.3f}%  test {100*res['underflow_test']:.3f}%", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
