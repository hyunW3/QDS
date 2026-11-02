import math, sys, types, json, torch, os
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.expanduser("~/hyun2/fp4_ptq"))
from fp4lib import E2M1, UNIF, BLK
src = open(os.path.expanduser("~/hyun2/fp4_ptq/ptq8b.py")).read()
env = dict(torch=torch, math=math, E2M1=E2M1, UNIF=UNIF, BLK=BLK, dev="cuda", args=types.SimpleNamespace(fmt="nvfp4", method="e2m1"))
exec(src[src.index('G = 16 if args.fmt'):src.index('M = args.method')], env)
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "meta-llama/Meta-Llama-3-8B"
tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
ids = []
for line in open(os.path.expanduser("~/hyun2/fp4_ptq/dclm_val_head2000.jsonl")):
    ids += [tok.bos_token_id] + tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) > 140 * 1024: break
seqs = torch.tensor(ids[:140 * 1024]).cuda().view(140, 1024)
LI = int(sys.argv[1]) if len(sys.argv) > 1 else 1
mods = {"q": m.model.layers[LI].self_attn.q_proj, "down": m.model.layers[LI].mlp.down_proj}
H = {k: 0 for k in mods}; Hs8 = {k: 0 for k in mods}; held = {k: [] for k in mods}; cnt = [0]
def hook(k):
    def f(mod, a):
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        if cnt[0] < 128: H[k] = H[k] + x.T @ x
        if cnt[0] < 8: Hs8[k] = Hs8[k] + x.T @ x
        if cnt[0] >= 136: held[k].append(x)
    return f
for k, mod in mods.items(): mod.register_forward_pre_hook(hook(k))
with torch.no_grad():
    for i, s in enumerate(seqs):
        cnt[0] = i; m(input_ids=s[None])
for k, mod in mods.items():
    W = mod.weight; Xh = torch.cat(held[k]); Y = Xh @ W.float().T; n = Y.pow(2).sum()
    for kw in [dict(), dict(four_six=True), dict(rot=True)]:
        r = env["nvfp4"](W, **kw); e_r = ((Xh @ r.T - Y).pow(2).sum() / n).item()
        g8 = env["gptq"](W, Hs8[k], **kw); e8 = ((Xh @ g8.T - Y).pow(2).sum() / n).item()
        g128 = env["gptq"](W, H[k], **kw); e128 = ((Xh @ g128.T - Y).pow(2).sum() / n).item()
        print(f"layer {LI} {k:5s} {str(kw):22s} held-out W-only out-err  RTN {e_r:.5f}  GPTQ(8k tok) {e8:.5f}  GPTQ(131k tok) {e128:.5f}", flush=True)
