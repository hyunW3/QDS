import math, sys, types, json, torch, os
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.expanduser("~/hyun2/fp4_ptq"))
from fp4lib import E2M1, UNIF, BLK
src = open(os.path.expanduser("~/hyun2/fp4_ptq/ptq8b.py")).read()
env = dict(torch=torch, math=math, E2M1=E2M1, UNIF=UNIF, BLK=BLK, dev="cuda", args=types.SimpleNamespace(fmt="nvfp4", method="e2m1_h"))
exec(src[src.index('G = 16 if args.fmt'):src.index('M = args.method')], env)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).cuda().eval()
ids = []
for line in open(os.path.expanduser("~/hyun2/fp4_ptq/dclm_val_head2000.jsonl")):
    ids += tok.encode(json.loads(line)["text"], add_special_tokens=False)
    if len(ids) > 4096: break
x = torch.tensor(ids[:4096]).cuda().view(4, 1024)
got = {}
for li in [0, 1, 35]:
    for n, mod in [("q", m.model.layers[li].self_attn.q_proj), ("down", m.model.layers[li].mlp.down_proj)]:
        mod.register_forward_pre_hook(lambda mm, a, k=(li, n): got.setdefault(k, []).append(a[0].reshape(-1, a[0].shape[-1]).float()))
with torch.no_grad():
    for s in x: m(input_ids=s[None])
for (li, n), xs in got.items():
    X = torch.cat(xs); W = (m.model.layers[li].self_attn.q_proj if n == "q" else m.model.layers[li].mlp.down_proj).weight
    Hs = X.T @ X
    print(li, n, "X absmax %.1f" % X.abs().max().item(), "H finite", torch.isfinite(Hs).all().item(), "diag min %.3g max %.3g" % (Hs.diag().min().item(), Hs.diag().max().item()))
    for kw in [dict(), dict(rot=True)]:
        Q = env["gptq"](W, Hs, **kw)
        Y = X @ W.float().T; e = ((X @ Q.T - Y).pow(2).sum() / Y.pow(2).sum()).item()
        print("   ", kw, "finite", torch.isfinite(Q).all().item(), "out-err %.5f" % e)
