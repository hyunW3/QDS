"""Both-sided fold: v_proj gets columns x s_attn_in AND rows / s_kv; up_proj gets columns x s_mlp_in AND rows / s_down.
loss treats those layers' outputs as FP, so their NVFP4 quantization after the fold is never seen. Measure it on CPU."""
import torch, glob, json
from safetensors import safe_open
torch.set_num_threads(16); torch.set_grad_enabled(False)
snap = glob.glob("/mnt/ssd2/hyun2/hf_tmp/hub/models--meta-llama--Llama-3.1-8B/snapshots/*/")[0]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(snap + "config.json")); Hkv = cfg["num_key_value_heads"]; hd = cfg["hidden_size"] // cfg["num_attention_heads"]; REP = cfg["num_attention_heads"] // Hkv
def W(n):
    with safe_open(snap + idx[n], "pt") as f: return f.get_tensor(n).float()
G = 16; E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.]); TH = (E2M1[1:] + E2M1[:-1]) / 2
def rnd(A, s): return torch.sign(A) * E2M1[torch.bucketize((A.abs() / s).clamp(max=6), TH)] * s
def q46(A):
    N, K = A.shape; Ab = A.view(N, K // G, G); amax = Ab.abs().amax(-1, keepdim=True); gs = (amax.max() / 4 / 448).clamp(min=1e-30)
    cs = [((amax / d) / gs).clamp(max=448.).to(torch.float8_e4m3fn).float() * gs for d in (6., 4.)]
    z = ((cs[0] == 0) & (amax > 0)).float().mean().item(); cs = [torch.where(c == 0, torch.ones_like(c), c) for c in cs]
    a = rnd(Ab, cs[0]); b = rnd(Ab, cs[1])
    return torch.where((b - Ab).pow(2).sum(-1, keepdim=True) < (a - Ab).pow(2).sum(-1, keepdim=True), b, a).view(N, K), z
def qchunk(A):   # eval: 4096-row chunks
    outs = [q46(A[i:i + 4096]) for i in range(0, A.shape[0], 4096)]
    return torch.cat([o[0] for o in outs]), sum(o[1] * min(4096, A.shape[0] - i) for o, i in zip(outs, range(0, A.shape[0], 4096))) / A.shape[0]
D = "/mnt/ssd2/hyun2/experiment/QDS-repo/.claude/worktrees/qds-search-arms/remote_runs/info107/fp4_omni/out_s/"
RUNS = {"noclamp": "llama31-8b_nvfp4_diagomni_46_wt_fmt_noclamp", "qdsres": "llama31-8b_nvfp4_qdsres_46_wt"}
SC = {k: torch.load(D + v + ".scales.pt", map_location="cpu")["scales"] for k, v in RUNS.items()}
print("layer run     v_proj: relerr zero%  | up_proj: relerr zero%   (unfolded: v, up)")
tot = {k: [0, 0, 0, 0] for k in RUNS}
for i in range(32):
    v = W(f"model.layers.{i}.self_attn.v_proj.weight"); up = W(f"model.layers.{i}.mlp.up_proj.weight")
    base = [((qchunk(t)[0] - t).norm() / t.norm()).item() for t in (v, up)]
    for k in RUNS:
        s_o = SC[k][i]["o"].float(); s_kv = s_o.view(Hkv, REP, hd)[:, 0, :].reshape(-1); s_dn = SC[k][i]["down"].float()
        r = []
        for t, s, c in ((v, s_kv, SC[k][i]["attn_in"].float()), (up, s_dn, SC[k][i]["mlp_in"].float())):
            q, z = qchunk(t * c[None, :] / s[:, None]); r += [((q * s[:, None] / c[None, :] - t).norm() / t.norm()).item(), 100 * z]
        for j in range(4): tot[k][j] += r[j] / 32
        if i in (0, 1, 2, 15, 23, 31) or r[0] > 0.2 or r[2] > 0.2:
            print(f"{i:5d} {k:7s} {r[0]:13.3f} {r[1]:6.2f} | {r[2]:15.3f} {r[3]:6.2f}   ({base[0]:.3f}, {base[1]:.3f})", flush=True)
for k, t in tot.items(): print(f"mean over layers {k}: v_proj relerr {t[0]:.3f} zero {t[1]:.2f}% | up_proj relerr {t[2]:.3f} zero {t[3]:.2f}%")
