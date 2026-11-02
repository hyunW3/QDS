"""Aggregate Stage-0 JSONL into markdown tables."""
import json, math, sys
from collections import defaultdict
from statistics import mean

ROLE = {("fprop", "A"): "fwd_x", ("fprop", "B"): "fwd_w", ("dgrad", "A"): "bwd_dy(dgrad)",
        ("dgrad", "B"): "bwd_wT(dgrad)", ("wgrad", "A"): "bwd_dyT(wgrad)", ("wgrad", "B"): "bwd_xT(wgrad)"}
db = lambda x: -10 * math.log10(max(x, 1e-30))
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
models = list(dict.fromkeys(r["model"] for r in rows))
HFMAP = {"q_proj": "wq", "k_proj": "wk", "v_proj": "wv", "o_proj": "wo", "gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
mt = lambda r: HFMAP.get(r["layer"].split(".")[-1], r["layer"].split(".")[-1])


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


out = []
P = out.append
for m in models:
    P(f"\n## {m}\n")
    # ---------- T1/T2: tensor level, RTN on every operand (dy_round=rtn)
    for sf in ["fp32", "e4m3"]:
        P(f"\n### Tensor level, RTN, {sf} block scale (mean over {len(set(r['layer'] for r in sel(model=m)))} linears)\n")
        P("| operand | E2M1 SQNR I / H (Δ) | UNIF SQNR I / H (Δ) | winner I / H | E2M1 α I / H | UNIF α I / H | E2M1 b_G I / H | B_eff E2M1 I / H |")
        P("|---|---|---|---|---|---|---|---|")
        for (gemm, role), name in ROLE.items():
            v = {}
            for g in ["E2M1", "UNIF"]:
                for tr in ["I", "H"]:
                    rs = sel(model=m, gemm=gemm, grid=g, tr=tr, scale=sf, dy_round="rtn")
                    v[g, tr] = dict(s=mean(db(r[role]["nmse"]) for r in rs), a=mean(r[role]["alpha"] for r in rs),
                                    b=mean(r[role]["bG"] for r in rs), e=mean(r[role]["beff"] for r in rs))
            w = lambda tr: "E2M1" if v["E2M1", tr]["s"] > v["UNIF", tr]["s"] else "UNIF"
            P(f"| {name} | {v['E2M1','I']['s']:.2f} / {v['E2M1','H']['s']:.2f} ({v['E2M1','H']['s']-v['E2M1','I']['s']:+.2f}) "
              f"| {v['UNIF','I']['s']:.2f} / {v['UNIF','H']['s']:.2f} ({v['UNIF','H']['s']-v['UNIF','I']['s']:+.2f}) "
              f"| {w('I')} / {w('H')} | {v['E2M1','I']['a']:.4f} / {v['E2M1','H']['a']:.4f} "
              f"| {v['UNIF','I']['a']:.4f} / {v['UNIF','H']['a']:.4f} | {v['E2M1','I']['b']:+.4f} / {v['E2M1','H']['b']:+.4f} "
              f"| {v['E2M1','I']['e']:.3f} / {v['E2M1','H']['e']:.3f} |")
    # ---------- per module type, fwd_x (activation outliers live here)
    P("\n### fwd_x by module type (RTN, fp32 scale): E2M1 α I / H, SQNR E2M1 vs UNIF after RHT, α range over layers (H)\n")
    P("| module | E2M1 α I | E2M1 α H | α_H min..max | SQNR_H E2M1 | SQNR_H UNIF |")
    P("|---|---|---|---|---|---|")
    for mod in ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]:
        rI = [r for r in sel(model=m, gemm="fprop", grid="E2M1", tr="I", scale="fp32", dy_round="rtn") if mt(r) == mod]
        rH = [r for r in sel(model=m, gemm="fprop", grid="E2M1", tr="H", scale="fp32", dy_round="rtn") if mt(r) == mod]
        uH = [r for r in sel(model=m, gemm="fprop", grid="UNIF", tr="H", scale="fp32", dy_round="rtn") if mt(r) == mod]
        if not rI:
            continue
        aH = [r["A"]["alpha"] for r in rH]
        P(f"| {mod} | {mean(r['A']['alpha'] for r in rI):.4f} | {mean(aH):.4f} | {min(aH):.4f}..{max(aH):.4f} "
          f"| {mean(db(r['A']['nmse']) for r in rH):.2f} | {mean(db(r['A']['nmse']) for r in uH):.2f} |")
    # ---------- T3: GEMM output level
    P("\n### GEMM output level (mean over linears): output SQNR dB, α_Y, and SQNR after per-tensor 1/α correction\n")
    P("| GEMM | grid | T | scale | dY rnd | SQNR_Y | α_Y | SQNR_Y +αcorr | gain |")
    P("|---|---|---|---|---|---|---|---|---|")
    for gemm in ["fprop", "dgrad", "wgrad"]:
        for g in ["E2M1", "UNIF"]:
            for tr in ["I", "H"]:
                for sf in ["fp32", "e4m3"]:
                    for dr in (["rtn"] if gemm == "fprop" else ["rtn", "sr"]):
                        rs = sel(model=m, gemm=gemm, grid=g, tr=tr, scale=sf, dy_round=dr)
                        s = mean(db(r["g_nmse"]) for r in rs); sc = mean(db(r["g_nmse_acorr"]) for r in rs)
                        P(f"| {gemm} | {g} | {tr} | {sf} | {dr} | {s:.2f} | {mean(r['g_alpha'] for r in rs):.4f} | {sc:.2f} | {sc-s:+.2f} |")
print("\n".join(out))
