"""When to use QDS vs OCS: join criteria_stats.py (metrics) with placement_test.py (SQNR gains) per (layer, gemm)
-> criteria_tables.md.

Targets (dB, from the E2M1 NVFP4-style probe, 1x16, FP32 block scale):
  g_qds = QDS@4/6 - I@4/6     what QDS adds on top of 4/6
  g_ocs = OCS@4/6 - I@4/6     what OCS adds on top of 4/6
  d     = QDS@4/6 - OCS@4/6   > 0: QDS is the better choice for this GEMM
Reports Spearman correlations per group and a one-metric threshold rule for the sign of d, fitted
leave-one-model-out (threshold chosen on the other models, evaluated on the held-out one).
"""
import json, math
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[1]
SRC = [("qwen2.5-0.5b", "hf"), ("olmo2-1b", "hf"), ("qwen3-1.7b", "hf"), ("olmoe-1b-7b", "moe")]
MET = ["conc_A", "conc_B", "rho_A", "rho_B", "sq_crest", "crest_A", "crest_B", "tok_A", "dfp", "top46"]
db = lambda e: -10 * math.log10(max(e, 1e-30))


def group(layer):
    return "expert" if ".experts." in layer else ("attn" if ".self_attn." in layer else "mlp")


rows = []
for tag, d in SRC:
    pp, pc = D / d / f"place_{tag}.jsonl", D / d / f"crit_{tag}.jsonl"
    if not (pp.exists() and pc.exists()): continue
    P = {(r["layer"], r["gemm"]): r for r in map(json.loads, open(pp)) if "QDS@4/6" in r}
    for c in map(json.loads, open(pc)):
        p = P.get((c["layer"], c["gemm"]))
        if p is None: continue
        r = dict(model=tag, layer=c["layer"], gemm=c["gemm"], grp=group(c["layer"]), top46=p["I|top46"],
                 g_qds=db(p["QDS@4/6"]) - db(p["I@4/6"]), g_ocs=db(p["OCS@4/6"]) - db(p["I@4/6"]))
        r["d"] = r["g_qds"] - r["g_ocs"]
        r.update({m: c[m] for m in MET if m in c})
        rows.append(r)


def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def spear(x, y):
    if len(x) < 5: return float("nan")
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def sel(f):
    return [r for r in rows if f(r)]


out = ["# When to use QDS vs OCS (`criteria_stats.py` + `placement_test.py`)", "",
       "g_qds / g_ocs = dB that QDS / OCS adds on top of 4/6 (E2M1, 1x16); d = g_qds - g_ocs (> 0: QDS better).", ""]
groups = [("dense (all)", lambda r: r["model"] != "olmoe-1b-7b"), ("dense fprop", lambda r: r["model"] != "olmoe-1b-7b" and r["gemm"] == "fprop"),
          ("OLMoE experts", lambda r: r["grp"] == "expert"), ("OLMoE attention", lambda r: r["model"] == "olmoe-1b-7b" and r["grp"] == "attn"),
          ("all", lambda r: True)]
out += ["## Means", "", "| group | n | g_qds | g_ocs | d | share QDS better | " + " | ".join(MET) + " |", "|---|---|---|---|---|---|" + "---|" * len(MET)]
for gname, f in groups:
    R = sel(f)
    if not R: continue
    mean = lambda k: np.mean([r[k] for r in R if k in r])
    out.append(f"| {gname} | {len(R)} | {mean('g_qds'):+.2f} | {mean('g_ocs'):+.2f} | {mean('d'):+.2f} | {100*np.mean([r['d'] > 0 for r in R]):.0f}% | "
               + " | ".join(f"{mean(m):.3f}" for m in MET) + " |")
out += ["", "## Spearman correlation of each metric with the targets (per group)", ""]
for target in ("g_qds", "g_ocs", "d"):
    out += [f"### {target}", "", "| group | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
    for gname, f in groups:
        R = sel(f)
        if not R: continue
        out.append(f"| {gname} | " + " | ".join(f"{spear([r[m] for r in R], [r[target] for r in R]):+.2f}" for m in MET) + " |")
    out.append("")

# one-metric threshold rule for the QDS-vs-OCS choice, leave-one-model-out
models = sorted({r["model"] for r in rows})


def fit(R, m):
    """best (threshold, direction): choose QDS if dir*(x - t) > 0; maximize the mean realized gain."""
    xs = np.array([r[m] for r in R]); best = None
    for t in np.unique(np.quantile(xs, np.linspace(0.02, 0.98, 49))):
        for sgn in (1, -1):
            g = np.mean([r["g_qds"] if sgn * (r[m] - t) > 0 else r["g_ocs"] for r in R])
            if best is None or g > best[0]: best = (g, t, sgn)
    return best


out += ["## One-metric rule (choose QDS if the metric is on one side of a threshold, else OCS), leave-one-model-out", "",
        "Realized mean gain over 4/6 (dB) on the held-out model's GEMMs; 'acc' = share of GEMMs where the rule picks the better one.", "",
        "| metric | " + " | ".join(f"{m} gain / acc" for m in models) + " | mean gain |", "|---|" + "---|" * len(models) + "---|"]
base = {}
for m_out in models:
    T = [r for r in rows if r["model"] == m_out]
    base[m_out] = dict(qds=np.mean([r["g_qds"] for r in T]), ocs=np.mean([r["g_ocs"] for r in T]),
                       oracle=np.mean([max(r["g_qds"], r["g_ocs"]) for r in T]))
for name in ("always QDS", "always OCS", "oracle"):
    k = {"always QDS": "qds", "always OCS": "ocs", "oracle": "oracle"}[name]
    out.append(f"| {name} | " + " | ".join(f"{base[m][k]:+.2f}" for m in models) + f" | {np.mean([base[m][k] for m in models]):+.2f} |")
for met in MET:
    cells, gains = [], []
    for m_out in models:
        Rtr = [r for r in rows if r["model"] != m_out and met in r]; T = [r for r in rows if r["model"] == m_out and met in r]
        if not Rtr or not T: cells.append("-"); continue
        _, t, sgn = fit(Rtr, met)
        pick_q = [sgn * (r[met] - t) > 0 for r in T]
        g = np.mean([r["g_qds"] if q else r["g_ocs"] for r, q in zip(T, pick_q)])
        acc = np.mean([q == (r["d"] > 0) for r, q in zip(T, pick_q)])
        cells.append(f"{g:+.2f} / {100*acc:.0f}% (t={t:.3g}, {'>' if sgn > 0 else '<'})"); gains.append(g)
    out.append(f"| {met} | " + " | ".join(cells) + f" | {np.mean(gains):+.2f} |")
(D / "scaling/criteria_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
